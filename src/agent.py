"""LangGraph agent for interactive Genie query refinement."""
from typing import Literal, TypedDict, Optional
import os
import mlflow
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import GenieMessage
from databricks_langchain import ChatDatabricks, CheckpointSaver


class AgentState(TypedDict):
    """State for the Genie question clarification agent."""
    user_question: str
    conversation_id: str | None
    current_message: GenieMessage | None
    assumptions: str | None
    user_approved: bool
    feedback: str | None
    final_result: dict | None


class GenieAgent:
    """LangGraph agent that interactively refines Genie queries."""
    
    def __init__(
        self,
        space_id: str,
        databricks_profile: Optional[str] = None,
        workspace_client: Optional[WorkspaceClient] = None,
        lakebase_instance: Optional[str] = None,
        use_lakebase: bool = True,
    ):
        """Initialize the agent with Databricks configuration.
        
        Args:
            space_id: Genie Space ID
            databricks_profile: Databricks CLI profile name (optional)
            workspace_client: Pre-configured WorkspaceClient (optional)
            lakebase_instance: Lakebase instance name for checkpoint storage
            use_lakebase: If True, use Lakebase for checkpoints; if False, use in-memory
        """
        if workspace_client is not None:
            self.client = workspace_client
        elif databricks_profile:
            self.client = WorkspaceClient(profile=databricks_profile)
        else:
            self.client = WorkspaceClient()
        self.space_id = space_id
        self.lakebase_instance = lakebase_instance or os.environ.get("LAKEBASE_INSTANCE_NAME")
        self.use_lakebase = use_lakebase and self.lakebase_instance is not None
        self.llm = ChatDatabricks(
            endpoint="databricks-gpt-5-2",
            temperature=0.1
        )
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine with checkpointing."""
        workflow = StateGraph(AgentState)

        workflow.add_node("get_initial_plan", self._get_initial_plan)
        workflow.add_node("review_with_llm", self._review_with_llm)
        workflow.add_node("approval_gate", self._approval_gate)
        workflow.add_node("gather_feedback", self._gather_feedback)
        workflow.add_node("refine_plan", self._refine_plan)
        workflow.add_node("show_results", self._show_results)

        workflow.set_entry_point("get_initial_plan")

        # Conditional edge: if get_initial_plan failed, go to END; otherwise continue
        def route_after_initial_plan(state: AgentState) -> str:
            if state.get("final_result") and state["final_result"].get("status") == "failed":
                return END
            return "review_with_llm"
        
        workflow.add_conditional_edges("get_initial_plan", route_after_initial_plan)
        workflow.add_edge("review_with_llm", "approval_gate")
        workflow.add_edge("gather_feedback", "refine_plan")
        workflow.add_edge("refine_plan", "review_with_llm")
        workflow.add_edge("show_results", END)

        if self.use_lakebase:
            checkpointer = CheckpointSaver(
                instance_name=self.lakebase_instance,
                workspace_client=self.client,
            )
            checkpointer.setup()
        else:
            checkpointer = MemorySaver()
        self.checkpointer = checkpointer
        return workflow.compile(checkpointer=checkpointer)
    
    @mlflow.trace(span_type="AGENT", name="get_initial_plan")
    def _get_initial_plan(self, state: AgentState) -> AgentState:
        """Get initial plan from Genie based on user question.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with conversation and message
        """
        from databricks.sdk.errors import OperationFailed
        
        try:
            # Start conversation with Genie
            message = self.client.genie.start_conversation_and_wait(
                space_id=self.space_id,
                content=state["user_question"]
            )
            
            return {
                **state,
                "conversation_id": message.conversation_id,
                "current_message": message
            }
        except OperationFailed as e:
            # Genie couldn't process the query - return helpful error state
            # This happens for non-data queries like "Hello, world!"
            return {
                **state,
                "conversation_id": None,
                "current_message": None,
                "final_result": {
                    "status": "failed",
                    "error": str(e),
                    "summary": (
                        "I couldn't process that query. I'm designed to answer questions about your data. "
                        "Please ask a specific question about your database, like 'What are the top 10 customers by revenue?' "
                        "or 'Show me sales trends over the last year.'"
                    ),
                }
            }
    
    @mlflow.trace(span_type="LLM", name="review_with_llm")
    def _review_with_llm(self, state: AgentState) -> AgentState:
        """Use LLM to extract top 3 assumptions from the query plan.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with assumptions
        """
        message = state["current_message"]
        if not message:
            return {**state, "assumptions": None}
        
        # Extract description and query from message
        description = self._get_from_query_attachment(message, 'description')
        query = self._get_from_query_attachment(message, 'query')
        
        if not description or not query:
            return {**state, "assumptions": None}
        
        prompt = f"""Given this database query interpretation and SQL plan, identify the top 3 most important assumptions being made.

INTERPRETATION:
{description}

SQL QUERY:
{query}

Please list exactly 3 key assumptions that are being made about:
- What data is being used
- How the question is being interpreted
- What filters or constraints are applied
- What calculations or aggregations are performed

Format your response as a numbered list (1., 2., 3.) with each assumption on its own line. Be concise and specific."""

        response = self.llm.invoke(prompt)
        
        assumptions = response.content
        
        return {
            **state,
            "assumptions": assumptions
        }
    
    @mlflow.trace(span_type="AGENT", name="approval_gate")
    def _approval_gate(self, state: AgentState) -> Command[Literal["show_results", "gather_feedback"]]:
        """Present plan and ask for approval (yes/no only).
        
        This node interrupts to show the plan and waits for approval decision.
        If not approved, routes to gather_feedback to get detailed feedback.
        
        Args:
            state: Current agent state
            
        Returns:
            Command routing to show_results or gather_feedback
        """
        message = state["current_message"]
        description = self._get_from_query_attachment(message, 'description')
        query = self._get_from_query_attachment(message, 'query')
        
        plan_text = f"""
        The following plan was constructed based on your question, with our assumptions highlighted.
        Does this appear correct? Answer 'yes' and we'll execute the plan or 'no' to refine based on your input.

        INTERPRETATION:
        {description}

        SQL QUERY:
        {query}

        ASSUMPTIONS:
        {state['assumptions']}
        """

        plan_to_present = {"question": plan_text}
        
        is_approved = interrupt(plan_to_present)
        
        state["user_approved"] = is_approved
        
        if is_approved.lower() in {"yes", "y", "true", "t", "1"}:
            return Command(goto="show_results", update=state)
        else:
            return Command(goto="gather_feedback", update=state)
    
    @mlflow.trace(span_type="AGENT", name="gather_feedback")
    def _gather_feedback(self, state: AgentState) -> Command[Literal["refine_plan"]]:
        """Gather feedback from user after they reject the plan.
        
        This node interrupts to ask what changes the user wants.
        Always routes to refine_plan after getting feedback.
        
        Args:
            state: Current agent state
            
        Returns:
            Command routing to refine_plan with feedback
        """

        feedback = interrupt("What would you like to change or add to the plan?")
        state["feedback"] = feedback
        
        return Command(goto="refine_plan", update=state)
    
    @mlflow.trace(span_type="AGENT", name="refine_plan")
    def _refine_plan(self, state: AgentState) -> AgentState:
        """Refine the plan based on user feedback.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with refined message
        """
        message = self.client.genie.create_message_and_wait(
            space_id=self.space_id,
            conversation_id=state["conversation_id"],
            content=state["feedback"]
        )
        
        return {
            **state,
            "current_message": message
        }
    
    @mlflow.trace(span_type="AGENT", name="show_results")
    def _show_results(self, state: AgentState) -> AgentState:
        """Show results from the already-executed query.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with final results
        """
        message = state["current_message"]

        summary = self._extract_text_summary(message)
        results = self._extract_results_from_message(message)
        
        result = {
            "summary": summary,
            "results": results
        }
        
        return {
            **state,
            "final_result": result
        }
    
    def _get_from_query_attachment(self, message: GenieMessage, attr_name: str) -> str | None:
        """Extract an attribute from the query attachment.
        
        Args:
            message: GenieMessage object
            attr_name: Name of the attribute to extract (e.g. 'description', 'query')
            
        Returns:
            Attribute value or None
        """
        if not message:
            return None
        try:
            for attachment in message.attachments:
                if attachment.query is not None:
                    return getattr(attachment.query, attr_name, None)
        except (AttributeError, TypeError):
            pass
        return None
    
    def _extract_text_summary(self, message: GenieMessage) -> str | None:
        """Extract plain text summary from message attachments.
        
        Args:
            message: GenieMessage object
            
        Returns:
            Plain text summary if available, None otherwise
        """
        try:
            if message.attachments:
                for attachment in message.attachments:
                    if attachment.text is not None:
                        return getattr(attachment.text, 'content', None)
        except (AttributeError, TypeError):
            pass
        return None
    
    def _extract_results_from_message(self, message: GenieMessage) -> dict:
        """Extract query results from message query_result and statement execution.
        
        Args:
            message: GenieMessage object with query results
            
        Returns:
            dict with columns and data
        """
        columns = []
        rows = []

        statement_id = None
        try:
            if message.query_result:
                statement_id = getattr(message.query_result, 'statement_id', None)
        except (AttributeError, TypeError):
            pass
        
        if not statement_id:
            return {"columns": columns, "rows": rows, "row_count": 0}
        
        try:
            stmt = self.client.statement_execution.get_statement(statement_id)
            return self._extract_results(stmt)
        except Exception:
            return {"columns": columns, "rows": rows, "row_count": 0}
    
    def _extract_results(self, statement_response) -> dict:
        """Extract results from a statement response.
        
        Args:
            statement_response: StatementResponse object
            
        Returns:
            dict with columns and data
        """
        columns = []
        rows = []

        try:
            schema = statement_response.manifest.schema
            columns = [col.name for col in schema.columns]
        except (AttributeError, TypeError):
            pass

        statement_id = getattr(statement_response, 'statement_id', None)
        if not statement_id:
            return {"columns": columns, "rows": rows, "row_count": 0}

        try:
            result = statement_response.result

            if result.data_array is not None:
                rows = result.data_array
            else:
                chunks = statement_response.manifest.chunks
                if chunks:
                    for chunk_info in chunks:
                        chunk_index = getattr(chunk_info, 'chunk_index', 0)
                        
                        try:
                            chunk_data = self.client.statement_execution.get_statement_result_chunk_n(
                                statement_id=statement_id,
                                chunk_index=chunk_index
                            )
                            
                            if chunk_data.data_array:
                                rows.extend(chunk_data.data_array)
                        except Exception:
                            pass 
        except (AttributeError, TypeError):
            pass
        
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows)
        }
    
    @mlflow.trace(span_type="AGENT", name="genie_agent_run")
    def run(self, user_question: str, thread_id: str = "default") -> dict:
        """Run the agent with a user question (will interrupt for human input).
        
        Args:
            user_question: The question to ask Genie
            thread_id: Unique ID for this conversation thread
            
        Returns:
            Dict with state including __interrupt__ key if paused
        """
        initial_state: AgentState = {
            "user_question": user_question,
            "conversation_id": None,
            "current_message": None,
            "assumptions": None,
            "user_approved": False,
            "feedback": None,
            "final_result": None
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        result = self.graph.invoke(initial_state, config)
        
        return result

    def resume(self, user_response: dict, thread_id: str = "default") -> dict:
        """Resume the agent after an interrupt with user input."""
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.invoke(Command(resume=user_response), config)

    def close(self) -> None:
        """Close resources such as the Lakebase connection pool."""
        if self.use_lakebase and hasattr(self, "checkpointer"):
            try:
                self.checkpointer.__exit__(None, None, None)
            except Exception:
                pass


def create_graph():
    """Factory function to create the Genie agent graph for LangGraph Studio.
    
    This function is called by LangGraph Studio/API to instantiate the graph.
    Configuration is read from environment variables.
    
    Environment variables required:
    - DATABRICKS_PROFILE: Databricks CLI profile name (default: "FE-EAST")
    - GENIE_SPACE_ID: ID of the Genie Space (required)
    - LAKEBASE_INSTANCE_NAME: Lakebase instance name (optional)
    
    Returns:
        Compiled LangGraph graph ready for execution
    """
    profile = os.getenv("DATABRICKS_PROFILE")
    space_id = os.getenv("GENIE_SPACE_ID")
    
    if not space_id:
        raise ValueError(
            "GENIE_SPACE_ID environment variable is required. "
            "Find this in your Genie Space URL."
        )
    
    agent = GenieAgent(
        space_id=space_id,
        databricks_profile=profile,
        lakebase_instance=os.getenv("LAKEBASE_INSTANCE_NAME"),
        use_lakebase=True,
    )
    return agent.graph

