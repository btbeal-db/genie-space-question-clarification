"""LangGraph agent for interactive Genie query refinement.

This agent uses the GenieAgent from databricks_langchain.genie to query
Genie spaces, with human-in-the-loop approval for query plans.
"""
from typing import Literal, TypedDict, Optional, List, Any
import os
import mlflow
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from databricks_langchain import ChatDatabricks, CheckpointSaver
from databricks_langchain.genie import GenieAgent as DatabricksGenieAgent


class AgentState(TypedDict):
    """State for the Genie question clarification agent.
    
    Uses a message-based format compatible with GenieAgent.
    """
    messages: List[BaseMessage]           # Chat history for Genie
    conversation_id: str | None           # Genie conversation ID (for follow-ups)
    query_reasoning: str | None           # Genie's interpretation of the query
    query_sql: str | None                 # Generated SQL
    query_result: str | None              # Query results (markdown)
    assumptions: str | None               # LLM-extracted assumptions
    user_approved: bool                   # Whether user approved the plan
    final_result: dict | None             # Final output


class GenieQueryAgent:
    """LangGraph agent that interactively refines Genie queries.
    
    Uses GenieAgent from databricks_langchain for Genie API interactions,
    with human-in-the-loop approval for query plans.
    """
    
    def __init__(
        self,
        space_id: str,
        lakebase_instance: Optional[str] = None,
        use_lakebase: bool = True,
    ):
        """Initialize the agent.
        
        Args:
            space_id: Genie Space ID
            lakebase_instance: Lakebase instance name for checkpoint storage
            use_lakebase: If True, use Lakebase for checkpoints; if False, use in-memory
        """
        self.space_id = space_id
        self.lakebase_instance = lakebase_instance or os.environ.get("LAKEBASE_INSTANCE_NAME")
        self.use_lakebase = use_lakebase and self.lakebase_instance is not None

        self.genie = DatabricksGenieAgent(
            genie_space_id=space_id,
            genie_agent_name="DataAnalyst",
            include_context=True,
        )
        
        self.llm = ChatDatabricks(
            endpoint="databricks-claude-sonnet-4",
            temperature=0.1
        )
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine with checkpointing."""
        workflow = StateGraph(AgentState)

        workflow.add_node("query_genie", self._query_genie)
        workflow.add_node("extract_assumptions", self._extract_assumptions)
        workflow.add_node("approval_gate", self._approval_gate)
        workflow.add_node("gather_feedback", self._gather_feedback)
        workflow.add_node("refine_query", self._refine_query)
        workflow.add_node("finalize_results", self._finalize_results)

        workflow.set_entry_point("query_genie")

        def route_after_query(state: AgentState) -> str:
            if state.get("final_result") and state["final_result"].get("status") == "failed":
                return END
            return "extract_assumptions"
        
        workflow.add_conditional_edges("query_genie", route_after_query)
        workflow.add_edge("extract_assumptions", "approval_gate")
        workflow.add_edge("gather_feedback", "refine_query")
        workflow.add_edge("refine_query", "extract_assumptions")
        workflow.add_edge("finalize_results", END)

        if self.use_lakebase:
            checkpointer = CheckpointSaver(
                instance_name=self.lakebase_instance,
            )
            checkpointer.setup()
        else:
            checkpointer = MemorySaver()
        
        self.checkpointer = checkpointer
        return workflow.compile(checkpointer=checkpointer)
    
    @mlflow.trace(span_type="AGENT", name="query_genie")
    def _query_genie(self, state: AgentState) -> AgentState:
        """Query Genie with the user's question.
        
        Uses GenieAgent which handles all the API complexity including waiting.
        """
        try:
            genie_input = {
                "messages": state["messages"],
            }

            if state.get("conversation_id"):
                genie_input["conversation_id"] = state["conversation_id"]

            result = self.genie.invoke(genie_input)
            query_reasoning = None
            query_sql = None
            query_result = None
            
            for msg in result.get("messages", []):
                if hasattr(msg, "name"):
                    if msg.name == "query_reasoning":
                        query_reasoning = msg.content
                    elif msg.name == "query_sql":
                        query_sql = msg.content
                    elif msg.name == "query_result":
                        query_result = msg.content
            
            return {
                **state,
                "conversation_id": result.get("conversation_id"),
                "query_reasoning": query_reasoning,
                "query_sql": query_sql,
                "query_result": query_result,
            }
            
        except Exception as e:
            error_msg = str(e)
            return {
                **state,
                "final_result": {
                    "status": "failed",
                    "error": error_msg,
                    "summary": (
                        "I couldn't process that query. I'm designed to answer questions about your data. "
                        "Please ask a specific question about your database, like 'What are the top 10 customers by revenue?' "
                        "or 'Show me sales trends over the last year.'"
                    ),
                }
            }
    
    @mlflow.trace(span_type="LLM", name="extract_assumptions")
    def _extract_assumptions(self, state: AgentState) -> AgentState:
        """Use LLM to extract top 3 assumptions from the query plan."""
        reasoning = state.get("query_reasoning")
        sql = state.get("query_sql")
        
        if not reasoning or not sql:
            return {**state, "assumptions": "Unable to extract assumptions - no query plan available."}
        
        prompt = f"""Given this database query interpretation and SQL plan, identify the top 3 most important assumptions being made.

INTERPRETATION:
{reasoning}

SQL QUERY:
{sql}

Please list exactly 3 key assumptions that are being made about:
- What data is being used
- How the question is being interpreted
- What filters or constraints are applied
- What calculations or aggregations are performed

Format your response as a numbered list (1., 2., 3.) with each assumption on its own line. Be concise and specific."""

        response = self.llm.invoke(prompt)
        
        return {
            **state,
            "assumptions": response.content
        }
    
    @mlflow.trace(span_type="AGENT", name="approval_gate")
    def _approval_gate(self, state: AgentState) -> Command[Literal["finalize_results", "gather_feedback"]]:
        """Present plan and ask for approval.
        
        This node interrupts to show the plan and waits for approval decision.
        """
        plan_text = f"""
The following plan was constructed based on your question, with assumptions highlighted.
Does this appear correct? Answer 'yes' to see results or 'no' to refine.

**INTERPRETATION:**
{state.get('query_reasoning', 'N/A')}

**SQL QUERY:**
```sql
{state.get('query_sql', 'N/A')}
```

**ASSUMPTIONS:**
{state.get('assumptions', 'N/A')}
"""
        
        user_response = interrupt({"question": plan_text})
        
        if isinstance(user_response, str):
            is_approved = user_response.lower() in {"yes", "y", "true", "t", "1", "ok", "okay", "approve", "approved"}
        else:
            is_approved = False
        
        if is_approved:
            return Command(goto="finalize_results", update={**state, "user_approved": True})
        else:
            return Command(goto="gather_feedback", update={**state, "user_approved": False})
    
    @mlflow.trace(span_type="AGENT", name="gather_feedback")
    def _gather_feedback(self, state: AgentState) -> Command[Literal["refine_query"]]:
        """Gather feedback from user after they reject the plan."""
        feedback = interrupt("What would you like to change or add to the plan?")
        
        new_messages = list(state.get("messages", []))
        new_messages.append(HumanMessage(content=feedback))
        
        return Command(goto="refine_query", update={**state, "messages": new_messages})
    
    @mlflow.trace(span_type="AGENT", name="refine_query")
    def _refine_query(self, state: AgentState) -> AgentState:
        """Refine the query based on user feedback.
        
        Calls Genie again with the follow-up message using the same conversation_id.
        """
        try:
            genie_input = {
                "messages": state["messages"],
                "conversation_id": state.get("conversation_id"),
            }
            
            result = self.genie.invoke(genie_input)
            
            query_reasoning = None
            query_sql = None
            query_result = None
            
            for msg in result.get("messages", []):
                if hasattr(msg, "name"):
                    if msg.name == "query_reasoning":
                        query_reasoning = msg.content
                    elif msg.name == "query_sql":
                        query_sql = msg.content
                    elif msg.name == "query_result":
                        query_result = msg.content
            
            return {
                **state,
                "conversation_id": result.get("conversation_id"),
                "query_reasoning": query_reasoning,
                "query_sql": query_sql,
                "query_result": query_result,
            }
        except Exception as e:
            return {
                **state,
                "final_result": {
                    "status": "failed",
                    "error": str(e),
                    "summary": f"Failed to refine query: {str(e)}"
                }
            }
    
    @mlflow.trace(span_type="AGENT", name="finalize_results")
    def _finalize_results(self, state: AgentState) -> AgentState:
        """Finalize and return the query results."""
        return {
            **state,
            "final_result": {
                "status": "complete",
                "summary": state.get("query_result", "Query completed."),
                "query_reasoning": state.get("query_reasoning"),
                "query_sql": state.get("query_sql"),
            }
        }
    
    def run(self, user_question: str, thread_id: str = "default") -> dict:
        """Run the agent with a user question.
        
        Args:
            user_question: The question to ask Genie
            thread_id: Unique ID for this conversation thread
            
        Returns:
            Dict with state including interrupt if paused
        """
        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_question)],
            "conversation_id": None,
            "query_reasoning": None,
            "query_sql": None,
            "query_result": None,
            "assumptions": None,
            "user_approved": False,
            "final_result": None,
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.invoke(initial_state, config)

    def resume(self, user_response: str, thread_id: str = "default") -> dict:
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

# This is useful for the langsmith local debugging/visualization
def create_graph(model_config: dict = None):
    """Factory function to create the Genie agent graph.
    
    Args:
        model_config: Configuration dict with genie_space_id and lakebase_instance_name.
                     Falls back to environment variables if not provided.
    
    Returns:
        Compiled LangGraph graph ready for execution
    """
    config = model_config or {}
    space_id = config.get("genie_space_id") or os.getenv("GENIE_SPACE_ID")
    lakebase_instance = config.get("lakebase_instance_name") or os.getenv("LAKEBASE_INSTANCE_NAME")
    
    if not space_id:
        raise ValueError(
            "genie_space_id is required. Pass it via model_config or GENIE_SPACE_ID env var."
        )
    
    agent = GenieQueryAgent(
        space_id=space_id,
        lakebase_instance=lakebase_instance,
        use_lakebase=lakebase_instance is not None,
    )
    return agent.graph
