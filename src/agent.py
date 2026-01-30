"""LangGraph agent for interactive Genie query refinement."""
from typing import Literal, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from genie_client import GenieClient
from langchain_databricks import ChatDatabricks


class AgentState(TypedDict):
    """State for the Genie question clarification agent."""
    user_question: str
    assumptions: str | None
    user_approved: bool
    feedback: str | None
    final_result: dict | None


class GenieAgent:
    """LangGraph agent that interactively refines Genie queries."""
    
    def __init__(self, genie_client: GenieClient):
        """Initialize the agent with a Genie client.
        
        Args:
            genie_client: Configured GenieClient instance
        """
        self.genie = genie_client
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
        workflow.add_node("present_plan", self._present_plan)
        workflow.add_node("refine_plan", self._refine_plan)
        workflow.add_node("show_results", self._show_results)

        workflow.set_entry_point("get_initial_plan")

        workflow.add_edge("get_initial_plan", "review_with_llm")
        workflow.add_edge("refine_plan", "review_with_llm")
        workflow.add_edge("review_with_llm", "present_plan")

        # Conditional: approved -> show results, not approved -> refine
        # Refining loops back through review and present
        workflow.add_conditional_edges(
            "present_plan",
            self._should_show_or_refine,
            {
                "show": "show_results",
                "refine": "refine_plan"
            }
        )
        
        workflow.add_edge("show_results", END)
        
        # Compile with checkpointer for human-in-the-loop
        # MemorySaver stores state in memory (for local testing)
        # For production, use a persistent checkpointer (e.g., SqliteSaver)
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)
    
    def _get_initial_plan(self, state: AgentState) -> AgentState:
        """Get initial plan from Genie based on user question.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state (GenieClient manages Genie state internally)
        """
        print(f"\nAsking Genie: '{state['user_question']}'")
        print("Waiting for Genie's response...\n")
        
        self.genie.start_conversation(state["user_question"])
        
        return state
    
    def _review_with_llm(self, state: AgentState) -> AgentState:
        """Use LLM to extract top 3 assumptions from the query plan.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with assumptions
        """
        description = self.genie.get_description()
        query = self.genie.get_query()
        
        if not description or not query:
            return {**state, "assumptions": None}
        
        print("Analyzing assumptions with LLM...\n")
        
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
        
        return {
            **state,
            "assumptions": response.content
        }
    
    def _present_plan(self, state: AgentState) -> AgentState:
        """Present the plan to user and wait for their response.
        
        This node uses interrupt() to pause execution and wait for human input.
        The agent will return the plan and resume when the user responds.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with user approval and feedback
        """
        description = self.genie.get_description()
        query = self.genie.get_query()
        
        # Package the plan to show to the user
        plan_to_present = {
            "description": description,
            "query": query,
            "assumptions": state["assumptions"]
        }
        
        # Interrupt and wait for human response
        # This will pause execution and return control
        # The user's response will be available when resumed
        user_response = interrupt(plan_to_present)
        
        # When resumed, user_response will contain their answer
        # Expected format: {"approved": true/false, "feedback": "optional feedback"}
        if user_response and isinstance(user_response, dict):
            approved = user_response.get("approved", False)
            feedback = user_response.get("feedback", None)
            
            return {
                **state,
                "user_approved": approved,
                "feedback": feedback
            }
        
        # Fallback if no valid response
        return {
            **state,
            "user_approved": False,
            "feedback": None
        }
    
    def _refine_plan(self, state: AgentState) -> AgentState:
        """Refine the plan based on user feedback.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state (GenieClient manages Genie state internally)
        """
        print(f"\nRefining plan based on feedback: '{state['feedback']}'")
        print("⏳ Waiting for Genie's response...\n")
        
        self.genie.refine_question(state["feedback"])
        
        return state
    
    def _show_results(self, state: AgentState) -> AgentState:
        """Show results from the already-executed query.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with final results
        """
        print("\nPlan approved! Fetching results...\n")
        
        result = self.genie.get_results()
        
        print("=" * 80)
        print("QUERY RESULTS")
        print("=" * 80)

        if result.get('summary'):
            print(f"\nSUMMARY:")
            print(result['summary'])

        if result.get('results'):
            self._display_results(result['results'])
        else:
            print("\nNo results available.")
        
        print("=" * 80)
        
        return {
            **state,
            "final_result": result
        }
    
    def _should_show_or_refine(self, state: AgentState) -> Literal["show", "refine"]:
        """Decide whether to show results or refine the plan.
        
        Args:
            state: Current agent state
            
        Returns:
            Next step: "show" or "refine"
        """
        return "show" if state["user_approved"] else "refine"
    
    def _display_results(self, results: dict) -> None:
        """Display query results in a formatted table.
        
        Args:
            results: dict with 'columns', 'rows', and 'row_count'
        """
        columns = results.get('columns', [])
        rows = results.get('rows', [])
        row_count = results.get('row_count', 0)
        
        if not rows:
            print("\nNo results returned.")
            return
        
        print(f"\nResults ({row_count} row{'s' if row_count != 1 else ''}):")
        print()

        # This is fairly unnecessary -- just a way to format sql results
        col_widths = []
        for i, col_name in enumerate(columns):
            max_width = len(col_name)
            for row in rows:
                if i < len(row):
                    cell_value = str(row[i]) if row[i] is not None else 'NULL'
                    max_width = max(max_width, len(cell_value))
            col_widths.append(min(max_width, 50))

        header = " | ".join(col.ljust(col_widths[i]) for i, col in enumerate(columns))
        print(header)
        print("-" * len(header))

        for row in rows:
            row_str = " | ".join(
                str(row[i] if i < len(row) and row[i] is not None else 'NULL').ljust(col_widths[i])
                for i in range(len(columns))
            )
            print(row_str)
        
        print()
    
    def run(self, user_question: str, thread_id: str = "default") -> dict:
        """Run the agent with a user question (will interrupt for human input).
        
        Args:
            user_question: The question to ask Genie
            thread_id: Unique ID for this conversation thread
            
        Returns:
            Dict with either the plan (if interrupted) or final result
        """
        initial_state: AgentState = {
            "user_question": user_question,
            "assumptions": None,
            "user_approved": False,
            "feedback": None,
            "final_result": None
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Stream the execution - this will pause at interrupt points
        for event in self.graph.stream(initial_state, config, stream_mode="values"):
            # event contains the current state
            pass
        
        # Return the final state or interruption info
        return event
    
    def resume(self, user_response: dict, thread_id: str = "default") -> dict:
        """Resume execution after human input.
        
        Args:
            user_response: User's response with format {"approved": bool, "feedback": str}
            thread_id: Same thread ID used in run()
            
        Returns:
            Dict with either updated plan (if refined) or final result
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        # Resume from checkpoint with user's response
        for event in self.graph.stream(user_response, config, stream_mode="values"):
            pass
        
        return event

