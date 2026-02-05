"""MLflow ResponsesAgent wrapper for GenieQueryAgent serving.

This module provides a ResponsesAgent implementation that wraps the GenieQueryAgent
for deployment on Databricks Model Serving.

Usage
=====
    from src.agent import GenieQueryAgent
    from src.serving_model import GenieResponsesAgent
    
    # Create the agent with configuration
    agent = GenieQueryAgent(
        space_id="your-space-id",
        lakebase_instance="your-lakebase",  # optional
    )
    
    # Wrap in ResponsesAgent
    responses_agent = GenieResponsesAgent(agent)
    
    # Log to MLflow
    mlflow.pyfunc.log_model(python_model=responses_agent, ...)

The databricks_langchain components auto-authenticate in the serving environment.
"""
from __future__ import annotations

from typing import Generator
from uuid import uuid4

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.agent import GenieQueryAgent, AgentState


class GenieResponsesAgent(ResponsesAgent):
    """Serve GenieQueryAgent as a ResponsesAgent for Databricks Model Serving.
    
    This agent wraps the LangGraph-based GenieQueryAgent and exposes it via the
    OpenAI-compatible Responses API format.
    """

    def __init__(self, agent: GenieQueryAgent):
        """Initialize with a configured GenieQueryAgent.
        
        Args:
            agent: A fully configured GenieQueryAgent instance.
        """
        super().__init__()
        self._agent = agent
    
    @property
    def agent(self) -> GenieQueryAgent:
        """Get the underlying GenieQueryAgent."""
        return self._agent
    
    def close(self) -> None:
        """Close the agent and release resources."""
        if self._agent is not None:
            self._agent.close()
    
    def __del__(self):
        """Ensure cleanup on garbage collection."""
        self.close()

    def _get_last_user_message(self, input_items: list) -> str:
        """Extract the last user message content from input items."""
        normalized = to_chat_completions_input(input_items)
        for item in reversed(normalized):
            if item.get("role") == "user":
                return item.get("content", "")
        return ""

    def _get_graph_state(self, config: dict):
        """Get the current graph state for a thread."""
        try:
            return self.agent.graph.get_state(config)
        except Exception:
            return None

    def _get_interrupt_value(self, state):
        """Get the raw interrupt value from graph state, if any."""
        if not state or not state.tasks:
            return None
            
        for task in state.tasks:
            if task.interrupts:
                return task.interrupts[0].value
        
        return None

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Handle a prediction request.
        
        - If thread_id exists: resume from checkpoint with user's response
        - If no thread_id: start a new conversation
        """
        custom_inputs = request.custom_inputs or {}
        thread_id = custom_inputs.get("thread_id") or str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        user_message = self._get_last_user_message(request.input)
        
        if custom_inputs.get("thread_id"):
            # Resume from checkpoint
            result = self.agent.graph.invoke(Command(resume=user_message), config)
        else:
            # New conversation
            initial_state: AgentState = {
                "messages": [HumanMessage(content=user_message)],
                "conversation_id": None,
                "query_reasoning": None,
                "query_sql": None,
                "query_result": None,
                "assumptions": None,
                "user_approved": False,
                "final_result": None,
            }
            result = self.agent.graph.invoke(initial_state, config)
        
        # Check final state - did we hit an interrupt or complete?
        state = self._get_graph_state(config)
        interrupt_value = self._get_interrupt_value(state)
        
        if interrupt_value is not None:
            return self._build_interrupt_response(interrupt_value, thread_id, result)
        else:
            return self._build_complete_response(result, thread_id)

    def _build_interrupt_response(
        self, interrupt_value, thread_id: str, result: dict
    ) -> ResponsesAgentResponse:
        """Build a response when the graph has interrupted."""
        if isinstance(interrupt_value, dict):
            message_text = interrupt_value.get("question", str(interrupt_value))
        elif isinstance(interrupt_value, str):
            message_text = interrupt_value
        else:
            message_text = str(interrupt_value) if interrupt_value else "Awaiting input..."
        
        query_reasoning = result.get("query_reasoning", "")
        query_sql = result.get("query_sql", "")
        assumptions = result.get("assumptions", "")
        
        response_id = f"resp-{uuid4().hex[:8]}"
        return ResponsesAgentResponse(
            id=response_id,
            output=[
                self.create_text_output_item(
                    text=message_text,
                    id=f"msg-{uuid4().hex[:8]}"
                )
            ],
            custom_outputs={
                "thread_id": thread_id,
                "status": "awaiting_input",
                "interrupt": interrupt_value,
                "query_reasoning": query_reasoning,
                "query_sql": query_sql,
                "assumptions": assumptions,
            }
        )

    def _build_complete_response(
        self, result: dict, thread_id: str
    ) -> ResponsesAgentResponse:
        """Build a response when the graph has completed."""
        final_result = result.get("final_result", {})
        summary = final_result.get("summary", "Query completed.")
        
        response_id = f"resp-{uuid4().hex[:8]}"
        return ResponsesAgentResponse(
            id=response_id,
            output=[
                self.create_text_output_item(
                    text=summary,
                    id=f"msg-{uuid4().hex[:8]}"
                )
            ],
            custom_outputs={
                "thread_id": thread_id,
                "status": "complete",
                "query_reasoning": final_result.get("query_reasoning"),
                "query_sql": final_result.get("query_sql"),
            }
        )

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Handle a streaming prediction request."""
        custom_inputs = request.custom_inputs or {}
        thread_id = custom_inputs.get("thread_id") or str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        user_message = self._get_last_user_message(request.input)
        msg_id = f"msg-{uuid4().hex[:8]}"
        final_text = ""
        
        try:
            if custom_inputs.get("thread_id"):
                result = self.agent.graph.invoke(Command(resume=user_message), config)
            else:
                initial_state: AgentState = {
                    "messages": [HumanMessage(content=user_message)],
                    "conversation_id": None,
                    "query_reasoning": None,
                    "query_sql": None,
                    "query_result": None,
                    "assumptions": None,
                    "user_approved": False,
                    "final_result": None,
                }
                result = self.agent.graph.invoke(initial_state, config)
            
            state = self._get_graph_state(config)
            interrupt_value = self._get_interrupt_value(state)
            
            if interrupt_value is not None:
                if isinstance(interrupt_value, dict):
                    message_text = interrupt_value.get("question", str(interrupt_value))
                elif isinstance(interrupt_value, str):
                    message_text = interrupt_value
                else:
                    message_text = str(interrupt_value) if interrupt_value else "I need more information."
                
                final_text = f"{message_text}\n\n---\n*thread_id: {thread_id}*"
            else:
                final_result = result.get("final_result", {})
                final_text = final_result.get("summary") or "Query executed successfully."
            
            yield ResponsesAgentStreamEvent(
                **self.create_text_delta(delta=final_text, item_id=msg_id)
            )
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_text_output_item(text=final_text, id=msg_id),
            )
            
        except Exception as e:
            final_text = f"An error occurred: {str(e)}"
            yield ResponsesAgentStreamEvent(
                **self.create_text_delta(delta=final_text, item_id=msg_id)
            )
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_text_output_item(text=final_text, id=msg_id),
            )


genie_space_id = "01f0f218b8c814f3aeb2bcb24c8aa8b5"
lakebase_instance_name = "bbeal-genie-clarification-app"   
mlflow.langchain.autolog()

query_agent = GenieQueryAgent(
    space_id=genie_space_id,
    lakebase_instance=lakebase_instance_name,
    use_lakebase=lakebase_instance_name is not None,
)

MODEL = GenieResponsesAgent(query_agent)

mlflow.models.set_model(MODEL)