"""MLflow ResponsesAgent wrapper for GenieAgent serving.

This module provides a ResponsesAgent implementation that wraps the GenieAgent
for deployment on Databricks Model Serving. The agent uses the OpenAI-compatible
Responses API format for seamless integration with downstream clients.

Input/Output Schema
===================

Request Format (ResponsesAgentRequest):
---------------------------------------
{
    "input": [
        {"role": "user", "content": "What are the top sales by region?"}
    ],
    "custom_inputs": {                    # Optional
        "thread_id": "abc-123-def-456"    # Include to resume a conversation
    }
}

Response Format (ResponsesAgentResponse):
-----------------------------------------
{
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Here's the plan..."}]
        }
    ],
    "custom_outputs": {
        "thread_id": "abc-123-def-456",   # Always returned, use to continue
        "status": "awaiting_input",       # or "complete"
        "interrupt": {...},               # Raw interrupt value (if awaiting)
        "results": {...}                  # Query results (if complete)
    }
}

Conversation Flow
=================

1. **Start**: Send question without thread_id
   - Returns plan with status="awaiting_input", thread_id, and interrupt payload

2. **Continue**: Send response with the thread_id
   - If approved: returns results with status="complete"
   - If rejected or more input needed: returns status="awaiting_input" with new interrupt

The interrupt payload is passed through as-is from the agent, keeping the
serving model decoupled from the agent's internal structure.

Configuration
=============
Configuration can be passed via:

1. MLflow's model_config (production):
    mlflow.pyfunc.log_model(
        python_model="src/serving_model.py",
        model_config={
            "genie_space_id": "your-space-id",
            "lakebase_instance_name": "your-lakebase-instance",  # optional
        },
        ...
    )

2. Environment variables (local testing):
    export GENIE_SPACE_ID="your-space-id"
    export LAKEBASE_INSTANCE_NAME="your-lakebase-instance"
"""
from __future__ import annotations

import os
from typing import Generator
from uuid import uuid4

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from langgraph.types import Command

from src.agent import GenieAgent

# Load .env file for local development (no-op if file doesn't exist)
load_dotenv()


class GenieResponsesAgent(ResponsesAgent):
    """Serve GenieAgent as a ResponsesAgent for Databricks Model Serving.
    
    This agent wraps the LangGraph-based GenieAgent and exposes it via the
    OpenAI-compatible Responses API format. State is maintained in Lakebase
    (Postgres) so the serving endpoint remains stateless.
    
    Input/Output Contract:
    - Users send text messages in `input`
    - Users track `thread_id` from `custom_outputs` and pass it back in `custom_inputs`
    - All graph state is managed transparently via Lakebase checkpointing
    """

    def __init__(self):
        """Initialize the agent.
        
        Configuration is read from (in order of priority):
        1. model_config (set when logging the model - available at serving time)
        2. Environment variables (for local testing)
        
        Config keys:
        - genie_space_id / GENIE_SPACE_ID: Required. The Genie Space ID to query.
        - lakebase_instance_name / LAKEBASE_INSTANCE_NAME: Optional. Lakebase instance.
        
        Note: Agent creation is deferred until first predict() call because
        model_config is not available during log_model() import.
        """
        super().__init__()
        self._agent = None  # Lazy initialization
    
    def close(self) -> None:
        """Close the agent and release resources (e.g., Lakebase connection pool)."""
        if self._agent is not None:
            self._agent.close()
            self._agent = None
    
    def __del__(self):
        """Ensure cleanup on garbage collection."""
        self.close()
    
    @property
    def agent(self) -> GenieAgent:
        """Lazy-load the GenieAgent on first access.
        
        This is necessary because model_config is not available when the module
        is first imported during mlflow.pyfunc.log_model(). It's only available
        at serving time after MLflow injects it.
        """
        if self._agent is None:
            config = getattr(self, "model_config", {}) or {}
            
            # Try model_config first, fall back to environment variables
            space_id = config.get("genie_space_id") or os.getenv("GENIE_SPACE_ID")
            lakebase_instance = config.get("lakebase_instance_name") or os.getenv("LAKEBASE_INSTANCE_NAME")

            if not space_id:
                raise ValueError(
                    "genie_space_id is required. Either:\n"
                    "1. Pass model_config={'genie_space_id': '...'} when logging the model\n"
                    "2. Set GENIE_SPACE_ID environment variable for local testing"
                )

            self._agent = GenieAgent(
                space_id=space_id,
                workspace_client=WorkspaceClient(),
                lakebase_instance=lakebase_instance,
                use_lakebase=lakebase_instance is not None,
            )
        return self._agent

    def _get_last_user_message(self, input_items: list) -> str:
        """Extract the last user message content from input items."""
        for item in reversed(input_items):
            if hasattr(item, 'role') and item.role == "user":
                return item.content if hasattr(item, 'content') else str(item)
            elif isinstance(item, dict) and item.get("role") == "user":
                return item.get("content", "")
        return ""

    def _get_graph_state(self, config: dict):
        """Get the current graph state for a thread.
        
        Returns:
            StateSnapshot or None if no state exists
        """
        try:
            return self.agent.graph.get_state(config)
        except Exception:
            return None

    def _get_interrupt_value(self, state):
        """Get the raw interrupt value from graph state, if any.
        
        Args:
            state: StateSnapshot from graph.get_state()
            
        Returns:
            The interrupt value (whatever the agent passed to interrupt()), or None
        """
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
        
        Args:
            request: ResponsesAgentRequest with input messages and optional custom_inputs
            
        Returns:
            ResponsesAgentResponse with output messages and custom_outputs
        """
        custom_inputs = request.custom_inputs or {}
        thread_id = custom_inputs.get("thread_id") or str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        user_message = self._get_last_user_message(request.input)
        
        if custom_inputs.get("thread_id"):
            # Resume from checkpoint - LangGraph handles the rest
            result = self.agent.graph.invoke(Command(resume=user_message), config)
        else:
            # New conversation
            from src.agent import AgentState
            initial_state: AgentState = {
                "user_question": user_message,
                "conversation_id": None,
                "current_message": None,
                "assumptions": None,
                "user_approved": False,
                "feedback": None,
                "final_result": None
            }
            result = self.agent.graph.invoke(initial_state, config)
        
        # Check final state - did we hit an interrupt or complete?
        state = self._get_graph_state(config)
        interrupt_value = self._get_interrupt_value(state)
        
        if interrupt_value is not None:
            return self._build_interrupt_response(interrupt_value, thread_id)
        else:
            return self._build_complete_response(result, thread_id)

    def _build_interrupt_response(
        self, interrupt_value, thread_id: str
    ) -> ResponsesAgentResponse:
        """Build a response when the graph has interrupted.
        
        Args:
            interrupt_value: Raw value from interrupt() - passed through as-is
            thread_id: The thread ID for this conversation
            
        Returns:
            ResponsesAgentResponse with interrupt payload in custom_outputs
        """
        # Convert interrupt value to displayable text
        if isinstance(interrupt_value, dict):
            message_text = interrupt_value.get("question", str(interrupt_value))
        elif isinstance(interrupt_value, str):
            message_text = interrupt_value
        else:
            message_text = str(interrupt_value) if interrupt_value else "Awaiting input..."
        
        return ResponsesAgentResponse(
            output=[
                self.create_text_output_item(
                    text=message_text,
                    id=f"msg-{uuid4().hex[:8]}"
                )
            ],
            custom_outputs={
                "thread_id": thread_id,
                "status": "awaiting_input",
                "interrupt": interrupt_value,  # Pass through raw value
            }
        )

    def _build_complete_response(
        self, result: dict, thread_id: str
    ) -> ResponsesAgentResponse:
        """Build a response when the graph has completed.
        
        Args:
            result: The final state from the graph
            thread_id: The thread ID for this conversation
            
        Returns:
            ResponsesAgentResponse with final results
        """
        final_result = result.get("final_result", {})
        summary = final_result.get("summary", "Query completed.")
        results_data = final_result.get("results", {})
        
        # Build the response text
        if summary:
            response_text = summary
        else:
            response_text = "Query executed successfully."
        
        return ResponsesAgentResponse(
            output=[
                self.create_text_output_item(
                    text=response_text,
                    id=f"msg-{uuid4().hex[:8]}"
                )
            ],
            custom_outputs={
                "thread_id": thread_id,
                "status": "complete",
                "results": results_data,
            }
        )

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Handle a streaming prediction request.
        
        Args:
            request: ResponsesAgentRequest with input messages and optional custom_inputs
            
        Yields:
            ResponsesAgentStreamEvent for each piece of the response
        """
        custom_inputs = request.custom_inputs or {}
        thread_id = custom_inputs.get("thread_id") or str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        user_message = self._get_last_user_message(request.input)
        msg_id = f"msg-{uuid4().hex[:8]}"
        
        if custom_inputs.get("thread_id"):
            yield ResponsesAgentStreamEvent(
                **self.create_text_delta(delta="Resuming...\n", item_id=msg_id)
            )
            result = self.agent.graph.invoke(Command(resume=user_message), config)
        else:
            yield ResponsesAgentStreamEvent(
                **self.create_text_delta(delta="Processing your question...\n", item_id=msg_id)
            )
            from src.agent import AgentState
            initial_state: AgentState = {
                "user_question": user_message,
                "conversation_id": None,
                "current_message": None,
                "assumptions": None,
                "user_approved": False,
                "feedback": None,
                "final_result": None
            }
            result = self.agent.graph.invoke(initial_state, config)
        
        # Check final state
        state = self._get_graph_state(config)
        interrupt_value = self._get_interrupt_value(state)
        
        if interrupt_value is not None:
            # Convert to displayable text
            if isinstance(interrupt_value, dict):
                message_text = interrupt_value.get("question", str(interrupt_value))
            elif isinstance(interrupt_value, str):
                message_text = interrupt_value
            else:
                message_text = str(interrupt_value) if interrupt_value else "Awaiting input..."
            
            yield ResponsesAgentStreamEvent(
                **self.create_text_delta(delta=message_text, item_id=msg_id)
            )
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_text_output_item(text=message_text, id=msg_id),
            )
        else:
            final_result = result.get("final_result", {})
            summary = final_result.get("summary") or "Query executed successfully."
            
            yield ResponsesAgentStreamEvent(
                **self.create_text_delta(delta=summary, item_id=msg_id)
            )
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_text_output_item(text=summary, id=msg_id),
            )


# Required for MLflow models-from-code logging
AGENT = GenieResponsesAgent()
mlflow.models.set_model(AGENT)
