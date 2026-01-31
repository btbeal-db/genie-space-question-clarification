"""MLflow model wrapper for GenieAgent serving."""
from __future__ import annotations

import atexit
import os
from typing import Any, Dict

import mlflow

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional for serving
    load_dotenv = None

from databricks.sdk import WorkspaceClient

from src.agent import GenieAgent


class GenieServingModel(mlflow.pyfunc.PythonModel):
    """Serve GenieAgent in Databricks Model Serving."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        if load_dotenv is not None:
            load_dotenv()

        profile = os.getenv("DATABRICKS_PROFILE")
        space_id = os.getenv("GENIE_SPACE_ID")
        lakebase_instance = os.getenv("LAKEBASE_INSTANCE_NAME")

        if not space_id:
            raise ValueError("GENIE_SPACE_ID is required for serving")

        if profile:
            workspace_client = WorkspaceClient(profile=profile)
        else:
            workspace_client = WorkspaceClient()

        self.agent = GenieAgent(
            space_id=space_id,
            databricks_profile=profile,
            workspace_client=workspace_client,
            lakebase_instance=lakebase_instance,
            use_lakebase=lakebase_instance is not None,
        )

        atexit.register(self.agent.close)

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: Any) -> Dict[str, Any]:
        import pandas as pd

        if isinstance(model_input, pd.DataFrame):
            input_dict = model_input.to_dict(orient="records")[0]
        else:
            input_dict = model_input

        action = input_dict.get("action", "run")
        thread_id = input_dict.get("thread_id", "default")

        if action == "run":
            question = input_dict.get("question")
            if not question:
                return {"error": "Missing 'question'"}
            return self.agent.run(question, thread_id=thread_id)

        if action == "resume":
            resume_payload = input_dict.get("resume_payload")
            if resume_payload is None:
                return {"error": "Missing 'resume_payload'"}
            return self.agent.resume(resume_payload, thread_id=thread_id)

        return {"error": f"Unknown action: {action}"}
