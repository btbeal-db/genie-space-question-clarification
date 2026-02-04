"""Deploy GenieAgent as a Databricks Model Serving endpoint."""
from __future__ import annotations

import os
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput


def main() -> None:
    profile = os.getenv("DATABRICKS_PROFILE")
    space_id = os.getenv("GENIE_SPACE_ID")
    lakebase_instance = os.getenv("LAKEBASE_INSTANCE_NAME")

    if not space_id:
        raise ValueError("GENIE_SPACE_ID is required")

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Users/your.name/genie-agent")
    catalog = os.getenv("UC_CATALOG", "main")
    schema = os.getenv("UC_SCHEMA", "default")
    model_short_name = os.getenv("MODEL_NAME", "genie_question_clarification_agent")
    model_name = f"{catalog}.{schema}.{model_short_name}"
    endpoint_name = os.getenv("ENDPOINT_NAME", "genie-agent-endpoint")
    workload_size = os.getenv("WORKLOAD_SIZE", "Small")

    if profile:
        client = WorkspaceClient(profile=profile)
    else:
        client = WorkspaceClient()

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_name)

    from src.serving_model import GenieServingModel
    from mlflow.models import infer_signature

    # Define input/output examples for signature
    input_example = {
        "action": "run",
        "question": "What is the average patient age?",
        "thread_id": "example-thread"
    }
    output_example = {
        "__interrupt__": {
            "plan": "Sample plan",
            "assumptions": ["Assumption 1", "Assumption 2"]
        }
    }
    signature = infer_signature(input_example, output_example)

    with mlflow.start_run(run_name="genie-agent-serving") as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=GenieServingModel(),
            code_paths=["src"],
            pip_requirements=[
                "mlflow==3.5.1",
                "databricks-langchain[memory]>=0.9.0",
                "databricks-sdk>=0.35.0",
                "langgraph>=0.2.0",
                "langchain-core>=0.3.0",
                "langchain-community>=0.3.0",
            ],
            signature=signature,
            input_example=input_example,
        )

        model_uri = model_info.model_uri

    registered = mlflow.register_model(model_uri, model_name)

    served_entities = [
        ServedEntityInput(
            entity_name=model_name,
            entity_version=registered.version,
            workload_size=workload_size,
            scale_to_zero_enabled=True,
        )
    ]

    try:
        client.serving_endpoints.get(endpoint_name)
        client.serving_endpoints.update_config(
            name=endpoint_name,
            served_entities=served_entities,
        )
    except Exception:
        client.serving_endpoints.create(
            name=endpoint_name,
            config=EndpointCoreConfigInput(served_entities=served_entities),
        )

    print("Deployment started")
    print(f"Run ID: {run.info.run_id}")
    print(f"Model: {model_name} (version {registered.version})")
    print(f"Endpoint: {endpoint_name}")
    print("Set endpoint environment variables:")
    print(f"  GENIE_SPACE_ID={space_id}")
    if lakebase_instance:
        print(f"  LAKEBASE_INSTANCE_NAME={lakebase_instance}")
    if profile:
        print(f"  DATABRICKS_PROFILE={profile}")


if __name__ == "__main__":
    main()
