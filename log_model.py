"""Log and register the GenieResponsesAgent to Unity Catalog.

This script logs the model locally and registers it to Unity Catalog.
After registration, you can create a Model Serving endpoint via UI or SDK.

Usage:
    # Set your Databricks profile
    export DATABRICKS_CONFIG_PROFILE=FE-EAST
    
    # Run the script (uses .env file for config)
    python log_model.py
    
    # Or with custom settings
    python log_model.py --space-id YOUR_SPACE_ID --model-name catalog.schema.model_name
"""

import argparse
import os

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from dotenv import load_dotenv

# Load .env file for configuration
load_dotenv()


# Default configuration - reads from .env file or uses fallback values
DEFAULT_CONFIG = {
    "genie_space_id": os.getenv("GENIE_SPACE_ID", "01f0f218b8c814f3aeb2bcb24c8aa8b5"),
    "lakebase_instance_name": os.getenv("LAKEBASE_INSTANCE_NAME", "bbeal-genie-clarification-app"),
    "model_name": os.getenv("MODEL_NAME", "btbeal.genie_question_refinement.genie_clarification_agent"),
    "endpoint_name": os.getenv("ENDPOINT_NAME", "genie-clarification-agent"),
    "experiment_name": os.getenv("MLFLOW_EXPERIMENT_NAME", "/Users/brennan.beal@databricks.com/genie-question-clarification"),
}

PIP_REQUIREMENTS = [
    "mlflow>=3.0.0",
    "pydantic>=2.0",
    "databricks-langchain>=0.12.0",  # 0.12+ has CheckpointSaver
    "langgraph>=0.2.0",
    "databricks-sdk>=0.35.0",
    "python-dotenv>=1.0.0",
    "sniffio>=1.3.0,<2",  # Pin to avoid broken 0.0.0 version on PyPI
]


def log_and_register(
    genie_space_id: str,
    lakebase_instance_name: str | None,
    model_name: str,
    experiment_name: str | None = None,
) -> str:
    """Log the model and register to Unity Catalog.
    
    Args:
        genie_space_id: The Genie Space ID to query
        lakebase_instance_name: Optional Lakebase instance for checkpointing
        model_name: Full UC path (catalog.schema.model_name)
        experiment_name: Optional MLflow experiment name
    
    Returns:
        The registered model version
    """
    # Get workspace host from SDK (respects DATABRICKS_CONFIG_PROFILE)
    client = WorkspaceClient()
    host = client.config.host
    print(f"Using Databricks workspace: {host}")
    
    # Configure MLflow to use the correct workspace
    mlflow.set_tracking_uri(f"databricks://{os.environ.get('DATABRICKS_CONFIG_PROFILE', 'DEFAULT')}")
    mlflow.set_registry_uri("databricks-uc")
    
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    
    # Ensure env vars are set for validation during logging
    # (dotenv already loaded at module level, but override if args provided)
    os.environ["GENIE_SPACE_ID"] = genie_space_id
    if lakebase_instance_name:
        os.environ["LAKEBASE_INSTANCE_NAME"] = lakebase_instance_name
    
    print(f"Logging model with config:")
    print(f"  genie_space_id: {genie_space_id}")
    print(f"  lakebase_instance_name: {lakebase_instance_name}")
    print(f"  model_name: {model_name}")
    
    with mlflow.start_run() as run:
        # Log using models-from-code approach
        # Note: input_example=False skips validation prediction (which would fail
        # because MLflow tests with "Hello, world!" which isn't a valid Genie query)
        model_info = mlflow.pyfunc.log_model(
            python_model="src/serving_model.py",
            name="model",
            model_config={
                "genie_space_id": genie_space_id,
                "lakebase_instance_name": lakebase_instance_name,
            },
            pip_requirements="requirements.txt",
            code_paths=["src"],
            input_example=False,
        )
        
        print(f"Model logged: {model_info.model_uri}")
        print(f"Run ID: {run.info.run_id}")
        
        # Register to Unity Catalog
        print(f"\nRegistering to Unity Catalog: {model_name}")
        registered = mlflow.register_model(model_info.model_uri, model_name)
        print(f"Registered version: {registered.version}")
        
        return registered.version


def create_or_update_endpoint(
    model_name: str,
    model_version: str,
    endpoint_name: str,
):
    """Create or update a Model Serving endpoint.
    
    Args:
        model_name: Full UC path (catalog.schema.model_name)
        model_version: The version to serve
        endpoint_name: Name for the serving endpoint
    """
    from databricks.sdk.errors import NotFound, InvalidParameterValue
    
    client = WorkspaceClient()
    
    served_entities = [
        ServedEntityInput(
            entity_name=model_name,
            entity_version=model_version,
            workload_size="Small",
            scale_to_zero_enabled=True,
        )
    ]
    
    config = EndpointCoreConfigInput(name=endpoint_name, served_entities=served_entities)
    
    # Check if endpoint exists
    endpoint_exists = False
    try:
        client.serving_endpoints.get(endpoint_name)
        endpoint_exists = True
    except NotFound:
        pass
    
    if endpoint_exists:
        print(f"\nUpdating existing endpoint: {endpoint_name}")
        try:
            client.serving_endpoints.update_config(
                name=endpoint_name,
                served_entities=served_entities,
            )
        except InvalidParameterValue as e:
            # Can't update non-agent endpoint to agent endpoint - need to delete and recreate
            if "Agent endpoint cannot be updated" in str(e):
                print(f"Endpoint type mismatch. Deleting and recreating endpoint...")
                client.serving_endpoints.delete(endpoint_name)
                print(f"Deleted old endpoint. Creating new agent endpoint...")
                client.serving_endpoints.create(
                    name=endpoint_name,
                    config=config,
                )
            else:
                raise
    else:
        print(f"\nCreating new endpoint: {endpoint_name}")
        client.serving_endpoints.create(
            name=endpoint_name,
            config=config,
        )
    
    print(f"Endpoint URL: https://<workspace>/serving-endpoints/{endpoint_name}/invocations")


def main():
    parser = argparse.ArgumentParser(description="Log and deploy GenieResponsesAgent")
    parser.add_argument("--space-id", default=DEFAULT_CONFIG["genie_space_id"],
                        help="Genie Space ID")
    parser.add_argument("--lakebase", default=DEFAULT_CONFIG["lakebase_instance_name"],
                        help="Lakebase instance name (optional)")
    parser.add_argument("--model-name", default=DEFAULT_CONFIG["model_name"],
                        help="Unity Catalog model path")
    parser.add_argument("--endpoint-name", default=DEFAULT_CONFIG["endpoint_name"],
                        help="Serving endpoint name")
    parser.add_argument("--experiment", default=DEFAULT_CONFIG["experiment_name"],
                        help="MLflow experiment name")
    parser.add_argument("--deploy", action="store_true",
                        help="Also create/update the serving endpoint")
    parser.add_argument("--log-only", action="store_true",
                        help="Only log locally, don't register to UC")
    
    args = parser.parse_args()
    
    if args.log_only:
        # Log locally without registering
        print("Logging model locally (not registering to UC)...")
        mlflow.set_experiment("local-genie-agent")
        
        # Ensure env vars are set (dotenv already loaded, but override with args)
        os.environ["GENIE_SPACE_ID"] = args.space_id
        if args.lakebase:
            os.environ["LAKEBASE_INSTANCE_NAME"] = args.lakebase
        
        with mlflow.start_run() as run:
            model_info = mlflow.pyfunc.log_model(
                python_model="src/serving_model.py",
                artifact_path="model",
                model_config={
                    "genie_space_id": args.space_id,
                    "lakebase_instance_name": args.lakebase,
                },
                pip_requirements="requirements.txt",
            )
            print(f"Model logged locally!")
            print(f"Run ID: {run.info.run_id}")
            print(f"\nTo serve locally:")
            print(f"  mlflow models serve -m runs:/{run.info.run_id}/model -p 5000")
        return
    
    # Log and register to UC
    version = log_and_register(
        genie_space_id=args.space_id,
        lakebase_instance_name=args.lakebase,
        model_name=args.model_name,
        experiment_name=args.experiment,
    )
    
    if args.deploy:
        create_or_update_endpoint(
            model_name=args.model_name,
            model_version=version,
            endpoint_name=args.endpoint_name,
        )
    else:
        print(f"\nTo deploy the endpoint, run:")
        print(f"  python log_model.py --deploy")
        print(f"\nOr create via Databricks UI:")
        print(f"  1. Go to Machine Learning > Serving")
        print(f"  2. Create endpoint with model: {args.model_name} version {version}")


if __name__ == "__main__":
    main()

