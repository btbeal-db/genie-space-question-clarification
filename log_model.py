"""Log and deploy the GenieResponsesAgent using Databricks Agent Framework.

This script:
1. Creates a GenieQueryAgent with configuration
2. Wraps it in a GenieResponsesAgent
3. Logs to MLflow with resource declarations
4. Optionally deploys using databricks.agents.deploy()

Usage:
    export DATABRICKS_CONFIG_PROFILE=FE-EAST
    
    # Log and register
    python log_model.py
    
    # Log, register, and deploy
    python log_model.py --deploy
"""

import argparse
import os

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.models.resources import (
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
)

from src.agent import GenieQueryAgent
from src.serving_model import GenieResponsesAgent

# Default configuration
DEFAULT_CONFIG = {
    "genie_space_id": "01f0f218b8c814f3aeb2bcb24c8aa8b5",
    "lakebase_instance_name": "bbeal-genie-clarification-app",
    "model_name": "btbeal.genie_question_refinement.genie_clarification_agent",
    "endpoint_name": "genie-clarification-agent",
    "experiment_name": "/Users/brennan.beal@databricks.com/genie-question-clarification",
    "llm_endpoint": "databricks-claude-sonnet-4",
}


def log_and_register(
    genie_space_id: str,
    lakebase_instance_name: str | None,
    model_name: str,
    experiment_name: str | None = None,
    llm_endpoint: str = "databricks-claude-sonnet-4",
) -> str:
    """Log the model and register to Unity Catalog.
    
    Returns:
        The registered model version
    """
    client = WorkspaceClient()
    host = client.config.host
    print(f"Using Databricks workspace: {host}")
    
    # Configure MLflow
    mlflow.set_tracking_uri(f"databricks://{os.environ.get('DATABRICKS_CONFIG_PROFILE', 'DEFAULT')}")
    mlflow.set_registry_uri("databricks-uc")
    
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    
    # Declare resources
    resources = [
        DatabricksGenieSpace(genie_space_id=genie_space_id),
        DatabricksServingEndpoint(endpoint_name=llm_endpoint),
    ]
    
    print(f"Creating agent with config:")
    print(f"  Genie Space: {genie_space_id}")
    print(f"  LLM Endpoint: {llm_endpoint}")
    print(f"  Lakebase: {lakebase_instance_name or 'None (using MemorySaver)'}")
    
    # Step 1: Create the GenieQueryAgent
    query_agent = GenieQueryAgent(
        space_id=genie_space_id,
        lakebase_instance=lakebase_instance_name,
        use_lakebase=lakebase_instance_name is not None,
    )
    
    # Step 2: Wrap in ResponsesAgent
    responses_agent = GenieResponsesAgent(query_agent)
    
    # Step 3: Log to MLflow
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            python_model=responses_agent,
            name="model",
            pip_requirements="requirements.txt",
            code_paths=["src"],
            resources=resources,
        )
        
        print(f"\nModel logged: {model_info.model_uri}")
        print(f"Run ID: {run.info.run_id}")
        
        # Register to Unity Catalog
        print(f"\nRegistering to Unity Catalog: {model_name}")
        registered = mlflow.register_model(model_info.model_uri, model_name)
        print(f"Registered version: {registered.version}")
        
        return registered.version


def deploy_agent(
    model_name: str,
    model_version: str,
    endpoint_name: str = None,
):
    """Deploy using databricks.agents.deploy()."""
    from databricks.agents import deploy
    
    print(f"\nDeploying agent using databricks.agents.deploy()...")
    print(f"  Model: {model_name}")
    print(f"  Version: {model_version}")
    
    deployment = deploy(
        model_name=model_name,
        model_version=int(model_version),
        scale_to_zero=True,
        endpoint_name=endpoint_name,
    )
    
    print(f"\nDeployment successful!")
    print(f"  Endpoint: {deployment.endpoint_name}")
    print(f"  Query endpoint: {deployment.query_endpoint}")


def main():
    parser = argparse.ArgumentParser(description="Log and deploy GenieResponsesAgent")
    parser.add_argument("--space-id", default=DEFAULT_CONFIG["genie_space_id"])
    parser.add_argument("--lakebase", default=DEFAULT_CONFIG["lakebase_instance_name"])
    parser.add_argument("--model-name", default=DEFAULT_CONFIG["model_name"])
    parser.add_argument("--endpoint-name", default=DEFAULT_CONFIG["endpoint_name"])
    parser.add_argument("--experiment", default=DEFAULT_CONFIG["experiment_name"])
    parser.add_argument("--llm-endpoint", default=DEFAULT_CONFIG["llm_endpoint"])
    parser.add_argument("--deploy", action="store_true", help="Deploy after registering")
    
    args = parser.parse_args()
    
    version = log_and_register(
        genie_space_id=args.space_id,
        lakebase_instance_name=args.lakebase,
        model_name=args.model_name,
        experiment_name=args.experiment,
        llm_endpoint=args.llm_endpoint,
    )
    
    if args.deploy:
        deploy_agent(
            model_name=args.model_name,
            model_version=version,
            endpoint_name=args.endpoint_name,
        )
    else:
        print(f"\nTo deploy: python log_model.py --deploy")


if __name__ == "__main__":
    main()
