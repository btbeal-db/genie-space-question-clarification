import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.models.resources import (
    DatabricksGenieSpace, 
    DatabricksServingEndpoint, 
    DatabricksSQLWarehouse,
    DatabricksLakebase,
    DatabricksTable
)

from dotenv import load_dotenv
load_dotenv()
import os

profile = os.getenv("DATABRICKS_CONFIG_PROFILE", "DEFAULT")

genie_space_id = "01f0f218b8c814f3aeb2bcb24c8aa8b5"
lakebase_instance_name = "bbeal-genie-clarification-app"
sql_warehouse_id = "148ccb90800933a1"
serving_endpoint_name = "databricks-claude-sonnet-4"
table_name = "btbeal.genie_question_refinement.mock_healthcare_data"

resources = [
    DatabricksGenieSpace(genie_space_id=genie_space_id),
    DatabricksLakebase(database_instance_name=lakebase_instance_name),
    DatabricksSQLWarehouse(warehouse_id=sql_warehouse_id),
    DatabricksServingEndpoint(endpoint_name=serving_endpoint_name),
    DatabricksTable(table_name=table_name)
]


if __name__ == "__main__":
    client = WorkspaceClient()
    host = client.config.host
    print(f"Using Databricks workspace: {host}")

    # Configure MLflow
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment("/Users/brennan.beal@databricks.com/genie-question-clarification")
    with mlflow.start_run() as run:
            model_info = mlflow.pyfunc.log_model(
                python_model="src/serving_model.py",
                name="model",
                pip_requirements="requirements.txt",
                code_paths=["src"],
                resources=resources,
                input_example=False,
            )