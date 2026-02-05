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
import os
load_dotenv()

profile = os.getenv("DATABRICKS_CONFIG_PROFILE", "DEFAULT")
experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")

genie_space_id = os.getenv("GENIE_SPACE_ID")
lakebase_instance_name = os.getenv("LAKEBASE_INSTANCE_NAME")
sql_warehouse_id = os.getenv("SQL_WAREHOUSE_ID")
serving_endpoint_name = "databricks-claude-sonnet-4"
table_name = os.getenv("TABLE_NAME")
model_name = os.getenv("MODEL_NAME")

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

    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
            model_info = mlflow.pyfunc.log_model(
                python_model="src/serving_model.py",
                name="model",
                pip_requirements="requirements.txt",
                code_paths=["src"],
                resources=resources,
                input_example=False,
            )

            registered = mlflow.register_model(model_info.model_uri, model_name)
