import os

import mlflow
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient

# Load env vars
load_dotenv()

# Configure MLflow to use Databricks
# Check for profile or fall back to workspace host
profile = os.environ.get('DATABRICKS_CONFIG_PROFILE')
if profile:
    tracking_uri = f"databricks://{profile}"
else:
    # Fall back to host-based URI
    tracking_uri = "databricks"

print(f"Using MLflow tracking URI: {tracking_uri}")
mlflow.set_tracking_uri(tracking_uri)

# Debug: Show which workspace the SDK is connecting to
client = WorkspaceClient()
model_uri = "runs:/a6910c7821124503abf4443060b6940f/model"

dependencies = mlflow.pyfunc.get_model_dependencies(model_uri)
print(dependencies)