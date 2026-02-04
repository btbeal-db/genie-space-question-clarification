import os

import mlflow
from dotenv import load_dotenv

from mlflow.models import validate_serving_input

# Load env vars
load_dotenv()

# Configure MLflow to use Databricks (respects DATABRICKS_CONFIG_PROFILE)
mlflow.set_tracking_uri(f"databricks://{os.environ.get('DATABRICKS_CONFIG_PROFILE', 'DEFAULT')}")

# Use runs:/<run_id>/model format (not the UI URL)
model_uri = "runs:/a6910c7821124503abf4443060b6940f/model"

# ResponsesAgent uses "input" format (not "messages")
serving_payload = """{
  "input": [
    {
      "role": "user",
      "content": "What is the longest name in the database?"
    }
  ]
}
"""

# Validate the serving payload works on the model
print(f"Validating model: {model_uri}")
print(validate_serving_input(model_uri, serving_payload))
print("✅ Validation passed!")