"""Test the GenieResponsesAgent locally."""
import os

os.environ["DATABRICKS_CONFIG_PROFILE"] = "FE-EAST"

from src.serving_model import MODEL
from mlflow.types.responses import ResponsesAgentRequest

if __name__ == "__main__":
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": "What is the average patient age?"}]
    )
    
    print("Testing predict...")
    response = MODEL.predict(request)
    
    print(f"\nResponse ID: {response.id}")
    print(f"Status: {response.custom_outputs.get('status')}")
    print(f"Thread ID: {response.custom_outputs.get('thread_id')}")
    
    # Print the output
    for output in response.output:
        print(f"\nOutput type: {output.type}")
        if hasattr(output, 'content'):
            for content in output.content:
                if hasattr(content, 'text'):
                    print(f"Text: {content.text[:1000]}")
    
    print(f"\nQuery Reasoning: {response.custom_outputs.get('query_reasoning', '')[:200]}...")
    print(f"Query SQL: {response.custom_outputs.get('query_sql', '')[:200]}...")
