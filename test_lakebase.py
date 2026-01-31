"""Minimal Lakebase checkpoint test."""
import os
from dotenv import load_dotenv
from src.agent import GenieAgent


def main():
    load_dotenv()
    profile = os.environ.get("DATABRICKS_PROFILE", "FE-EAST")
    space_id = os.environ.get("GENIE_SPACE_ID")
    lakebase_instance = os.environ.get("LAKEBASE_INSTANCE_NAME")

    if not space_id:
        print("GENIE_SPACE_ID is required, currently: ", space_id)
        return

    if not lakebase_instance:
        print("LAKEBASE_INSTANCE_NAME is required")
        return

    agent = GenieAgent(
        space_id=space_id,
        databricks_profile=profile,
        lakebase_instance=lakebase_instance,
        use_lakebase=True,
    )

    try:
        result = agent.run("test lakebase checkpoint", thread_id="lakebase_test")

        if "__interrupt__" in result:
            print("Lakebase check: ok")
        else:
            print("Lakebase check: unexpected result")
    finally:
        agent.close()


if __name__ == "__main__":
    main()
