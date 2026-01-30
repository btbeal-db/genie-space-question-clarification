"""CLI entry point for Genie Question Clarification Agent."""
import sys
from genie_client import GenieClient
from agent import GenieAgent


def main():
    """Run the Genie question clarification agent."""
    print("=" * 80)
    print("DATABRICKS GENIE QUESTION CLARIFICATION AGENT")
    print("=" * 80)
    print()
    print("This agent will help you refine your questions to Genie Space.")
    print("You'll be able to review and adjust the query plan before execution.")
    print()

    PROFILE = "FE-EAST"
    SPACE_NAME = "Healthcare Genie"
    SPACE_ID = "01f0f218b8c814f3aeb2bcb24c8aa8b5"
    
    try:
        print(f"Connecting to Databricks (profile: {PROFILE})...")
        genie_client = GenieClient(profile=PROFILE, space_name=SPACE_NAME, space_id=SPACE_ID)
        print("Genie Space: ",genie_client.client.genie.get_space(space_id=SPACE_ID))
        print(f"Connected to Genie Space: '{SPACE_NAME}'")
        print()
        print("=" * 80)
        user_question = input("What would you like to ask Genie? ")
        
        if not user_question.strip():
            print("No question provided. Exiting.")
            return

        agent = GenieAgent(genie_client)
        agent.run(user_question)
        
        print("\nDone! Thank you for using the Genie Question Clarification Agent.")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

