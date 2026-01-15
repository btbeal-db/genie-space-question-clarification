# Genie Question Clarification Agent 🧞

A LangGraph-powered agent that helps you interactively refine questions for Databricks Genie Space. The agent presents Genie's query plan to you before execution, allowing you to provide feedback and refinements.

## Features

- 🔄 **Interactive Refinement Loop**: Review and adjust Genie's query plan before execution
- 🤖 **LangGraph Agent**: State machine architecture with dedicated LLM review node
- 🎯 **Direct Genie Integration**: Uses Databricks SDK to communicate with Genie Space
- 🧠 **AI-Powered Assumption Extraction**: Uses Databricks Foundation Models (Llama 3.1 405B) to identify top 3 assumptions in query plans
- 💬 **Simple CLI**: Easy-to-use command-line interface

## Architecture

The agent follows this workflow:

```
User Question → Submit to Genie → Get Plan → LLM Review → Present to User
                                              (Extract        ↓
                                              Assumptions) [Approved?]
                                                         ↙        ↘
                                                    YES: Execute   NO: Collect Feedback
                                                         ↓              ↓
                                                      Results      Refine Query
                                                                       ↓
                                                                  (Loop back to LLM Review)
```

## Prerequisites

- Python 3.10 or higher
- [UV](https://github.com/astral-sh/uv) for dependency management
- Databricks CLI configured with profile "FE-EAST"
- Access to the "Healthcare Genie" Genie Space
- Access to Databricks Foundation Model API (for LLM-based assumption extraction)

## Installation

1. **Clone or navigate to the project directory**

2. **Install UV** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Sync dependencies**:
   ```bash
   uv sync
   ```

4. **Ensure Databricks authentication is configured**:
   The agent uses your existing Databricks CLI profile for both Genie API and Foundation Model API access.

## Usage

### Running the Agent

Simply run the main script:

```bash
uv run src/main.py
```

Or alternatively:

```bash
uv run python src/main.py
```

### Example Session

```
🧞 DATABRICKS GENIE QUESTION CLARIFICATION AGENT
================================================================================

This agent will help you refine your questions to Genie Space.
You'll be able to review and adjust the query plan before execution.

🔌 Connecting to Databricks (profile: FE-EAST)...
✅ Connected to Genie Space: 'Healthcare Genie'

================================================================================
❓ What would you like to ask Genie? What is the average patient wait time?

🤔 Asking Genie: 'What is the average patient wait time?'
⏳ Waiting for Genie's response...

================================================================================
📋 GENIE'S PLAN
================================================================================
Description: I'll calculate the average patient wait time from the visits table.

SQL Query:
SELECT AVG(wait_time_minutes) as avg_wait_time
FROM patient_visits
WHERE visit_date >= '2024-01-01'
================================================================================

Does this plan look correct? (yes/no): no
What would you like to change or add? Filter only for emergency department visits

🔄 Refining plan based on feedback: 'Filter only for emergency department visits'
⏳ Waiting for Genie's response...

================================================================================
📋 GENIE'S PLAN
================================================================================
Description: I'll calculate the average patient wait time for emergency department visits.

SQL Query:
SELECT AVG(wait_time_minutes) as avg_wait_time
FROM patient_visits
WHERE visit_date >= '2024-01-01'
  AND department = 'Emergency'
================================================================================

Does this plan look correct? (yes/no): yes

✅ Plan approved! Executing query...
⏳ Running query in Genie...

================================================================================
🎉 QUERY EXECUTED
================================================================================
Status: completed
Statement ID: 01234567-89ab-cdef-0123-456789abcdef

💡 You can view the full results in the Databricks Genie Space UI
================================================================================

✨ Done! Thank you for using the Genie Question Clarification Agent.
```

## Project Structure

```
genie-space-question-clarification/
├── pyproject.toml          # UV/Python project configuration
├── .python-version         # Python version specification
├── README.md               # This file
└── src/
    ├── __init__.py
    ├── main.py            # CLI entry point
    ├── agent.py           # LangGraph agent implementation
    └── genie_client.py    # Databricks Genie API wrapper
```

## Configuration

The agent is currently configured for:
- **Databricks Profile**: `FE-EAST`
- **Genie Space**: `Healthcare Genie`

To modify these settings, edit the constants in `src/main.py`:

```python
PROFILE = "FE-EAST"
SPACE_NAME = "Healthcare Genie"
```

## How It Works

### Components

1. **GenieClient** (`genie_client.py`):
   - Wraps the Databricks SDK Genie API
   - Manages conversations with Genie Space
   - Handles query submission, refinement, and execution

2. **GenieAgent** (`agent.py`):
   - LangGraph state machine with 5 nodes:
     - `get_initial_plan`: Submit question to Genie
     - `review_with_llm`: Extract top 3 assumptions using Databricks Foundation Model
     - `present_plan`: Show plan, assumptions, and query to user and collect feedback
     - `refine_plan`: Send feedback to Genie for refinement
     - `execute_plan`: Execute the approved query
   - Uses conditional edges to route between refinement and execution
   - LLM review runs on every Genie response (initial and refinements)

3. **Main** (`main.py`):
   - CLI interface
   - Initializes clients and agent
   - Handles user input and output

## Limitations

This is a demo application with intentional simplifications:
- No result data retrieval (only execution confirmation)
- Single-question sessions (no multi-turn conversations beyond refinement)
- Minimal error handling for edge cases
- CLI-only interface

## Future Enhancements

Possible improvements for production use:
- Retrieve and display actual query results
- Support for multiple questions in one session
- Web UI for better visualization
- Enhanced error handling and retry logic
- Logging and telemetry
- Support for multiple Genie spaces
- Configuration file for settings

## License

This is a demonstration project for internal use.

