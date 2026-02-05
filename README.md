# Genie Question Clarification Agent

A LangGraph-powered agent that helps you interactively refine questions for Databricks Genie Space before execution.

## Prerequisites

- Python 3.10+
- [UV](https://github.com/astral-sh/uv) package manager
- Databricks CLI configured with a valid profile
- Access to a Genie Space and Databricks Foundation Model API

## Installation

(Note the project is fairly minimal so can reproduce pyproject.toml however you see fit)

I prefer to...

Install UV (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Sync dependencies:
```bash
uv sync
```

## Limitations
This is a demo application and not meant to be used in production without modification. Simplifications are:
- No result data retrieval (only execution confirmation)
- Single-question sessions (no multi-turn conversations beyond refinement)
- Minimal error handling for edge cases
- CLI-only interface

