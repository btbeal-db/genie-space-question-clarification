"""Databricks Genie Space API client wrapper."""
from typing import Optional
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import GenieMessage


class GenieClient:
    """Client for interacting with Databricks Genie Space."""
    
    def __init__(self, profile: str, space_name: str, space_id: str):
        """Initialize Genie client with Databricks CLI profile.
        
        Args:
            profile: Databricks CLI profile name
            space_name: Name of the Genie space to query
        """
        self.client = WorkspaceClient(profile=profile)
        self.space_name = space_name
        self.space_id = space_id
        self.conversation_id = None
    
    def start_conversation(self, question: str) -> dict:
        """Start a new conversation with Genie and get the initial plan.
        
        Args:
            question: User's question to ask Genie
            
        Returns:
            dict with 'plan' (description of what Genie will do) and 'conversation_id'
        """

        response = self.client.genie.start_conversation_and_wait(
            space_id=self.space_id,
            content=question
        )
        
        self.conversation_id = response.conversation_id

        query_info = self._extract_query_info(response)
        
        return {
            "description": query_info["description"],
            "query": query_info["query"],
            "conversation_id": response.conversation_id,
            "message_id": response.id
        }
    
    def refine_question(self, feedback: str) -> dict:
        """Send refinement feedback and get updated plan.
        
        Args:
            feedback: User's feedback/refinement request
            
        Returns:
            dict with updated 'plan' and 'conversation_id'
        """
        if not self.conversation_id:
            raise ValueError("No active conversation. Call start_conversation first.")

        response = self.client.genie.create_message_and_wait(
            space_id=self.space_id,
            conversation_id=self.conversation_id,
            content=feedback
        )
        
        query_info = self._extract_query_info(response)
        
        return {
            "description": query_info["description"],
            "query": query_info["query"],
            "conversation_id": response.conversation_id,
            "message_id": response.id
        }
    
    def execute_plan(self, message_id: str) -> dict:
        """Execute the approved plan and get results.
        
        Args:
            message_id: ID of the message containing the plan to execute
            
        Returns:
            dict with execution info and results
        """
        execute_response = self.client.genie.execute_message_query(
            space_id=self.space_id,
            conversation_id=self.conversation_id,
            message_id=message_id
        )

        statement_id = None
        if hasattr(execute_response, 'statement_response') and execute_response.statement_response:
            statement_id = execute_response.statement_response.statement_id

        max_attempts = 30
        for attempt in range(max_attempts):
            result_response = self.client.genie.get_message_query_result(
                space_id=self.space_id,
                conversation_id=self.conversation_id,
                message_id=message_id
            )
            
            if hasattr(result_response, 'statement_response') and result_response.statement_response:
                stmt = result_response.statement_response

                status_state = None
                if hasattr(stmt, 'status') and stmt.status and hasattr(stmt.status, 'state'):
                    status_state = stmt.status.state.value if hasattr(stmt.status.state, 'value') else str(stmt.status.state)

                if status_state == 'SUCCEEDED':
                    results = self._extract_results(stmt)
                    return {
                        "executed": True,
                        "statement_id": statement_id,
                        "status": status_state,
                        "results": results
                    }
                elif status_state in ['FAILED', 'CANCELED']:
                    error_msg = None
                    if hasattr(stmt, 'status') and hasattr(stmt.status, 'error'):
                        error_msg = str(stmt.status.error)
                    return {
                        "executed": True,
                        "statement_id": statement_id,
                        "status": status_state,
                        "error": error_msg
                    }

            time.sleep(2)

        return {
            "executed": True,
            "statement_id": statement_id,
            "status": "TIMEOUT",
            "error": "Query did not complete within timeout period"
        }
    
    def _extract_results(self, statement_response) -> dict:
        """Extract results from a statement response.
        
        Args:
            statement_response: StatementResponse object
            
        Returns:
            dict with columns and data
        """
        columns = []
        rows = []

        try:
            schema = statement_response.manifest.schema
            columns = [col.name for col in schema.columns]
        except (AttributeError, TypeError):
            pass  # No schema available

        statement_id = getattr(statement_response, 'statement_id', None)
        if not statement_id:
            return {"columns": columns, "rows": rows, "row_count": 0}

        try:
            result = statement_response.result

            if result.data_array is not None:
                rows = result.data_array
            else:
                chunks = statement_response.manifest.chunks
                if chunks:
                    print(f"\nFetching {len(chunks)} chunk(s)...")
                    
                    for chunk_info in chunks:
                        chunk_index = getattr(chunk_info, 'chunk_index', 0)
                        
                        try:
                            chunk_data = self.client.statement_execution.get_statement_result_chunk_n(
                                statement_id=statement_id,
                                chunk_index=chunk_index
                            )
                            
                            if chunk_data.data_array:
                                rows.extend(chunk_data.data_array)
                        except Exception as e:
                            print(f"Warning: Failed to fetch chunk {chunk_index}: {e}")
        except (AttributeError, TypeError):
            pass  # No result data available
        
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows)
        }
    
    def _extract_query_info(self, message: GenieMessage) -> dict:
        """Extract query information from Genie message attachments.
        
        Args:
            message: Genie message response
            
        Returns:
            dict with 'description' and 'query' from GenieQueryAttachment, or None values if not found
        """
        try:
            for attachment in message.attachments:
                if attachment.query is not None:
                    return {
                        "description": getattr(attachment.query, 'description', None),
                        "query": getattr(attachment.query, 'query', None)
                    }
        except (AttributeError, TypeError):
            pass
        
        return {"description": None, "query": None}
    

