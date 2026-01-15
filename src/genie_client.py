"""Databricks Genie Space API client wrapper."""
from typing import Optional
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import GenieMessage


class GenieClient:
    """Client for interacting with Databricks Genie Space.
    
    This client manages the Genie conversation state and provides methods
    to interact with Genie and extract information from the current response.
    """
    
    def __init__(self, profile: str, space_name: str, space_id: str):
        """Initialize Genie client with Databricks CLI profile.
        
        Args:
            profile: Databricks CLI profile name
            space_name: Name of the Genie space to query
            space_id: ID of the Genie space
        """
        self.client = WorkspaceClient(profile=profile)
        self.space_name = space_name
        self.space_id = space_id
        self.conversation_id = None
        self.current_message = None
    
    def start_conversation(self, question: str) -> None:
        """Start a new conversation with Genie and get the initial plan.
        
        Args:
            question: User's question to ask Genie
        """
        response = self.client.genie.start_conversation_and_wait(
            space_id=self.space_id,
            content=question
        )
        
        self.conversation_id = response.conversation_id
        self.current_message = response
    
    def refine_question(self, feedback: str) -> None:
        """Send refinement feedback and get updated plan.
        
        Args:
            feedback: User's feedback/refinement request
        """
        if not self.conversation_id:
            raise ValueError("No active conversation. Call start_conversation first.")

        response = self.client.genie.create_message_and_wait(
            space_id=self.space_id,
            conversation_id=self.conversation_id,
            content=feedback
        )
        
        self.current_message = response
    
    def get_results(self) -> dict:
        """Get results from the current message (which has already been executed).
        
        Since start_conversation_and_wait and create_message_and_wait execute
        the query automatically, we just need to extract the results.
        
        Returns:
            dict with summary and results
        """
        if not self.current_message:
            raise ValueError("No current message. Call start_conversation first.")
        
        summary = self._extract_text_summary(self.current_message)
        results = self._extract_results_from_message(self.current_message)
        
        return {
            "summary": summary,
            "results": results
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
    
    def _extract_text_summary(self, message: GenieMessage) -> str | None:
        """Extract plain text summary from message attachments.
        
        Args:
            message: GenieMessage object
            
        Returns:
            Plain text summary if available, None otherwise
        """
        try:
            if message.attachments:
                for attachment in message.attachments:
                    if attachment.text is not None:
                        return getattr(attachment.text, 'content', None)
        except (AttributeError, TypeError):
            pass
        
        return None
    
    def _extract_results_from_message(self, message: GenieMessage) -> dict:
        """Extract query results from message query_result and statement execution.
        
        Args:
            message: GenieMessage object with query results
            
        Returns:
            dict with columns and data
        """
        columns = []
        rows = []

        statement_id = None
        try:
            if message.query_result:
                statement_id = getattr(message.query_result, 'statement_id', None)
        except (AttributeError, TypeError):
            pass
        
        if not statement_id:
            return {"columns": columns, "rows": rows, "row_count": 0}
        
        try:
            stmt = self.client.statement_execution.get_statement(statement_id)
            return self._extract_results(stmt)
        except Exception as e:
            print(f"Warning: Could not fetch statement results: {e}")
            return {"columns": columns, "rows": rows, "row_count": 0}
    
    def get_description(self) -> str | None:
        """Get the query description from the current message."""
        return self._get_from_query_attachment('description')
    
    def get_query(self) -> str | None:
        """Get the SQL query from the current message."""
        return self._get_from_query_attachment('query')
    
    def get_message_id(self) -> str | None:
        """Get the ID of the current message."""
        return getattr(self.current_message, 'id', None) if self.current_message else None
    
    def get_attachment_id(self) -> str | None:
        """Get the query attachment ID from the current message."""
        if not self.current_message:
            return None
        try:
            for attachment in self.current_message.attachments:
                if attachment.query is not None:
                    return getattr(attachment, 'attachment_id', None)
        except (AttributeError, TypeError):
            pass
        return None
    
    def _get_from_query_attachment(self, attr_name: str) -> str | None:
        """Extract an attribute from the query attachment.
        
        Args:
            attr_name: Name of the attribute to extract (e.g. 'description', 'query')
            
        Returns:
            Attribute value or None
        """
        if not self.current_message:
            return None
        try:
            for attachment in self.current_message.attachments:
                if attachment.query is not None:
                    return getattr(attachment.query, attr_name, None)
        except (AttributeError, TypeError):
            pass
        return None
