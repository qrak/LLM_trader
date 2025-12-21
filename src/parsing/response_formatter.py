"""
Response Formatter Utility

Handles formatting of AI model responses and error responses.
Extracted from ModelManager to follow SRP (Single Responsibility Principle).
"""

import json
from typing import Dict, Any


class ResponseFormatter:
    """Formats AI model responses and error responses into standardized formats."""
    
    @staticmethod
    def format_error_response(error_message: str) -> str:
        """
        Create a standardized error response in JSON format.
        
        Args:
            error_message: Error message to include in the response
            
        Returns:
            Formatted error response with JSON structure and markdown
        """
        json_fallback = {
            "analysis": {
                "summary": f"Analysis unavailable: {error_message}. Please try again later.",
            }
        }
        return f"```json\n{json.dumps(json_fallback, indent=2)}\n```\n\nThe analysis failed due to a technical issue. Please try again later."
    
    @staticmethod
    def format_provider_error(provider: str, error_detail: str) -> Dict[str, Any]:
        """
        Create a standardized provider error dictionary.
        
        Args:
            provider: Provider name that failed
            error_detail: Details about the error
            
        Returns:
            Error dictionary in ResponseDict format
        """
        return {"error": f"{provider} failed: {error_detail}"}
    
    @staticmethod
    def format_final_fallback_error(provider: str, error_detail: str) -> Dict[str, Any]:
        """
        Create error response for final fallback failure.
        
        Args:
            provider: Last provider attempted
            error_detail: Details about the error
            
        Returns:
            Error dictionary indicating all providers failed
        """
        return {"error": f"All models failed. Last attempt ({provider}): {error_detail}"}
