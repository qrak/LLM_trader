import asyncio
from typing import Optional, Dict, Any

import aiohttp

from src.logger.logger import Logger


class BaseApiClient:
    """Base class for API clients with common functionality."""
    
    def __init__(self, api_key: str, base_url: str, logger: Logger) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.logger = logger
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        await self.close()
    
    async def close(self) -> None:
        """Close the aiohttp session."""
        if self.session:
            try:
                self.logger.debug(f"Closing {self.__class__.__name__} session")
                await self.session.close()
                self.session = None
            except Exception as e:
                self.logger.error(f"Error closing session in {self.__class__.__name__}: {e}")
    
    def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure a session exists and return it."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def _handle_error_response(self, response: aiohttp.ClientResponse, model: str) -> Dict[str, Any]:
        """Handle error responses from APIs."""
        try:
            error_text = await response.text()
        except Exception:
            error_text = "Failed to read error response"
        
        if error_text is None:
            error_text = "No error details available"
            
        # Log as debug - higher level will log with proper context
        self.logger.debug(f"API Error for model {model}: Status {response.status} - {error_text}")
        
        error_details = {
            401: "Authentication error with API key. Check your API key.",
            403: "Permission denied. Your API key may not have access to this model.",
            404: f"Model {model} not found or doesn't support this operation.",
            408: "Request timeout. The server took too long to respond.",
            429: "Rate limit exceeded. Consider upgrading your plan." if "Rate limit exceeded" in error_text else "Too many requests. Temporary rate limit."
        }
        
        # Log detailed error info as debug - higher level decides what to show user
        if response.status in error_details:
            self.logger.debug(error_details[response.status])
            
        if response.status >= 500:
            self.logger.debug(f"Server error. The service may be experiencing issues.")
            
        if "Rate limit exceeded" in error_text:
            return {"error": "rate_limit", "details": error_text}
        
        if response.status == 408 or "timeout" in error_text.lower():
            return {"error": "timeout", "details": error_text}
                
        return {"error": f"http_{response.status}", "details": error_text}
    
    async def _handle_common_exceptions(self, model: str, operation: str = "request") -> Optional[Dict[str, Any]]:
        """
        Common exception handling for all API clients.
        Returns a dictionary for specific exceptions that should be handled differently,
        None for exceptions that should be re-raised.
        """
        try:
            raise  # Re-raise the current exception
        except asyncio.TimeoutError as e:
            self.logger.error(f"Timeout error when {operation} for model {model}: {e}")
            return {"error": "timeout", "details": str(e)}
        except aiohttp.ClientPayloadError as e:
            # Incomplete payload transfer - retryable
            self.logger.error(f"Payload transfer error when {operation} for model {model}: {type(e).__name__} - {e}")
            return {"error": {"code": 502, "message": str(e)}, "details": "Incomplete payload transfer"}
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error when {operation} for model {model}: {type(e).__name__} - {e}")
            return {"error": {"code": 503, "message": str(e)}, "details": "Network error"}
        except Exception as e:
            self.logger.error(f"Unexpected error when {operation} for model {model}: {type(e).__name__} - {e}")
            return None
    
    async def _make_post_request(self, 
                                url: str,
                                headers: Dict[str, str], 
                                payload: Dict[str, Any],
                                model: str,
                                timeout: int = 300) -> Optional[Dict[str, Any]]:
        """
        Common POST request method with standardized error handling.
        
        Args:
            url: The URL to send the request to
            headers: Request headers
            payload: Request payload
            model: Model name for logging
            timeout: Request timeout in seconds
            
        Returns:
            Response data or error dictionary
        """
        session = self._ensure_session()
        
        try:
            request_id = id(payload.get('messages', payload))
            self.logger.debug(f"Sending request #{request_id} to {self.__class__.__name__} with model: {model}")

            async with session.post(url, headers=headers, json=payload, timeout=timeout) as response:
                if response.status != 200:
                    return await self._handle_error_response(response, model)
                
                response_json = await response.json()
                if "error" in response_json:
                    self.logger.error(f"API returned error payload for model {model}: {response_json['error']}")
                    return response_json
                    
                self.logger.debug(f"Received successful response for model {model}")
                return response_json
                    
        except Exception:
            return await self._handle_common_exceptions(model, "requesting")