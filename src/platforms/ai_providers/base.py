"""
Base client for AI providers with shared functionality.
Implements common patterns: context managers, image processing, error handling.
"""
import io
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union, TypedDict

from src.logger.logger import Logger


class UsageDict(TypedDict, total=False):
    """Token usage and cost information from API response."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float


class ResponseDict(TypedDict, total=False):
    """Type for API responses with usage tracking."""
    error: str
    choices: List[Dict[str, Any]]
    usage: UsageDict
    id: str
    model: str


class BaseAIClient(ABC):
    """Abstract base class for AI provider clients."""

    def __init__(self, logger: Logger) -> None:
        self.logger = logger

    async def __aenter__(self):
        """Async context manager entry - calls _initialize_client."""
        await self._initialize_client()
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        """Async context manager exit."""
        await self.close()

    @abstractmethod
    async def _initialize_client(self) -> None:
        """Initialize the underlying SDK client. Called by __aenter__."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close and cleanup the client."""
        pass

    @abstractmethod
    async def chat_completion(
        self, model: str, messages: List[Dict[str, Any]], model_config: Dict[str, Any]
    ) -> Optional[ResponseDict]:
        """Send a chat completion request."""
        pass

    @abstractmethod
    async def chat_completion_with_chart_analysis(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        chart_image: Union[io.BytesIO, bytes, str],
        model_config: Dict[str, Any]
    ) -> Optional[ResponseDict]:
        """Send a chat completion request with chart image analysis."""
        pass

    def process_chart_image(self, chart_image: Union[io.BytesIO, bytes, str]) -> bytes:
        """
        Process chart image from various input formats to bytes.

        Args:
            chart_image: Image as BytesIO stream, raw bytes, or file path string

        Returns:
            Image data as bytes
        """
        if isinstance(chart_image, io.BytesIO):
            chart_image.seek(0)
            img_data = chart_image.read()
            chart_image.seek(0)
            return img_data
        elif isinstance(chart_image, str):
            with open(chart_image, 'rb') as f:
                return f.read()
        return chart_image

    def handle_common_errors(self, exception: Exception) -> Optional[ResponseDict]:
        """
        Handle common API errors across all providers.

        Args:
            exception: The exception that occurred

        Returns:
            Error response dictionary or None for unhandled errors
        """
        error_message = str(exception).lower()
        if "quota" in error_message or "rate limit" in error_message or "resource_exhausted" in error_message:
            self.logger.error(f"Rate limit or quota exceeded: {exception}")
            return {"error": "rate_limit", "details": str(exception)}  # type: ignore
        if "authentication" in error_message or "api key" in error_message or "invalid_api_key" in error_message:
            self.logger.error(f"Authentication error: {exception}")
            return {"error": "authentication", "details": str(exception)}  # type: ignore
        if "timeout" in error_message:
            self.logger.error(f"Timeout error: {exception}")
            return {"error": "timeout", "details": str(exception)}  # type: ignore
        if "503" in str(exception) or "overloaded" in error_message or "unavailable" in error_message:
            self.logger.error(f"Service unavailable/overloaded: {exception}")
            return {"error": "overloaded", "details": str(exception)}  # type: ignore
        if "connection" in error_message or "econnreset" in error_message:
            self.logger.error(f"Connection error: {exception}")
            return {"error": "connection", "details": str(exception)}  # type: ignore
        return None

    def create_response(
        self,
        content: str,
        role: str = "assistant",
        usage: Optional[Dict[str, int]] = None,
        model: Optional[str] = None,
        response_id: Optional[str] = None
    ) -> ResponseDict:
        """
        Create a standardized ResponseDict from content.

        Args:
            content: Response text content
            role: Message role (default: assistant)
            usage: Optional token usage dict
            model: Optional model identifier
            response_id: Optional response ID

        Returns:
            Standardized ResponseDict
        """
        result: ResponseDict = {
            "choices": [{
                "message": {
                    "content": content,
                    "role": role
                }
            }]
        }
        if usage:
            result["usage"] = usage  # type: ignore
        if model:
            result["model"] = model
        if response_id:
            result["id"] = response_id
        return result
