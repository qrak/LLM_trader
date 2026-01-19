"""
Base client for AI providers with shared functionality.
Implements common patterns: context managers, image processing, error handling.
"""
import io
import re
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union, Callable

from src.logger.logger import Logger
from .response_models import ChatResponseModel, ChoiceModel, MessageModel, UsageModel


class BaseAIClient(ABC):
    """Abstract base class for AI provider clients."""

    def __init__(self, logger: Logger) -> None:
        self.logger = logger
        # Common unsupported parameters to pre-filter
        self._known_unsupported_params = {'thinking_budget', 'thinking_config', 'top_k'}

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
    ) -> Optional[ChatResponseModel]:
        """Send a chat completion request."""
        pass

    @abstractmethod
    async def chat_completion_with_chart_analysis(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        chart_image: Union[io.BytesIO, bytes, str],
        model_config: Dict[str, Any]
    ) -> Optional[ChatResponseModel]:
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

    def _sanitize_error_message(self, message: str) -> str:
        """
        Sanitize error message by redaction of sensitive information like API keys.

        Args:
            message: The raw error message string.

        Returns:
            Sanitized string with API keys redacted.
        """
        sanitized = message
        # If the instance has an api_key attribute, try to redact it
        if hasattr(self, 'api_key') and self.api_key:
            if isinstance(self.api_key, str) and len(self.api_key) > 5:
                sanitized = sanitized.replace(self.api_key, "[REDACTED_API_KEY]")

        return sanitized

    def handle_common_errors(self, exception: Exception) -> Optional[ChatResponseModel]:
        """
        Handle common API errors across all providers.

        Args:
            exception: The exception that occurred

        Returns:
            Error response or None for unhandled errors
        """
        # Sanitize exception string before logging
        error_message_raw = str(exception)
        error_message_sanitized = self._sanitize_error_message(error_message_raw)
        error_message_lower = error_message_sanitized.lower()

        if "quota" in error_message_lower or "rate limit" in error_message_lower or "resource_exhausted" in error_message_lower:
            self.logger.error(f"Rate limit or quota exceeded: {error_message_sanitized}")
            return ChatResponseModel.from_error(f"rate_limit: {error_message_sanitized}")
        if "authentication" in error_message_lower or "api key" in error_message_lower or "invalid_api_key" in error_message_lower:
            self.logger.error(f"Authentication error: {error_message_sanitized}")
            return ChatResponseModel.from_error(f"authentication: {error_message_sanitized}")
        if "timeout" in error_message_lower:
            self.logger.error(f"Timeout error: {error_message_sanitized}")
            return ChatResponseModel.from_error(f"timeout: {error_message_sanitized}")
        if "503" in error_message_raw or "overloaded" in error_message_lower or "unavailable" in error_message_lower:
            self.logger.error(f"Service unavailable/overloaded: {error_message_sanitized}")
            return ChatResponseModel.from_error(f"overloaded: {error_message_sanitized}")
        if "connection" in error_message_lower or "econnreset" in error_message_lower:
            self.logger.error(f"Connection error: {error_message_sanitized}")
            return ChatResponseModel.from_error(f"connection: {error_message_sanitized}")
        return None

    def convert_pydantic_response(
        self,
        response: Any,
        wrapper_attr: Optional[str] = None
    ) -> ChatResponseModel:
        """
        Convert any Pydantic SDK response to ChatResponseModel.
        Used by: BlockRun (wrapper_attr='response'), OpenRouter (no wrapper)

        Args:
            response: SDK response (Pydantic model)
            wrapper_attr: Unwrap attribute (e.g., 'response' for ChatResponseWithCost)
        """
        if response is None:
            return ChatResponseModel.from_error("Empty response from SDK")
        inner = getattr(response, wrapper_attr, response) if wrapper_attr else response
        try:
            return ChatResponseModel(
                choices=[
                    ChoiceModel(
                        message=MessageModel(
                            role=getattr(choice.message, 'role', 'assistant') if choice.message else "assistant",
                            content=getattr(choice.message, 'content', '') if choice.message else ""
                        ),
                        finish_reason=getattr(choice, 'finish_reason', None)
                    )
                    for choice in (inner.choices or [])
                ],
                usage=UsageModel(
                    prompt_tokens=getattr(inner.usage, 'prompt_tokens', 0) or 0,
                    completion_tokens=getattr(inner.usage, 'completion_tokens', 0) or 0,
                    total_tokens=getattr(inner.usage, 'total_tokens', 0) or 0
                ) if inner.usage else None,
                id=getattr(inner, 'id', None),
                model=getattr(inner, 'model', None)
            )
        except (AttributeError, TypeError) as e:
            self.logger.error(f"Failed to create response model: {e}")
            return ChatResponseModel.from_error(f"Response creation error: {e}")

    def _detect_unsupported_param(self, error_msg: str) -> Optional[str]:
        """
        Detect which parameter caused the error from error message.
        Shared logic for all providers to handle SDK strictness.
        """
        # Python keyword argument error
        match = re.search(r"unexpected keyword argument '(\w+)'", error_msg)
        if match:
            return match.group(1)
        # API error message format 1
        match = re.search(r"unknown (parameter|argument)[:\s]+['\"]?(\w+)['\"]?", error_msg, re.IGNORECASE)
        if match:
            return match.group(2)
        # API error message format 2 (e.g. "Additional properties are not allowed ('top_k' was unexpected)")
        match = re.search(r"Additional properties are not allowed \('(\w+)' was unexpected\)", error_msg)
        if match:
            return match.group(1)
        return None

    async def _execute_with_param_retry(
        self, 
        func: Callable[..., Any], 
        config: Dict[str, Any], 
        **fixed_args: Any
    ) -> Any:
        """
        Execute an SDK function with automatic retry handling for unsupported parameters.
        
        Args:
            func: Async function to call (e.g., client.chat.completions.create)
            config: Configuration dictionary that might contain unsupported params (will be unpacked)
            **fixed_args: Fixed named arguments to pass to the function (e.g., model, messages)
            
        Returns:
            The result of the function call
            
        Raises:
            Exception: If the call fails after retries or for non-parameter reasons
        """
        # Start with a copy of config and pre-filter known unsupported params
        current_config = {k: v for k, v in config.items() if k not in self._known_unsupported_params}
        rejected_params = set()
        max_retries = 3
        
        for attempt in range(max_retries + 1):
            try:
                # Call function with fixed args AND unpacked config
                return await func(**fixed_args, **current_config)
            except Exception as e:
                # Only retry if we haven't exhausted retries
                if attempt == max_retries:
                    raise
                
                error_msg = str(e)
                bad_param = self._detect_unsupported_param(error_msg)
                
                # If we found a bad parameter that is currently in our config
                if bad_param and bad_param in current_config:
                    self.logger.warning(
                        f"Parameter '{bad_param}' not supported by provider/model. "
                        f"Retrying without it (Attempt {attempt + 1}/{max_retries})"
                    )
                    rejected_params.add(bad_param)
                    # Create new config without the bad parameter
                    current_config = {k: v for k, v in current_config.items() if k not in rejected_params}
                    continue
                
                # If it's not a parameter error or we can't identify the parameter, re-raise
                raise

    def create_response(
        self,
        content: str,
        role: str = "assistant",
        usage: Optional[UsageModel] = None,
        model: Optional[str] = None,
        response_id: Optional[str] = None
    ) -> ChatResponseModel:
        """
        Create a ChatResponseModel from content.
        Used by: Google, LMStudio (providers with custom extraction logic)

        Args:
            content: Response text content
            role: Message role (default: assistant)
            usage: Optional token usage
            model: Optional model identifier
            response_id: Optional response ID

        Returns:
            ChatResponseModel instance
        """
        return ChatResponseModel.from_content(
            content=content,
            role=role,
            usage=usage,
            model=model,
            response_id=response_id
        )
