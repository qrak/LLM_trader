"""
Google GenAI client implementation using the official Google GenAI SDK.
Supports both text-only and multimodal (text + image) requests for pattern analysis.
"""
import io
from typing import Optional, Dict, Any, List, Union

from google import genai
from google.genai import types

from src.logger.logger import Logger
from src.platforms.ai_providers.base import BaseAIClient
from src.platforms.ai_providers.response_models import ChatResponseModel, UsageModel
from src.utils.decorators import retry_api_call


class GoogleAIClient(BaseAIClient):
    """Client for handling Google AI API requests using the official Google GenAI SDK."""

    def __init__(self, api_key: str, model: str, logger: Logger) -> None:
        """
        Initialize the GoogleAIClient.

        Args:
            api_key: Google AI API key
            model: Model name (e.g., 'gemini-2.5-flash')
            logger: Logger instance
        """
        super().__init__(logger)
        self.api_key = api_key
        self.model = model
        self.client: Optional[genai.Client] = None

    async def _initialize_client(self) -> None:
        """Initialize the Google GenAI client."""
        self.client = genai.Client(api_key=self.api_key)

    async def close(self) -> None:
        """Close the client."""
        if self.client:
            self.client = None
            self.logger.debug("GoogleAIClient closed successfully")

    def _ensure_client(self) -> genai.Client:
        """Ensure a client exists and return it."""
        if not self.client:
            self.client = genai.Client(api_key=self.api_key)
        return self.client

    def _extract_text_from_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Extract combined text content from OpenAI-style messages."""
        text_parts = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                text_parts.append(f"System: {content}")
            else:
                text_parts.append(content)
        return "\n\n".join(text_parts)

    def _extract_text_from_response(self, response) -> str:
        """Extract text content from Google AI response, handling non-text parts gracefully."""
        try:
            text_parts = []
            non_text_parts = []
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                    else:
                        non_text_parts.append(type(part).__name__)
            if non_text_parts:
                self.logger.debug(f"Google AI response contains non-text parts: {non_text_parts}. Extracting text only.")
            return "\n".join(text_parts)
        except Exception as e:
            self.logger.error(f"Failed to extract text from Google AI response: {e}")
            return ""

    def _extract_usage_metadata(self, response) -> Optional[UsageModel]:
        """Extract token usage metadata from Google AI response."""
        try:
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                metadata = response.usage_metadata
                return UsageModel(
                    prompt_tokens=getattr(metadata, 'prompt_token_count', 0),
                    completion_tokens=getattr(metadata, 'candidates_token_count', 0),
                    total_tokens=getattr(metadata, 'total_token_count', 0),
                )
        except Exception as e:
            self.logger.debug(f"Failed to extract usage metadata: {e}")
        return None

    def _create_generation_config(self, model_config: Dict[str, Any], include_thinking: bool = True) -> types.GenerateContentConfig:
        """Create a generation config from model configuration dictionary."""
        thinking_config = None
        if include_thinking:
            thinking_level = model_config.get("thinking_level", "high")
            if thinking_level and thinking_level in ("minimal", "low", "medium", "high"):
                thinking_config = types.ThinkingConfig(thinking_level=thinking_level)
        return types.GenerateContentConfig(
            temperature=model_config.get("temperature", 0.7),
            top_p=model_config.get("top_p", 0.9),
            top_k=model_config.get("top_k", 40),
            max_output_tokens=model_config.get("max_tokens", 32768),
            thinking_config=thinking_config,
        )

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def chat_completion(
        self, model: str, messages: List[Dict[str, Any]], model_config: Dict[str, Any]
    ) -> Optional[ChatResponseModel]:
        """
        Send a chat completion request to the Google AI API.

        Args:
            model: Model name (overrides default if provided)
            messages: List of OpenAI-style messages
            model_config: Configuration parameters for the model

        Returns:
            ChatResponseModel or None if failed
        """
        client = self._ensure_client()
        prompt = self._extract_text_from_messages(messages)
        effective_model = model if model else self.model
        for include_thinking in (True, False):
            try:
                generation_config = self._create_generation_config(model_config, include_thinking=include_thinking)
                self.logger.debug(f"Sending request to Google AI with model: {effective_model} (thinking={include_thinking})")
                response = await client.aio.models.generate_content(
                    model=effective_model,
                    contents=prompt,
                    config=generation_config
                )
                content_text = self._extract_text_from_response(response)
                usage = self._extract_usage_metadata(response)
                self.logger.debug("Received successful response from Google AI")
                return self.create_response(content_text, usage=usage)
            except Exception as e:
                error_str = str(e).lower()
                if include_thinking and ("thinking" in error_str or "400" in error_str or "invalid" in error_str):
                    self.logger.warning(f"Model may not support thinking_config, retrying without it: {e}")
                    continue
                self.logger.error(f"Error during Google AI request: {e}")
                return self._handle_exception(e)
        return None

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def chat_completion_with_chart_analysis(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        chart_image: Union[io.BytesIO, bytes, str],
        model_config: Dict[str, Any]
    ) -> Optional[ChatResponseModel]:
        """
        Send a chat completion request with a chart image for pattern analysis.

        Args:
            model: Model name (overrides default if provided)
            messages: List of OpenAI-style messages
            chart_image: Chart image as BytesIO, bytes, or file path string
            model_config: Configuration parameters for the model

        Returns:
            ChatResponseModel or None if failed
        """
        client = self._ensure_client()
        prompt = self._extract_text_from_messages(messages)
        effective_model = model if model else self.model
        img_data = self.process_chart_image(chart_image)
        image_part = types.Part.from_bytes(data=img_data, mime_type='image/png')
        contents = [prompt, image_part]
        for include_thinking in (True, False):
            try:
                generation_config = self._create_generation_config(model_config, include_thinking=include_thinking)
                self.logger.debug(f"Sending chart analysis to Google AI: {effective_model} (thinking={include_thinking}, {len(img_data)} bytes)")
                response = await client.aio.models.generate_content(
                    model=effective_model,
                    contents=contents,
                    config=generation_config
                )
                content_text = self._extract_text_from_response(response)
                usage = self._extract_usage_metadata(response)
                self.logger.debug("Received successful chart analysis response from Google AI")
                return self.create_response(content_text, usage=usage)
            except Exception as e:
                error_str = str(e).lower()
                if include_thinking and ("thinking" in error_str or "400" in error_str or "invalid" in error_str):
                    self.logger.warning(f"Model may not support thinking_config for chart analysis, retrying without it: {e}")
                    continue
                self.logger.error(f"Error during Google AI chart analysis request: {e}")
                return self._handle_exception(e)
        return None

    def _handle_exception(self, exception: Exception) -> Optional[ChatResponseModel]:
        """Handle Google AI specific exceptions, falling back to common handler."""
        result = self.handle_common_errors(exception)
        if result:
            return result
        sanitized_error = self._sanitize_error_message(str(exception))
        self.logger.error(f"Unexpected Google AI error: {sanitized_error}")
        return None
