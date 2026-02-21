"""
OpenRouter client implementation using the official OpenRouter SDK.
Supports text-only and multimodal (text + image) requests with cost tracking.
"""
import asyncio
import io
import base64
from typing import Optional, Dict, Any, List, Union

from PIL import Image
from openrouter import OpenRouter

from src.logger.logger import Logger
from src.platforms.ai_providers.base import BaseAIClient
from src.platforms.ai_providers.response_models import ChatResponseModel
from src.utils.decorators import retry_api_call


class OpenRouterClient(BaseAIClient):
    """Client for handling OpenRouter API requests using the official SDK."""

    def __init__(self, api_key: str, base_url: str, logger: Logger) -> None:
        super().__init__(logger)
        self.api_key = api_key
        self.base_url = base_url
        self._client: Optional[OpenRouter] = None

    async def _initialize_client(self) -> None:
        """Initialize the OpenRouter SDK client."""
        self._client = OpenRouter(api_key=self.api_key)

    async def close(self) -> None:
        """Close the SDK client."""
        if self._client:
            self.logger.debug("Closing OpenRouterClient SDK session")
            self._client = None

    def _ensure_client(self) -> OpenRouter:
        """Ensure a client exists and return it."""
        if not self._client:
            self._client = OpenRouter(api_key=self.api_key)
        return self._client



    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def chat_completion(
        self, model: str, messages: list, model_config: Dict[str, Any]
    ) -> Optional[ChatResponseModel]:
        """Send a chat completion request to the OpenRouter API using the SDK."""
        client = self._ensure_client()
        try:
            self.logger.debug(f"Sending request to OpenRouter SDK with model: {model}")

            # Use base class shared retry logic
            response = await self._execute_with_param_retry(
                client.chat.send_async,
                model_config,
                model=model,
                messages=messages
            )
            return self.convert_pydantic_response(response)
        except Exception as e:
            return self._handle_exception(e)

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
            model: Model name to use
            messages: List of OpenAI-style messages
            chart_image: Chart image as BytesIO, bytes, or file path string
            model_config: Configuration parameters for the model

        Returns:
            ChatResponseModel or None if failed
        """
        client = self._ensure_client()
        try:
            img_data = self.process_chart_image(chart_image)
            base64_image = base64.b64encode(img_data).decode('utf-8')
            user_text = self._extract_user_text_from_messages(messages)
            multimodal_content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
            multimodal_messages = self._prepare_multimodal_messages(
                messages, multimodal_content
            )
            self.logger.debug(f"Sending chart analysis request to OpenRouter SDK ({len(img_data)} bytes)")

            # Use base class shared retry logic
            response = await self._execute_with_param_retry(
                client.chat.send_async,
                model_config,
                model=model,
                messages=multimodal_messages
            )
            if response:
                self.logger.debug("Received successful chart analysis response from OpenRouter SDK")
            return self.convert_pydantic_response(response)
        except Exception as e:
            self.logger.error(f"Error during OpenRouter chart analysis request: {str(e)}")
            return self._handle_exception(e)

    async def get_generation_cost(self, generation_id: str, retry_delay: float = 0.5) -> Optional[Dict[str, Any]]:
        """
        Retrieve cost and stats for a specific generation.

        Args:
            generation_id: The generation ID from completion response
            retry_delay: Seconds to wait before querying (API may need time to index)

        Returns:
            Dictionary with token counts and costs
        """
        await asyncio.sleep(retry_delay)
        client = self._ensure_client()
        try:
            generation = client.generations.get_generation(id=generation_id)
            if generation and generation.data:
                data = generation.data
                return {
                    "model": getattr(data, 'model', 'unknown'),
                    "total_cost": getattr(data, 'total_cost', 0),
                    "prompt_tokens": getattr(data, 'tokens_prompt', 0),
                    "completion_tokens": getattr(data, 'tokens_completion', 0),
                    "native_prompt_tokens": getattr(data, 'native_tokens_prompt', 0),
                    "native_completion_tokens": getattr(data, 'native_tokens_completion', 0),
                }
            return None
        except Exception as e:
            error_msg = str(e)
            if "not found" in error_msg.lower():
                self.logger.debug(f"Generation stats not yet available for {generation_id[:20]}... (will be indexed shortly)")
            else:
                self.logger.warning(f"Could not retrieve generation stats: {error_msg}")
            return None

    def _extract_user_text_from_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Extract text content from the last user message."""
        for message in reversed(messages):
            if message["role"] == "user":
                return message["content"]
        return ""

    def _prepare_multimodal_messages(
        self,
        messages: List[Dict[str, Any]],
        multimodal_content: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert messages to OpenRouter multimodal format."""
        multimodal_messages = []
        for message in messages:
            if message["role"] == "system":
                multimodal_messages.append({
                    "role": "user",
                    "content": f"System instructions: {message['content']}"
                })
            elif message["role"] == "user" and message == messages[-1]:
                multimodal_messages.append({
                    "role": "user",
                    "content": multimodal_content
                })
            else:
                multimodal_messages.append(message)
        return multimodal_messages

    def _process_image(self, image: Union[Image.Image, bytes, str]) -> bytes:
        """Process PIL Image and return as bytes."""
        if isinstance(image, Image.Image):
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            return img_buffer.getvalue()
        return self.process_chart_image(image)

    def _handle_exception(self, exception: Exception) -> Optional[ChatResponseModel]:
        """Handle OpenRouter specific exceptions, falling back to common handler."""
        result = self.handle_common_errors(exception)
        if result:
            return result
        self.logger.error(f"Unexpected OpenRouter error: {exception}")
        return None
