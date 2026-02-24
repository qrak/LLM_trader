"""
BlockRun.AI client implementation using the official blockrun-llm SDK.
Supports text-only and multimodal (text + image) requests with x402 micropayments.
"""
import io
import base64
from typing import Optional, Dict, Any, List, Union

from src.logger.logger import Logger
from src.platforms.ai_providers.base import BaseAIClient
from src.platforms.ai_providers.response_models import ChatResponseModel
from src.utils.decorators import retry_api_call

try:
    from blockrun_llm import AsyncLLMClient
except ImportError:
    AsyncLLMClient = None


class BlockRunClient(BaseAIClient):
    """Client for handling BlockRun.AI API requests using the official SDK."""

    def __init__(self, wallet_key: str, base_url: str, logger: Logger) -> None:
        super().__init__(logger)
        self._wallet_key = wallet_key
        self.base_url = base_url
        self._client: Optional[Any] = None

    async def _initialize_client(self) -> None:
        """Initialize the BlockRun SDK client."""
        try:
            if AsyncLLMClient is None:
                raise ImportError("blockrun-llm SDK is required but not installed")
            self._client = AsyncLLMClient(private_key=self._wallet_key, api_url=self.base_url)
            self.logger.debug("BlockRun SDK client initialized successfully")
        except ImportError:
            self.logger.error("BlockRun SDK not installed. Run: pip install blockrun-llm")
            raise

    async def close(self) -> None:
        """Close the SDK client."""
        if self._client:
            self.logger.debug("Closing BlockRunClient SDK session")
            self._client = None

    async def _ensure_client(self) -> Any:
        """Ensure a client exists and return it, initializing if needed."""
        if not self._client:
            await self._initialize_client()
        return self._client

    def _redact_private_key(self, message: str) -> str:
        """Redact private key from error messages and logs."""
        if self._wallet_key and len(self._wallet_key) > 10:
            return message.replace(self._wallet_key, f"{self._wallet_key[:6]}...{self._wallet_key[-4:]}")
        return message

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def chat_completion(
        self, model: str, messages: List[Dict[str, Any]], model_config: Dict[str, Any]
    ) -> Optional[ChatResponseModel]:
        """
        Send a chat completion request to the BlockRun API using the SDK.

        Args:
            model: Model name in provider/model format (e.g., openai/gpt-4o, anthropic/claude-sonnet-4)
            messages: List of OpenAI-style messages
            model_config: Configuration parameters for the model

        Returns:
            ChatResponseModel or None if failed
        """
        client = await self._ensure_client()
        try:
            self.logger.debug("Sending request to BlockRun SDK with model: %s", model)
            effective_model = self._ensure_provider_prefix(model)

            # Use base class retry logic for unsupported parameters
            response = await self._execute_with_param_retry(
                client.chat_completion,
                model_config,
                model=effective_model,
                messages=messages
            )
            return self.convert_pydantic_response(response, wrapper_attr='response')
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
            model: Model name in provider/model format (e.g., openai/gpt-4o, anthropic/claude-sonnet-4)
            messages: List of OpenAI-style messages
            chart_image: Chart image as BytesIO, bytes, or file path string
            model_config: Configuration parameters for the model

        Returns:
            ChatResponseModel or None if failed
        """
        client = await self._ensure_client()
        try:
            img_data = self.process_chart_image(chart_image)
            base64_image = base64.b64encode(img_data).decode('utf-8')
            user_text = self._extract_all_user_text_from_messages(messages)
            multimodal_content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
            multimodal_messages = self._prepare_multimodal_messages(
                messages, multimodal_content
            )
            self.logger.debug("Sending chart analysis request to BlockRun SDK (%s bytes)", len(img_data))
            effective_model = self._ensure_provider_prefix(model)

            # Use base class retry logic for unsupported parameters
            response = await self._execute_with_param_retry(
                client.chat_completion,
                model_config,
                model=effective_model,
                messages=multimodal_messages
            )
            if response:
                self.logger.debug("Received successful chart analysis response from BlockRun SDK")
            return self.convert_pydantic_response(response, wrapper_attr='response')
        except Exception as e:
            self.logger.error("Error during BlockRun chart analysis request: %s", self._redact_private_key(str(e)))
            return self._handle_exception(e)

    def _ensure_provider_prefix(self, model: str) -> str:
        """Ensure model has provider/model format. If no slash is present, assume openai."""
        if "/" not in model:
            return f"openai/{model}"
        return model

    def _extract_all_user_text_from_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Extract and concatenate text content from all user messages."""
        user_texts = []
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content", "")
                if content:
                    user_texts.append(content)
        return "\n\n".join(user_texts) if user_texts else ""

    def _prepare_multimodal_messages(
        self,
        messages: List[Dict[str, Any]],
        multimodal_content: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert messages to BlockRun multimodal format."""
        multimodal_messages = []
        for i, message in enumerate(messages):
            if message.get("role") == "system":
                multimodal_messages.append({
                    "role": "user",
                    "content": f"System instructions: {message['content']}"
                })
            elif message.get("role") == "user" and i == len(messages) - 1:
                multimodal_messages.append({
                    "role": "user",
                    "content": multimodal_content
                })
            else:
                multimodal_messages.append(message)
        return multimodal_messages

    def _handle_exception(self, exception: Exception) -> Optional[ChatResponseModel]:
        """Handle BlockRun specific exceptions, falling back to common handler."""
        redacted_error = self._redact_private_key(str(exception))
        self.logger.error("BlockRun API error: %s", redacted_error)
        result = self.handle_common_errors(exception)
        if result:
            return result
        return ChatResponseModel.from_error(redacted_error)
