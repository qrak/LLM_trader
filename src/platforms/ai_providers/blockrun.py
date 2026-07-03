"""
BlockRun.AI client implementation using the official blockrun-llm SDK (v1.4.7+).
Supports text-only and multimodal (text + image) chat with x402 micropayments.
"""
import io
import base64
from typing import Any, Union

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
        self._base_url = base_url
        self._client: Any | None = None

    async def _initialize_client(self) -> None:
        """Initialize the BlockRun SDK client."""
        if AsyncLLMClient is None:
            raise ImportError("blockrun-llm SDK is required but not installed")
        self._client = AsyncLLMClient(
            private_key=self._wallet_key,
            api_url=self._base_url,
        )
        self.logger.debug("BlockRun SDK client initialized successfully")

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
        self, model: str, messages: list[dict[str, Any]], model_config: dict[str, Any]
    ) -> ChatResponseModel | None:
        """
        Send a chat completion request to the BlockRun API using the SDK.

        Args:
            model: Model name in provider/model format (e.g., openai/gpt-4o, anthropic/claude-sonnet-4)
            messages: list of OpenAI-style messages
            model_config: Configuration parameters (temperature, max_tokens, etc.)

        Returns:
            ChatResponseModel or None if failed
        """
        client = await self._ensure_client()
        try:
            self.logger.debug("Sending request to BlockRun SDK with model: %s", model)
            effective_model = self._ensure_provider_prefix(model)

            # Extract config params — SDK now takes them as kwargs, not a dict
            kwargs = self._build_chat_kwargs(model_config)
            response = await client.chat_completion(
                model=effective_model,
                messages=messages,
                **kwargs,
            )
            return self._convert_response(response)
        except Exception as e:
            return self._handle_exception(e)

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def chat_completion_with_chart_analysis(
        self,
        model: str,
        messages: list[dict[str, Any]],
        chart_image: Union[io.BytesIO, bytes, str],
        model_config: dict[str, Any],
    ) -> ChatResponseModel | None:
        """
        Send a chat completion request with a chart image for pattern analysis.

        Args:
            model: Model name in provider/model format
            messages: list of OpenAI-style messages
            chart_image: Chart image as BytesIO, bytes, or file path string
            model_config: Configuration parameters

        Returns:
            ChatResponseModel or None if failed
        """
        client = await self._ensure_client()
        try:
            img_data = self.process_chart_image(chart_image)
            base64_image = base64.b64encode(img_data).decode("utf-8")
            user_text = self._extract_all_user_text_from_messages(messages)
            multimodal_content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            ]
            multimodal_messages = self._prepare_multimodal_messages(messages, multimodal_content)
            self.logger.debug("Sending chart analysis request to BlockRun SDK (%s bytes)", len(img_data))
            effective_model = self._ensure_provider_prefix(model)

            kwargs = self._build_chat_kwargs(model_config)
            response = await client.chat_completion(
                model=effective_model,
                messages=multimodal_messages,
                **kwargs,
            )
            if response:
                self.logger.debug("Received successful chart analysis response from BlockRun SDK")
            return self._convert_response(response)
        except Exception as e:
            self.logger.error("Error during BlockRun chart analysis request: %s", self._redact_private_key(str(e)))
            return self._handle_exception(e)

    def _build_chat_kwargs(self, model_config: dict[str, Any]) -> dict[str, Any]:
        """Convert model_config dict to SDK-compatible kwargs."""
        kwargs: dict[str, Any] = {}
        if "temperature" in model_config and model_config["temperature"] is not None:
            kwargs["temperature"] = model_config["temperature"]
        if "max_tokens" in model_config and model_config["max_tokens"] is not None:
            kwargs["max_tokens"] = model_config["max_tokens"]
        if "top_p" in model_config and model_config["top_p"] is not None:
            kwargs["top_p"] = model_config["top_p"]
        if "top_k" in model_config and model_config["top_k"] is not None:
            kwargs["top_k"] = model_config["top_k"]
        return kwargs

    def _convert_response(self, response: Any) -> ChatResponseModel | None:
        """Convert SDK ChatResponse to our internal ChatResponseModel."""
        if response is None:
            return None
        # SDK v1.4.7 returns ChatResponse with choices[0].message.content
        content = ""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and choice.message:
                content = choice.message.content or ""
        else:
            # Fallback: try dict-like access
            try:
                content = response["choices"][0]["message"]["content"]
            except (KeyError, TypeError, IndexError):
                content = ""

        if not content:
            self.logger.warning("BlockRun returned empty content")
            return ChatResponseModel.from_error("BlockRun returned empty content")

        return ChatResponseModel.from_content(
            content=content,
            model=getattr(response, "model", "unknown"),
        )

    def _ensure_provider_prefix(self, model: str) -> str:
        """Ensure model has provider/model format. Default to openai/ prefix if missing."""
        if "/" not in model:
            return f"openai/{model}"
        return model

    def _extract_all_user_text_from_messages(self, messages: list[dict[str, Any]]) -> str:
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
        messages: list[dict[str, Any]],
        multimodal_content: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert messages to BlockRun multimodal format."""
        multimodal_messages = []
        for i, message in enumerate(messages):
            if message.get("role") == "system":
                multimodal_messages.append({
                    "role": "user",
                    "content": f"System instructions: {message['content']}",
                })
            elif message.get("role") == "user" and i == len(messages) - 1:
                multimodal_messages.append({
                    "role": "user",
                    "content": multimodal_content,
                })
            else:
                multimodal_messages.append(message)
        return multimodal_messages

    def _handle_exception(self, exception: Exception) -> ChatResponseModel | None:
        """Handle BlockRun specific exceptions, falling back to common handler."""
        redacted_error = self._redact_private_key(str(exception))
        self.logger.error("BlockRun API error: %s", redacted_error)
        result = self.handle_common_errors(exception)
        if result:
            return result
        return ChatResponseModel.from_error(redacted_error)
