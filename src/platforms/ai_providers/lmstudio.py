"""
LM Studio client implementation using the server's OpenAI-compatible endpoint.
Supports text-only streaming and multimodal (text + image) requests for local inference.
"""
import base64
import io
from typing import Any

from openai import AsyncOpenAI

from src.logger.logger import Logger
from src.platforms.ai_providers.base import BaseAIClient
from src.platforms.ai_providers.response_models import ChatResponseModel, UsageModel
from src.utils.decorators import retry_api_call


class LMStudioClient(BaseAIClient):
    """Client for handling LM Studio API requests via its OpenAI-compatible endpoint."""

    def __init__(self, base_url: str, logger: Logger) -> None:
        super().__init__(logger)
        self.base_url = base_url
        self._client: AsyncOpenAI | None = None
        self._cached_model: str | None = None
        self._known_unsupported_params.add("openrouter_reasoning_effort")

    async def _initialize_client(self) -> None:
        """Initialize the LM Studio API client."""
        self._client = self._create_client()

    def _create_client(self) -> AsyncOpenAI:
        """Create an OpenAI-compatible client pointed at the LM Studio server."""
        return AsyncOpenAI(base_url=self.base_url, api_key="lm-studio")

    def _ensure_client(self) -> AsyncOpenAI:
        """Ensure a client exists and return it."""
        if not self._client:
            self._client = self._create_client()
        return self._client

    async def close(self) -> None:
        """Close the SDK client."""
        client = self._client
        if not client:
            return
        try:
            self.logger.debug("Closing LMStudioClient SDK session")
            await client.close()
        finally:
            self._client = None

    async def _get_model_or_auto_select(self, client: AsyncOpenAI, model: str) -> str:
        """Get model name or auto-select from loaded models (cached)."""
        if model:
            return model
        if self._cached_model:
            return self._cached_model
        loaded_models = await client.models.list()
        if loaded_models.data:
            self._cached_model = loaded_models.data[0].id
            self.logger.info("Auto-selected loaded model: %s", self._cached_model)
            return self._cached_model
        raise ValueError("No model specified and no models loaded in LM Studio")

    def _build_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rewrite messages for the LM Studio endpoint: system prompts become "System: ..." user messages (legacy shape 1:1)."""
        prepared = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                prepared.append({"role": "user", "content": f"System: {content}"})
            elif role == "user":
                prepared.append({"role": "user", "content": content})
            elif role == "assistant":
                prepared.append({"role": "assistant", "content": content})
        return prepared

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def chat_completion(  # type: ignore[reportIncompatibleMethodOverride]
        self, model: str, messages: list, model_config: dict[str, Any]
    ) -> ChatResponseModel | None:
        """Send a chat completion request to the LM Studio API."""
        client = self._ensure_client()
        try:
            model = await self._get_model_or_auto_select(client, model)
            self.logger.debug("Sending request to LM Studio API with model: %s", model)
            response = await self._execute_with_param_retry(
                client.chat.completions.create,
                model_config,
                model=model,
                messages=self._build_messages(messages)
            )
            self.logger.debug("Received successful response from LM Studio API")
            return self.convert_pydantic_response(response)
        except Exception as e:  # noqa: BLE001
            self.logger.error("Error during LM Studio request: %s", str(e))
            return self._handle_exception(e)

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def chat_completion_with_chart_analysis(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        model: str,
        messages: list[dict[str, Any]],
        chart_image: io.BytesIO | bytes | str,
        model_config: dict[str, Any]
    ) -> ChatResponseModel | None:
        """
        Send a chat completion request with a chart image for pattern analysis.
        Returns:
            ChatResponseModel or None if failed
        """
        client = self._ensure_client()
        try:
            img_data = self.process_chart_image(chart_image)
            base64_image = base64.b64encode(img_data).decode("utf-8")
            user_text = self._extract_user_text_from_messages(messages)
            multimodal_content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
            prepared = self._attach_multimodal_content(self._build_messages(messages), multimodal_content)
            model = await self._get_model_or_auto_select(client, model)
            self.logger.debug("Sending chart analysis request to LM Studio API with model: %s", model)
            response = await self._execute_with_param_retry(
                client.chat.completions.create,
                model_config,
                model=model,
                messages=prepared
            )
            self.logger.debug("Received successful chart analysis response from LM Studio API")
            return self.convert_pydantic_response(response)
        except Exception as e:  # noqa: BLE001
            self.logger.error("Error during LM Studio chart analysis request: %s", str(e))
            return self._handle_exception(e)

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def stream_chat_completion(
        self,
        model: str,
        messages: list,
        model_config: dict[str, Any],
        callback=None
    ) -> ChatResponseModel | None:
        """Send a streaming chat completion request to the LM Studio API.

        Partial-output semantics (1:1 with the previous SDK client): if the stream
        fails after any content arrived, return what was received instead of raising.
        """
        client = self._ensure_client()
        try:
            model = await self._get_model_or_auto_select(client, model)
            self.logger.debug("Sending streaming request to LM Studio API with model: %s", model)
            stream = await self._execute_with_param_retry(
                client.chat.completions.create,
                model_config,
                model=model,
                messages=self._build_messages(messages),
                stream=True
            )
            complete_content = ""
            stream_usage = None
            try:
                async for chunk in stream:
                    if chunk.usage:
                        stream_usage = chunk.usage
                    if not chunk.choices:
                        continue
                    text = chunk.choices[0].delta.content or ""
                    if not text:
                        continue
                    complete_content += text
                    if callback:
                        await callback(text)
            except Exception:
                if not complete_content:
                    raise
            self.logger.debug("Streaming response from LM Studio completed")
            if stream_usage:
                usage = UsageModel(
                    prompt_tokens=stream_usage.prompt_tokens or 0,
                    completion_tokens=stream_usage.completion_tokens or 0,
                    total_tokens=stream_usage.total_tokens or 0,
                )
            else:
                usage = UsageModel(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            return self.create_response(content=complete_content, usage=usage)
        except Exception as e:  # noqa: BLE001
            self.logger.error("Error during LM Studio streaming request: %s", str(e))
            return self._handle_exception(e)

    def _handle_exception(self, exception: Exception) -> ChatResponseModel | None:
        """Handle LM Studio specific exceptions, falling back to common handler."""
        error_message = str(exception)
        if "ErrorDeviceLost" in error_message or "vk::Queue::submit" in error_message:
            friendly_msg = (
                "GPU Error detected on LM Studio server. "
                "Your GPU driver may have crashed or run out of memory. "
                "Try using a smaller model or reducing 'n_gpu_layers' in LM Studio."
            )
            self.logger.error("LM Studio GPU Crash: %s", friendly_msg)
            return ChatResponseModel.from_error(f"gpu_crash: {friendly_msg}")
        result = self.handle_common_errors(exception)
        if result:
            return result
        self.logger.error("LM Studio Error: %s", error_message)
        return None
