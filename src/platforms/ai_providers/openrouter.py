"""
OpenRouter client implementation using the raw OpenAI SDK (AsyncOpenAI).
Supports text-only and multimodal (text + image) requests with cost tracking.
"""
import asyncio
import base64
import io
from typing import Any

import httpx
from openai import AsyncOpenAI

from src.logger.logger import Logger
from src.platforms.ai_providers.base import BaseAIClient
from src.platforms.ai_providers.response_models import ChatResponseModel
from src.utils.decorators import retry_api_call


class OpenRouterClient(BaseAIClient):
    """Client for handling OpenRouter API requests via its OpenAI-compatible endpoint."""

    def __init__(self, api_key: str, base_url: str, logger: Logger) -> None:
        super().__init__(logger)
        self.api_key = api_key
        self.base_url = base_url
        self._client: AsyncOpenAI | None = None
        # presence_penalty is never forwarded to OpenRouter — kept filtered for parity
        # with the pre-consolidation dedicated-SDK client (SDK 0.11+ dropped it)
        self._known_unsupported_params.add("presence_penalty")

    async def _initialize_client(self) -> None:
        """Initialize the OpenRouter API client."""
        self._client = self._create_client()

    def _create_client(self) -> AsyncOpenAI:
        """Create an OpenAI-compatible client pointed at the OpenRouter API."""
        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

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
            self.logger.debug("Closing OpenRouterClient SDK session")
            await client.close()
        finally:
            self._client = None

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def chat_completion(  # type: ignore[reportIncompatibleMethodOverride]
        self, model: str, messages: list, model_config: dict[str, Any]
    ) -> ChatResponseModel | None:
        """Send a chat completion request to the OpenRouter API."""
        client = self._ensure_client()
        try:
            self.logger.debug("Sending request to OpenRouter API with model: %s", model)

            # shared config dict: copy before popping reasoning; sent via extra_body
            call_config = dict(model_config)
            reasoning_effort = call_config.pop("openrouter_reasoning_effort", None)
            extra_kwargs = {}
            if reasoning_effort:
                extra_kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}

            # Use base class shared retry logic
            response = await self._execute_with_param_retry(
                client.chat.completions.create,
                call_config,
                model=model,
                messages=messages,
                **extra_kwargs
            )
            return self.convert_pydantic_response(response)
        except Exception as e:  # noqa: BLE001
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

        Args:
            model: Model name to use
            messages: list of OpenAI-style messages
            chart_image: Chart image as BytesIO, bytes, or file path string
            model_config: Configuration parameters for the model

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
            multimodal_messages = self._prepare_multimodal_messages(
                messages, multimodal_content
            )
            self.logger.debug("Sending chart analysis request to OpenRouter API (%s bytes)", len(img_data))

            # shared config dict: copy before popping reasoning; sent via extra_body
            call_config = dict(model_config)
            reasoning_effort = call_config.pop("openrouter_reasoning_effort", None)
            extra_kwargs = {}
            if reasoning_effort:
                extra_kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}

            # Use base class shared retry logic
            response = await self._execute_with_param_retry(
                client.chat.completions.create,
                call_config,
                model=model,
                messages=multimodal_messages,
                **extra_kwargs
            )
            if response:
                self.logger.debug("Received successful chart analysis response from OpenRouter API")
            return self.convert_pydantic_response(response)
        except Exception as e:  # noqa: BLE001
            self.logger.error("Error during OpenRouter chart analysis request: %s", str(e))
            return self._handle_exception(e)

    async def get_generation_cost(self, generation_id: str, retry_delay: float = 0.5) -> dict[str, Any] | None:
        """
        Retrieve cost and stats for a specific generation via the REST endpoint.

        Args:
            generation_id: The generation ID from completion response
            retry_delay: Seconds to wait before querying (API may need time to index)

        Returns:
            Dictionary with token counts and costs
        """
        await asyncio.sleep(retry_delay)
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"{self.base_url.rstrip('/')}/generation",
                    params={"id": generation_id},
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
            if response.status_code == 404:
                self.logger.debug(
                    "Generation stats not yet available for %s... (will be indexed shortly)",
                    generation_id[:20]
                )
                return None
            response.raise_for_status()
            data = response.json().get("data")
            if not data:
                return None
            return {
                "model": data.get("model", "unknown"),
                "total_cost": data.get("total_cost", 0),
                "prompt_tokens": data.get("tokens_prompt", 0),
                "completion_tokens": data.get("tokens_completion", 0),
                "native_prompt_tokens": data.get("native_tokens_prompt", 0),
                "native_completion_tokens": data.get("native_tokens_completion", 0),
            }
        except Exception as e:  # noqa: BLE001
            self.logger.warning("Could not retrieve generation stats: %s", e)
            return None

    def _handle_exception(self, exception: Exception) -> ChatResponseModel | None:
        """Handle OpenRouter specific exceptions, falling back to common handler."""
        result = self.handle_common_errors(exception)
        if result:
            return result
        self.logger.error("Unexpected OpenRouter error: %s", exception)
        return None
