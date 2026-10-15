"""
DeepSeek client implementation using the official OpenAI-compatible API.

Supports text-only and multimodal (text + image) requests via the official
api.deepseek.com endpoint. The regular flash model accepts images natively
(verified live 2026-09-12) — one model serves text and charts; there is no
separate vision model.
"""
import io
from typing import Any

from openai import AsyncOpenAI

from src.logger.logger import Logger
from src.platforms.ai_providers.base import BaseAIClient
from src.platforms.ai_providers.response_models import ChatResponseModel
from src.utils.decorators import retry_api_call


class DeepSeekClient(BaseAIClient):
    """Client for handling DeepSeek API requests using the official OpenAI-compatible API."""

    def __init__(self, api_key: str, base_url: str, logger: Logger) -> None:
        super().__init__(logger)
        self.api_key = api_key
        self.base_url = base_url

    def _build_client(self) -> AsyncOpenAI:
        """Create an OpenAI-compatible client pointed at the DeepSeek API."""
        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30, retry_on_empty=True)
    async def chat_completion(  # type: ignore[reportIncompatibleMethodOverride]
        self, model: str, messages: list, model_config: dict[str, Any]
    ) -> ChatResponseModel | None:
        """Send a chat completion request to the DeepSeek API."""
        client = self._ensure_client()
        try:
            self.logger.debug("Sending request to DeepSeek API with model: %s", model)
            response = await self._execute_with_param_retry(
                client.chat.completions.create,
                model_config,
                model=model,
                messages=messages
            )
            self._log_usage(response)
            return self.convert_pydantic_response(response)
        except Exception as e:  # noqa: BLE001
            return self._handle_exception(e)

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30, retry_on_empty=True)
    async def chat_completion_with_chart_analysis(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        model: str,
        messages: list[dict[str, Any]],
        chart_image: io.BytesIO | bytes | str,
        model_config: dict[str, Any]
    ) -> ChatResponseModel | None:
        """
        Send a chat completion request with a chart image for pattern analysis.

        Images go inside user messages (official docs used to require the
        vision-exp model; the regular flash model accepts images as of
        2026-09-12), so the last user message carries the multimodal block.
        Returns:
            ChatResponseModel or None if failed
        """
        client = self._ensure_client()
        try:
            user_text = self._extract_user_text_from_messages(messages)
            img_data, multimodal_content = self._build_multimodal_content(user_text, chart_image)
            multimodal_messages = self._attach_multimodal_content(messages, multimodal_content)
            self.logger.debug("Sending chart analysis request to DeepSeek API (%s bytes)", len(img_data))
            response = await self._execute_with_param_retry(
                client.chat.completions.create,
                model_config,
                model=model,
                messages=multimodal_messages
            )
            self._log_usage(response)
            return self.convert_pydantic_response(response)
        except Exception as e:  # noqa: BLE001
            self.logger.error("Error during DeepSeek chart analysis request: %s", str(e))
            return self._handle_exception(e)

    def _log_usage(self, response: Any) -> None:
        """Log DeepSeek token usage, including reasoning and cache-hit counters."""
        usage = getattr(response, "usage", None)
        if not usage:
            return
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = details.reasoning_tokens if details else 0
        self.logger.info(
            "DeepSeek token breakdown: prompt=%s, completion=%s (incl. reasoning=%s), total=%s, cache_hit=%s, cache_miss=%s",
            usage.prompt_tokens,
            usage.completion_tokens,
            reasoning,
            usage.total_tokens,
            getattr(usage, "prompt_cache_hit_tokens", 0),
            getattr(usage, "prompt_cache_miss_tokens", 0),
        )
