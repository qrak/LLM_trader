
from typing import Any

from src.logger.logger import Logger
from src.platforms.ai_providers.openai_compatible import OpenAICompatibleClient
from src.platforms.ai_providers.response_models import ChatResponseModel


class DeepSeekClient(OpenAICompatibleClient):
    def __init__(self, api_key: str, base_url: str, logger: Logger, retry_timeframe: str | None = None) -> None:
        super().__init__(api_key, base_url, logger, retry_timeframe=retry_timeframe)
        self.retry_on_empty = True

    async def _chat(
        self, model: str, messages: list[dict[str, Any]], model_config: dict[str, Any]
    ) -> ChatResponseModel | None:

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

    def _log_usage(self, response: Any) -> None:

        usage = getattr(response, "usage", None)
        if not usage:
            return
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = details.reasoning_tokens if details else 0
        choices = getattr(response, "choices", None) or []
        finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
        self.logger.info(
            "DeepSeek token breakdown: prompt=%s, completion=%s (incl. reasoning=%s), total=%s, "
            "cache_hit=%s, cache_miss=%s, finish_reason=%s",
            usage.prompt_tokens,
            usage.completion_tokens,
            reasoning,
            usage.total_tokens,
            getattr(usage, "prompt_cache_hit_tokens", 0),
            getattr(usage, "prompt_cache_miss_tokens", 0),
            finish_reason or "?",
        )
