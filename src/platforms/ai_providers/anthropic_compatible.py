
from typing import Any

import httpx

from src.logger.logger import Logger
from src.platforms.ai_providers.base import BaseAIClient
from src.platforms.ai_providers.response_models import (
    ChatResponseModel,
    ChoiceModel,
    MessageModel,
    UsageModel,
)


class AnthropicCompatibleClient(BaseAIClient):
    API_VERSION = "2023-06-01"
    REQUEST_TIMEOUT_SECONDS = 600.0
    FINISH_REASONS = {  # noqa: RUF012
        "max_tokens": "length",
        "end_turn": "stop",
        "stop_sequence": "stop",
        "refusal": "content_filter",
    }

    def __init__(
        self, api_key: str | None, base_url: str, logger: Logger, retry_timeframe: str | None = None,
    ) -> None:
        super().__init__(logger, retry_timeframe)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _build_client(self) -> httpx.AsyncClient:

        headers = {"anthropic-version": self.API_VERSION}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=self.REQUEST_TIMEOUT_SECONDS,
        )

    async def close(self) -> None:

        client = self._client
        if not client:
            return
        try:
            await client.aclose()
        finally:
            self._client = None

    async def _chat(
        self, model: str, messages: list[dict[str, Any]], model_config: dict[str, Any]
    ) -> ChatResponseModel | None:

        client = self._ensure_client()
        try:
            self.logger.debug("Sending request to %s with model: %s", type(self).__name__, model)
            response = await client.post("/messages", json=self._request_body(model, messages, model_config))
            response.raise_for_status()
            return self._to_response(response.json())
        except Exception as e:  # noqa: BLE001
            self.logger.error("Error during %s request: %s", type(self).__name__, self._sanitize_error_message(str(e)))
            return self._handle_exception(e)

    @staticmethod
    def _request_body(model: str, messages: list[dict[str, Any]], model_config: dict[str, Any]) -> dict[str, Any]:

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": model_config["max_tokens"],
            "messages": [
                {"role": message["role"], "content": message["content"]}
                for message in messages if message["role"] != "system"
            ],
        }
        system = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
        if system:
            body["system"] = system
        return body

    def _to_response(self, payload: dict[str, Any]) -> ChatResponseModel:

        content = "".join(
            block.get("text", "") for block in payload.get("content") or [] if block.get("type") == "text"
        )
        usage = payload.get("usage") or {}
        prompt_tokens = usage.get("input_tokens") or 0
        completion_tokens = usage.get("output_tokens") or 0
        stop_reason = str(payload.get("stop_reason") or "")
        return ChatResponseModel(
            choices=[ChoiceModel(
                message=MessageModel(content=content),
                finish_reason=self.FINISH_REASONS.get(stop_reason, stop_reason) or None,
            )],
            usage=UsageModel(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            id=payload.get("id"),
            model=payload.get("model"),
        )
