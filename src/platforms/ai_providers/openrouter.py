
import asyncio
from typing import Any

import httpx

from src.logger.logger import Logger
from src.platforms.ai_providers.openai_compatible import OpenAICompatibleClient


class OpenRouterClient(OpenAICompatibleClient):
    def __init__(self, api_key: str, base_url: str, logger: Logger, retry_timeframe: str | None = None) -> None:
        super().__init__(api_key, base_url, logger, retry_timeframe=retry_timeframe)
        self._known_unsupported_params.discard("openrouter_reasoning_effort")
        self._known_unsupported_params.add("presence_penalty")

    def _split_config(self, model_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:

        call_config = dict(model_config)
        reasoning_effort = call_config.pop("openrouter_reasoning_effort", None)
        if not reasoning_effort:
            return call_config, {}
        return call_config, {"extra_body": {"reasoning": {"effort": reasoning_effort}}}

    async def get_generation_cost(self, generation_id: str, retry_delay: float = 0.5) -> dict[str, Any] | None:

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
