
from openai import AsyncOpenAI

from src.logger.logger import Logger
from src.platforms.ai_providers.base import BaseAIClient


class OpenAICompatibleClient(BaseAIClient):
    USER_AGENT = "llm-trader/1.0"

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        logger: Logger,
        default_headers: dict[str, str] | None = None,
        retry_timeframe: str | None = None,
    ) -> None:
        super().__init__(logger, retry_timeframe)
        self.api_key = api_key
        self.base_url = base_url
        self.default_headers = default_headers or {}
        self._known_unsupported_params.add("openrouter_reasoning_effort")

    def _build_client(self) -> AsyncOpenAI:

        return AsyncOpenAI(
            api_key=self.api_key or "not-needed",
            base_url=self.base_url,
            default_headers=self.default_headers,
            max_retries=0,
        )
