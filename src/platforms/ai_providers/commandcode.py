
from openai import AsyncOpenAI

from src.logger.logger import Logger
from src.platforms.ai_providers.base import BaseAIClient


class CommandCodeClient(BaseAIClient):
    USER_AGENT = "llm-trader/1.0"

    def __init__(self, api_key: str, base_url: str, logger: Logger) -> None:
        super().__init__(logger)
        self.api_key = api_key
        self.base_url = base_url

    def _build_client(self) -> AsyncOpenAI:

        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers={"User-Agent": self.USER_AGENT},
        )
