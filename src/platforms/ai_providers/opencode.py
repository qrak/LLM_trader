
import uuid

from openai import AsyncOpenAI

from src.logger.logger import Logger
from src.platforms.ai_providers.base import BaseAIClient


class OpenCodeClient(BaseAIClient):
    """Client for OpenCode Go through the OpenAI-compatible inference endpoint."""

    USER_AGENT = "llm-trader/1.0"

    def __init__(self, api_key: str, base_url: str, logger: Logger, session_id: str | None = None) -> None:
        super().__init__(logger)
        self.api_key = api_key
        self.base_url = base_url
        self.session_id = session_id or f"llm-trader-{uuid.uuid4().hex[:16]}"

    def _build_client(self) -> AsyncOpenAI:

        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers={
                "User-Agent": self.USER_AGENT,
                "x-opencode-session": self.session_id,
            },
        )
