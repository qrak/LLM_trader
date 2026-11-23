

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.platforms.ai_providers import (
        BaseAIClient,
        DeepSeekClient,
        GoogleAIClient,
        OpenRouterClient,
    )
    from src.platforms.ai_providers.response_models import ChatResponseModel


@dataclass
class ProviderMetadata:
    name: str
    client: Any | None
    default_model: str
    config: dict[str, Any]

    def is_available(self) -> bool:

        return self.client is not None


@dataclass
class InvocationResult:
    success: bool
    response: ChatResponseModel | None
    provider: str
    model: str
    error_message: str | None = None

    @property
    def error(self) -> str | None:

        if self.error_message:
            return self.error_message
        if self.response:
            if self.response.error:
                return self.response.error
            if self.response.choices and self.response.choices[0].error:
                err_dict = self.response.choices[0].error
                msg = err_dict.get("message") or err_dict.get("code") or str(err_dict)
                return f"choice error: {msg}"
        return None


@dataclass
class ProviderClients:
    google: GoogleAIClient | None = None
    openrouter: OpenRouterClient | None = None
    opencode: BaseAIClient | None = None
    commandcode: BaseAIClient | None = None
    custom: BaseAIClient | None = None
    deepseek: DeepSeekClient | None = None

    def all(self) -> list[BaseAIClient]:

        return [
            client for client in (
                self.google, self.openrouter, self.opencode, self.commandcode, self.custom, self.deepseek,
            ) if client is not None
        ]
