"""Dataclasses for AI provider configuration and invocation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.platforms.ai_providers import (
        BlockRunClient,
        GoogleAIClient,
        LMStudioClient,
        OpenRouterClient,
    )
    from src.platforms.ai_providers.response_models import ChatResponseModel


@dataclass
class ProviderMetadata:
    """Configuration metadata for an AI provider."""
    name: str
    client: Any | None
    default_model: str
    config: dict[str, Any]
    supports_chart: bool
    paid_client: Any | None = None
    fallback_model: str | None = None

    def is_available(self) -> bool:
        """Check if the provider's client is available."""
        return self.client is not None


@dataclass
class InvocationResult:
    """Result of a provider invocation attempt."""
    success: bool
    response: ChatResponseModel | None
    provider: str
    model: str
    used_paid_tier: bool = False
    error_message: str | None = None

    @property
    def error(self) -> str | None:
        """Extract error message from explicit field or response if present."""
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
    """Container for all AI provider clients (runtime objects, not serializable)."""
    google: GoogleAIClient | None = None
    google_paid: GoogleAIClient | None = None
    openrouter: OpenRouterClient | None = None
    lmstudio: LMStudioClient | None = None
    blockrun: BlockRunClient | None = None
