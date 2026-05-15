from __future__ import annotations
"""Dataclasses for AI provider configuration and invocation results."""
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.platforms.ai_providers import OpenRouterClient, GoogleAIClient, LMStudioClient, BlockRunClient
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

    def is_available(self) -> bool:
        """Check if the provider's client is available."""
        return self.client is not None


@dataclass
class InvocationResult:
    """Result of a provider invocation attempt."""
    success: bool
    response: "ChatResponseModel" | None
    provider: str
    model: str
    used_paid_tier: bool = False

    @property
    def error(self) -> str | None:
        """Extract error message from response if present."""
        if self.response and self.response.error:
            return self.response.error
        return None


@dataclass
class ProviderClients:
    """Container for all AI provider clients (runtime objects, not serializable)."""
    google: "GoogleAIClient" | None = None
    google_paid: "GoogleAIClient" | None = None
    openrouter: "OpenRouterClient" | None = None
    lmstudio: "LMStudioClient" | None = None
    blockrun: "BlockRunClient" | None = None

    @classmethod
    def from_factory_dict(cls, clients: dict[str, Any]) -> "ProviderClients":
        """Create from ProviderFactory.create_all_clients() output."""
        return cls(
            google=clients.get('google'),
            google_paid=clients.get('google_paid'),
            openrouter=clients.get('openrouter'),
            lmstudio=clients.get('lmstudio'),
            blockrun=clients.get('blockrun')
        )
