"""Dataclasses for AI provider configuration and invocation results."""
from dataclasses import dataclass
from typing import Optional, Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from src.platforms.ai_providers import OpenRouterClient, GoogleAIClient, LMStudioClient, BlockRunClient


@dataclass
class ProviderMetadata:
    """Configuration metadata for an AI provider."""
    name: str
    client: Optional[Any]
    default_model: str
    config: Dict[str, Any]
    supports_chart: bool
    has_rate_limits: bool
    paid_client: Optional[Any] = None

    def is_available(self) -> bool:
        """Check if the provider's client is available."""
        return self.client is not None


@dataclass
class InvocationResult:
    """Result of a provider invocation attempt."""
    success: bool
    response: Optional[Dict[str, Any]]
    provider: str
    model: str
    used_paid_tier: bool = False

    @property
    def error(self) -> Optional[str]:
        """Extract error message from response if present."""
        if self.response and "error" in self.response:
            return self.response["error"]
        return None


@dataclass
class ProviderClients:
    """Container for all AI provider clients (runtime objects, not serializable)."""
    google: Optional["GoogleAIClient"] = None
    google_paid: Optional["GoogleAIClient"] = None
    openrouter: Optional["OpenRouterClient"] = None
    lmstudio: Optional["LMStudioClient"] = None
    blockrun: Optional["BlockRunClient"] = None

    @classmethod
    def from_factory_dict(cls, clients: Dict[str, Any]) -> "ProviderClients":
        """Create from ProviderFactory.create_all_clients() output."""
        return cls(
            google=clients.get('google'),
            google_paid=clients.get('google_paid'),
            openrouter=clients.get('openrouter'),
            lmstudio=clients.get('lmstudio'),
            blockrun=clients.get('blockrun')
        )
