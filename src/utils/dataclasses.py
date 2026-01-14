"""Dataclasses for utility modules.

Provides typed data structures for token counting, cost tracking, and other utility operations.
"""

from dataclasses import dataclass
from typing import Optional

from src.utils.data_utils import SerializableMixin


@dataclass(slots=True)
class TokenUsageStats(SerializableMixin):
    """Token usage statistics from a single API request."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: Optional[float] = None


@dataclass(slots=True)
class SessionCosts(SerializableMixin):
    """Cumulative session costs by provider."""
    openrouter: float = 0.0
    google: float = 0.0
    lmstudio: float = 0.0

    @property
    def total(self) -> float:
        """Get total cost across all providers."""
        return self.openrouter + self.google + self.lmstudio


@dataclass(slots=True)
class ProviderCostStats(SerializableMixin):
    """Persistent cost statistics for a single provider."""
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
