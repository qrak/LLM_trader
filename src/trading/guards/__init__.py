"""Pre-execution guard pipeline for order validation.

Every order INTENT must pass through this pipeline before reaching
READY_FOR_REVIEW state. Guards are first-class policy objects that
can be composed, reordered, and extended without touching the trading
strategy core.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.trading.order_lifecycle import OrderIntent


class GuardResult(BaseModel):
    """Immutable result of a single guard check.

    Attributes:
        guard_name: Unique identifier for the guard (e.g., "max_position_size").
        passed: Whether the intent passed this guard.
        reason: Human-readable explanation of the result.
        metadata: Arbitrary structured data for audit trail enrichment.
        checked_at: Timestamp of the check.
    """

    model_config = ConfigDict(frozen=True)

    guard_name: str
    passed: bool
    reason: str = ""
    metadata: dict = Field(default_factory=dict)
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GuardProtocol(Protocol):
    """Protocol that every guard MUST satisfy (structural subtyping).

    Guards are stateless policy objects. They receive an OrderIntent
    and return a GuardResult. Side effects (logging, audit storage)
    are handled externally by the pipeline or the caller.
    """

    name: str

    def check(self, intent: OrderIntent, /, *, capital: float, config: Any) -> GuardResult:
        """Evaluate the intent against this guard's policy.

        Args:
            intent: OrderIntent instance with current state.
            capital: Current available capital.
            config: Configuration protocol providing risk parameters.

        Returns:
            GuardResult indicating pass/fail with reasoning.
        """
        raise NotImplementedError


__all__ = [
    "GuardResult",
    "GuardProtocol",
]
