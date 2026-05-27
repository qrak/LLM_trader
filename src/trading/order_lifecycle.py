"""Lightweight order intent model for the trading layer."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class OrderLifecycle(str, Enum):
    """Minimal states for order processing."""

    INTENT = "INTENT"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    REJECTED = "REJECTED"
    EXECUTED = "EXECUTED"

    @property
    def is_terminal(self) -> bool:
        """Return True if this state represents a terminal outcome."""
        return self in (OrderLifecycle.EXECUTED, OrderLifecycle.REJECTED)

    @property
    def is_blocking(self) -> bool:
        """Return True if this state blocks execution."""
        return self is OrderLifecycle.REJECTED


_ALLOWED_TRANSITIONS: dict[OrderLifecycle, tuple[OrderLifecycle, ...]] = {
    OrderLifecycle.INTENT: (OrderLifecycle.READY_FOR_REVIEW, OrderLifecycle.REJECTED),
    OrderLifecycle.READY_FOR_REVIEW: (OrderLifecycle.EXECUTED, OrderLifecycle.REJECTED),
    OrderLifecycle.REJECTED: (),
    OrderLifecycle.EXECUTED: (),
}


def can_transition(from_state: OrderLifecycle, to_state: OrderLifecycle) -> bool:
    """Validate that a state transition is allowed."""
    return to_state in _ALLOWED_TRANSITIONS.get(from_state, ())


class OrderIntent(BaseModel):
    """Carries the minimum entry intent needed by guards and audit records."""

    model_config = ConfigDict(frozen=False)

    order_id: str
    state: OrderLifecycle = OrderLifecycle.INTENT
    signal: str
    direction: str
    symbol: str
    confidence: str
    current_price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    position_size: float | None = None
    reasoning: str = ""
    confluence_factors: tuple = Field(default_factory=tuple)
    market_conditions: object | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def transition_to(self, target: OrderLifecycle, reason: str = "") -> bool:
        """Attempt to move the order to *target* state."""
        if not can_transition(self.state, target):
            return False
        self.state = target
        return True
