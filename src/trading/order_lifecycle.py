"""Order lifecycle state machine for the trading layer.

Defines the complete lifecycle of a trade order from initial intent
through guard validation, review, approval, and execution.

States flow: INTENT -> READY_FOR_REVIEW -> APPROVED/REJECTED -> EXECUTED/CANCELLED
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class OrderLifecycle(str, Enum):
    """Canonical states for order processing.

    Orders normally pass through INTENT -> READY_FOR_REVIEW before
    reaching APPROVED. Guard failures may reject an order directly
    from INTENT. EXECUTED, REJECTED, and CANCELLED are terminal states.
    """

    INTENT = "INTENT"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        """Return True if this state represents a terminal outcome."""
        return self in (OrderLifecycle.EXECUTED, OrderLifecycle.CANCELLED, OrderLifecycle.REJECTED)

    @property
    def is_blocking(self) -> bool:
        """Return True if this state blocks execution."""
        return self in (OrderLifecycle.REJECTED, OrderLifecycle.CANCELLED)


_ALLOWED_TRANSITIONS: dict[OrderLifecycle, tuple[OrderLifecycle, ...]] = {
    OrderLifecycle.INTENT: (OrderLifecycle.READY_FOR_REVIEW, OrderLifecycle.REJECTED, OrderLifecycle.CANCELLED),
    OrderLifecycle.READY_FOR_REVIEW: (OrderLifecycle.APPROVED, OrderLifecycle.REJECTED),
    OrderLifecycle.APPROVED: (OrderLifecycle.EXECUTED, OrderLifecycle.CANCELLED),
    OrderLifecycle.REJECTED: (),  # Terminal
    OrderLifecycle.EXECUTED: (),  # Terminal
    OrderLifecycle.CANCELLED: (),  # Terminal
}


def can_transition(from_state: OrderLifecycle, to_state: OrderLifecycle) -> bool:
    """Validate that a state transition is allowed."""
    return to_state in _ALLOWED_TRANSITIONS.get(from_state, ())


class OrderIntent(BaseModel):
    """Carries a trading intent through its lifecycle.

    Attributes:
        order_id: Unique identifier for audit trail correlation.
        state: Current lifecycle state.
        signal: BUY or SELL.
        direction: LONG or SHORT (derived from signal).
        symbol: Trading pair (e.g., BTC/USDC).
        confidence: AI confidence level (HIGH, MEDIUM, LOW).
        current_price: Price at intent generation.
        stop_loss: Proposed stop loss price.
        take_profit: Proposed take profit price.
        position_size: AI-suggested position size (0.0-1.0).
        reasoning: AI reasoning for the decision.
        confluence_factors: Tuple of (name, score) pairs.
        market_conditions: Snapshot of market state at intent time.
        created_at: Timestamp of intent creation.
        state_history: Ordered log of every state transition with timestamps.
    """

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
    state_history: list[dict] = Field(default_factory=list)

    def transition_to(self, target: OrderLifecycle, reason: str = "") -> bool:
        """Attempt to move the order to *target* state.

        Returns:
            True if the transition was performed, False if it was invalid.
        """
        if not can_transition(self.state, target):
            return False
        previous = self.state
        self.state = target
        self.state_history.append(
            {
                "from": previous.value,
                "to": target.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
            }
        )
        return True
