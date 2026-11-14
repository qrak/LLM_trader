

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.trading.order_lifecycle import OrderIntent


class GuardResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    guard_name: str
    passed: bool
    reason: str = ""
    metadata: dict = Field(default_factory=dict)
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GuardProtocol(Protocol):
    name: str

    def check(self, intent: OrderIntent, /, *, capital: float, config: Any) -> GuardResult:

        raise NotImplementedError


__all__ = [
    "GuardProtocol",
    "GuardResult",
]
