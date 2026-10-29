

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


class ProviderHealth:
    """Consecutive-failure circuit breaker with a half-open probe."""

    RATE_LIMIT_CATEGORY = "rate_limit"

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        cooldown_seconds: float = 120.0,
        max_cooldown_seconds: float = 900.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_seconds = max(1.0, float(cooldown_seconds))
        self.max_cooldown_seconds = max(self.cooldown_seconds, float(max_cooldown_seconds))
        self._clock = clock
        self._state: dict[str, dict[str, Any]] = {}

    def _entry(self, provider: str) -> dict[str, Any]:
        return self._state.setdefault(
            provider,
            {"failures": 0, "opened_until": None, "cooldown": self.cooldown_seconds, "last_error": None},
        )

    def allows(self, provider: str) -> bool:

        opened_until = self._entry(provider)["opened_until"]
        return opened_until is None or self._clock() >= opened_until

    def record_success(self, provider: str) -> None:
        entry = self._entry(provider)
        entry.update(failures=0, opened_until=None, cooldown=self.cooldown_seconds, last_error=None)

    def record_failure(self, provider: str, *, category: str = "error", detail: str | None = None) -> None:
        if category == self.RATE_LIMIT_CATEGORY:
            return
        entry = self._entry(provider)
        entry["failures"] += 1
        entry["last_error"] = detail
        if entry["opened_until"] is None and entry["failures"] < self.failure_threshold:
            return
        entry["opened_until"] = self._clock() + entry["cooldown"]
        entry["cooldown"] = min(entry["cooldown"] * 2, self.max_cooldown_seconds)

    def snapshot(self) -> dict[str, dict[str, Any]]:

        now = self._clock()
        return {
            provider: {
                "failures": entry["failures"],
                "open": entry["opened_until"] is not None and now < entry["opened_until"],
                "seconds_left": round(max(0.0, (entry["opened_until"] or now) - now), 1),
                "last_error": entry["last_error"],
            }
            for provider, entry in self._state.items()
        }
