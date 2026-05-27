"""Guard pipeline — orchestrates pre-execution guard checks for an order intent."""

from __future__ import annotations

from typing import Any

from . import GuardProtocol, GuardResult


class GuardPipeline:
    """Runs guards sequentially against an intent and returns their results."""

    def __init__(self, guards: list[GuardProtocol]) -> None:
        self._guards = list(guards)

    def evaluate(self, intent, /, *, capital: float, config: Any) -> list[GuardResult]:
        results: list[GuardResult] = []
        for guard in self._guards:
            result = guard.check(intent, capital=capital, config=config)
            results.append(result)
            if not result.passed:
                break
        return results

    @property
    def guard_names(self) -> list[str]:
        return [g.name for g in self._guards]

    def invalidate_cooldown_cache(self) -> None:
        """Invalidate the CooldownWindowGuard cache after trade execution.

        Safe no-op when no CooldownWindowGuard is in the pipeline or when
        the guard doesn't support caching (e.g., older versions).
        """
        for guard in self._guards:
            if guard.name == "cooldown_window" and hasattr(guard, "invalidate_cache"):
                guard.invalidate_cache()
                return
