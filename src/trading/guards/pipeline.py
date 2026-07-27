"""Guard pipeline — orchestrates pre-execution guard checks for an order intent."""

from __future__ import annotations

from typing import Any

from . import GuardProtocol, GuardResult


class GuardPipeline:
    """Runs guards sequentially against an intent and returns their results."""

    def __init__(self, guards: list[GuardProtocol]) -> None:
        self._guards = list(guards)

    def evaluate(self, intent, /, *, capital: float, config: Any) -> list[GuardResult]:
        """Evaluate all guards sequentially against an order intent.

        Args:
            intent: Order intent to validate.
            capital: Available portfolio capital.
            config: Runtime configuration mapping or object.

        Returns:
            List of GuardResult instances for all executed guards.
        """
        results: list[GuardResult] = []
        for guard in self._guards:
            try:
                result = guard.check(intent, capital=capital, config=config)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                results.append(
                    GuardResult(
                        guard_name=getattr(guard, "name", "unknown_guard"),
                        passed=False,
                        reason=f"Guard '{getattr(guard, 'name', 'unknown')}' failed closed due to error: {exc}",
                    )
                )
                break
            results.append(result)
            if not result.passed:
                break
        return results

    @property
    def guard_names(self) -> list[str]:
        """Return the list of guard names registered in this pipeline."""
        return [g.name for g in self._guards]

    def invalidate_cooldown_cache(self) -> None:
        """Invalidate the CooldownWindowGuard cache after trade execution.

        Safe no-op when no CooldownWindowGuard is in the pipeline or when
        the guard doesn't support caching (e.g., older versions).
        """
        for guard in self._guards:
            if guard.name == "cooldown_window":
                guard.invalidate_cache()
                return
