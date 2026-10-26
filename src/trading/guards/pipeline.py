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
        Returns:
            List of GuardResult instances for all executed guards.
        """
        results: list[GuardResult] = []
        for guard in self._guards:
            try:
                result = guard.check(intent, capital=capital, config=config)
            except Exception as exc:  # pylint: disable=broad-exception-caught  # noqa: BLE001
                results.append(
                    GuardResult(
                        guard_name=guard.name,
                        passed=False,
                        reason=f"Guard '{guard.name}' failed closed due to error: {exc}",
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

