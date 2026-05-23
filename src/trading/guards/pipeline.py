"""Guard pipeline — orchestrates pre-execution guard checks for an order intent."""

from __future__ import annotations

from typing import Any

from . import GuardProtocol, GuardResult


class GuardPipeline:
    """Runs guards sequentially against an intent. Fail-fast, records to audit."""

    def __init__(self, guards: list[GuardProtocol], audit_trail: Any = None) -> None:
        self._guards = list(guards)
        self._audit_trail = audit_trail

    def set_audit_trail(self, audit_trail: Any) -> None:
        self._audit_trail = audit_trail

    def evaluate(self, intent, /, *, capital: float, config: Any) -> list[GuardResult]:
        results: list[GuardResult] = []
        for guard in self._guards:
            result = guard.check(intent, capital=capital, config=config)
            results.append(result)
            if self._audit_trail is not None:
                self._audit_trail.record(
                    order_id=intent.order_id, event_type="guard_check",
                    actor=guard.name, result="passed" if result.passed else "failed",
                    reason=result.reason, metadata=result.metadata)
            if not result.passed:
                break
        return results

    @property
    def guard_names(self) -> list[str]:
        return [g.name for g in self._guards]
