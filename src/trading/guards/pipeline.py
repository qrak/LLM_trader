"""Guard pipeline — orchestrates pre-execution guard checks for an order intent.

Composes multiple GuardProtocol implementations into a sequential
pipeline. Every intent must pass ALL guards before reaching
READY_FOR_REVIEW state.
"""

from __future__ import annotations

from typing import Any

from . import GuardProtocol, GuardResult


class GuardPipeline:
    """Orchestrates pre-execution guard checks for order intents.

    All guards are evaluated sequentially. The pipeline stops at the
    first failure (fail-fast) and records results for audit.
    """

    def __init__(self, guards: list[GuardProtocol], audit_trail: Any = None) -> None:
        self._guards = list(guards)
        self._audit_trail = audit_trail

    def set_audit_trail(self, audit_trail: Any) -> None:
        """Attach the audit trail used for guard-check records."""
        self._audit_trail = audit_trail

    def evaluate(
        self, intent, /, *, capital: float, config: Any
    ) -> list[GuardResult]:
        """Run all guards against the intent.

        Args:
            intent: OrderIntent to validate.
            capital: Current available capital.
            config: Configuration protocol.

        Returns:
            List of GuardResult in evaluation order. On failure, the
            list is truncated at the failing guard (fail-fast).
        """
        results: list[GuardResult] = []

        for guard in self._guards:
            result = guard.check(intent, capital=capital, config=config)
            results.append(result)

            # Emit audit record for every guard check
            if self._audit_trail is not None:
                self._audit_trail.record(
                    order_id=intent.order_id,
                    event_type="guard_check",
                    actor=guard.name,
                    result="passed" if result.passed else "failed",
                    reason=result.reason,
                    metadata=result.metadata,
                )

            if not result.passed:
                break

        return results

    @property
    def guard_names(self) -> list[str]:
        """Return the ordered list of guard names for reporting."""
        return [g.name for g in self._guards]
