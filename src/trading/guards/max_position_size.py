"""Max Position Size Guard — enforces the hard cap on capital allocation."""

from __future__ import annotations

import math

from . import GuardResult


class MaxPositionSizeGuard:
    """Reject intents that would exceed the configured max position size.

    If the AI explicitly requests a finite positive position size above
    MAX_POSITION_SIZE, this guard rejects it rather than silently clamping.
    Missing or invalid sizes are left to RiskManager's existing fallback
    sizing policy so enabling the guard does not disable safe fallbacks.
    """

    name = "max_position_size"

    def check(self, intent, /, *, capital: float, config) -> GuardResult:
        try:
            max_size = float(config.MAX_POSITION_SIZE)
        except (TypeError, ValueError):
            return GuardResult(
                guard_name=self.name,
                passed=False,
                reason="Configured MAX_POSITION_SIZE is not a valid number",
                metadata={"max_size": config.MAX_POSITION_SIZE},
            )

        if not math.isfinite(max_size) or max_size <= 0:
            return GuardResult(
                guard_name=self.name,
                passed=False,
                reason="Configured MAX_POSITION_SIZE must be a positive finite decimal",
                metadata={"max_size": max_size},
            )

        requested = intent.position_size
        if requested is None:
            return GuardResult(
                guard_name=self.name,
                passed=True,
                reason="No position_size provided; RiskManager fallback sizing will apply",
                metadata={"max_size": max_size},
            )

        requested_size = float(requested)
        if not math.isfinite(requested_size) or requested_size <= 0:
            return GuardResult(
                guard_name=self.name,
                passed=True,
                reason="Invalid position_size provided; RiskManager fallback sizing will apply",
                metadata={"max_size": max_size, "requested": requested},
            )

        if requested_size > max_size:
            return GuardResult(
                guard_name=self.name,
                passed=False,
                reason=f"Position size {requested_size * 100:.1f}% exceeds maximum {max_size * 100:.1f}%",
                metadata={
                    "max_size": max_size,
                    "requested": requested_size,
                    "direction": intent.direction,
                },
            )

        return GuardResult(
            guard_name=self.name,
            passed=True,
            reason=f"Position size {requested_size * 100:.1f}% within limit {max_size * 100:.1f}%",
            metadata={
                "max_size": max_size,
                "requested": requested_size,
                "direction": intent.direction,
            },
        )
