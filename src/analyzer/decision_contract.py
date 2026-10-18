"""Structural contract for a model decision payload.

Pydantic models validate the provider response shape; this contract states what a
decision must satisfy BEFORE it can become an order intent (levels, direction, size).
It is pure: no I/O, no config, no provider types, so it can run in CI on recorded
payloads without a network or an exchange.
"""

from __future__ import annotations

import math
from typing import Any, TypeGuard

SIGNALS = frozenset({"BUY", "SELL", "HOLD", "CLOSE", "UPDATE"})


def _is_number(value: Any) -> TypeGuard[float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _positive(value: Any) -> float | None:
    return float(value) if _is_number(value) and value > 0 else None


def validate_decision_payload(payload: Any) -> list[str]:
    """Return a list of contract violations; empty means the payload is usable."""
    if not isinstance(payload, dict):
        return ["payload must be an object"]
    analysis = payload.get("analysis")
    if not isinstance(analysis, dict):
        return ["analysis must be an object"]

    problems: list[str] = []
    signal = str(analysis.get("signal") or "").upper()
    if signal not in SIGNALS:
        problems.append(f"analysis.signal {signal!r} is not one of {sorted(SIGNALS)}")

    # Confidence is guaranteed by TradingAnalysisModel on the real provider path; here it
    # only has to be sane WHEN present, so a missing field does not look like a broken
    # execution contract.
    if "confidence" in analysis:
        confidence = analysis.get("confidence")
        if not (isinstance(confidence, int) and not isinstance(confidence, bool) and 0 <= confidence <= 100):
            problems.append("analysis.confidence must be an integer 0..100")

    if signal in ("BUY", "SELL"):
        entry = _positive(analysis.get("entry_price"))
        stop = _positive(analysis.get("stop_loss"))
        target = _positive(analysis.get("take_profit"))
        for name in ("entry_price", "stop_loss", "take_profit"):
            if _positive(analysis.get(name)) is None:
                problems.append(f"analysis.{name} must be a positive number for {signal}")

        if entry is not None and stop is not None and target is not None:
            if stop == target:
                problems.append("stop_loss and take_profit must differ")
            elif signal == "BUY" and not stop < entry < target:
                problems.append("BUY levels must satisfy stop_loss < entry_price < take_profit")
            elif signal == "SELL" and not target < entry < stop:
                problems.append("SELL levels must satisfy take_profit < entry_price < stop_loss")

        if _positive(analysis.get("risk_reward_ratio")) is None:
            problems.append("analysis.risk_reward_ratio must be a positive number")

        size = analysis.get("position_size")
        if _is_number(size):
            if not 0 < size <= 1:
                problems.append("analysis.position_size must be in (0, 1]")
        else:
            problems.append("analysis.position_size must be in (0, 1]")

    if signal == "UPDATE" and not any(
        _positive(analysis.get(key)) is not None for key in ("stop_loss", "take_profit")
    ):
        problems.append("UPDATE requires a new stop_loss or take_profit")

    return problems
