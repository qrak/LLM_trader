"""Shared risk/reward entry-floor policy."""

import math
from collections.abc import Mapping
from typing import Any


def configured_min_rr(value: Any) -> float:
    """Normalize the configured entry floor, preserving zero as disabled."""
    try:
        floor = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(floor):
        return 1.0
    return max(0.0, floor)


def resolve_entry_rr_floor(config: Any, thresholds: Mapping[str, Any] | None = None) -> float:
    """Return the shared entry floor from config and the brain's stricter threshold."""
    try:
        config_value = config.MIN_RR_ENTRY
    except (AttributeError, TypeError, ValueError):
        config_value = 1.0
    config_floor = configured_min_rr(config_value)
    try:
        brain_floor = float((thresholds or {}).get("rr_borderline_min", config_floor))
    except (TypeError, ValueError):
        return config_floor
    if not math.isfinite(brain_floor):
        return config_floor
    return max(config_floor, brain_floor)


def format_rr_floor(value: float) -> str:
    """Render the enforced floor without hiding meaningful decimal precision."""
    if value.is_integer():
        return f"{value:.1f}"
    return format(value, ".15g")
