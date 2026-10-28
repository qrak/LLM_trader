"""One definition of the brain's default thresholds.

The provider publishes these values to the prompt and ``TemplateManager`` renders them. When
both modules carried their own ``.get(key, default)`` copy, a change to one produced a prompt
that described a gate the runtime did not use — a plausible value winning over the real one.
Keep every unlearned default here; learned values always override a default.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

THRESHOLD_DEFAULTS: Final[dict[str, Any]] = {
    "adx_strong_threshold": 25,
    "avg_sl_pct": 2.5,
    "min_rr_recommended": 2.0,
    "confidence_threshold": 60,
    "safe_mae_pct": 0,
    "safe_mae_sample_count": 0,
    "adx_weak_threshold": 20,
    "min_confluences_weak": 2,
    "min_confluences_standard": 2,
    "position_reduce_mixed": 0.20,
    "position_reduce_divergent": 0.35,
    "min_position_size": 0.02,
    "rr_borderline_min": 0.0,
    "rr_strong_setup": 2.5,
    "sl_tightening": None,
}


def apply_defaults(learned: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the defaults overlaid with learned values.

    A learned ``None`` means "not learned" and must not erase a default. The result never
    claims a default was learned: ``learned_keys`` stays the caller's responsibility.
    """
    merged = dict(THRESHOLD_DEFAULTS)
    for key, value in (learned or {}).items():
        if value is not None:
            merged[key] = value
    return merged
