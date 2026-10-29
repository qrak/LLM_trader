

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

    merged = dict(THRESHOLD_DEFAULTS)
    for key, value in (learned or {}).items():
        if value is not None:
            merged[key] = value
    return merged
