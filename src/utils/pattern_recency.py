"""Recency rules for detected indicator patterns.

A pattern older than its category's staleness window must not be scored or
shown. The prompt section and the deterministic pattern-quality score have to
see the same pattern set, otherwise the quality number counts signals the model
never received.

The windows are expressed in hours and converted to bars with the timeframe,
so a 4h cross older than 40 hours is stale, while an MA cross stays relevant
for 200 hours.
"""

from src.utils.timeframe_validator import TimeframeValidator

# How far back a pattern of a given category stays actionable (hours).
STALENESS_TARGET_HOURS: dict[str, int] = {
    "rsi": 40,
    "macd": 40,
    "stochastic": 40,
    "ma_crossover": 200,
    "divergence": 80,
    "volatility": 20,
    "volume": 40,
}

DEFAULT_STALENESS_TARGET_HOURS = 40

# Share of the loaded history that still counts as "recent" per category,
# capped by the hour-based window above.
_RECENT_HISTORY_RATIO: dict[str, float] = {
    "ma_crossover": 0.30,
    "divergence": 0.10,
    "volatility": 0.05,
    "volume": 0.05,
}
_DEFAULT_RECENT_HISTORY_RATIO = 0.15

_RECENT_HISTORY_FLOOR: dict[str, int] = {
    "ma_crossover": 0,
    "divergence": 20,
    "volatility": 10,
    "volume": 10,
}
_DEFAULT_RECENT_HISTORY_FLOOR = 20


def staleness_threshold(category: str, timeframe: str | None) -> int:
    """Hour-based window for a category, expressed in bars of the timeframe.

    The frame is required: a missing one used to fall back to 240 minutes, which made a
    pattern window silently describe a 4h bot no matter what the analysis was running.
    """
    if not timeframe:
        raise ValueError("timeframe is required for pattern recency")
    minutes_per_candle = TimeframeValidator.to_minutes(timeframe)

    target_minutes = STALENESS_TARGET_HOURS.get(category, DEFAULT_STALENESS_TARGET_HOURS) * 60
    return max(1, target_minutes // minutes_per_candle)


def recent_history_window(category: str, total_candles: int) -> int:
    """History-relative window: a pattern deeper than this share is not recent."""
    ratio = _RECENT_HISTORY_RATIO.get(category, _DEFAULT_RECENT_HISTORY_RATIO)
    floor = _RECENT_HISTORY_FLOOR.get(category, _DEFAULT_RECENT_HISTORY_FLOOR)
    return max(floor, int(total_candles * ratio))


def recency_threshold(category: str, timeframe: str | None, total_candles: int) -> int:
    """Effective recency window: the stricter of the hour and history windows."""
    return min(staleness_threshold(category, timeframe), recent_history_window(category, total_candles))


def is_pattern_recent(
    pattern_index: int | None,
    total_candles: int | None,
    category: str,
    timeframe: str | None,
) -> bool:
    """Whether a pattern is still within its category's recency window.

    Patterns without an index, or a context without candle count, are kept:
    they cannot be proven stale.
    """
    if pattern_index is None or not total_candles or total_candles <= 0:
        return True
    age = (total_candles - 1) - pattern_index
    if age < 0:
        return True
    return age <= recency_threshold(category, timeframe, total_candles)


def filter_recent_patterns(
    patterns: dict[str, list[dict]],
    total_candles: int | None,
    timeframe: str | None,
) -> dict[str, list[dict]]:
    """Drop stale patterns from a detector result, keeping the dict shape.

    Applied right after detection so every consumer (prompt formatter, quality
    scorer) works on the same set.
    """
    return {
        category: [
            pattern
            for pattern in patterns_list
            if is_pattern_recent(pattern.get("index"), total_candles, category, timeframe)
        ]
        for category, patterns_list in patterns.items()
    }
