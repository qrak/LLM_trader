

from src.utils.timeframe_validator import TimeframeValidator

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

    if not timeframe:
        raise ValueError("timeframe is required for pattern recency")
    minutes_per_candle = TimeframeValidator.to_minutes(timeframe)

    target_minutes = STALENESS_TARGET_HOURS.get(category, DEFAULT_STALENESS_TARGET_HOURS) * 60
    return max(1, target_minutes // minutes_per_candle)


def recent_history_window(category: str, total_candles: int) -> int:

    ratio = _RECENT_HISTORY_RATIO.get(category, _DEFAULT_RECENT_HISTORY_RATIO)
    floor = _RECENT_HISTORY_FLOOR.get(category, _DEFAULT_RECENT_HISTORY_FLOOR)
    return max(floor, int(total_candles * ratio))


def recency_threshold(category: str, timeframe: str | None, total_candles: int) -> int:

    return min(staleness_threshold(category, timeframe), recent_history_window(category, total_candles))


def is_pattern_recent(
    pattern_index: int | None,
    total_candles: int | None,
    category: str,
    timeframe: str | None,
) -> bool:

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

    return {
        category: [
            pattern
            for pattern in patterns_list
            if is_pattern_recent(pattern.get("index"), total_candles, category, timeframe)
        ]
        for category, patterns_list in patterns.items()
    }
