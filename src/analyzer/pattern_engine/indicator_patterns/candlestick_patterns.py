

import numpy as np
from numba import njit

BODY_LONG_AVG = 10
BODY_DOJI_AVG = 10
SHADOW_VERY_SHORT_AVG = 10
NEAR_AVG = 5
FAR_AVG = 5

PATTERN_IDS = {
    1: "hammer",
    2: "inverted_hammer",
    3: "shooting_star",
    4: "hanging_man",
    5: "bullish_engulfing",
    6: "bearish_engulfing",
    7: "morning_star",
    8: "evening_star",
    9: "piercing_line",
    10: "dark_cloud_cover",
    11: "three_white_soldiers",
    12: "three_black_crows",
    13: "doji",
}


@njit(cache=True)
def _avg_body(opens, closes, index, period):
    total = 0.0
    for offset in range(1, period + 1):
        total += abs(closes[index - offset] - opens[index - offset])
    return total / period


@njit(cache=True)
def _avg_range(highs, lows, index, period):
    total = 0.0
    for offset in range(1, period + 1):
        total += highs[index - offset] - lows[index - offset]
    return total / period


@njit(cache=True)
def _is_downtrend(closes, index, bars):
    if index - bars < 0:
        return False
    return closes[index - 1] < closes[index - 1 - bars]


@njit(cache=True)
def _is_uptrend(closes, index, bars):
    if index - bars < 0:
        return False
    return closes[index - 1] > closes[index - 1 - bars]


@njit(cache=True)
def detect_candlestick_patterns_numba(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:

    count = len(closes)
    max_hits = lookback * 13
    ids = np.empty(max_hits, dtype=np.int64)
    indices = np.empty(max_hits, dtype=np.int64)
    found = 0
    start = max(BODY_LONG_AVG + 3, count - lookback)

    for index in range(start, count):
        body = abs(closes[index] - opens[index])
        candle_range = highs[index] - lows[index]
        if candle_range <= 0.0:
            continue
        upper_shadow = highs[index] - max(opens[index], closes[index])
        lower_shadow = min(opens[index], closes[index]) - lows[index]
        avg_body = _avg_body(opens, closes, index, BODY_LONG_AVG)
        avg_range = _avg_range(highs, lows, index, BODY_DOJI_AVG)
        bullish = closes[index] > opens[index]
        long_body = body > avg_body
        small_body = body < avg_body
        doji = body < avg_range * 0.1
        very_short_upper = upper_shadow < avg_range * 0.1
        very_short_lower = lower_shadow < avg_range * 0.1
        long_lower = lower_shadow > body * 2.0
        long_upper = upper_shadow > body * 2.0

        if doji:
            ids[found] = 13
            indices[found] = index
            found += 1

        if small_body and long_lower and very_short_upper:
            if _is_downtrend(closes, index, 3):
                ids[found] = 1
                indices[found] = index
                found += 1
            elif _is_uptrend(closes, index, 3):
                ids[found] = 4
                indices[found] = index
                found += 1

        if small_body and long_upper and very_short_lower:
            if _is_downtrend(closes, index, 3):
                ids[found] = 2
                indices[found] = index
                found += 1
            elif _is_uptrend(closes, index, 3):
                ids[found] = 3
                indices[found] = index
                found += 1

        if index >= 1:
            previous_body = abs(closes[index - 1] - opens[index - 1])
            previous_bearish = closes[index - 1] < opens[index - 1]
            previous_bullish = closes[index - 1] > opens[index - 1]
            if (
                previous_bearish
                and bullish
                and long_body
                and previous_body > avg_body
                and closes[index] >= opens[index - 1]
                and opens[index] <= closes[index - 1]
            ):
                ids[found] = 5
                indices[found] = index
                found += 1
            if (
                previous_bullish
                and not bullish
                and long_body
                and previous_body > avg_body
                and closes[index] <= opens[index - 1]
                and opens[index] >= closes[index - 1]
            ):
                ids[found] = 6
                indices[found] = index
                found += 1
            if previous_bearish and bullish and previous_body > avg_body:
                midpoint = (opens[index - 1] + closes[index - 1]) / 2.0
                if opens[index] < closes[index - 1] and closes[index] > midpoint:
                    ids[found] = 9
                    indices[found] = index
                    found += 1
            if previous_bullish and not bullish and previous_body > avg_body:
                midpoint = (opens[index - 1] + closes[index - 1]) / 2.0
                if opens[index] > closes[index - 1] and closes[index] < midpoint:
                    ids[found] = 10
                    indices[found] = index
                    found += 1

        if index >= 2:
            first_body = abs(closes[index - 2] - opens[index - 2])
            first_bearish = closes[index - 2] < opens[index - 2]
            first_bullish = closes[index - 2] > opens[index - 2]
            middle_small = abs(closes[index - 1] - opens[index - 1]) < avg_body
            first_midpoint = (opens[index - 2] + closes[index - 2]) / 2.0
            if (
                first_bearish
                and first_body > avg_body
                and middle_small
                and bullish
                and long_body
                and closes[index] > first_midpoint
            ):
                ids[found] = 7
                indices[found] = index
                found += 1
            if (
                first_bullish
                and first_body > avg_body
                and middle_small
                and not bullish
                and long_body
                and closes[index] < first_midpoint
            ):
                ids[found] = 8
                indices[found] = index
                found += 1
            if (
                first_bullish
                and closes[index - 1] > opens[index - 1]
                and bullish
                and first_body > avg_body
                and abs(closes[index - 1] - opens[index - 1]) > avg_body
                and body > avg_body
                and closes[index - 1] > closes[index - 2]
                and closes[index] > closes[index - 1]
                and opens[index] > opens[index - 1]
            ):
                ids[found] = 11
                indices[found] = index
                found += 1
            if (
                first_bearish
                and closes[index - 1] < opens[index - 1]
                and not bullish
                and first_body > avg_body
                and abs(closes[index - 1] - opens[index - 1]) > avg_body
                and body > avg_body
                and closes[index - 1] < closes[index - 2]
                and closes[index] < closes[index - 1]
                and opens[index] < opens[index - 1]
            ):
                ids[found] = 12
                indices[found] = index
                found += 1

    return ids[:found], indices[:found]
