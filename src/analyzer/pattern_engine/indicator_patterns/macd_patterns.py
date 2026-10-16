"""
MACD Pattern Detection - Pure NumPy/Numba Implementation

Detects MACD-based patterns:
1. Bullish crossover (MACD line crosses above signal line)
2. Bearish crossover (MACD line crosses below signal line)
3. Zero-line crossover (MACD crosses above/below zero)

All functions use @njit for performance.
"""


import numpy as np
from numba import njit


@njit(cache=True)
def detect_macd_crossover_numba(
    macd_line: np.ndarray,
    signal_line: np.ndarray
) -> tuple[bool, bool, int, float, float]:
    """
    Detect MACD line crossing signal line.
    Scans ENTIRE array for the most recent crossover event.
    Returns:
        (crossover_found, is_bullish, periods_ago, macd_value, signal_value)
        - crossover_found: True if crossover detected
        - is_bullish: True = bullish (MACD crossed above), False = bearish
        - periods_ago: How many periods ago crossover occurred
        - macd_value: MACD value at crossover
        - signal_value: Signal value at crossover
    """
    if len(macd_line) < 2 or len(signal_line) < 2:
        return (False, False, -1, 0.0, 0.0)

    actual_lookback = len(macd_line) - 1

    for i in range(actual_lookback):
        current_idx = len(macd_line) - 1 - i
        prev_idx = current_idx - 1

        if current_idx < 1:
            break

        macd_current = macd_line[current_idx]
        macd_prev = macd_line[prev_idx]
        signal_current = signal_line[current_idx]
        signal_prev = signal_line[prev_idx]

        if macd_prev <= signal_prev and macd_current > signal_current:
            return (True, True, i, macd_current, signal_current)

        if macd_prev >= signal_prev and macd_current < signal_current:
            return (True, False, i, macd_current, signal_current)

    return (False, False, -1, 0.0, 0.0)


@njit(cache=True)
def detect_macd_zero_cross_numba(
    macd_line: np.ndarray
) -> tuple[bool, bool, int, float]:
    """
    Detect MACD line crossing zero line.
    Scans ENTIRE array for the most recent zero-line crossover.

    Zero-line cross indicates momentum shift:
    - Crossing above zero = bullish momentum
    - Crossing below zero = bearish momentum
    Returns:
        (crossover_found, is_bullish, periods_ago, macd_value)
        - crossover_found: True if zero-line cross detected
        - is_bullish: True = crossed above zero, False = crossed below
        - periods_ago: How many periods ago crossover occurred
        - macd_value: MACD value at crossover
    """
    if len(macd_line) < 2:
        return (False, False, -1, 0.0)

    actual_lookback = len(macd_line) - 1

    for i in range(actual_lookback):
        current_idx = len(macd_line) - 1 - i
        prev_idx = current_idx - 1

        if current_idx < 1:
            break

        macd_current = macd_line[current_idx]
        macd_prev = macd_line[prev_idx]

        if macd_prev <= 0.0 and macd_current > 0.0:
            return (True, True, i, macd_current)

        if macd_prev >= 0.0 and macd_current < 0.0:
            return (True, False, i, macd_current)

    return (False, False, -1, 0.0)


@njit(cache=True)
def _macd_hist_trend_at(macd_hist: np.ndarray, end: int, lookback: int) -> int:
    """The MACD histogram direction rule evaluated as of bar `end`."""
    if end < lookback:
        return 0

    increasing = 0
    decreasing = 0

    for i in range(end - lookback + 1, end + 1):
        if macd_hist[i] > macd_hist[i - 1]:
            increasing += 1
        elif macd_hist[i] < macd_hist[i - 1]:
            decreasing += 1

    if increasing > decreasing:
        return 1
    if decreasing > increasing:
        return -1
    return 0


@njit(cache=True)
def get_macd_histogram_trend_numba(
    macd_hist: np.ndarray,
    lookback: int = 3
) -> int:
    """
    Get MACD histogram trend direction.
    Returns:
        1 = increasing (bullish), -1 = decreasing (bearish), 0 = neutral/mixed
    """
    if len(macd_hist) < lookback + 1:
        return 0

    return _macd_hist_trend_at(macd_hist, len(macd_hist) - 1, lookback)


@njit(cache=True)
def macd_histogram_trend_streak_numba(
    macd_hist: np.ndarray,
    lookback: int = 3
) -> int:
    """
    Consecutive bars the histogram direction has kept its current sign.

    The direction is a state (a majority of the last `lookback` changes), so this
    is how long it has already held: a long run is an established momentum, not a
    fresh turn. 0 for a mixed reading, which carries no state to measure.
    """
    n = len(macd_hist)
    if n < lookback + 1:
        return 0

    current = _macd_hist_trend_at(macd_hist, n - 1, lookback)
    if current == 0:
        return 0

    bars = 0
    for end in range(n - 1, lookback - 1, -1):
        if _macd_hist_trend_at(macd_hist, end, lookback) != current:
            break
        bars += 1
    return bars
