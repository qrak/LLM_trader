


import numpy as np
from numba import njit


@njit(cache=True)
def detect_macd_crossover_numba(
    macd_line: np.ndarray,
    signal_line: np.ndarray
) -> tuple[bool, bool, int, float, float]:

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

    if len(macd_hist) < lookback + 1:
        return 0

    return _macd_hist_trend_at(macd_hist, len(macd_hist) - 1, lookback)


@njit(cache=True)
def macd_histogram_trend_streak_numba(
    macd_hist: np.ndarray,
    lookback: int = 3
) -> int:

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
