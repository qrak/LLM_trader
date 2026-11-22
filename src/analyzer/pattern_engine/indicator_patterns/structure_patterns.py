

import numpy as np
from numba import njit
from numpy.typing import NDArray

PIVOT_WINDOW = 5
DOUBLE_TOLERANCE = 0.005


@njit(cache=True)
def find_pivots_numba(
    highs: NDArray[np.float64], lows: NDArray[np.float64], window: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:

    count = len(highs)
    high_idx = np.empty(count, dtype=np.int64)
    low_idx = np.empty(count, dtype=np.int64)
    high_count = 0
    low_count = 0
    for index in range(window, count - window):
        is_high = True
        is_low = True
        for offset in range(1, window + 1):
            if highs[index] <= highs[index - offset] or highs[index] <= highs[index + offset]:
                is_high = False
            if lows[index] >= lows[index - offset] or lows[index] >= lows[index + offset]:
                is_low = False
            if not is_high and not is_low:
                break
        if is_high:
            high_idx[high_count] = index
            high_count += 1
        if is_low:
            low_idx[low_count] = index
            low_count += 1
    return high_idx[:high_count], low_idx[:low_count]


@njit(cache=True)
def swing_structure_numba(
    highs: NDArray[np.float64], lows: NDArray[np.float64], window: int
) -> tuple[int, float, float, float, float]:

    high_idx, low_idx = find_pivots_numba(highs, lows, window)
    if len(high_idx) < 2 or len(low_idx) < 2:
        return 0, 0.0, 0.0, 0.0, 0.0
    last_high = highs[high_idx[len(high_idx) - 1]]
    previous_high = highs[high_idx[len(high_idx) - 2]]
    last_low = lows[low_idx[len(low_idx) - 1]]
    previous_low = lows[low_idx[len(low_idx) - 2]]
    higher_high = last_high > previous_high
    higher_low = last_low > previous_low
    if higher_high and higher_low:
        state = 1
    elif not higher_high and not higher_low:
        state = -1
    else:
        state = 0
    return state, float(last_high), float(last_low), float(previous_high), float(previous_low)


@njit(cache=True)
def break_of_structure_numba(
    closes: NDArray[np.float64], highs: NDArray[np.float64], lows: NDArray[np.float64], window: int
) -> tuple[int, float, int]:

    high_idx, low_idx = find_pivots_numba(highs, lows, window)
    if len(high_idx) == 0 or len(low_idx) == 0:
        return 0, 0.0, -1
    last_close = closes[len(closes) - 1]
    last_high_index = high_idx[len(high_idx) - 1]
    last_low_index = low_idx[len(low_idx) - 1]
    if last_close > highs[last_high_index]:
        return 1, float(highs[last_high_index]), len(closes) - 1 - last_high_index
    if last_close < lows[last_low_index]:
        return -1, float(lows[last_low_index]), len(closes) - 1 - last_low_index
    return 0, 0.0, -1


@njit(cache=True)
def double_top_bottom_numba(
    highs: NDArray[np.float64],
    lows: NDArray[np.float64],
    closes: NDArray[np.float64],
    window: int,
    tolerance: float,
) -> tuple[int, float, int]:

    high_idx, low_idx = find_pivots_numba(highs, lows, window)
    last_close = closes[len(closes) - 1]
    kind = 0
    level = 0.0
    bars_ago = -1
    if len(high_idx) >= 2:
        first = high_idx[len(high_idx) - 2]
        second = high_idx[len(high_idx) - 1]
        top_level = (highs[first] + highs[second]) / 2.0
        if (
            top_level > 0.0
            and abs(highs[first] - highs[second]) / top_level <= tolerance
            and last_close <= top_level
        ):
            kind = 1
            level = top_level
            bars_ago = len(highs) - 1 - second
    if len(low_idx) >= 2:
        first = low_idx[len(low_idx) - 2]
        second = low_idx[len(low_idx) - 1]
        bottom_level = (lows[first] + lows[second]) / 2.0
        if (
            bottom_level > 0.0
            and abs(lows[first] - lows[second]) / bottom_level <= tolerance
            and last_close >= bottom_level
        ):
            bottom_bars_ago = len(lows) - 1 - second
            if kind == 0 or bottom_bars_ago < bars_ago:
                kind = -1
                level = bottom_level
                bars_ago = bottom_bars_ago
    return kind, float(level), bars_ago
