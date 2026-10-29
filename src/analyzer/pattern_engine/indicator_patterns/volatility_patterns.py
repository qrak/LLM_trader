


import numpy as np
from numba import njit

BB_SQUEEZE_PERCENTILE = 20.0
BB_SQUEEZE_LOOKBACK = 120


@njit(cache=True)
def detect_atr_spike_numba(
    atr: np.ndarray,
    spike_threshold: float = 1.5,
    lookback: int = 14
) -> tuple[bool, int, float, float]:

    if len(atr) < lookback + 1:
        return (False, -1, 0.0, 0.0)

    current_atr = atr[-1]

    recent_atr = atr[-(lookback+1):-1]
    avg_atr = np.mean(recent_atr)

    if avg_atr < 0.0001:
        return (False, -1, current_atr, avg_atr)  # type: ignore

    spike_ratio = current_atr / avg_atr

    if spike_ratio >= spike_threshold:
        return (True, 0, current_atr, avg_atr)  # type: ignore

    return (False, -1, current_atr, avg_atr)  # type: ignore


@njit(cache=True)
def detect_bb_squeeze_numba(
    bb_upper: np.ndarray,
    bb_lower: np.ndarray,
    squeeze_percentile: float = 20.0,
    lookback: int = 120
) -> tuple[bool, float, float, float]:

    if len(bb_upper) < lookback or len(bb_lower) < lookback:
        return (False, 0.0, 0.0, 0.0)

    current_width, percentile_width, width_percentile = _bb_width_position_at(
        bb_upper, bb_lower, len(bb_upper) - 1, squeeze_percentile, lookback
    )

    if current_width <= percentile_width:
        return (True, current_width, percentile_width, width_percentile)

    return (False, current_width, percentile_width, width_percentile)


@njit(cache=True)
def _bb_width_position_at(
    bb_upper: np.ndarray,
    bb_lower: np.ndarray,
    end: int,
    squeeze_percentile: float,
    lookback: int
) -> tuple[float, float, float]:

    start = end - lookback + 1
    widths = bb_upper[start:end + 1] - bb_lower[start:end + 1]
    current_width = widths[lookback - 1]

    sorted_widths = np.sort(widths)
    percentile_index = int(lookback * squeeze_percentile / 100.0)
    percentile_index = max(0, min(percentile_index, lookback - 1))
    percentile_width = sorted_widths[percentile_index]

    narrower = 0
    for i in range(lookback):
        if widths[i] < current_width:
            narrower += 1

    return current_width, percentile_width, 100.0 * narrower / lookback


@njit(cache=True)
def bb_squeeze_streak_numba(
    bb_upper: np.ndarray,
    bb_lower: np.ndarray,
    squeeze_percentile: float = 20.0,
    lookback: int = 120
) -> int:

    n = len(bb_upper)
    if n < lookback or len(bb_lower) < lookback:
        return 0

    bars = 0
    for end in range(n - 1, lookback - 2, -1):
        current_width, percentile_width, _ = _bb_width_position_at(
            bb_upper, bb_lower, end, squeeze_percentile, lookback
        )
        if current_width > percentile_width:
            break
        bars += 1
    return bars


@njit(cache=True)
def _volatility_trend_at(atr: np.ndarray, end: int, lookback: int) -> int:

    if end + 1 < lookback:
        return 0

    increasing = 0
    decreasing = 0

    for i in range(end - lookback + 2, end + 1):
        diff = atr[i] - atr[i - 1]
        if diff > 0.0:
            increasing += 1
        elif diff < 0.0:
            decreasing += 1

    threshold = (lookback - 1) * 0.6

    if float(increasing) >= threshold:
        return 1
    if float(decreasing) >= threshold:
        return -1
    return 0


@njit(cache=True)
def detect_volatility_trend_numba(
    atr: np.ndarray,
    lookback: int = 10
) -> int:

    if len(atr) < lookback:
        return 0

    return _volatility_trend_at(atr, len(atr) - 1, lookback)


@njit(cache=True)
def volatility_trend_streak_numba(
    atr: np.ndarray,
    lookback: int = 10
) -> int:

    n = len(atr)
    if n < lookback:
        return 0

    current = _volatility_trend_at(atr, n - 1, lookback)
    if current == 0:
        return 0

    bars = 0
    for end in range(n - 1, lookback - 2, -1):
        if _volatility_trend_at(atr, end, lookback) != current:
            break
        bars += 1
    return bars


@njit(cache=True)
def detect_keltner_squeeze_numba(
    kc_upper: np.ndarray,
    kc_lower: np.ndarray,
    bb_upper: np.ndarray,
    bb_lower: np.ndarray
) -> tuple[bool, int, int]:

    n = len(kc_upper)
    if n < 1 or len(bb_upper) < n or len(kc_lower) < n or len(bb_lower) < n:
        return (False, 0, -1)

    inside = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        inside[i] = bb_upper[i] <= kc_upper[i] and bb_lower[i] >= kc_lower[i]

    bars_in_squeeze = 0
    while bars_in_squeeze < n and inside[n - 1 - bars_in_squeeze]:
        bars_in_squeeze += 1

    if bars_in_squeeze > 0:
        return (True, bars_in_squeeze, -1)

    for i in range(n - 1, 0, -1):
        if inside[i - 1] and not inside[i]:
            return (False, 0, (n - 1) - i)

    return (False, 0, -1)
