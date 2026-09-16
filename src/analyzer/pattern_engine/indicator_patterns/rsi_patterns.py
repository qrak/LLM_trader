"""
RSI Pattern Detection - Pure NumPy/Numba Implementation

Detects RSI-based patterns:
1. Oversold conditions (RSI < threshold)
2. Overbought conditions (RSI > threshold)
3. W-Bottoms (bullish reversal - double bottom in RSI)
4. M-Tops (bearish reversal - double top in RSI)

All functions use @njit for performance.
"""


import numpy as np
from numba import njit


@njit(cache=True)
def _detect_rsi_threshold_numba(
    rsi: np.ndarray,
    threshold: float,
    min_periods: int,
    above: bool
) -> tuple[bool, int, float]:
    """
    Detect a sustained RSI breach of ``threshold`` (single shared implementation).
    Returns:
        (is_in_zone, periods_ago, rsi_value)
    """
    if len(rsi) < 1:
        return (False, -1, 0.0)

    current_rsi = rsi[-1]

    if above:
        still_in_zone = current_rsi > threshold
    else:
        still_in_zone = current_rsi < threshold
    if not still_in_zone:
        return (False, -1, current_rsi)

    periods_in_zone = 0
    for i in range(len(rsi) - 1, -1, -1):
        if above:
            in_zone = rsi[i] > threshold
        else:
            in_zone = rsi[i] < threshold
        if in_zone:
            periods_in_zone += 1
        else:
            break

    if periods_in_zone >= min_periods:
        return (True, 0, current_rsi)

    return (False, -1, current_rsi)


@njit(cache=True)
def detect_rsi_oversold_numba(
    rsi: np.ndarray,
    threshold: float = 30.0,
    min_periods: int = 1
) -> tuple[bool, int, float]:
    """Detect oversold conditions in RSI (RSI below threshold)."""
    return _detect_rsi_threshold_numba(rsi, threshold, min_periods, False)


@njit(cache=True)
def detect_rsi_overbought_numba(
    rsi: np.ndarray,
    threshold: float = 70.0,
    min_periods: int = 1
) -> tuple[bool, int, float]:
    """Detect overbought conditions in RSI (RSI above threshold)."""
    return _detect_rsi_threshold_numba(rsi, threshold, min_periods, True)


@njit(cache=True)
def detect_rsi_w_bottom_numba(
    rsi: np.ndarray,
    prices: np.ndarray,
    threshold: float = 30.0,
    similarity_threshold: float = 5.0,
    lookback: int = 14
) -> tuple[bool, int, int, float, float]:
    """W-bottom in RSI (bullish reversal confirmation).

    Both bottoms oversold, the second higher in RSI while price makes an equal or
    lower low, bottoms within similarity_threshold.
    Returns (found, first_idx, second_idx, first_rsi, second_rsi).
    """
    if len(rsi) < lookback or len(prices) < lookback:
        return (False, -1, -1, 0.0, 0.0)

    recent_rsi = rsi[-lookback:]
    recent_prices = prices[-lookback:]

    if len(recent_rsi) < 3:
        return (False, -1, -1, 0.0, 0.0)

    second_bottom_value = recent_rsi[-1]
    second_price = recent_prices[-1]

    if second_bottom_value >= threshold:
        return (False, -1, -1, 0.0, 0.0)

    first_bottom_idx = -1
    first_bottom_value = 0.0
    first_price = 0.0

    for i in range(len(recent_rsi) - 3, -1, -1):
        if recent_rsi[i] < threshold:
            is_local_min = True
            if i > 0 and recent_rsi[i] >= recent_rsi[i-1]:
                is_local_min = False
            if i < len(recent_rsi) - 1 and recent_rsi[i] >= recent_rsi[i+1]:
                is_local_min = False

            if is_local_min:
                first_bottom_idx = i
                first_bottom_value = recent_rsi[i]
                first_price = recent_prices[i]
                break

    if first_bottom_idx == -1:
        return (False, -1, -1, 0.0, 0.0)

    if second_bottom_value <= first_bottom_value:
        return (False, -1, -1, 0.0, 0.0)

    rsi_diff = abs(second_bottom_value - first_bottom_value)
    if rsi_diff > similarity_threshold:
        return (False, -1, -1, 0.0, 0.0)

    if second_price > first_price * 1.02:
        return (False, -1, -1, 0.0, 0.0)

    return (True, first_bottom_idx, len(recent_rsi) - 1, first_bottom_value, second_bottom_value)


@njit(cache=True)
def detect_rsi_m_top_numba(
    rsi: np.ndarray,
    prices: np.ndarray,
    threshold: float = 70.0,
    similarity_threshold: float = 5.0,
    lookback: int = 14
) -> tuple[bool, int, int, float, float]:
    """M-top in RSI (bearish reversal confirmation).

    Mirror of the W-bottom: both peaks overbought, the second lower in RSI while
    price makes an equal or higher high.
    Returns (found, first_idx, second_idx, first_rsi, second_rsi).
    """
    if len(rsi) < lookback or len(prices) < lookback:
        return (False, -1, -1, 0.0, 0.0)

    recent_rsi = rsi[-lookback:]
    recent_prices = prices[-lookback:]

    if len(recent_rsi) < 3:
        return (False, -1, -1, 0.0, 0.0)

    second_top_value = recent_rsi[-1]
    second_price = recent_prices[-1]

    if second_top_value <= threshold:
        return (False, -1, -1, 0.0, 0.0)

    first_top_idx = -1
    first_top_value = 0.0
    first_price = 0.0

    for i in range(len(recent_rsi) - 3, -1, -1):
        if recent_rsi[i] > threshold:
            is_local_max = True
            if i > 0 and recent_rsi[i] <= recent_rsi[i-1]:
                is_local_max = False
            if i < len(recent_rsi) - 1 and recent_rsi[i] <= recent_rsi[i+1]:
                is_local_max = False

            if is_local_max:
                first_top_idx = i
                first_top_value = recent_rsi[i]
                first_price = recent_prices[i]
                break

    if first_top_idx == -1:
        return (False, -1, -1, 0.0, 0.0)

    if second_top_value >= first_top_value:
        return (False, -1, -1, 0.0, 0.0)

    rsi_diff = abs(second_top_value - first_top_value)
    if rsi_diff > similarity_threshold:
        return (False, -1, -1, 0.0, 0.0)

    if second_price < first_price * 0.98:
        return (False, -1, -1, 0.0, 0.0)

    return (True, first_top_idx, len(recent_rsi) - 1, first_top_value, second_top_value)
