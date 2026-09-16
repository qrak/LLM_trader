"""
Divergence Pattern Detection - Pure NumPy/Numba Implementation

Detects divergences between price and indicators:
1. Bullish divergence - Price lower low + Indicator higher low (reversal up)
2. Bearish divergence - Price higher high + Indicator lower high (reversal down)

Divergences are powerful reversal signals that often precede major trend changes.

All functions use @njit for performance.
"""


import numpy as np
from numba import njit


@njit(cache=True)
def _find_local_extrema_numba(
    data: np.ndarray,
    lookback: int,
    find_maxima: bool
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find local maxima or minima in data.
    Returns:
        (indices, values) of local extrema
    """
    n = len(data)
    if n < lookback * 2 + 1:
        empty_indices = np.empty(0, dtype=np.int64)
        empty_values = np.empty(0, dtype=np.float64)
        return (empty_indices, empty_values)

    max_extrema = n - 2 * lookback
    indices_temp = np.empty(max_extrema, dtype=np.int64)
    values_temp = np.empty(max_extrema, dtype=np.float64)
    count = 0

    for i in range(lookback, n - lookback):
        is_extrema = True

        for j in range(i - lookback, i + lookback + 1):
            if j == i:
                continue

            if find_maxima:
                if data[i] <= data[j]:
                    is_extrema = False
                    break
            elif data[i] >= data[j]:
                is_extrema = False
                break

        if is_extrema:
            indices_temp[count] = i
            values_temp[count] = data[i]
            count += 1

    if count == 0:
        empty_indices = np.empty(0, dtype=np.int64)
        empty_values = np.empty(0, dtype=np.float64)
        return (empty_indices, empty_values)

    indices = indices_temp[:count]
    values = values_temp[:count]

    return (indices, values)


@njit(cache=True)
def _find_matching_indicator_extrema(
    indicator_indices: np.ndarray,
    indicator_values: np.ndarray,
    price_idx: int,
    tolerance: int = 3
) -> tuple[int, float]:
    """
    Find indicator extrema near a price extrema.
    Returns:
        (indicator_idx, indicator_value) or (-1, 0.0) if not found
    """
    for j in range(len(indicator_indices)):
        if abs(indicator_indices[j] - price_idx) <= tolerance:
            return (indicator_indices[j], indicator_values[j])
    return (-1, 0.0)


@njit(cache=True)
def _detect_divergence_numba(
    prices: np.ndarray,
    indicator: np.ndarray,
    min_spacing: int,
    bullish: bool
) -> tuple[bool, int, int, float, float, float, float]:
    """
    Detect price/indicator divergence (single shared implementation).

    bullish=True:  price makes a lower low while the indicator makes a higher low.
    bullish=False: price makes a higher high while the indicator makes a lower high.

    Returns:
        (divergence_found, first_idx, second_idx,
         first_price, second_price, first_indicator, second_indicator)
    """
    if len(prices) < 10 or len(indicator) < 10:
        return (False, -1, -1, 0.0, 0.0, 0.0, 0.0)

    price_ext_idx, price_ext_values = _find_local_extrema_numba(prices, 10, not bullish)
    indicator_ext_idx, indicator_ext_values = _find_local_extrema_numba(indicator, 10, not bullish)

    if len(price_ext_idx) < 2 or len(indicator_ext_idx) < 2:
        return (False, -1, -1, 0.0, 0.0, 0.0, 0.0)

    for i in range(len(price_ext_idx) - 1, 0, -1):
        second_price_idx = price_ext_idx[i]
        first_price_idx = price_ext_idx[i - 1]

        if second_price_idx - first_price_idx < min_spacing:
            continue

        second_price = price_ext_values[i]
        first_price = price_ext_values[i - 1]

        if bullish:
            if second_price >= first_price:
                continue
            move_pct = (first_price - second_price) / first_price * 100
        else:
            if second_price <= first_price:
                continue
            move_pct = (second_price - first_price) / first_price * 100

        if move_pct < 0.5:
            continue

        first_indicator_idx, first_indicator_value = _find_matching_indicator_extrema(
            indicator_ext_idx, indicator_ext_values, first_price_idx
        )
        if first_indicator_idx == -1:
            continue

        second_indicator_idx, second_indicator_value = _find_matching_indicator_extrema(
            indicator_ext_idx, indicator_ext_values, second_price_idx
        )
        if second_indicator_idx == -1:
            continue

        if bullish:
            if second_indicator_value <= first_indicator_value:
                continue
        elif second_indicator_value >= first_indicator_value:
            continue

        return (
            True,
            first_price_idx,
            second_price_idx,
            first_price,
            second_price,
            first_indicator_value,
            second_indicator_value
        )

    return (False, -1, -1, 0.0, 0.0, 0.0, 0.0)


@njit(cache=True)
def detect_bullish_divergence_numba(
    prices: np.ndarray,
    indicator: np.ndarray,
    min_spacing: int = 5
) -> tuple[bool, int, int, float, float, float, float]:
    """Detect bullish divergence (price lower low, indicator higher low)."""
    return _detect_divergence_numba(prices, indicator, min_spacing, True)


@njit(cache=True)
def detect_bearish_divergence_numba(
    prices: np.ndarray,
    indicator: np.ndarray,
    min_spacing: int = 5
) -> tuple[bool, int, int, float, float, float, float]:
    """Detect bearish divergence (price higher high, indicator lower high)."""
    return _detect_divergence_numba(prices, indicator, min_spacing, False)
