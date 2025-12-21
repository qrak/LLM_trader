import numpy as np
from numba import njit

from src.indicators.indicators.trend import supertrend_numba


@njit(cache=True)
def _calculate_volume_filter_numba(volume, length, i):
    """
    Calculate volume filter for support/resistance detection (helper function).
    
    Args:
        volume: Volume array
        length: Lookback period
        i: Current index
        
    Returns:
        (rolling_avg_volume: float, volume_filter: bool)
    """
    rolling_avg_volume = np.mean(volume[i - length:i])
    volume_filter = volume[i] > rolling_avg_volume
    return rolling_avg_volume, volume_filter


@njit(cache=True)
def support_resistance_numba(high, low, length):
    n = len(high)
    rolling_resistance = np.full(n, np.nan)
    rolling_support = np.full(n, np.nan)

    for i in range(length - 1, n):
        rolling_resistance[i] = np.max(high[i - length + 1:i + 1])
        rolling_support[i] = np.min(low[i - length + 1:i + 1])

    return rolling_support, rolling_resistance


@njit(cache=True)
def support_resistance_numba_advanced(high, low, close, volume, length):
    n = len(close)
    pivot_points = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    volume_filter = np.full(n, False)
    rolling_avg_volume = np.full(n, np.nan)

    for i in range(length, n):
        rolling_avg_volume[i], volume_filter[i] = _calculate_volume_filter_numba(volume, length, i)

        pivot_points[i] = (high[i - 1] + low[i - 1] + close[i - 1]) / 3

        r1[i] = (2 * pivot_points[i]) - low[i - 1]
        s1[i] = (2 * pivot_points[i]) - high[i - 1]

    strong_support = np.where(volume_filter, s1, np.nan)
    strong_resistance = np.where(volume_filter, r1, np.nan)

    return strong_support, strong_resistance



@njit(cache=True)
def advanced_support_resistance_numba(high, low, close, volume, length=50, strength_threshold=2, persistence=1,
                                      volume_factor=2.0, price_factor=0.005):
    n = len(close)
    pivot_points = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    volume_filter = np.full(n, False)
    rolling_avg_volume = np.full(n, np.nan)

    support_strength = np.zeros(n)
    resistance_strength = np.zeros(n)

    strong_support = np.full(n, np.nan)
    strong_resistance = np.full(n, np.nan)

    for i in range(length, n):
        rolling_avg_volume[i], volume_filter[i] = _calculate_volume_filter_numba(volume, length, i)
        pivot_points[i] = (high[i - 1] + low[i - 1] + close[i - 1]) / 3

        r1[i] = (2 * pivot_points[i]) - low[i - 1]
        s1[i] = (2 * pivot_points[i]) - high[i - 1]
        r2[i] = pivot_points[i] + (high[i - 1] - low[i - 1])
        s2[i] = pivot_points[i] - (high[i - 1] - low[i - 1])

        if close[i] < s1[i]:
            support_strength[i] = support_strength[i - 1] + 1
        elif close[i] > r1[i]:
            resistance_strength[i] = resistance_strength[i - 1] + 1
        else:
            support_strength[i] = max(0, support_strength[i - 1] - 1)
            resistance_strength[i] = max(0, resistance_strength[i - 1] - 1)

        if volume_filter[i] and volume[i] > volume_factor * rolling_avg_volume[i]:
            if support_strength[i] >= strength_threshold and close[i] < (1 - price_factor) * s1[i]:
                for j in range(max(0, i - persistence + 1), i + 1):
                    strong_support[j] = min(s1[j], s2[j])
            if resistance_strength[i] >= strength_threshold and close[i] > (1 + price_factor) * r1[i]:
                for j in range(max(0, i - persistence + 1), i + 1):
                    strong_resistance[j] = max(r1[j], r2[j])

    return strong_support, strong_resistance


@njit(cache=True)
def find_support_resistance_numba(close, support, resistance, window):
    current_price = close[-1]

    valid_support = support[~np.isnan(support)][-window:]
    valid_resistance = resistance[~np.isnan(resistance)][-window:]

    if len(valid_support) > 0:
        nearest_support = np.max(valid_support[valid_support < current_price]) if np.any(
            valid_support < current_price) else np.min(valid_support)
        distance_to_support = (current_price - nearest_support) / current_price
    else:
        distance_to_support = 1

    if len(valid_resistance) > 0:
        nearest_resistance = np.min(valid_resistance[valid_resistance > current_price]) if np.any(
            valid_resistance > current_price) else np.max(valid_resistance)
        distance_to_resistance = (nearest_resistance - current_price) / current_price
    else:
        distance_to_resistance = 1

    return distance_to_support, distance_to_resistance

@njit(cache=True)
def fibonacci_retracement_numba(length, high, low):
    n = len(high)
    fib_levels = np.array([0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0])

    retracement_values = np.full((n, len(fib_levels)), np.nan)

    for i in range(length - 1, n):
        high_max = np.max(high[i - length + 1:i + 1])
        low_min = np.min(low[i - length + 1:i + 1])
        diff = high_max - low_min

        for j, level in enumerate(fib_levels):
            retracement_values[i, j] = low_min + diff * level

    return retracement_values

@njit(cache=True)
def pivot_points_numba(high, low, close):
    """Calculate standard pivot points and support/resistance levels
    
    Returns:
        Tuple of (pivot_point, r1, r2, r3, r4, s1, s2, s3, s4) arrays
    """
    n = len(high)
    pivot_point = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    r4 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    s4 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Calculate pivot point as simple average of H, L, C from previous period
        pivot_point[i] = (high[i - 1] + low[i - 1] + close[i - 1]) / 3
        
        # Calculate resistance levels
        r1[i] = (2 * pivot_point[i]) - low[i - 1]
        r2[i] = pivot_point[i] + (high[i - 1] - low[i - 1])
        # Additional higher resistance levels (extended multiples of the high-low range)
        r3[i] = pivot_point[i] + 2.0 * (high[i - 1] - low[i - 1])
        r4[i] = pivot_point[i] + 3.0 * (high[i - 1] - low[i - 1])
        
        # Calculate support levels  
        s1[i] = (2 * pivot_point[i]) - high[i - 1]
        s2[i] = pivot_point[i] - (high[i - 1] - low[i - 1])
        # Additional lower support levels (extended multiples of the high-low range)
        s3[i] = pivot_point[i] - 2.0 * (high[i - 1] - low[i - 1])
        s4[i] = pivot_point[i] - 3.0 * (high[i - 1] - low[i - 1])
    
    return pivot_point, r1, r2, r3, r4, s1, s2, s3, s4

@njit(cache=True)
def floating_levels_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                         length: int, multiplier: float, lookback: int,
                         level_up: float, level_down: float):
    supertrend, _ = supertrend_numba(high, low, close, length, multiplier)
    n = len(supertrend)
    flu = np.empty(n, dtype=np.float64)
    fld = np.empty(n, dtype=np.float64)
    flm = np.empty(n, dtype=np.float64)

    for i in range(lookback, n):
        mini = np.min(supertrend[i - lookback:i])
        maxi = np.max(supertrend[i - lookback:i])
        rrange = maxi - mini
        flu[i] = mini + level_up * rrange / 100.0
        fld[i] = mini + level_down * rrange / 100.0
        flm[i] = mini + 0.5 * rrange

    flu[:lookback] = np.nan
    fld[:lookback] = np.nan
    flm[:lookback] = np.nan

    return flu, fld, flm

@njit(cache=True)
def fibonacci_bollinger_bands_numba(src, volume, length, mult):
    vwma_values = np.empty_like(src)
    stdev_values = np.empty_like(src)
    basis = np.empty_like(src)
    dev = np.empty_like(src)
    upper_bands = np.empty((6, len(src)))
    lower_bands = np.empty((6, len(src)))
    fib_levels = np.array([0.236, 0.382, 0.5, 0.618, 0.764, 1.0])

    for i in range(len(src)):
        if i < length:
            vwma_values[i] = np.nan
            stdev_values[i] = np.nan
            basis[i] = np.nan
            dev[i] = np.nan
            for j in range(6):
                upper_bands[j, i] = np.nan
                lower_bands[j, i] = np.nan
        else:
            sum_pv = np.sum(src[i - length + 1:i + 1] * volume[i - length + 1:i + 1])
            sum_v = np.sum(volume[i - length + 1:i + 1])
            vwma_values[i] = sum_pv / sum_v if sum_v != 0 else np.nan

            mean = np.mean(src[i - length + 1:i + 1])
            variance = np.sum((src[i - length + 1:i + 1] - mean) ** 2) / length
            stdev_values[i] = np.sqrt(variance)

            basis[i] = vwma_values[i]
            dev[i] = mult * stdev_values[i]


            for j in range(6):
                upper_bands[j, i] = basis[i] + (fib_levels[j] * dev[i])
                lower_bands[j, i] = basis[i] - (fib_levels[j] * dev[i])

    return basis, upper_bands, lower_bands
