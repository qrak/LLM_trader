import math
import numpy as np
from numba import njit

from src.indicators.trend import supertrend_numba


@njit(cache=True)
def support_resistance_numba(high, low, length):
    n = len(high)
    rolling_resistance = np.full(n, np.nan)
    rolling_support = np.full(n, np.nan)

    if n < length:
        return rolling_support, rolling_resistance

    current_max = high[0]
    max_idx = 0
    current_min = low[0]
    min_idx = 0

    nan_count_high = 0
    nan_count_low = 0
    for i in range(length - 1):
        if math.isnan(high[i]):
            nan_count_high += 1
        if math.isnan(low[i]):
            nan_count_low += 1

    for i in range(1, length - 1):
        if high[i] >= current_max or math.isnan(current_max):
            current_max = high[i]
            max_idx = i
        if low[i] <= current_min or math.isnan(current_min):
            current_min = low[i]
            min_idx = i

    for i in range(length - 1, n):
        if math.isnan(high[i]):
            nan_count_high += 1
        if math.isnan(low[i]):
            nan_count_low += 1

        old_idx = i - length + 1

        if high[i] >= current_max or math.isnan(current_max):
            current_max = high[i]
            max_idx = i
        elif max_idx < old_idx:
            current_max = high[old_idx]
            max_idx = old_idx
            for j in range(old_idx + 1, i + 1):
                if high[j] >= current_max or math.isnan(current_max):
                    current_max = high[j]
                    max_idx = j

        if low[i] <= current_min or math.isnan(current_min):
            current_min = low[i]
            min_idx = i
        elif min_idx < old_idx:
            current_min = low[old_idx]
            min_idx = old_idx
            for j in range(old_idx + 1, i + 1):
                if low[j] <= current_min or math.isnan(current_min):
                    current_min = low[j]
                    min_idx = j

        if nan_count_high > 0:
            rolling_resistance[i] = np.nan
        else:
            rolling_resistance[i] = current_max

        if nan_count_low > 0:
            rolling_support[i] = np.nan
        else:
            rolling_support[i] = current_min

        if math.isnan(high[old_idx]):
            nan_count_high -= 1
        if math.isnan(low[old_idx]):
            nan_count_low -= 1

    return rolling_support, rolling_resistance


@njit(cache=True)
def support_resistance_numba_advanced(high, low, close, volume, length):
    n = len(close)
    pivot_points = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    volume_filter = np.full(n, False)
    rolling_avg_volume = np.full(n, np.nan)

    # Guard: seeding loop below needs at least `length` elements
    if n < length:
        strong_support = np.where(volume_filter, s1, np.nan)
        strong_resistance = np.where(volume_filter, r1, np.nan)
        return strong_support, strong_resistance

    # O(N) running sum replaces O(N*K) volume slice scans.
    # NaN-safe: track nan_count so vol_sum recovers once NaN scrolls out of the window.
    vol_sum = 0.0
    nan_count = 0
    for j in range(length):
        v = volume[j]
        if math.isnan(v):
            nan_count += 1
        else:
            vol_sum += v

    for i in range(length, n):
        if nan_count > 0:
            rolling_avg_volume[i] = np.nan
            volume_filter[i] = False
        else:
            rolling_avg_volume[i] = vol_sum / length
            volume_filter[i] = volume[i] > rolling_avg_volume[i]

        # Slide the window: remove oldest element, add newest
        old_val = volume[i - length]
        if math.isnan(old_val):
            nan_count -= 1
        else:
            vol_sum -= old_val
        new_val = volume[i]
        if math.isnan(new_val):
            nan_count += 1
        else:
            vol_sum += new_val

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

    # Guard: seeding loop below needs at least `length` elements
    if n < length:
        return strong_support, strong_resistance

    # O(N) running sum replaces O(N*K) volume slice scans.
    # NaN-safe: track nan_count so vol_sum recovers once NaN scrolls out of the window.
    vol_sum = 0.0
    nan_count = 0
    for j in range(length):
        v = volume[j]
        if math.isnan(v):
            nan_count += 1
        else:
            vol_sum += v

    for i in range(length, n):
        if nan_count > 0:
            rolling_avg_volume[i] = np.nan
            volume_filter[i] = False
        else:
            rolling_avg_volume[i] = vol_sum / length
            volume_filter[i] = volume[i] > rolling_avg_volume[i]

        # Slide the window: remove oldest element, add newest
        old_val = volume[i - length]
        if math.isnan(old_val):
            nan_count -= 1
        else:
            vol_sum -= old_val
        new_val = volume[i]
        if math.isnan(new_val):
            nan_count += 1
        else:
            vol_sum += new_val

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

    if n < length:
        return retracement_values

    current_max = high[0]
    max_idx = 0
    current_min = low[0]
    min_idx = 0

    nan_count_high = 0
    nan_count_low = 0
    for i in range(length - 1):
        if math.isnan(high[i]):
            nan_count_high += 1
        if math.isnan(low[i]):
            nan_count_low += 1

    for i in range(1, length - 1):
        if high[i] >= current_max or math.isnan(current_max):
            current_max = high[i]
            max_idx = i
        if low[i] <= current_min or math.isnan(current_min):
            current_min = low[i]
            min_idx = i

    for i in range(length - 1, n):
        if math.isnan(high[i]):
            nan_count_high += 1
        if math.isnan(low[i]):
            nan_count_low += 1

        old_idx = i - length + 1

        if high[i] >= current_max or math.isnan(current_max):
            current_max = high[i]
            max_idx = i
        elif max_idx < old_idx:
            current_max = high[old_idx]
            max_idx = old_idx
            for j in range(old_idx + 1, i + 1):
                if high[j] >= current_max or math.isnan(current_max):
                    current_max = high[j]
                    max_idx = j

        if low[i] <= current_min or math.isnan(current_min):
            current_min = low[i]
            min_idx = i
        elif min_idx < old_idx:
            current_min = low[old_idx]
            min_idx = old_idx
            for j in range(old_idx + 1, i + 1):
                if low[j] <= current_min or math.isnan(current_min):
                    current_min = low[j]
                    min_idx = j

        if nan_count_high > 0 or nan_count_low > 0:
            for j in range(len(fib_levels)):
                retracement_values[i, j] = np.nan
        else:
            high_max = current_max
            low_min = current_min
            diff = high_max - low_min

            for j, level in enumerate(fib_levels):
                retracement_values[i, j] = low_min + diff * level

        if math.isnan(high[old_idx]):
            nan_count_high -= 1
        if math.isnan(low[old_idx]):
            nan_count_low -= 1

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
def fibonacci_pivot_points_numba(high, low, close):
    """Calculate Fibonacci pivot points using Fibonacci ratios

    Uses Fibonacci ratios (0.382, 0.618, 1.0, 1.618) for support/resistance levels

    Returns:
        Tuple of (pivot_point, r1, r2, r3, s1, s2, s3) arrays
    """
    n = len(high)
    pivot_point = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)

    for i in range(1, n):
        # Calculate pivot point as simple average of H, L, C from previous period
        pivot_point[i] = (high[i - 1] + low[i - 1] + close[i - 1]) / 3

        # Calculate range from previous period
        range_val = high[i - 1] - low[i - 1]

        # Calculate Fibonacci resistance levels
        r1[i] = pivot_point[i] + (0.382 * range_val)
        r2[i] = pivot_point[i] + (0.618 * range_val)
        r3[i] = pivot_point[i] + (1.000 * range_val)

        # Calculate Fibonacci support levels
        s1[i] = pivot_point[i] - (0.382 * range_val)
        s2[i] = pivot_point[i] - (0.618 * range_val)
        s3[i] = pivot_point[i] - (1.000 * range_val)

    return pivot_point, r1, r2, r3, s1, s2, s3

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
    n = len(src)
    vwma_values = np.empty(n)
    stdev_values = np.empty(n)
    basis = np.empty(n)
    dev = np.empty(n)
    upper_bands = np.empty((6, n))
    lower_bands = np.empty((6, n))
    fib_levels = np.array([0.236, 0.382, 0.5, 0.618, 0.764, 1.0])

    for i in range(n):
        if i < length:
            vwma_values[i] = np.nan
            stdev_values[i] = np.nan
            basis[i] = np.nan
            dev[i] = np.nan
            for j in range(6):
                upper_bands[j, i] = np.nan
                lower_bands[j, i] = np.nan

    if n < length:
        return basis, upper_bands, lower_bands

    sum_pv = 0.0
    sum_v = 0.0
    sum_src = 0.0
    sum_src_sq = 0.0

    for i in range(length - 1):
        pv = src[i] * volume[i]
        sum_pv += pv
        sum_v += volume[i]
        sum_src += src[i]
        sum_src_sq += src[i] * src[i]

    inv_length = 1.0 / length

    for i in range(length - 1, n):
        # Recenter every 1000 items to prevent floating-point precision loss
        if (i - length + 1) % 1000 == 0 and i > length - 1:
            sum_pv = 0.0
            sum_v = 0.0
            sum_src = 0.0
            sum_src_sq = 0.0
            for j in range(i - length + 1, i):
                pv = src[j] * volume[j]
                sum_pv += pv
                sum_v += volume[j]
                sum_src += src[j]
                sum_src_sq += src[j] * src[j]

        pv = src[i] * volume[i]
        sum_pv += pv
        sum_v += volume[i]
        sum_src += src[i]
        sum_src_sq += src[i] * src[i]

        if sum_v != 0.0:
            vwma_values[i] = sum_pv / sum_v
        else:
            vwma_values[i] = np.nan

        mean = sum_src * inv_length
        variance = (sum_src_sq * inv_length) - (mean * mean)
        if variance < 0.0:
            variance = 0.0
        stdev_values[i] = np.sqrt(variance)

        basis[i] = vwma_values[i]
        dev[i] = mult * stdev_values[i]

        for j in range(6):
            upper_bands[j, i] = basis[i] + (fib_levels[j] * dev[i])
            lower_bands[j, i] = basis[i] - (fib_levels[j] * dev[i])

        old_idx = i - length + 1
        old_pv = src[old_idx] * volume[old_idx]
        sum_pv -= old_pv
        sum_v -= volume[old_idx]
        sum_src -= src[old_idx]
        sum_src_sq -= src[old_idx] * src[old_idx]

    return basis, upper_bands, lower_bands
