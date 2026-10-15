"""Support Resistance Indicators module.

Provides functionality for indicators.support_resistance.support_resistance_indicators.py.
"""
import math

import numpy as np
from numba import njit


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
def retested_support_resistance_numba(high, low, close, length=120, min_touches=3, price_tolerance=0.005):
    """Nearest levels with repeated, confirmed swing touches in a trailing window."""
    n = len(close)
    support = np.full(n, np.nan)
    resistance = np.full(n, np.nan)
    swing_lows = np.full(n, np.nan)
    swing_highs = np.full(n, np.nan)

    for j in range(1, n - 1):
        if low[j] <= low[j - 1] and low[j] < low[j + 1]:
            swing_lows[j] = low[j]
        if high[j] >= high[j - 1] and high[j] > high[j + 1]:
            swing_highs[j] = high[j]

    for i in range(length, n):
        start = max(1, i - length)
        price = close[i]
        for swings, levels, is_support in ((swing_lows, support, True), (swing_highs, resistance, False)):
            for j in range(start, i):
                anchor = swings[j]
                if math.isnan(anchor) or anchor <= 0:
                    continue
                count = 0
                total = 0.0
                for k in range(start, i):
                    candidate = swings[k]
                    if not math.isnan(candidate) and abs(candidate - anchor) <= anchor * price_tolerance:
                        count += 1
                        total += candidate
                if count < min_touches:
                    continue
                level = total / count
                if is_support and level < price and (math.isnan(levels[i]) or level > levels[i]):
                    levels[i] = level
                if not is_support and level > price and (math.isnan(levels[i]) or level < levels[i]):
                    levels[i] = level

    return support, resistance

@njit(cache=True)
def pivot_points_numba(high, low, close):
    """Calculate standard pivot points and support/resistance levels

    Returns: tuple of (pivot_point, r1, r2, r3, r4, s1, s2, s3, s4) arrays
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
        pivot_point[i] = (high[i - 1] + low[i - 1] + close[i - 1]) / 3

        r1[i] = (2 * pivot_point[i]) - low[i - 1]
        r2[i] = pivot_point[i] + (high[i - 1] - low[i - 1])
        r3[i] = pivot_point[i] + 2.0 * (high[i - 1] - low[i - 1])
        r4[i] = pivot_point[i] + 3.0 * (high[i - 1] - low[i - 1])

        s1[i] = (2 * pivot_point[i]) - high[i - 1]
        s2[i] = pivot_point[i] - (high[i - 1] - low[i - 1])
        s3[i] = pivot_point[i] - 2.0 * (high[i - 1] - low[i - 1])
        s4[i] = pivot_point[i] - 3.0 * (high[i - 1] - low[i - 1])

    return pivot_point, r1, r2, r3, r4, s1, s2, s3, s4

@njit(cache=True)
def fibonacci_pivot_points_numba(high, low, close):
    """Calculate Fibonacci pivot points using Fibonacci ratios

    Uses Fibonacci ratios (0.382, 0.618, 1.0, 1.618) for support/resistance levels

    Returns: tuple of (pivot_point, r1, r2, r3, s1, s2, s3) arrays
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
        pivot_point[i] = (high[i - 1] + low[i - 1] + close[i - 1]) / 3

        range_val = high[i - 1] - low[i - 1]

        r1[i] = pivot_point[i] + (0.382 * range_val)
        r2[i] = pivot_point[i] + (0.618 * range_val)
        r3[i] = pivot_point[i] + (1.000 * range_val)

        s1[i] = pivot_point[i] - (0.382 * range_val)
        s2[i] = pivot_point[i] - (0.618 * range_val)
        s3[i] = pivot_point[i] - (1.000 * range_val)

    return pivot_point, r1, r2, r3, s1, s2, s3
