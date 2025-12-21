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
from typing import Tuple


@njit(cache=True)
def detect_macd_crossover_numba(
    macd_line: np.ndarray,
    signal_line: np.ndarray,
    lookback: int = 5
) -> Tuple[bool, bool, int, float, float]:
    """
    Detect MACD line crossing signal line.
    
    Scans ENTIRE array for the most recent crossover event.
    Note: lookback parameter kept for backward compatibility but ignored.
    
    Args:
        macd_line: MACD line values (most recent last)
        signal_line: Signal line values (most recent last)
        lookback: Deprecated - scans entire array
        
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
    
    # Scan ENTIRE array for most recent crossover (starting from most recent)
    actual_lookback = len(macd_line) - 1
    
    # Check all periods for crossover
    for i in range(actual_lookback):
        current_idx = len(macd_line) - 1 - i
        prev_idx = current_idx - 1
        
        if current_idx < 1:
            break
        
        macd_current = macd_line[current_idx]
        macd_prev = macd_line[prev_idx]
        signal_current = signal_line[current_idx]
        signal_prev = signal_line[prev_idx]
        
        # Bullish crossover: MACD was below, now above signal
        if macd_prev <= signal_prev and macd_current > signal_current:
            return (True, True, i, macd_current, signal_current)
        
        # Bearish crossover: MACD was above, now below signal
        if macd_prev >= signal_prev and macd_current < signal_current:
            return (True, False, i, macd_current, signal_current)
    
    return (False, False, -1, 0.0, 0.0)


@njit(cache=True)
def detect_macd_zero_cross_numba(
    macd_line: np.ndarray,
    lookback: int = 5
) -> Tuple[bool, bool, int, float]:
    """
    Detect MACD line crossing zero line.
    
    Scans ENTIRE array for the most recent zero-line crossover.
    Note: lookback parameter kept for backward compatibility but ignored.
    
    Zero-line cross indicates momentum shift:
    - Crossing above zero = bullish momentum
    - Crossing below zero = bearish momentum
    
    Args:
        macd_line: MACD line values (most recent last)
        lookback: Deprecated - scans entire array
        
    Returns:
        (crossover_found, is_bullish, periods_ago, macd_value)
        - crossover_found: True if zero-line cross detected
        - is_bullish: True = crossed above zero, False = crossed below
        - periods_ago: How many periods ago crossover occurred
        - macd_value: MACD value at crossover
    """
    if len(macd_line) < 2:
        return (False, False, -1, 0.0)
    
    # Scan ENTIRE array for most recent zero-line crossover
    actual_lookback = len(macd_line) - 1
    
    # Check all periods for zero-line crossover
    for i in range(actual_lookback):
        current_idx = len(macd_line) - 1 - i
        prev_idx = current_idx - 1
        
        if current_idx < 1:
            break
        
        macd_current = macd_line[current_idx]
        macd_prev = macd_line[prev_idx]
        
        # Bullish: crossed above zero
        if macd_prev <= 0.0 and macd_current > 0.0:
            return (True, True, i, macd_current)
        
        # Bearish: crossed below zero
        if macd_prev >= 0.0 and macd_current < 0.0:
            return (True, False, i, macd_current)
    
    return (False, False, -1, 0.0)


@njit(cache=True)
def get_macd_histogram_trend_numba(
    macd_hist: np.ndarray,
    lookback: int = 3
) -> int:
    """
    Get MACD histogram trend direction.
    
    Args:
        macd_hist: MACD histogram values (most recent last)
        lookback: Periods to analyze for trend
        
    Returns:
        1 = increasing (bullish), -1 = decreasing (bearish), 0 = neutral/mixed
    """
    if len(macd_hist) < lookback + 1:
        return 0
    
    recent_hist = macd_hist[-lookback-1:]
    
    # Count increasing vs decreasing periods
    increasing = 0
    decreasing = 0
    
    for i in range(1, len(recent_hist)):
        if recent_hist[i] > recent_hist[i-1]:
            increasing += 1
        elif recent_hist[i] < recent_hist[i-1]:
            decreasing += 1
    
    if increasing > decreasing:
        return 1
    elif decreasing > increasing:
        return -1
    else:
        return 0
