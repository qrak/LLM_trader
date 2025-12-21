"""
Volatility Pattern Detection - Pure NumPy/Numba Implementation

Detects volatility-based patterns:
1. ATR spikes (sudden volatility increase - risk warning)
2. Bollinger Band squeezes (low volatility - breakout imminent)

All functions use @njit for performance.
"""

import numpy as np
from numba import njit
from typing import Tuple


@njit(cache=True)
def detect_atr_spike_numba(
    atr: np.ndarray,
    spike_threshold: float = 1.5,
    lookback: int = 14
) -> Tuple[bool, int, float, float]:
    """
    Detect sudden ATR spikes (volatility explosion).
    
    ATR spike = Current ATR is significantly higher than recent average.
    Indicates sudden volatility increase, often at trend reversals or breakouts.
    
    Args:
        atr: ATR values (most recent last)
        spike_threshold: Multiplier for spike detection (default 1.5x)
        lookback: Periods for average calculation
        
    Returns:
        (spike_detected, periods_ago, current_atr, average_atr)
    """
    if len(atr) < lookback + 1:
        return (False, -1, 0.0, 0.0)
    
    current_atr = atr[-1]
    
    # Calculate average ATR over lookback period (excluding current)
    recent_atr = atr[-(lookback+1):-1]
    avg_atr = np.mean(recent_atr)
    
    # Avoid division by zero
    if avg_atr < 0.0001:
        return (False, -1, current_atr, avg_atr)
    
    # Check if current ATR is significantly higher
    spike_ratio = current_atr / avg_atr
    
    if spike_ratio >= spike_threshold:
        return (True, 0, current_atr, avg_atr)
    
    return (False, -1, current_atr, avg_atr)


@njit(cache=True)
def detect_bb_squeeze_numba(
    bb_upper: np.ndarray,
    bb_lower: np.ndarray,
    squeeze_percentile: float = 20.0,
    lookback: int = 20
) -> Tuple[bool, float, float]:
    """
    Detect Bollinger Band squeeze (low volatility).
    
    BB squeeze = Band width is in lowest X percentile over lookback period.
    Indicates consolidation and potential for big move (breakout or breakdown).
    
    Args:
        bb_upper: Bollinger Band upper values
        bb_lower: Bollinger Band lower values
        squeeze_percentile: Percentile threshold (default 20 = lowest 20%)
        lookback: Periods to compare against
        
    Returns:
        (squeeze_detected, current_width, percentile_width)
        - squeeze_detected: True if squeeze detected
        - current_width: Current band width
        - percentile_width: Percentile threshold width
    """
    if len(bb_upper) < lookback or len(bb_lower) < lookback:
        return (False, 0.0, 0.0)
    
    # Calculate band widths over lookback period
    recent_upper = bb_upper[-lookback:]
    recent_lower = bb_lower[-lookback:]
    
    widths = recent_upper - recent_lower
    current_width = widths[-1]
    
    # Calculate percentile threshold
    # Sort widths to find percentile value
    sorted_widths = np.sort(widths)
    percentile_index = int(len(sorted_widths) * squeeze_percentile / 100.0)
    percentile_index = max(0, min(percentile_index, len(sorted_widths) - 1))
    percentile_width = sorted_widths[percentile_index]
    
    # Check if current width is in lowest percentile
    if current_width <= percentile_width:
        return (True, current_width, percentile_width)
    
    return (False, current_width, percentile_width)


@njit(cache=True)
def detect_volatility_trend_numba(
    atr: np.ndarray,
    lookback: int = 10
) -> int:
    """
    Detect trend in volatility (increasing, decreasing, or stable).
    
    Args:
        atr: ATR values (most recent last)
        lookback: Periods to analyze for trend
        
    Returns:
        1 = increasing volatility, -1 = decreasing volatility, 0 = stable
    """
    if len(atr) < lookback:
        return 0
    
    recent_atr = atr[-lookback:]
    
    # Count increasing vs decreasing periods
    increasing = 0
    decreasing = 0
    
    for i in range(1, len(recent_atr)):
        diff = recent_atr[i] - recent_atr[i-1]
        if diff > 0.0:  # Any positive change
            increasing += 1
        elif diff < 0.0:  # Any negative change
            decreasing += 1
    
    # Determine trend (need at least 60% consistency)
    # Number of transitions is len(recent_atr) - 1
    num_transitions = len(recent_atr) - 1
    if num_transitions == 0:
        return 0
    
    threshold = num_transitions * 0.6
    
    if float(increasing) >= threshold:
        return 1
    elif float(decreasing) >= threshold:
        return -1
    else:
        return 0


@njit(cache=True)
def detect_keltner_squeeze_numba(
    kc_upper: np.ndarray,
    kc_lower: np.ndarray,
    bb_upper: np.ndarray,
    bb_lower: np.ndarray
) -> bool:
    """
    Detect TTM Squeeze (Bollinger Bands inside Keltner Channels).
    
    This is John Carter's TTM Squeeze indicator:
    - When BB are inside KC, volatility is extremely low (squeeze)
    - Breakout is imminent when squeeze releases
    
    Args:
        kc_upper: Keltner Channel upper
        kc_lower: Keltner Channel lower
        bb_upper: Bollinger Band upper
        bb_lower: Bollinger Band lower
        
    Returns:
        True if squeeze detected (BB inside KC)
    """
    if len(kc_upper) < 1 or len(bb_upper) < 1:
        return False
    
    # Check current values
    kc_u = kc_upper[-1]
    kc_l = kc_lower[-1]
    bb_u = bb_upper[-1]
    bb_l = bb_lower[-1]
    
    # Squeeze = BB inside KC
    if bb_u <= kc_u and bb_l >= kc_l:
        return True
    
    return False
