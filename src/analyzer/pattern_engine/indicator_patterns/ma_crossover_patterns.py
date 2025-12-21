"""
Moving Average Crossover Patterns - Pure NumPy/Numba implementation

Detects Golden Cross, Death Cross, and short-term MA crossovers.
Classic long-term trend reversal signals.
"""

import numpy as np
from numba import njit


@njit
def _detect_ma_crossover_numba(sma_50: np.ndarray, sma_200: np.ndarray, is_bullish: bool) -> tuple:
    """
    Generic MA crossover detection (helper function).
    
    Scans ENTIRE array for the most recent crossover event.
    
    Args:
        sma_50: 50-period SMA array
        sma_200: 200-period SMA array
        is_bullish: True for golden cross (50 crosses above 200), False for death cross (50 crosses below 200)
        
    Returns:
        (found: bool, periods_ago: int, sma_50_value: float, sma_200_value: float)
    """
    if len(sma_50) < 2 or len(sma_200) < 2:
        return False, 0, 0.0, 0.0
    
    # Scan ENTIRE array for most recent crossover (starting from most recent)
    for i in range(1, len(sma_50)):
        idx = len(sma_50) - i - 1
        
        # Skip if any values are NaN
        if np.isnan(sma_50[idx]) or np.isnan(sma_50[idx + 1]):
            continue
        if np.isnan(sma_200[idx]) or np.isnan(sma_200[idx + 1]):
            continue
        
        # Detect crossover based on direction
        if is_bullish:
            # Golden Cross: was below, now above
            was_below = sma_50[idx] <= sma_200[idx]
            now_above = sma_50[idx + 1] > sma_200[idx + 1]
            crossover = was_below and now_above
        else:
            # Death Cross: was above, now below
            was_above = sma_50[idx] >= sma_200[idx]
            now_below = sma_50[idx + 1] < sma_200[idx + 1]
            crossover = was_above and now_below
        
        if crossover:
            # Use values at crossover point, not current values
            return True, i, float(sma_50[idx + 1]), float(sma_200[idx + 1])
    
    return False, 0, 0.0, 0.0


@njit
def detect_golden_cross_numba(sma_50: np.ndarray, sma_200: np.ndarray) -> tuple:
    """
    Detect Golden Cross: 50 SMA crosses above 200 SMA.
    
    Strong bullish signal indicating potential long-term uptrend.
    
    Args:
        sma_50: 50-period SMA array
        sma_200: 200-period SMA array
        
    Returns:
        (found: bool, periods_ago: int, sma_50_value: float, sma_200_value: float)
    """
    return _detect_ma_crossover_numba(sma_50, sma_200, True)


@njit
def detect_death_cross_numba(sma_50: np.ndarray, sma_200: np.ndarray) -> tuple:
    """
    Detect Death Cross: 50 SMA crosses below 200 SMA.
    
    Strong bearish signal indicating potential long-term downtrend.
    
    Args:
        sma_50: 50-period SMA array
        sma_200: 200-period SMA array
        
    Returns:
        (found: bool, periods_ago: int, sma_50_value: float, sma_200_value: float)
    """
    return _detect_ma_crossover_numba(sma_50, sma_200, False)


@njit
def detect_short_term_crossover_numba(sma_20: np.ndarray, sma_50: np.ndarray) -> tuple:
    """
    Detect short-term MA crossover: 20 SMA crosses 50 SMA.
    
    Scans ENTIRE array for the most recent crossover event.
    Medium-term trend signal for swing trading.
    
    Args:
        sma_20: 20-period SMA array
        sma_50: 50-period SMA array
        
    Returns:
        (found: bool, is_bullish: bool, periods_ago: int, sma_20_value: float, sma_50_value: float)
    """
    if len(sma_20) < 2 or len(sma_50) < 2:
        return False, False, 0, 0.0, 0.0
    
    # Scan ENTIRE array for most recent crossover (starting from most recent)
    for i in range(1, len(sma_20)):
        idx = len(sma_20) - i - 1
        
        # Skip if any values are NaN
        if np.isnan(sma_20[idx]) or np.isnan(sma_20[idx + 1]):
            continue
        if np.isnan(sma_50[idx]) or np.isnan(sma_50[idx + 1]):
            continue
        
        # Bullish crossover: 20 crosses above 50
        was_below = sma_20[idx] <= sma_50[idx]
        now_above = sma_20[idx + 1] > sma_50[idx + 1]
        
        if was_below and now_above:
            # Use values at crossover point, not current values
            return True, True, i, float(sma_20[idx + 1]), float(sma_50[idx + 1])
        
        # Bearish crossover: 20 crosses below 50
        was_above = sma_20[idx] >= sma_50[idx]
        now_below = sma_20[idx + 1] < sma_50[idx + 1]
        
        if was_above and now_below:
            # Use values at crossover point, not current values
            return True, False, i, float(sma_20[idx + 1]), float(sma_50[idx + 1])
    
    return False, False, 0, 0.0, 0.0


@njit
def check_ma_alignment_numba(sma_20: np.ndarray, sma_50: np.ndarray, sma_200: np.ndarray) -> tuple:
    """
    Check alignment of moving averages for trend confirmation.
    
    Bullish alignment: 20 > 50 > 200
    Bearish alignment: 20 < 50 < 200
    
    Args:
        sma_20: 20-period SMA array
        sma_50: 50-period SMA array
        sma_200: 200-period SMA array
        
    Returns:
        (is_bullish_aligned: bool, is_bearish_aligned: bool, sma_20_val: float, sma_50_val: float, sma_200_val: float)
    """
    if len(sma_20) == 0 or len(sma_50) == 0 or len(sma_200) == 0:
        return False, False, 0.0, 0.0, 0.0
    
    v20 = sma_20[-1]
    v50 = sma_50[-1]
    v200 = sma_200[-1]
    
    if np.isnan(v20) or np.isnan(v50) or np.isnan(v200):
        return False, False, 0.0, 0.0, 0.0
    
    # Bullish: 20 > 50 > 200
    is_bullish = v20 > v50 and v50 > v200
    
    # Bearish: 20 < 50 < 200
    is_bearish = v20 < v50 and v50 < v200
    
    return is_bullish, is_bearish, float(v20), float(v50), float(v200)
