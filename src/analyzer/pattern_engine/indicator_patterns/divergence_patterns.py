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
from typing import Tuple


@njit(cache=True)
def _find_local_extrema_numba(
    data: np.ndarray,
    lookback: int,
    find_maxima: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find local maxima or minima in data.
    
    Args:
        data: Array to search
        lookback: Window size for local extrema
        find_maxima: True for maxima, False for minima
        
    Returns:
        (indices, values) of local extrema
    """
    n = len(data)
    if n < lookback * 2 + 1:
        # Return empty arrays with proper types
        empty_indices = np.empty(0, dtype=np.int64)
        empty_values = np.empty(0, dtype=np.float64)
        return (empty_indices, empty_values)
    
    # Pre-allocate max possible size
    max_extrema = n - 2 * lookback
    indices_temp = np.empty(max_extrema, dtype=np.int64)
    values_temp = np.empty(max_extrema, dtype=np.float64)
    count = 0
    
    for i in range(lookback, n - lookback):
        is_extrema = True
        
        # Check if current point is extrema in window
        for j in range(i - lookback, i + lookback + 1):
            if j == i:
                continue
            
            if find_maxima:
                if data[i] <= data[j]:
                    is_extrema = False
                    break
            else:  # Find minima
                if data[i] >= data[j]:
                    is_extrema = False
                    break
        
        if is_extrema:
            indices_temp[count] = i
            values_temp[count] = data[i]
            count += 1
    
    # Return only filled portion
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
) -> Tuple[int, float]:
    """
    Find indicator extrema near a price extrema.
    
    Args:
        indicator_indices: Array of indicator extrema indices
        indicator_values: Array of indicator extrema values
        price_idx: Index of price extrema to match
        tolerance: Maximum periods difference to consider a match
        
    Returns:
        (indicator_idx, indicator_value) or (-1, 0.0) if not found
    """
    for j in range(len(indicator_indices)):
        if abs(indicator_indices[j] - price_idx) <= tolerance:
            return (indicator_indices[j], indicator_values[j])
    return (-1, 0.0)


@njit(cache=True)
def detect_bullish_divergence_numba(
    prices: np.ndarray,
    indicator: np.ndarray,
    lookback: int = 20,
    min_spacing: int = 5
) -> Tuple[bool, int, int, float, float, float, float]:
    """
    Detect bullish divergence between price and indicator.
    
    Bullish divergence = Price making lower low, indicator making higher low
    This suggests weakening bearish momentum and potential reversal up.
    
    Scans ENTIRE array for the most recent divergence pattern.
    Note: lookback parameter kept for backward compatibility but now scans full array.
    
    Args:
        prices: Price values (most recent last)
        indicator: Indicator values (RSI, MACD, Stoch, etc.)
        lookback: (Ignored - kept for backward compatibility, now scans entire array)
        min_spacing: Minimum periods between the two lows
        
    Returns:
        (divergence_found, first_idx, second_idx, 
         first_price, second_price, first_indicator, second_indicator)
    """
    if len(prices) < 10 or len(indicator) < 10:
        return (False, -1, -1, 0.0, 0.0, 0.0, 0.0)
    
    # FIXED: Scan ENTIRE array instead of limiting to last 'lookback' periods
    # This ensures divergences anywhere in the dataset are detected
    # Find local minima in both price and indicator
    price_low_indices, price_low_values = _find_local_extrema_numba(
        prices, lookback=2, find_maxima=False
    )
    indicator_low_indices, indicator_low_values = _find_local_extrema_numba(
        indicator, lookback=2, find_maxima=False
    )
    
    if len(price_low_indices) < 2 or len(indicator_low_indices) < 2:
        return (False, -1, -1, 0.0, 0.0, 0.0, 0.0)
    
    # Check most recent pair of lows
    # Price: second low should be lower than first low
    # Indicator: second low should be higher than first low
    
    for i in range(len(price_low_indices) - 1, 0, -1):
        second_price_idx = price_low_indices[i]
        first_price_idx = price_low_indices[i - 1]
        
        # Check spacing
        if second_price_idx - first_price_idx < min_spacing:
            continue
        
        second_price = price_low_values[i]
        first_price = price_low_values[i - 1]
        
        # Price must make lower low
        if second_price >= first_price:
            continue
        
        # Find corresponding indicator lows around same times
        first_indicator_idx, first_indicator_value = _find_matching_indicator_extrema(
            indicator_low_indices, indicator_low_values, first_price_idx
        )
        if first_indicator_idx == -1:
            continue
        
        second_indicator_idx, second_indicator_value = _find_matching_indicator_extrema(
            indicator_low_indices, indicator_low_values, second_price_idx
        )
        if second_indicator_idx == -1:
            continue
        
        # Indicator must make higher low (divergence!)
        if second_indicator_value > first_indicator_value:
            # Bullish divergence detected!
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
def detect_bearish_divergence_numba(
    prices: np.ndarray,
    indicator: np.ndarray,
    lookback: int = 20,
    min_spacing: int = 5
) -> Tuple[bool, int, int, float, float, float, float]:
    """
    Detect bearish divergence between price and indicator.
    
    Bearish divergence = Price making higher high, indicator making lower high
    This suggests weakening bullish momentum and potential reversal down.
    
    Scans ENTIRE array for the most recent divergence pattern.
    Note: lookback parameter kept for backward compatibility but now scans full array.
    
    Args:
        prices: Price values (most recent last)
        indicator: Indicator values (RSI, MACD, Stoch, etc.)
        lookback: (Ignored - kept for backward compatibility, now scans entire array)
        min_spacing: Minimum periods between the two highs
        
    Returns:
        (divergence_found, first_idx, second_idx, 
         first_price, second_price, first_indicator, second_indicator)
    """
    if len(prices) < 10 or len(indicator) < 10:
        return (False, -1, -1, 0.0, 0.0, 0.0, 0.0)
    
    # FIXED: Scan ENTIRE array instead of limiting to last 'lookback' periods
    # This ensures divergences anywhere in the dataset are detected
    # Find local maxima in both price and indicator
    price_high_indices, price_high_values = _find_local_extrema_numba(
        prices, lookback=2, find_maxima=True
    )
    indicator_high_indices, indicator_high_values = _find_local_extrema_numba(
        indicator, lookback=2, find_maxima=True
    )
    
    if len(price_high_indices) < 2 or len(indicator_high_indices) < 2:
        return (False, -1, -1, 0.0, 0.0, 0.0, 0.0)
    
    # Check most recent pair of highs
    # Price: second high should be higher than first high
    # Indicator: second high should be lower than first high
    
    for i in range(len(price_high_indices) - 1, 0, -1):
        second_price_idx = price_high_indices[i]
        first_price_idx = price_high_indices[i - 1]
        
        # Check spacing
        if second_price_idx - first_price_idx < min_spacing:
            continue
        
        second_price = price_high_values[i]
        first_price = price_high_values[i - 1]
        
        # Price must make higher high
        if second_price <= first_price:
            continue
        
        # Find corresponding indicator highs around same times
        first_indicator_idx, first_indicator_value = _find_matching_indicator_extrema(
            indicator_high_indices, indicator_high_values, first_price_idx
        )
        if first_indicator_idx == -1:
            continue
        
        second_indicator_idx, second_indicator_value = _find_matching_indicator_extrema(
            indicator_high_indices, indicator_high_values, second_price_idx
        )
        if second_indicator_idx == -1:
            continue
        
        # Indicator must make lower high (divergence!)
        if second_indicator_value < first_indicator_value:
            # Bearish divergence detected!
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
