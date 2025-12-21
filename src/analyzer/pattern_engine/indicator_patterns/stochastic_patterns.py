"""
Stochastic Oscillator Patterns - Pure NumPy/Numba implementation

Detects oversold/overbought conditions and %K/%D crossovers.
Excellent momentum indicator for identifying reversal points.
"""

import numpy as np
from numba import njit


@njit
def detect_stoch_oversold_numba(stoch_k: np.ndarray, threshold: float = 20.0) -> tuple:
    """
    Detect oversold Stochastic condition.
    
    Stochastic below 20 indicates oversold - potential bullish reversal.
    
    Args:
        stoch_k: Stochastic %K array
        threshold: Oversold threshold (default 20)
        
    Returns:
        (is_oversold: bool, periods_ago: int, stoch_value: float)
    """
    if len(stoch_k) == 0:
        return False, 0, 0.0
    
    current_value = stoch_k[-1]
    
    if np.isnan(current_value):
        return False, 0, 0.0
    
    if current_value < threshold:
        # Find how many periods ago we entered oversold
        periods_ago = 0
        for i in range(len(stoch_k) - 1, -1, -1):
            if np.isnan(stoch_k[i]) or stoch_k[i] >= threshold:
                break
            periods_ago = len(stoch_k) - i - 1
        
        return True, periods_ago, float(current_value)
    
    return False, 0, 0.0


@njit
def detect_stoch_overbought_numba(stoch_k: np.ndarray, threshold: float = 80.0) -> tuple:
    """
    Detect overbought Stochastic condition.
    
    Stochastic above 80 indicates overbought - potential bearish reversal.
    
    Args:
        stoch_k: Stochastic %K array
        threshold: Overbought threshold (default 80)
        
    Returns:
        (is_overbought: bool, periods_ago: int, stoch_value: float)
    """
    if len(stoch_k) == 0:
        return False, 0, 0.0
    
    current_value = stoch_k[-1]
    
    if np.isnan(current_value):
        return False, 0, 0.0
    
    if current_value > threshold:
        # Find how many periods ago we entered overbought
        periods_ago = 0
        for i in range(len(stoch_k) - 1, -1, -1):
            if np.isnan(stoch_k[i]) or stoch_k[i] <= threshold:
                break
            periods_ago = len(stoch_k) - i - 1
        
        return True, periods_ago, float(current_value)
    
    return False, 0, 0.0


@njit
def _detect_stochastic_crossover_numba(stoch_k: np.ndarray, stoch_d: np.ndarray, 
                                       is_bullish: bool, threshold: float) -> tuple:
    """
    Generic stochastic crossover detection (helper function).
    
    Scans ENTIRE array for the most recent crossover event.
    
    Args:
        stoch_k: Stochastic %K array (fast line)
        stoch_d: Stochastic %D array (slow line)
        is_bullish: True for bullish crossover (K crosses above D), False for bearish (K crosses below D)
        threshold: Oversold/overbought threshold value
        
    Returns:
        (found: bool, periods_ago: int, k_value: float, d_value: float, is_in_zone: bool)
    """
    if len(stoch_k) < 2 or len(stoch_d) < 2:
        return False, 0, 0.0, 0.0, False
    
    # Scan ENTIRE array for most recent crossover (starting from most recent)
    for i in range(1, len(stoch_k)):
        idx = len(stoch_k) - i - 1
        
        # Skip if any values are NaN
        if np.isnan(stoch_k[idx]) or np.isnan(stoch_k[idx + 1]):
            continue
        if np.isnan(stoch_d[idx]) or np.isnan(stoch_d[idx + 1]):
            continue
        
        # Detect crossover based on direction
        if is_bullish:
            # Bullish: %K crosses above %D
            was_below = stoch_k[idx] <= stoch_d[idx]
            now_above = stoch_k[idx + 1] > stoch_d[idx + 1]
            crossover = was_below and now_above
        else:
            # Bearish: %K crosses below %D
            was_above = stoch_k[idx] >= stoch_d[idx]
            now_below = stoch_k[idx + 1] < stoch_d[idx + 1]
            crossover = was_above and now_below
        
        if crossover:
            # Use values at crossover point, not current values
            k_val = float(stoch_k[idx + 1])
            d_val = float(stoch_d[idx + 1])
            
            # Check if in oversold (bullish) or overbought (bearish) zone
            if is_bullish:
                in_zone = k_val < threshold or d_val < threshold
            else:
                in_zone = k_val > threshold or d_val > threshold
            
            return True, i, k_val, d_val, in_zone
    
    return False, 0, 0.0, 0.0, False


@njit
def detect_stoch_bullish_crossover_numba(stoch_k: np.ndarray, stoch_d: np.ndarray, oversold_threshold: float = 30.0) -> tuple:
    """
    Detect bullish Stochastic crossover: %K crosses above %D while in oversold territory.
    
    Strong bullish signal when occurring below 30.
    
    Args:
        stoch_k: Stochastic %K array (fast line)
        stoch_d: Stochastic %D array (slow line)
        oversold_threshold: Consider oversold below this (default 30)
        
    Returns:
        (found: bool, periods_ago: int, k_value: float, d_value: float, is_in_oversold: bool)
    """
    return _detect_stochastic_crossover_numba(stoch_k, stoch_d, True, oversold_threshold)


@njit
def detect_stoch_bearish_crossover_numba(stoch_k: np.ndarray, stoch_d: np.ndarray, overbought_threshold: float = 70.0) -> tuple:
    """
    Detect bearish Stochastic crossover: %K crosses below %D while in overbought territory.
    
    Strong bearish signal when occurring above 70.
    
    Args:
        stoch_k: Stochastic %K array (fast line)
        stoch_d: Stochastic %D array (slow line)
        overbought_threshold: Consider overbought above this (default 70)
        
    Returns:
        (found: bool, periods_ago: int, k_value: float, d_value: float, is_in_overbought: bool)
    """
    return _detect_stochastic_crossover_numba(stoch_k, stoch_d, False, overbought_threshold)


@njit
def detect_stoch_divergence_numba(stoch_k: np.ndarray, prices: np.ndarray, lookback: int = 20) -> tuple:
    """
    Detect divergence between Stochastic and price.
    
    Bullish divergence: Price makes lower low, Stochastic makes higher low
    Bearish divergence: Price makes higher high, Stochastic makes lower high
    
    Args:
        stoch_k: Stochastic %K array
        prices: Price array (typically close prices)
        lookback: Periods to look back for divergence
        
    Returns:
        (found: bool, is_bullish: bool, price_change: float, stoch_change: float)
    """
    if len(stoch_k) < lookback or len(prices) < lookback:
        return False, False, 0.0, 0.0
    
    # Get recent segment
    recent_stoch = stoch_k[-lookback:]
    recent_prices = prices[-lookback:]
    
    # Skip if any NaN values
    for i in range(len(recent_stoch)):
        if np.isnan(recent_stoch[i]) or np.isnan(recent_prices[i]):
            return False, False, 0.0, 0.0
    
    # Find local extrema
    stoch_min_idx = np.argmin(recent_stoch)
    stoch_max_idx = np.argmax(recent_stoch)
    price_min_idx = np.argmin(recent_prices)
    price_max_idx = np.argmax(recent_prices)
    
    # Bullish divergence: price lower low, stoch higher low
    if price_min_idx > 0 and stoch_min_idx > 0:
        # Find previous low in first half
        mid_point = len(recent_prices) // 2
        if price_min_idx > mid_point:
            prev_price_low = np.min(recent_prices[:mid_point])
            prev_stoch_low = np.min(recent_stoch[:mid_point])
            
            current_price_low = recent_prices[price_min_idx]
            current_stoch_low = recent_stoch[stoch_min_idx]
            
            if current_price_low < prev_price_low and current_stoch_low > prev_stoch_low:
                price_change = float(current_price_low - prev_price_low)
                stoch_change = float(current_stoch_low - prev_stoch_low)
                return True, True, price_change, stoch_change
    
    # Bearish divergence: price higher high, stoch lower high
    if price_max_idx > 0 and stoch_max_idx > 0:
        mid_point = len(recent_prices) // 2
        if price_max_idx > mid_point:
            prev_price_high = np.max(recent_prices[:mid_point])
            prev_stoch_high = np.max(recent_stoch[:mid_point])
            
            current_price_high = recent_prices[price_max_idx]
            current_stoch_high = recent_stoch[stoch_max_idx]
            
            if current_price_high > prev_price_high and current_stoch_high < prev_stoch_high:
                price_change = float(current_price_high - prev_price_high)
                stoch_change = float(current_stoch_high - prev_stoch_high)
                return True, False, price_change, stoch_change
    
    return False, False, 0.0, 0.0
