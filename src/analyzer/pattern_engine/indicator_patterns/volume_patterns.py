"""
Volume Pattern Detection - Pure NumPy/Numba implementation

Detects volume spikes, dry-ups, and abnormal volume conditions.
Critical for confirming breakouts and identifying accumulation/distribution.
"""

import numpy as np
from numba import njit


@njit
def _calculate_average_volume_numba(volume: np.ndarray, lookback: int) -> tuple:
    """
    Calculate average volume with NaN handling (helper function).
    
    Args:
        volume: Volume array
        lookback: Number of periods to look back
        
    Returns:
        (is_valid: bool, avg_volume: float, valid_count: int)
    """
    if len(volume) < lookback + 1:
        return False, 0.0, 0
    
    current_vol = volume[-1]
    
    if np.isnan(current_vol) or current_vol < 0:
        return False, 0.0, 0
    
    # Calculate average volume (excluding current)
    recent_volume = volume[-(lookback + 1):-1]
    
    # Skip if any NaN values
    valid_count = 0
    vol_sum = 0.0
    for i in range(len(recent_volume)):
        if not np.isnan(recent_volume[i]) and recent_volume[i] > 0:
            vol_sum += recent_volume[i]
            valid_count += 1
    
    if valid_count < lookback // 2:  # Need at least half the data
        return False, 0.0, valid_count
    
    avg_vol = vol_sum / valid_count
    
    if avg_vol <= 0:
        return False, 0.0, valid_count
    
    return True, avg_vol, valid_count


@njit
def detect_volume_spike_numba(volume: np.ndarray, multiplier: float = 2.5, lookback: int = 20) -> tuple:
    """
    Detect volume spike: current volume significantly above average.
    
    Strong confirmation signal for breakouts when volume > multiplier * avg_volume.
    
    Args:
        volume: Volume array
        multiplier: Spike threshold multiplier (default 2.0 = 200% of average)
        lookback: Periods for average calculation (default 20)
        
    Returns:
        (is_spike: bool, current_volume: float, avg_volume: float, spike_ratio: float)
    """
    is_valid, avg_vol, _ = _calculate_average_volume_numba(volume, lookback)
    
    if not is_valid:
        return False, 0.0, 0.0, 0.0
    
    current_vol = volume[-1]
    spike_ratio = current_vol / avg_vol
    
    if spike_ratio >= multiplier:
        return True, float(current_vol), float(avg_vol), float(spike_ratio)
    
    return False, float(current_vol), float(avg_vol), float(spike_ratio)


@njit
def detect_volume_dryup_numba(volume: np.ndarray, threshold: float = 0.5, lookback: int = 20) -> tuple:
    """
    Detect volume dry-up: current volume significantly below average.
    
    Often precedes major moves - low volume indicates consolidation.
    
    Args:
        volume: Volume array
        threshold: Dry-up threshold (default 0.5 = 50% of average)
        lookback: Periods for average calculation (default 20)
        
    Returns:
        (is_dryup: bool, current_volume: float, avg_volume: float, dryup_ratio: float)
    """
    is_valid, avg_vol, _ = _calculate_average_volume_numba(volume, lookback)
    
    if not is_valid:
        return False, 0.0, 0.0, 0.0
    
    current_vol = volume[-1]
    dryup_ratio = current_vol / avg_vol
    
    if dryup_ratio <= threshold:
        return True, float(current_vol), float(avg_vol), float(dryup_ratio)
    
    return False, float(current_vol), float(avg_vol), float(dryup_ratio)


@njit
def detect_volume_price_divergence_numba(volume: np.ndarray, prices: np.ndarray, lookback: int = 20) -> tuple:
    """
    Detect divergence between volume and price movement.
    
    Bearish: Price rising but volume declining (weak rally)
    Bullish: Price falling but volume declining (weak selloff)
    
    Args:
        volume: Volume array
        prices: Price array (typically close prices)
        lookback: Periods to analyze (default 20)
        
    Returns:
        (found: bool, is_bearish: bool, price_change_pct: float, volume_change_pct: float)
    """
    if len(volume) < lookback or len(prices) < lookback:
        return False, False, 0.0, 0.0
    
    # Get recent segments
    recent_volume = volume[-lookback:]
    recent_prices = prices[-lookback:]
    
    # Skip if any NaN values
    for i in range(len(recent_volume)):
        if np.isnan(recent_volume[i]) or np.isnan(recent_prices[i]):
            return False, False, 0.0, 0.0
    
    # Calculate price trend
    first_half_prices = recent_prices[:lookback // 2]
    second_half_prices = recent_prices[lookback // 2:]
    
    avg_first_price = np.mean(first_half_prices)
    avg_second_price = np.mean(second_half_prices)
    
    price_change_pct = ((avg_second_price / avg_first_price) - 1.0) * 100.0 if avg_first_price > 0 else 0.0
    
    # Calculate volume trend
    first_half_volume = recent_volume[:lookback // 2]
    second_half_volume = recent_volume[lookback // 2:]
    
    avg_first_volume = np.mean(first_half_volume)
    avg_second_volume = np.mean(second_half_volume)
    
    volume_change_pct = ((avg_second_volume / avg_first_volume) - 1.0) * 100.0 if avg_first_volume > 0 else 0.0
    
    # Bearish divergence: price up, volume down
    if price_change_pct > 2.0 and volume_change_pct < -15.0:
        return True, True, float(price_change_pct), float(volume_change_pct)
    
    # Bullish divergence: price down, volume down (weak selling)
    if price_change_pct < -2.0 and volume_change_pct < -15.0:
        return True, False, float(price_change_pct), float(volume_change_pct)
    
    return False, False, 0.0, 0.0


@njit
def detect_accumulation_distribution_numba(volume: np.ndarray, prices: np.ndarray, lookback: int = 10) -> tuple:
    """
    Detect accumulation (buying pressure) or distribution (selling pressure).
    
    Accumulation: Volume increases on up days, decreases on down days
    Distribution: Volume increases on down days, decreases on up days
    
    Args:
        volume: Volume array
        prices: Price array (typically close prices)
        lookback: Recent periods to analyze (default 10)
        
    Returns:
        (found: bool, is_accumulation: bool, strength: float, up_volume_ratio: float)
    """
    if len(volume) < lookback + 1 or len(prices) < lookback + 1:
        return False, False, 0.0, 0.0
    
    recent_volume = volume[-(lookback + 1):]
    recent_prices = prices[-(lookback + 1):]
    
    # Skip if any NaN values
    for i in range(len(recent_volume)):
        if np.isnan(recent_volume[i]) or np.isnan(recent_prices[i]):
            return False, False, 0.0, 0.0
    
    up_volume = 0.0
    down_volume = 0.0
    
    # Calculate volume on up days vs down days
    for i in range(1, len(recent_prices)):
        price_change = recent_prices[i] - recent_prices[i - 1]
        vol = recent_volume[i]
        
        if price_change > 0:
            up_volume += vol
        elif price_change < 0:
            down_volume += vol
    
    total_volume = up_volume + down_volume
    
    if total_volume <= 0:
        return False, False, 0.0, 0.0
    
    up_volume_ratio = up_volume / total_volume
    
    # Accumulation: >60% volume on up days
    if up_volume_ratio > 0.60:
        strength = float(up_volume_ratio - 0.5) * 2.0  # Scale 0.6-1.0 to 0.2-1.0
        return True, True, strength, float(up_volume_ratio)
    
    # Distribution: >60% volume on down days
    if up_volume_ratio < 0.40:
        strength = float(0.5 - up_volume_ratio) * 2.0  # Scale 0-0.4 to 1.0-0.2
        return True, False, strength, float(up_volume_ratio)
    
    return False, False, 0.0, float(up_volume_ratio)


@njit
def detect_climax_volume_numba(volume: np.ndarray, multiplier: float = 3.0, lookback: int = 50) -> tuple:
    """
    Detect climax volume: extreme volume spike indicating potential exhaustion.
    
    Very high volume often marks trend reversals (buying/selling climax).
    
    Args:
        volume: Volume array
        multiplier: Climax threshold (default 3.0 = 300% of average)
        lookback: Periods for average calculation (default 50)
        
    Returns:
        (is_climax: bool, current_volume: float, avg_volume: float, climax_ratio: float)
    """
    is_valid, avg_vol, _ = _calculate_average_volume_numba(volume, lookback)
    
    if not is_valid:
        return False, 0.0, 0.0, 0.0
    
    current_vol = volume[-1]
    climax_ratio = current_vol / avg_vol
    
    if climax_ratio >= multiplier:
        return True, float(current_vol), float(avg_vol), float(climax_ratio)
    
    return False, float(current_vol), float(avg_vol), float(climax_ratio)
