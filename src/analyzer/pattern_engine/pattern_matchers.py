import numpy as np
from numba import njit


@njit(cache=True)
def _extract_swing_indices(swing_array: np.ndarray) -> np.ndarray:
    """Extract indices where swing points occur."""
    return np.where(swing_array)[0]


@njit(cache=True)
def _is_monotonic_sequence(values: np.ndarray, ascending: bool) -> bool:
    """Check if values form a monotonic sequence (strictly ascending or descending)."""
    for j in range(len(values) - 1):
        if ascending:
            if values[j + 1] <= values[j]:
                return False
        else:  # descending
            if values[j + 1] >= values[j]:
                return False
    return True


@njit(cache=True)
def detect_head_shoulder_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                               swing_highs: np.ndarray, swing_lows: np.ndarray,
                               tolerance: float = 0.035) -> np.ndarray:
    n = len(high)
    patterns = np.zeros(n, dtype=np.int32)
    
    swing_high_indices = _extract_swing_indices(swing_highs)
    
    if len(swing_high_indices) < 3:
        return patterns
    
    for i in range(len(swing_high_indices) - 2):
        left_idx = swing_high_indices[i]
        head_idx = swing_high_indices[i + 1]
        right_idx = swing_high_indices[i + 2]
        
        left_shoulder = high[left_idx]
        head = high[head_idx]
        right_shoulder = high[right_idx]
        
        if head > left_shoulder and head > right_shoulder:
            shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
            if shoulder_diff < tolerance:
                patterns[head_idx] = 1
    
    swing_low_indices = _extract_swing_indices(swing_lows)
    
    if len(swing_low_indices) < 3:
        return patterns
    
    for i in range(len(swing_low_indices) - 2):
        left_idx = swing_low_indices[i]
        head_idx = swing_low_indices[i + 1]
        right_idx = swing_low_indices[i + 2]
        
        left_shoulder = low[left_idx]
        head = low[head_idx]
        right_shoulder = low[right_idx]
        
        if head < left_shoulder and head < right_shoulder:
            shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
            if shoulder_diff < tolerance:
                patterns[head_idx] = 2
    
    return patterns


@njit(cache=True)
def detect_double_top_bottom_numba(high: np.ndarray, low: np.ndarray,
                                   swing_highs: np.ndarray, swing_lows: np.ndarray,
                                   tolerance: float = 0.035) -> np.ndarray:
    n = len(high)
    patterns = np.zeros(n, dtype=np.int32)
    
    swing_high_indices = _extract_swing_indices(swing_highs)
    
    if len(swing_high_indices) >= 2:
        for i in range(len(swing_high_indices) - 1):
            idx1 = swing_high_indices[i]
            idx2 = swing_high_indices[i + 1]
            
            price1 = high[idx1]
            price2 = high[idx2]
            
            price_diff = abs(price1 - price2) / price1
            if price_diff < tolerance:
                patterns[idx2] = 1
    
    swing_low_indices = _extract_swing_indices(swing_lows)
    
    if len(swing_low_indices) >= 2:
        for i in range(len(swing_low_indices) - 1):
            idx1 = swing_low_indices[i]
            idx2 = swing_low_indices[i + 1]
            
            price1 = low[idx1]
            price2 = low[idx2]
            
            price_diff = abs(price1 - price2) / price1
            if price_diff < tolerance:
                patterns[idx2] = 2
    
    return patterns


@njit(cache=True)
def detect_triangle_numba(high: np.ndarray, low: np.ndarray,
                         swing_highs: np.ndarray, swing_lows: np.ndarray,
                         min_swings: int = 5) -> np.ndarray:
    n = len(high)
    patterns = np.zeros(n, dtype=np.int32)
    
    swing_high_indices = _extract_swing_indices(swing_highs)
    swing_low_indices = _extract_swing_indices(swing_lows)
    
    if len(swing_high_indices) < min_swings or len(swing_low_indices) < min_swings:
        return patterns
    
    for i in range(len(swing_high_indices) - min_swings + 1):
        indices = swing_high_indices[i:i + min_swings]
        values = high[indices]
        
        if _is_monotonic_sequence(values, ascending=False):
            patterns[indices[-1]] = 1
    
    for i in range(len(swing_low_indices) - min_swings + 1):
        indices = swing_low_indices[i:i + min_swings]
        values = low[indices]
        
        if _is_monotonic_sequence(values, ascending=True):
            patterns[indices[-1]] = 2
    
    return patterns


@njit(cache=True)
def detect_wedge_numba(high: np.ndarray, low: np.ndarray,
                      swing_highs: np.ndarray, swing_lows: np.ndarray,
                      min_swings: int = 5) -> np.ndarray:
    n = len(high)
    patterns = np.zeros(n, dtype=np.int32)
    
    swing_high_indices = _extract_swing_indices(swing_highs)
    swing_low_indices = _extract_swing_indices(swing_lows)
    
    if len(swing_high_indices) < min_swings or len(swing_low_indices) < min_swings:
        return patterns
    
    for i in range(min(len(swing_high_indices), len(swing_low_indices)) - min_swings + 1):
        high_indices = swing_high_indices[i:i + min_swings]
        low_indices = swing_low_indices[i:i + min_swings]
        
        high_values = high[high_indices]
        low_values = low[low_indices]
        
        high_rising = _is_monotonic_sequence(high_values, ascending=True)
        low_rising = _is_monotonic_sequence(low_values, ascending=True)
        high_falling = _is_monotonic_sequence(high_values, ascending=False)
        low_falling = _is_monotonic_sequence(low_values, ascending=False)
        
        if high_rising and low_rising:
            patterns[max(high_indices[-1], low_indices[-1])] = 1
        elif high_falling and low_falling:
            patterns[max(high_indices[-1], low_indices[-1])] = 2
    
    return patterns


@njit(cache=True)
def detect_channel_numba(high: np.ndarray, low: np.ndarray,
                        swing_highs: np.ndarray, swing_lows: np.ndarray,
                        min_swings: int = 4, slope_tolerance: float = 0.1) -> np.ndarray:
    n = len(high)
    patterns = np.zeros(n, dtype=np.int32)
    
    swing_high_indices = _extract_swing_indices(swing_highs)
    swing_low_indices = _extract_swing_indices(swing_lows)
    
    if len(swing_high_indices) < min_swings or len(swing_low_indices) < min_swings:
        return patterns
    
    for i in range(min(len(swing_high_indices), len(swing_low_indices)) - min_swings + 1):
        high_indices = swing_high_indices[i:i + min_swings].astype(np.float64)
        low_indices = swing_low_indices[i:i + min_swings].astype(np.float64)
        
        high_values = high[swing_high_indices[i:i + min_swings]].astype(np.float64)
        low_values = low[swing_low_indices[i:i + min_swings]].astype(np.float64)
        
        high_x_mean = np.mean(high_indices)
        high_y_mean = np.mean(high_values)
        low_x_mean = np.mean(low_indices)
        low_y_mean = np.mean(low_values)
        
        high_slope_num = 0.0
        high_slope_den = 0.0
        for j in range(len(high_indices)):
            x_diff = high_indices[j] - high_x_mean
            high_slope_num += x_diff * (high_values[j] - high_y_mean)
            high_slope_den += x_diff * x_diff
        
        low_slope_num = 0.0
        low_slope_den = 0.0
        for j in range(len(low_indices)):
            x_diff = low_indices[j] - low_x_mean
            low_slope_num += x_diff * (low_values[j] - low_y_mean)
            low_slope_den += x_diff * x_diff
        
        if high_slope_den > 0 and low_slope_den > 0:
            high_slope = high_slope_num / high_slope_den
            low_slope = low_slope_num / low_slope_den
            
            slope_diff = abs(high_slope - low_slope)
            if slope_diff < slope_tolerance:
                last_idx = max(swing_high_indices[i + min_swings - 1], swing_low_indices[i + min_swings - 1])
                if high_slope > 0:
                    patterns[last_idx] = 1
                elif high_slope < 0:
                    patterns[last_idx] = 2
    
    return patterns


@njit(cache=True)
def detect_multiple_tops_bottoms_numba(high: np.ndarray, low: np.ndarray,
                                      swing_highs: np.ndarray, swing_lows: np.ndarray,
                                      min_count: int = 3, tolerance: float = 0.035) -> np.ndarray:
    n = len(high)
    patterns = np.zeros(n, dtype=np.int32)
    
    swing_high_indices = _extract_swing_indices(swing_highs)
    
    if len(swing_high_indices) >= min_count:
        for i in range(len(swing_high_indices) - min_count + 1):
            indices = swing_high_indices[i:i + min_count]
            values = high[indices]
            
            mean_value = np.mean(values)
            all_close = True
            for val in values:
                if abs(val - mean_value) / mean_value > tolerance:
                    all_close = False
                    break
            
            if all_close:
                patterns[indices[-1]] = 1
    
    swing_low_indices = _extract_swing_indices(swing_lows)
    
    if len(swing_low_indices) >= min_count:
        for i in range(len(swing_low_indices) - min_count + 1):
            indices = swing_low_indices[i:i + min_count]
            values = low[indices]
            
            mean_value = np.mean(values)
            all_close = True
            for val in values:
                if abs(val - mean_value) / mean_value > tolerance:
                    all_close = False
                    break
            
            if all_close:
                patterns[indices[-1]] = 2
    
    return patterns
