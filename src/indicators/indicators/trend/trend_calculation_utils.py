"""
Trend calculation utilities for trend indicators.
Extracted to reduce complexity in trend_indicators.py
"""
from typing import NamedTuple, Tuple
import numpy as np
from numba import njit


@njit(cache=True)
def calculate_directional_movement(high: np.ndarray, low: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate positive and negative directional movement."""
    n = len(high)
    dm_pos = np.full(n, np.nan)
    dm_neg = np.full(n, np.nan)
    
    for i in range(1, n):
        dm_pos[i] = high[i] - high[i - 1] if high[i] - high[i - 1] > low[i - 1] - low[i] else 0
        dm_neg[i] = low[i - 1] - low[i] if low[i - 1] - low[i] > high[i] - high[i - 1] else 0
    
    return dm_pos, dm_neg


@njit(cache=True)
def calculate_smoothed_values(values: np.ndarray, length: int) -> np.ndarray:
    """Calculate Wilder's smoothed values (used in ADX)."""
    n = len(values)
    smoothed = np.full(n, np.nan)
    
    # Initialize first smoothed value
    smoothed[length - 1] = np.sum(values[1:length])
    
    # Calculate subsequent smoothed values
    length_recip = 1 / length
    for i in range(length, n):
        smoothed[i] = smoothed[i - 1] - (smoothed[i - 1] * length_recip) + values[i]
    
    return smoothed


@njit(cache=True)
def calculate_directional_indicators(dm_pos14: np.ndarray, dm_neg14: np.ndarray, 
                                   tr14: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate PDI, NDI, and DX from smoothed values."""
    n = len(dm_pos14)
    pdi = np.full(n, np.nan)
    ndi = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    
    for i in range(n):
        if not np.isnan(tr14[i]) and tr14[i] != 0:
            pdi[i] = 100 * (dm_pos14[i] / tr14[i])
            ndi[i] = 100 * (dm_neg14[i] / tr14[i])
        else:
            pdi[i] = 0
            ndi[i] = 0
        
        if not np.isnan(pdi[i]) and not np.isnan(ndi[i]) and pdi[i] + ndi[i] != 0:
            dx[i] = 100 * abs((pdi[i] - ndi[i]) / (pdi[i] + ndi[i]))
        else:
            dx[i] = 0
    
    return pdi, ndi, dx


@njit(cache=True)
def calculate_ichimoku_lines(high: np.ndarray, low: np.ndarray, 
                           conversion_length: int, base_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Ichimoku conversion and base lines."""
    n = len(high)
    conversion_line = np.full(n, np.nan)
    base_line = np.full(n, np.nan)
    
    for i in range(base_length - 1, n):
        if i >= conversion_length - 1:
            conversion_line[i] = (np.max(high[i - conversion_length + 1: i + 1]) +
                                np.min(low[i - conversion_length + 1: i + 1])) / 2
        
        base_line[i] = (np.max(high[i - base_length + 1: i + 1]) +
                       np.min(low[i - base_length + 1: i + 1])) / 2
    
    return conversion_line, base_line


@njit(cache=True)
def calculate_ichimoku_spans(high: np.ndarray, low: np.ndarray, 
                           conversion_line: np.ndarray, base_line: np.ndarray,
                           lagging_span2_length: int, displacement: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Ichimoku leading spans A and B."""
    n = len(high)
    leading_span_a = np.full(n, np.nan)
    leading_span_b = np.full(n, np.nan)
    
    for i in range(lagging_span2_length - 1, n):
        if i + displacement < n and not np.isnan(conversion_line[i]) and not np.isnan(base_line[i]):
            leading_span_a[i + displacement] = (conversion_line[i] + base_line[i]) / 2
        
        if i + displacement < n:
            leading_span_b[i + displacement] = (np.max(high[i - lagging_span2_length + 1: i + 1]) +
                                              np.min(low[i - lagging_span2_length + 1: i + 1])) / 2
    
    return leading_span_a, leading_span_b


@njit(cache=True)
def calculate_band_adjustments(close: np.ndarray, upperband: np.ndarray, 
                             lowerband: np.ndarray, i: int) -> Tuple[float, float]:
    """Calculate adjusted upper and lower bands for Supertrend."""
    upper = upperband[i]
    lower = lowerband[i]
    
    if i > 0:
        if close[i - 1] <= upperband[i - 1]:
            upper = min(upperband[i], upperband[i - 1])
        
        if close[i - 1] >= lowerband[i - 1]:
            lower = max(lowerband[i], lowerband[i - 1])
    
    return upper, lower


@njit(cache=True)
def calculate_vortex_components(high: np.ndarray, low: np.ndarray, 
                              close: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate True Range and Vortex Movement components."""
    n = len(high)
    tr = np.zeros(n)
    vmp = np.zeros(n)
    vmm = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        vmp[i] = abs(high[i] - low[i - 1])
        vmm[i] = abs(low[i] - high[i - 1])
    
    return tr, vmp, vmm


@njit(cache=True)
def calculate_pfe_efficiency(close: np.ndarray, start_idx: int, n: int) -> float:
    """Calculate Price Efficiency for a given window."""
    # Calculate sum of squared differences manually
    sum_square_diffs = 0.0
    for j in range(start_idx, start_idx + n - 1):
        diff = close[j + 1] - close[j]
        sum_square_diffs += diff * diff
    
    if sum_square_diffs <= 0:
        return np.nan
    
    term1 = np.sqrt((close[start_idx + n - 1] - close[start_idx]) ** 2 + n ** 2)
    pi = 100 * term1 / np.sqrt(sum_square_diffs)
    
    # Apply directional sign
    if close[start_idx + n - 1] < close[start_idx + n - 2]:
        pi = -pi
    
    return pi
