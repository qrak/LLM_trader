"""
Parabolic SAR utilities extracted to reduce complexity.
"""
from typing import Tuple
import numpy as np
from numba import njit


@njit(cache=True)
def initialize_sar_arrays(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize SAR calculation arrays."""
    sar = np.full(n, np.nan)
    ep = np.full(n, np.nan)
    af = np.full(n, np.nan)
    return sar, ep, af


@njit(cache=True)
def get_initial_sar_state(high: np.ndarray, low: np.ndarray, step: float) -> Tuple[int, float, float, float]:
    """Get initial SAR state based on first two periods."""
    trend = -1 if high[0] > low[1] else 1
    
    if trend == 1:
        sar_value = np.min(low[:2])
        ep_value = high[0]
    else:
        sar_value = np.max(high[:2])
        ep_value = low[0]
    
    return trend, sar_value, ep_value, step


@njit(cache=True)
def update_bullish_sar(i: int, high: np.ndarray, low: np.ndarray, 
                      sar: np.ndarray, ep: np.ndarray, af: np.ndarray,
                      step: float, max_step: float) -> int:
    """Update SAR values for bullish trend."""
    new_sar = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])
    
    if low[i] > new_sar:
        # Continue bullish trend
        sar[i] = min(new_sar, low[i - 1], low[i])
        
        if high[i] > ep[i - 1]:
            ep[i] = high[i]
            af[i] = min(af[i - 1] + step, max_step)
        else:
            ep[i] = ep[i - 1]
            af[i] = af[i - 1]
        return 1
    else:
        # Trend reversal to bearish
        sar[i] = ep[i - 1]
        ep[i] = low[i]
        af[i] = step
        return -1


@njit(cache=True)
def update_bearish_sar(i: int, high: np.ndarray, low: np.ndarray, 
                      sar: np.ndarray, ep: np.ndarray, af: np.ndarray,
                      step: float, max_step: float) -> int:
    """Update SAR values for bearish trend."""
    new_sar = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])
    
    if high[i] < new_sar:
        # Continue bearish trend
        sar[i] = max(new_sar, high[i - 1], high[i])
        
        if low[i] < ep[i - 1]:
            ep[i] = low[i]
            af[i] = min(af[i - 1] + step, max_step)
        else:
            ep[i] = ep[i - 1]
            af[i] = af[i - 1]
        return -1
    else:
        # Trend reversal to bullish
        sar[i] = ep[i - 1]
        ep[i] = high[i]
        af[i] = step
        return 1
