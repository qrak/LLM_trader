from typing import Tuple

import numpy as np
from numba import njit

from src.indicators.indicators.overlap import ema_numba
from src.indicators.indicators.volatility import atr_numba
from .trend_calculation_utils import (
    calculate_directional_movement, calculate_smoothed_values, 
    calculate_directional_indicators, calculate_ichimoku_lines,
    calculate_ichimoku_spans, calculate_band_adjustments,
    calculate_vortex_components, calculate_pfe_efficiency
)
from .sar_utils import (
    initialize_sar_arrays, get_initial_sar_state,
    update_bullish_sar, update_bearish_sar
)


@njit(cache=True)
def adx_numba(high, low, close, length):
    """Calculate ADX (Average Directional Index)."""
    n = len(high)
    
    # Calculate True Range
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    
    # Calculate directional movements
    dm_pos, dm_neg = calculate_directional_movement(high, low)
    
    # Calculate smoothed values
    tr14 = calculate_smoothed_values(tr, length)
    dm_pos14 = calculate_smoothed_values(dm_pos, length)
    dm_neg14 = calculate_smoothed_values(dm_neg, length)
    
    # Calculate directional indicators and DX
    pdi, ndi, dx = calculate_directional_indicators(dm_pos14, dm_neg14, tr14)
    
    # Calculate ADX
    adx = np.full(n, np.nan)
    length_recip = 1 / length
    
    # Initialize ADX
    if length * 2 - 2 < n:
        adx[length * 2 - 2] = np.nanmean(dx[length - 1:length * 2 - 1])
    
    # Calculate subsequent ADX values
    for i in range(length * 2 - 1, n):
        if not np.isnan(adx[i - 1]) and not np.isnan(dx[i]):
            adx[i] = ((adx[i - 1] * (length - 1)) + dx[i]) * length_recip

    return adx, pdi, ndi


@njit(cache=True)
def supertrend_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     length: int = 10, multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Supertrend indicator."""
    n = len(close)
    atr = atr_numba(high, low, close, length)

    hl2 = (high + low) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    
    trend = np.full(n, np.nan)
    direction = np.full(n, 1)

    # Initialize first values
    trend[0] = lowerband[0] if direction[0] == 1 else upperband[0]

    for i in range(1, n):
        # Adjust bands based on previous close
        upperband[i], lowerband[i] = calculate_band_adjustments(close, upperband, lowerband, i)

        # Determine trend direction
        if close[i] > upperband[i - 1]:
            direction[i] = 1
        elif close[i] < lowerband[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

        trend[i] = lowerband[i] if direction[i] == 1 else upperband[i]

    return trend, direction

@njit(cache=True)
def ichimoku_cloud_numba(high, low, conversion_length=9, base_length=26, 
                        lagging_span2_length=52, displacement=26):
    """Calculate Ichimoku Cloud components using configuration."""
    # Calculate conversion and base lines
    conversion_line, base_line = calculate_ichimoku_lines(
        high, low, conversion_length, base_length
    )
    
    # Calculate leading spans
    leading_span_a, leading_span_b = calculate_ichimoku_spans(
        high, low, conversion_line, base_line, lagging_span2_length, displacement
    )

    return conversion_line, base_line, leading_span_a, leading_span_b

@njit(cache=True)
def parabolic_sar_numba(high, low, step=0.02, max_step=0.2):
    """Calculate Parabolic SAR using extracted utilities."""
    n = len(high)
    sar, ep, af = initialize_sar_arrays(n)

    # Initialize state
    trend, sar[0], ep[0], af[0] = get_initial_sar_state(high, low, step)

    # Calculate SAR for each period
    for i in range(1, n):
        if trend == 1:
            trend = update_bullish_sar(i, high, low, sar, ep, af, step, max_step)
        else:
            trend = update_bearish_sar(i, high, low, sar, ep, af, step, max_step)

    return sar

@njit(cache=True)
def trix_numba(close, length=18, scalar=100, drift=1):
    n = len(close)
    trix = np.full(n, np.nan)

    ema1 = ema_numba(close, length)
    ema2 = ema_numba(ema1, length)
    ema3 = ema_numba(ema2, length)

    for i in range(length + drift, n):
        if ema3[i - drift] != 0:
            trix[i] = scalar * ((ema3[i] - ema3[i - drift]) / ema3[i - drift])

    return trix


@njit(cache=True)
def vortex_indicator_numba(high, low, close, length):
    """Calculate Vortex Indicator using extracted utilities."""
    n = len(high)
    vi_plus = np.full(n, np.nan)
    vi_minus = np.full(n, np.nan)

    # Calculate components
    tr, vmp, vmm = calculate_vortex_components(high, low, close)

    # Calculate rolling sums and Vortex Indicator
    for i in range(length, n):
        tr_sum = np.sum(tr[i - length + 1:i + 1])
        vmp_sum = np.sum(vmp[i - length + 1:i + 1])
        vmm_sum = np.sum(vmm[i - length + 1:i + 1])

        if tr_sum != 0:
            vi_plus[i] = vmp_sum / tr_sum
            vi_minus[i] = vmm_sum / tr_sum

    return vi_plus, vi_minus


@njit(cache=True)
def pfe_numba(close, n, m):
    """Calculate Polarized Fractal Efficiency using extracted utilities."""
    length = len(close)
    p = np.full(length, np.nan)
    pfe = np.full(length, np.nan)

    # Calculate efficiency values
    for i in range(n - 1, length):
        p[i] = calculate_pfe_efficiency(close, i - n + 1, n)

    # Calculate EMA of efficiency values
    multiplier = 2 / (m + 1)
    start_idx = n - 1

    # Initialize with first valid value
    if start_idx < length and not np.isnan(p[start_idx]):
        pfe[start_idx] = p[start_idx]

    # Calculate EMA
    for i in range(start_idx + 1, length):
        if not np.isnan(p[i]) and not np.isnan(pfe[i - 1]):
            pfe[i] = ((p[i] - pfe[i - 1]) * multiplier) + pfe[i - 1]

    return pfe


@njit(cache=True)
def td_sequential_numba(close, length=9):
    """
    Calculate TD Sequential indicator.
    Returns the count of consecutive higher/lower closes.
    Positive values indicate bullish counts, negative values indicate bearish counts.
    """
    n = len(close)
    td_seq = np.full(n, np.nan)
    
    for i in range(4, n):  # Need at least 4 periods for comparison
        # Count consecutive higher closes
        bullish_count = 0
        bearish_count = 0
        
        # Look back up to 'length' periods
        for j in range(min(length, i)):
            idx = i - j
            if idx >= 4:  # Need 4 periods for comparison
                if close[idx] > close[idx - 4]:
                    bullish_count += 1
                    bearish_count = 0  # Reset bearish count
                elif close[idx] < close[idx - 4]:
                    bearish_count += 1
                    bullish_count = 0  # Reset bullish count
                else:
                    break  # Break on equal close
        
        # Assign value based on the count
        if bullish_count > 0:
            td_seq[i] = bullish_count
        elif bearish_count > 0:
            td_seq[i] = -bearish_count
        else:
            td_seq[i] = 0
            
    return td_seq

