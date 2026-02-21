"""
Trend Indicators implementation using Numba.

This module provides optimized implementations of various trend indicators
including ADX, Supertrend, Ichimoku Cloud, Parabolic SAR, TRIX,
Vortex Indicator, PFE, and TD Sequential.
"""
from typing import Tuple

import numpy as np
from numba import njit

from src.indicators.overlap import ema_numba
from src.indicators.volatility import atr_numba
from .trend_calculation_utils import (
    calculate_directional_movement, calculate_smoothed_values,
    calculate_directional_indicators, calculate_ichimoku_lines,
    calculate_ichimoku_spans, calculate_band_adjustments,
    calculate_vortex_components
)
from .sar_utils import (
    initialize_sar_arrays, get_initial_sar_state,
    update_bullish_sar, update_bearish_sar
)


@njit(cache=True)
def adx_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray,
              length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate ADX (Average Directional Index) and directional indicators.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (ADX, +DI, -DI) arrays.
    """
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

    adx = np.full(n, np.nan)
    length_recip = 1 / length

    if length * 2 - 1 < n:
        adx[length * 2 - 1] = np.nanmean(dx[length:length * 2])

    for i in range(length * 2, n):
        if not np.isnan(adx[i - 1]) and not np.isnan(dx[i]):
            adx[i] = ((adx[i - 1] * (length - 1)) + dx[i]) * length_recip

    return adx, pdi, ndi


@njit(cache=True)
def supertrend_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     length: int = 10, multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Supertrend indicator.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (Supertrend Line, Trend Direction [1/-1])
    """
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
def ichimoku_cloud_numba(high: np.ndarray, low: np.ndarray,
                         conversion_length: int = 9, base_length: int = 26,
                         lagging_span2_length: int = 52,
                         displacement: int = 26) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Ichimoku Cloud components.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            (Conversion Line, Base Line, Leading Span A, Leading Span B)
    """
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
    """Calculate Polarized Fractal Efficiency (PFE).

    Formula:
    PFE = 100 * Sqrt((C[i] - C[i-n])^2 + n^2) / Sum(Sqrt((C[j] - C[j-1])^2 + 1))

    This implementation uses O(N) rolling sum for the denominator.
    """
    length = len(close)
    pfe = np.full(length, np.nan)
    p = np.full(length, np.nan)

    # Need at least n+1 points to calculate n-period PFE
    if length <= n:
        return pfe

    # Calculate segment lengths: Sqrt(diff^2 + 1)
    # segment_lengths[i] = length of segment from i-1 to i
    segment_lengths = np.zeros(length)
    for i in range(1, length):
        diff = close[i] - close[i - 1]
        segment_lengths[i] = np.sqrt(diff * diff + 1.0)

    # Rolling sum of segment lengths over window n
    # We sum n segments: from i-n+1 to i

    current_sum = 0.0
    # Initialize first window sum (indices 1 to n)
    for i in range(1, n + 1):
        current_sum += segment_lengths[i]

    # Calculate PFE for first valid point at index n
    # dy = close[n] - close[0] (n intervals)
    dy = close[n] - close[0]
    # Numerator uses n intervals as dx
    numerator = np.sqrt(dy * dy + n * n)

    # Sign based on trend direction
    sign = 1.0 if dy > 0 else (-1.0 if dy < 0 else 0.0)

    if current_sum != 0:
        p[n] = sign * 100 * numerator / current_sum
    else:
        p[n] = 0.0

    # Iterate for the rest
    for i in range(n + 1, length):
        # Update rolling sum: add new segment (i), remove old segment (i-n)
        current_sum += segment_lengths[i] - segment_lengths[i - n]

        dy = close[i] - close[i - n]
        numerator = np.sqrt(dy * dy + n * n)
        sign = 1.0 if dy > 0 else (-1.0 if dy < 0 else 0.0)

        if current_sum > 0:
             p[i] = sign * 100 * numerator / current_sum
        else:
             p[i] = 0.0

    # Calculate EMA of p values
    multiplier = 2 / (m + 1)

    # Initialize EMA with first valid p
    start_idx = n
    if not np.isnan(p[start_idx]):
        pfe[start_idx] = p[start_idx]

    for i in range(start_idx + 1, length):
        if not np.isnan(p[i]):
            if np.isnan(pfe[i-1]):
                pfe[i] = p[i]
            else:
                pfe[i] = ((p[i] - pfe[i - 1]) * multiplier) + pfe[i - 1]

    return pfe


@njit(cache=True)
def td_sequential_numba(close, length=9):
    """
    Calculate TD Sequential indicator.
    Returns the count of consecutive higher/lower closes.
    Positive values indicate bullish counts, negative values indicate bearish counts.
    Optimization: O(N) forward pass instead of O(N*L) backward nested loop.
    """
    n = len(close)
    td_seq = np.full(n, np.nan)

    # Need at least 4 periods for comparison
    # We can iterate and build state incrementally
    for i in range(4, n):
        c = close[i]
        c4 = close[i - 4]

        # Get previous state (handling start of sequence)
        prev = td_seq[i - 1]
        if np.isnan(prev):
            prev = 0.0

        if c > c4:
            # Bullish
            if prev >= 0:
                val = prev + 1
            else:
                val = 1
        elif c < c4:
            # Bearish
            if prev <= 0:
                val = prev - 1
            else:
                val = -1
        else:
            # Equal - reset
            val = 0.0

        # Cap at length if specified
        if abs(val) > length:
            val = np.sign(val) * length

        td_seq[i] = val

    return td_seq
