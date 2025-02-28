from typing import Tuple

import numpy as np
from numba import njit

from indicators.indicators.overlap import ema_numba
from indicators.indicators.volatility import atr_numba


@njit(cache=True)
def adx_numba(high, low, close, length):
    n = len(high)
    tr = np.full(n, np.nan)
    dm_pos = np.full(n, np.nan)
    dm_neg = np.full(n, np.nan)
    tr14 = np.full(n, np.nan)
    dm_pos14 = np.full(n, np.nan)
    dm_neg14 = np.full(n, np.nan)
    pdi = np.full(n, np.nan)
    ndi = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    adx = np.full(n, np.nan)

    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        dm_pos[i] = high[i] - high[i - 1] if high[i] - high[i - 1] > low[i - 1] - low[i] else 0
        dm_neg[i] = low[i - 1] - low[i] if low[i - 1] - low[i] > high[i] - high[i - 1] else 0

    tr14[length - 1] = np.sum(tr[1:length])
    dm_pos14[length - 1] = np.sum(dm_pos[1:length])
    dm_neg14[length - 1] = np.sum(dm_neg[1:length])

    length_recip = 1 / length
    for i in range(length, n):
        tr14[i] = tr14[i - 1] - (tr14[i - 1] * length_recip) + tr[i]
        dm_pos14[i] = dm_pos14[i - 1] - (dm_pos14[i - 1] * length_recip) + dm_pos[i]
        dm_neg14[i] = dm_neg14[i - 1] - (dm_neg14[i - 1] * length_recip) + dm_neg[i]

        if tr14[i] != 0:
            pdi[i] = 100 * (dm_pos14[i] / tr14[i])
            ndi[i] = 100 * (dm_neg14[i] / tr14[i])
        else:
            pdi[i] = 0
            ndi[i] = 0

        if pdi[i] + ndi[i] != 0:
            dx[i] = 100 * abs((pdi[i] - ndi[i]) / (pdi[i] + ndi[i]))
        else:
            dx[i] = 0

        if i == (length * 2 - 2):
            adx[length * 2 - 2] = np.nanmean(dx[length - 1:length * 2 - 1])

        if i > (length * 2 - 2):
            adx[i] = ((adx[i - 1] * (length - 1)) + dx[i]) * length_recip

    return adx, pdi, ndi


@njit(cache=True)
def supertrend_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     length: int = 10, multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    n = len(close)
    atr = atr_numba(high, low, close, length)

    hl2 = (high + low) / 2
    upperband = np.full(n, np.nan)
    lowerband = np.full(n, np.nan)
    trend = np.full(n, np.nan)
    direction = np.full(n, 1)

    upperband[0] = hl2[0] + multiplier * atr[0]
    lowerband[0] = hl2[0] - multiplier * atr[0]
    trend[0] = upperband[0]

    for i in range(1, n):
        upperband[i] = hl2[i] + multiplier * atr[i]
        lowerband[i] = hl2[i] - multiplier * atr[i]

        if close[i - 1] <= upperband[i - 1]:
            upperband[i] = min(upperband[i], upperband[i - 1])
        else:
            upperband[i] = upperband[i]

        if close[i - 1] >= lowerband[i - 1]:
            lowerband[i] = max(lowerband[i], lowerband[i - 1])
        else:
            lowerband[i] = lowerband[i]

        if close[i] > upperband[i - 1]:
            direction[i] = 1
        elif close[i] < lowerband[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

        trend[i] = lowerband[i] if direction[i] == 1 else upperband[i]

    return trend, direction

@njit(cache=True)
def ichimoku_cloud_numba(high, low, conversion_length, base_length, lagging_span2_length, displacement):
    n = len(high)
    conversion_line = np.full(n, np.nan)
    base_line = np.full(n, np.nan)
    leading_span_a = np.full(n, np.nan)
    leading_span_b = np.full(n, np.nan)

    for i in range(base_length - 1, n):
        conversion_line[i] = (np.max(high[i - conversion_length + 1: i + 1]) +
                              np.min(low[i - conversion_length + 1: i + 1])) / 2

        base_line[i] = (np.max(high[i - base_length + 1: i + 1]) +
                        np.min(low[i - base_length + 1: i + 1])) / 2

        if i + displacement < n:
            leading_span_a[i + displacement] = (conversion_line[i] + base_line[i]) / 2

        if i >= lagging_span2_length - 1 and i + displacement < n:
            leading_span_b[i + displacement] = (np.max(high[i - lagging_span2_length + 1: i + 1]) +
                                                np.min(low[i - lagging_span2_length + 1: i + 1])) / 2

    return conversion_line, base_line, leading_span_a, leading_span_b

@njit(cache=True)
def parabolic_sar_numba(high, low, step=0.02, max_step=0.2):
    n = len(high)
    sar = np.full(n, np.nan)
    ep = np.full(n, np.nan)
    af = np.full(n, np.nan)

    trend = 1
    sar[0] = low[0]
    ep[0] = high[0]
    af[0] = step

    for i in range(1, n):
        sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])

        if trend == 1:
            if low[i] > sar[i]:
                sar[i] = min(sar[i], low[i - 1], low[i])
            else:
                trend = -1
                sar[i] = ep[i - 1]
                ep[i] = low[i]
                af[i] = step
        else:
            if high[i] < sar[i]:
                sar[i] = max(sar[i], high[i - 1], high[i])
            else:
                trend = 1
                sar[i] = ep[i - 1]
                ep[i] = high[i]
                af[i] = step

        if trend == 1:
            if high[i] > ep[i - 1]:
                ep[i] = high[i]
                af[i] = min(af[i - 1] + step, max_step)
            else:
                ep[i] = ep[i - 1]
                af[i] = af[i - 1]
        else:
            if low[i] < ep[i - 1]:
                ep[i] = low[i]
                af[i] = min(af[i - 1] + step, max_step)
            else:
                ep[i] = ep[i - 1]
                af[i] = af[i - 1]

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
    n = len(high)
    tr = np.zeros(n)
    vmp = np.zeros(n)
    vmm = np.zeros(n)
    vi_plus = np.full(n, np.nan)
    vi_minus = np.full(n, np.nan)

    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        vmp[i] = abs(high[i] - low[i - 1])
        vmm[i] = abs(low[i] - high[i - 1])

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
    length = len(close)
    p = np.full(length, np.nan)
    pfe = np.full(length, np.nan)

    # Calculate differences manually instead of using np.diff
    for i in range(n - 1, length):
        # Calculate sum of squared differences manually
        sum_square_diffs = 0.0
        for j in range(i - n + 1, i):
            diff = close[j + 1] - close[j]
            sum_square_diffs += diff * diff

        if sum_square_diffs > 0:  # Avoid division by zero
            term1 = np.sqrt((close[i] - close[i - n]) ** 2 + n ** 2)
            pi = 100 * term1 / np.sqrt(sum_square_diffs)
            if close[i] < close[i - 1]:
                pi = -pi
            p[i] = pi

    # Calculate EMA of p
    multiplier = 2 / (m + 1)
    pfe[n - 1:] = p[n - 1:]  # Copy initial values

    for i in range(n + m - 1, length):
        pfe[i] = ((p[i] - pfe[i - 1]) * multiplier) + pfe[i - 1]

    return pfe

