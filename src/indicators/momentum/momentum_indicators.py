"""Momentum Indicators module.

Provides functionality for indicators.momentum.momentum_indicators.py.
"""
import math
from typing import Any

import numpy as np
from numba import njit

from src.indicators.overlap import ema_numba


@njit(cache=True)
def rsi_numba(close: np.ndarray, length: int) -> np.ndarray:
    n = len(close)
    gains = np.zeros(n)
    losses = np.zeros(n)

    for i in range(1, n):
        diff = float(close[i] - close[i - 1])
        gains[i] = max(0, diff)
        losses[i] = max(0, -diff)

    rsi = np.full(n, np.nan)
    avg_gain = np.sum(gains[1:length + 1]) / length
    avg_loss = np.sum(losses[1:length + 1]) / length

    if avg_loss == 0:
        rsi[length] = 100
    else:
        rs = avg_gain / avg_loss
        rsi[length] = 100 - (100 / (1 + rs))

    for i in range(length + 1, n):
        avg_gain = ((avg_gain * (length - 1)) + gains[i]) / length
        avg_loss = ((avg_loss * (length - 1)) + losses[i]) / length
        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))

    return rsi

@njit(cache=True)
def macd_numba(close: np.ndarray, fast_length: int = 12, slow_length: int = 26,
               signal_length: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    macd_line = np.full(n, np.nan, dtype=np.float64)
    signal_line = np.full(n, np.nan, dtype=np.float64)
    histogram = np.full(n, np.nan, dtype=np.float64)

    alpha_fast = 2.0 / (fast_length + 1)
    alpha_slow = 2.0 / (slow_length + 1)
    alpha_signal = 2.0 / (signal_length + 1)

    fast_ema = np.mean(close[:fast_length])
    slow_ema = np.mean(close[:slow_length])
    signal = 0.0

    for i in range(n):
        fast_ema = close[i] * alpha_fast + fast_ema * (1 - alpha_fast)
        slow_ema = close[i] * alpha_slow + slow_ema * (1 - alpha_slow)

        if i >= slow_length - 1:
            macd = fast_ema - slow_ema
            macd_line[i] = macd

            if i == slow_length - 1:
                signal = macd
            elif i > slow_length - 1:
                signal = macd * alpha_signal + signal * (1 - alpha_signal)
                signal_line[i] = signal
                histogram[i] = macd - signal

    return macd_line, signal_line, histogram

@njit(cache=True)
def stochastic_numba(high, low, close, period_k, smooth_k, period_d):
    n = len(close)
    k_values = np.full(n, np.nan)
    d_values = np.full(n, np.nan)

    for i in range(period_k - 1, n):
        start_idx = i - period_k + 1
        end_idx = i + 1

        high_max = high[start_idx]
        low_min = low[start_idx]

        for j in range(start_idx + 1, end_idx):
            val_h = high[j]
            val_l = low[j]
            high_max = max(high_max, val_h)
            low_min = min(low_min, val_l)

        if high_max != low_min:
            k_values[i] = 100 * (close[i] - low_min) / (high_max - low_min)

    smoothed_k = np.full(n, np.nan)
    sum_k = 0.0
    nan_count_k = 0
    start_k = period_k - 1

    if n >= start_k + smooth_k:
        for i in range(start_k, start_k + smooth_k - 1):
            if math.isnan(k_values[i]):
                nan_count_k += 1
            else:
                sum_k += k_values[i]

        for i in range(start_k + smooth_k - 1, n):
            if math.isnan(k_values[i]):
                nan_count_k += 1
            else:
                sum_k += k_values[i]

            if nan_count_k > 0:
                smoothed_k[i] = np.nan
            else:
                smoothed_k[i] = sum_k / smooth_k

            old_idx = i - smooth_k + 1
            if math.isnan(k_values[old_idx]):
                nan_count_k -= 1
            else:
                sum_k -= k_values[old_idx]

    sum_d = 0.0
    nan_count_d = 0
    start_d = period_k + smooth_k - 2

    if n >= start_d + period_d:
        for i in range(start_d, start_d + period_d - 1):
            if math.isnan(smoothed_k[i]):
                nan_count_d += 1
            else:
                sum_d += smoothed_k[i]

        for i in range(start_d + period_d - 1, n):
            if math.isnan(smoothed_k[i]):
                nan_count_d += 1
            else:
                sum_d += smoothed_k[i]

            if nan_count_d > 0:
                d_values[i] = np.nan
            else:
                d_values[i] = sum_d / period_d

            old_idx = i - period_d + 1
            if math.isnan(smoothed_k[old_idx]):
                nan_count_d -= 1
            else:
                sum_d -= smoothed_k[old_idx]

    return smoothed_k, d_values

@njit(cache=True)
def roc_numba(close, length=1):
    n = len(close)
    roc = np.empty(n, dtype=np.float64)
    roc[:length] = np.nan

    roc[length:] = ((close[length:] / close[:-length]) - 1) * 100

    return roc

@njit(cache=True)
def momentum_numba(close, length=1):
    n = len(close)
    mom = np.full(n, np.nan)

    for i in range(length, n):
        mom[i] = close[i] - close[i - length]

    return mom

@njit(cache=True)
def williams_r_numba(high, low, close, length):
    n = len(close)
    williams_r = np.full(n, np.nan)

    for i in range(length - 1, n):
        start_idx = i - length + 1
        end_idx = i + 1

        highest_high = high[start_idx]
        lowest_low = low[start_idx]

        for j in range(start_idx + 1, end_idx):
            h_val = high[j]
            l_val = low[j]
            highest_high = max(highest_high, h_val)
            lowest_low = min(lowest_low, l_val)

        if highest_high != lowest_low:
            williams_r[i] = ((highest_high - close[i]) / (highest_high - lowest_low)) * -100

    return williams_r

@njit(cache=True)
def tsi_numba(close, long_length, short_length):
    """
    True Strength Index (TSI) - Optimized implementation.

    Double smoothed momentum indicator.
    1. Calculates momentum (m = close - prev_close)
    2. Smoothes m with EMA (long_length) -> ema1
    3. Smoothes ema1 with EMA (short_length) -> ema2
    4. TSI = 100 * (ema2(m) / ema2(|m|))

    Optimized for performance:
    - Single pass execution (O(N))
    - Minimal memory allocation
    - Fixed off-by-one initialization error
    """
    n = len(close)
    tsi = np.full(n, np.nan)

    alpha_long = 2.0 / (long_length + 1)
    alpha_short = 2.0 / (short_length + 1)

    m_sum = 0.0
    abs_m_sum = 0.0

    for i in range(1, long_length + 1):
        if i < n:
            val = close[i] - close[i - 1]
            m_sum += val
            abs_m_sum += abs(val)

    if n <= long_length:
        return tsi

    curr_ema1 = m_sum / long_length
    curr_abs_ema1 = abs_m_sum / long_length


    ema1_sum = curr_ema1
    abs_ema1_sum = curr_abs_ema1

    prev_ema1 = curr_ema1
    prev_abs_ema1 = curr_abs_ema1

    start_ema2_init = long_length + 1
    end_ema2_init = long_length + short_length - 1

    if end_ema2_init >= n:
        return tsi

    for i in range(start_ema2_init, end_ema2_init + 1):
        m = close[i] - close[i - 1]
        abs_m = abs(m)

        curr_ema1 = (m - prev_ema1) * alpha_long + prev_ema1
        curr_abs_ema1 = (abs_m - prev_abs_ema1) * alpha_long + prev_abs_ema1

        ema1_sum += curr_ema1
        abs_ema1_sum += curr_abs_ema1

        prev_ema1 = curr_ema1
        prev_abs_ema1 = curr_abs_ema1

    curr_ema2 = ema1_sum / short_length
    curr_abs_ema2 = abs_ema1_sum / short_length

    if curr_abs_ema2 != 0:
        tsi[end_ema2_init] = (curr_ema2 / curr_abs_ema2) * 100.0
    else:
        tsi[end_ema2_init] = 0.0

    prev_ema2 = curr_ema2
    prev_abs_ema2 = curr_abs_ema2

    for i in range(end_ema2_init + 1, n):
        m = close[i] - close[i - 1]
        abs_m = abs(m)

        curr_ema1 = (m - prev_ema1) * alpha_long + prev_ema1
        curr_abs_ema1 = (abs_m - prev_abs_ema1) * alpha_long + prev_abs_ema1

        curr_ema2 = (curr_ema1 - prev_ema2) * alpha_short + prev_ema2
        curr_abs_ema2 = (curr_abs_ema1 - prev_abs_ema2) * alpha_short + prev_abs_ema2

        if curr_abs_ema2 != 0:
            tsi[i] = (curr_ema2 / curr_abs_ema2) * 100.0
        else:
            tsi[i] = tsi[i - 1]

        prev_ema1 = curr_ema1
        prev_abs_ema1 = curr_abs_ema1
        prev_ema2 = curr_ema2
        prev_abs_ema2 = curr_abs_ema2

    return tsi

@njit(cache=True)
def rmi_numba(close, length, momentum_length):
    n = len(close)
    rmi = np.full(n, np.nan)

    if n <= momentum_length:
        return rmi

    momentum = np.zeros(n - momentum_length)
    for i in range(len(momentum)):
        momentum[i] = close[i + momentum_length] - close[i]

    up = np.maximum(momentum, 0)
    down = np.maximum(-momentum, 0)

    m_len = len(momentum)
    if m_len < length:
        return rmi

    sum_up = 0.0
    sum_down = 0.0

    for i in range(length - 1):
        sum_up += up[i]
        sum_down += down[i]

    for i in range(length - 1, m_len):
        sum_up += up[i]
        sum_down += down[i]

        avg_up = sum_up / length
        avg_down = sum_down / length

        if avg_down == 0:
            rmi[i + momentum_length] = 100
        else:
            rs = avg_up / avg_down
            rmi[i + momentum_length] = 100 - (100 / (1 + rs))

        sum_up -= up[i - length + 1]
        sum_down -= down[i - length + 1]

    return rmi

@njit(cache=True)
def ppo_numba(close, fast_length, slow_length):
    n = len(close)
    ppo = np.full(n, np.nan)

    fast_ema = ema_numba(close, fast_length)
    slow_ema = ema_numba(close, slow_length)

    for i in range(slow_length - 1, n):
        if slow_ema[i] != 0:
            ppo[i] = ((fast_ema[i] - slow_ema[i]) / slow_ema[i]) * 100

    return ppo

@njit(cache=True)
def coppock_curve_numba(close, wl1=14, wl2=11, wma_length=10):
    roc_long = roc_numba(close, wl1)
    roc_short = roc_numba(close, wl2)
    coppock_arr = roc_long + roc_short
    ewma_coppock = ema_numba(coppock_arr, wma_length)
    return ewma_coppock

@njit(cache=True)
def detect_rsi_divergence(close_prices, rsi_values, length=14):
    divergence = np.zeros_like(close_prices)
    n = len(close_prices)
    for i in range(length, n):
        price_diff = close_prices[i] - close_prices[i - length]
        rsi_diff = rsi_values[i] - rsi_values[i - length]
        if price_diff < 0 and rsi_diff > 0:
            divergence[i] = 1
        elif price_diff > 0 and rsi_diff < 0:
            divergence[i] = -1
        else:
            divergence[i] = 0
    return divergence

@njit(cache=True)
def calculate_relative_strength_numba(pair_close, benchmark_close, window=14):
    n = len(pair_close)
    rs_array = np.zeros(n)

    for i in range(window, n):
        if math.isnan(pair_close[i]) or math.isnan(benchmark_close[i]) or benchmark_close[i] == 0:
            rs_array[i] = 0.0
            continue

        pair_return = np.log(pair_close[i] / pair_close[i - window])
        benchmark_return = np.log(benchmark_close[i] / benchmark_close[i - window])

        rs_value = pair_return - benchmark_return

        rs_array[i] = min(max(float(rs_value), -0.5), 0.5)

    return rs_array

@njit(cache=True)
def _uo_numba(high, low, close, fast, medium, slow, fast_w, medium_w, slow_w, drift):
    n = len(high)
    uo = np.full(n, np.nan)
    bp = np.zeros(n)
    tr = np.zeros(n)

    for i in range(drift, n):
        pc = close[i - drift]

        bp[i] = close[i] - min(low[i], pc)
        tr[i] = max(high[i], pc) - min(low[i], pc)

    def calc_average(bp_sum, tr_sum):
        return bp_sum / tr_sum if tr_sum != 0 else 0.0

    start_idx = slow + drift - 1

    if start_idx >= n:
        return uo

    bp_sum_fast = np.sum(bp[start_idx - fast + 1:start_idx + 1])
    tr_sum_fast = np.sum(tr[start_idx - fast + 1:start_idx + 1])

    bp_sum_medium = np.sum(bp[start_idx - medium + 1:start_idx + 1])
    tr_sum_medium = np.sum(tr[start_idx - medium + 1:start_idx + 1])

    bp_sum_slow = np.sum(bp[start_idx - slow + 1:start_idx + 1])
    tr_sum_slow = np.sum(tr[start_idx - slow + 1:start_idx + 1])

    avg_fast = calc_average(bp_sum_fast, tr_sum_fast)
    avg_medium = calc_average(bp_sum_medium, tr_sum_medium)
    avg_slow = calc_average(bp_sum_slow, tr_sum_slow)

    uo[start_idx] = 100 * ((avg_fast * fast_w) + (avg_medium * medium_w) + (avg_slow * slow_w)) / (
            fast_w + medium_w + slow_w)

    for i in range(start_idx + 1, n):
        bp_sum_fast += bp[i] - bp[i - fast]
        tr_sum_fast += tr[i] - tr[i - fast]

        bp_sum_medium += bp[i] - bp[i - medium]
        tr_sum_medium += tr[i] - tr[i - medium]

        bp_sum_slow += bp[i] - bp[i - slow]
        tr_sum_slow += tr[i] - tr[i - slow]

        avg_fast = calc_average(bp_sum_fast, tr_sum_fast)
        avg_medium = calc_average(bp_sum_medium, tr_sum_medium)
        avg_slow = calc_average(bp_sum_slow, tr_sum_slow)

        uo[i] = 100 * ((avg_fast * fast_w) + (avg_medium * medium_w) + (avg_slow * slow_w)) / (
                fast_w + medium_w + slow_w)

    return uo


@njit(cache=True)
def kst_numba(
    close: np.ndarray,
    roc1_length: int = 5,
    roc2_length: int = 10,
    roc3_length: int = 15,
    roc4_length: int = 20,
    sma1_length: int = 3,
    sma2_length: int = 5,
    sma3_length: int = 7,
    sma4_length: int = 9
) -> np.ndarray:
    """Know Sure Thing - single pass, sliding-window SMAs/ROCs, no temp arrays.

    Default ROC periods 5/10/15/20 smoothed by SMA 3/5/7/9; returns the weighted sum.
    """
    n = len(close)
    kst = np.full(n, np.nan)

    start_idx1 = roc1_length + sma1_length - 1
    start_idx2 = roc2_length + sma2_length - 1
    start_idx3 = roc3_length + sma3_length - 1
    start_idx4 = roc4_length + sma4_length - 1

    valid_start = max(start_idx1, start_idx2, start_idx3, start_idx4)

    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0
    sum4 = 0.0

    min_roc_len = min(roc1_length, roc2_length, roc3_length, roc4_length)

    for i in range(min_roc_len, n):
        if i >= roc1_length:
            roc = ((close[i] / close[i - roc1_length]) - 1) * 100
            sum1 += roc
            if i >= roc1_length + sma1_length:
                old_roc = ((close[i - sma1_length] / close[i - sma1_length - roc1_length]) - 1) * 100
                sum1 -= old_roc

        if i >= roc2_length:
            roc = ((close[i] / close[i - roc2_length]) - 1) * 100
            sum2 += roc
            if i >= roc2_length + sma2_length:
                old_roc = ((close[i - sma2_length] / close[i - sma2_length - roc2_length]) - 1) * 100
                sum2 -= old_roc

        if i >= roc3_length:
            roc = ((close[i] / close[i - roc3_length]) - 1) * 100
            sum3 += roc
            if i >= roc3_length + sma3_length:
                old_roc = ((close[i - sma3_length] / close[i - sma3_length - roc3_length]) - 1) * 100
                sum3 -= old_roc

        if i >= roc4_length:
            roc = ((close[i] / close[i - roc4_length]) - 1) * 100
            sum4 += roc
            if i >= roc4_length + sma4_length:
                old_roc = ((close[i - sma4_length] / close[i - sma4_length - roc4_length]) - 1) * 100
                sum4 -= old_roc

        if i >= valid_start:
            rcma1 = sum1 / sma1_length
            rcma2 = sum2 / sma2_length
            rcma3 = sum3 / sma3_length
            rcma4 = sum4 / sma4_length

            kst[i] = rcma1 * 1 + rcma2 * 2 + rcma3 * 3 + rcma4 * 4

    return kst


def uo_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    config: Any
) -> np.ndarray:
    """Ultimate Oscillator over 7/14/28 periods, weights from the config dict.

    UO = 100 * (4*Avg7 + 2*Avg14 + Avg28) / 7, where AvgN = SUM(BP,N)/SUM(TR,N)
    uses the previous close. Source: tradingview.com/wiki/Ultimate_Oscillator_(UO).
    """
    return _uo_numba(
        high, low, close,
        config["fast"], config["medium"], config["slow"],
        config["fast_w"], config["medium_w"], config["slow_w"],
        config["drift"]
    )
