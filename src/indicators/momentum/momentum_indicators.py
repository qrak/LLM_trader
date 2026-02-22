from typing import Tuple, Any
from dataclasses import dataclass

import numpy as np
from numba import njit

from src.indicators.overlap import ema_numba


@dataclass
class UltimateOscillatorConfig:
    """Configuration for Ultimate Oscillator."""
    fast: int = 7
    medium: int = 14
    slow: int = 28
    fast_w: float = 4.0
    medium_w: float = 2.0
    slow_w: float = 1.0
    drift: int = 1


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
               signal_length: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        # Optimization: Avoid creating slice arrays in the loop
        start_idx = i - period_k + 1
        end_idx = i + 1

        # Manual max/min finding to avoid slice allocation
        high_max = high[start_idx]
        low_min = low[start_idx]

        for j in range(start_idx + 1, end_idx):
            val_h = high[j]
            val_l = low[j]
            if val_h > high_max:
                high_max = val_h
            if val_l < low_min:
                low_min = val_l

        if high_max != low_min:
            k_values[i] = 100 * (close[i] - low_min) / (high_max - low_min)

    smoothed_k = np.full(n, np.nan)
    for i in range(period_k + smooth_k - 2, n):
        smoothed_k[i] = np.mean(k_values[i - smooth_k + 1:i + 1])
        if i >= period_k + smooth_k + period_d - 3:
            d_values[i] = np.mean(smoothed_k[i - period_d + 1:i + 1])

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

    # Optimization: Manual max/min finding to avoid slice allocation in loop
    for i in range(length - 1, n):
        start_idx = i - length + 1
        end_idx = i + 1

        highest_high = high[start_idx]
        lowest_low = low[start_idx]

        # Inner loop over the window - efficient for small windows (default 14)
        for j in range(start_idx + 1, end_idx):
            h_val = high[j]
            l_val = low[j]
            if h_val > highest_high:
                highest_high = h_val
            if l_val < lowest_low:
                lowest_low = l_val

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

    # Pre-calculate alpha values
    alpha_long = 2.0 / (long_length + 1)
    alpha_short = 2.0 / (short_length + 1)

    # Calculate initial window for EMA1 (momentum)
    m_sum = 0.0
    abs_m_sum = 0.0

    # Calculate sum for initial EMA1
    # We sum m[1]...m[long_length]
    for i in range(1, long_length + 1):
        if i < n:
            val = close[i] - close[i - 1]
            m_sum += val
            abs_m_sum += abs(val)

    if n <= long_length:
        return tsi

    # Initial EMA1 values at index long_length
    curr_ema1 = m_sum / long_length
    curr_abs_ema1 = abs_m_sum / long_length

    # Now we need to calculate EMA2.
    # EMA2 is the EMA of EMA1.
    # Its initial value is the mean of EMA1 from long_length to long_length + short_length - 1

    # Accumulate sums for EMA2 initialization
    ema1_sum = curr_ema1
    abs_ema1_sum = curr_abs_ema1

    # Store previous EMA1 values
    prev_ema1 = curr_ema1
    prev_abs_ema1 = curr_abs_ema1

    # Calculate EMA1 for the window required to initialize EMA2
    # Range: long_length + 1 to long_length + short_length - 1
    start_ema2_init = long_length + 1
    end_ema2_init = long_length + short_length - 1

    if end_ema2_init >= n:
        # Not enough data for full initialization
        return tsi

    for i in range(start_ema2_init, end_ema2_init + 1):
        # Calculate m
        m = close[i] - close[i - 1]
        abs_m = abs(m)

        # Calculate new EMA1
        curr_ema1 = (m - prev_ema1) * alpha_long + prev_ema1
        curr_abs_ema1 = (abs_m - prev_abs_ema1) * alpha_long + prev_abs_ema1

        # Accumulate for EMA2 initialization
        ema1_sum += curr_ema1
        abs_ema1_sum += curr_abs_ema1

        # Update prev
        prev_ema1 = curr_ema1
        prev_abs_ema1 = curr_abs_ema1

    # Initial EMA2 values at index end_ema2_init
    curr_ema2 = ema1_sum / short_length
    curr_abs_ema2 = abs_ema1_sum / short_length

    # Calculate TSI for the first valid point
    if curr_abs_ema2 != 0:
        tsi[end_ema2_init] = (curr_ema2 / curr_abs_ema2) * 100.0
    else:
        # If it's the very first point and denominator is 0, we can't use previous value.
        # Default to 0.0 or keep NaN? Usually 0.0 is safer than NaN for indicators.
        tsi[end_ema2_init] = 0.0

    prev_ema2 = curr_ema2
    prev_abs_ema2 = curr_abs_ema2

    # Main loop for the rest of the data
    for i in range(end_ema2_init + 1, n):
        # Calculate m
        m = close[i] - close[i - 1]
        abs_m = abs(m)

        # Calculate new EMA1
        curr_ema1 = (m - prev_ema1) * alpha_long + prev_ema1
        curr_abs_ema1 = (abs_m - prev_abs_ema1) * alpha_long + prev_abs_ema1

        # Calculate new EMA2
        curr_ema2 = (curr_ema1 - prev_ema2) * alpha_short + prev_ema2
        curr_abs_ema2 = (curr_abs_ema1 - prev_abs_ema2) * alpha_short + prev_abs_ema2

        # Calculate TSI
        if curr_abs_ema2 != 0:
            tsi[i] = (curr_ema2 / curr_abs_ema2) * 100.0
        else:
            tsi[i] = tsi[i - 1]

        # Update prev
        prev_ema1 = curr_ema1
        prev_abs_ema1 = curr_abs_ema1
        prev_ema2 = curr_ema2
        prev_abs_ema2 = curr_abs_ema2

    return tsi

@njit(cache=True)
def rmi_numba(close, length, momentum_length):
    n = len(close)
    rmi = np.full(n, np.nan)

    momentum = np.zeros(n - momentum_length)
    for i in range(len(momentum)):
        momentum[i] = close[i + momentum_length] - close[i]

    up = np.maximum(momentum, 0)
    down = np.maximum(-momentum, 0)

    for i in range(length - 1, len(momentum)):
        avg_up = np.mean(up[i - length + 1:i + 1])
        avg_down = np.mean(down[i - length + 1:i + 1])

        if avg_down == 0:
            rmi[i + momentum_length] = 100
        else:
            rs = avg_up / avg_down
            rmi[i + momentum_length] = 100 - (100 / (1 + rs))

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
    # Use ema_numba to handle NaNs correctly and avoid lookahead bias from np.roll
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
        if np.isnan(pair_close[i]) or np.isnan(benchmark_close[i]) or benchmark_close[i] == 0:
            rs_array[i] = 0.0
            continue

        # Calculate percentage changes
        pair_return = np.log(pair_close[i] / pair_close[i - window])
        benchmark_return = np.log(benchmark_close[i] / benchmark_close[i - window])

        # Calculate relative strength
        rs_value = pair_return - benchmark_return

        # Cap the value to prevent extreme scores
        rs_array[i] = min(max(float(rs_value), -0.5), 0.5)  # Cap at ±0.5

    return rs_array

# Optimized with sliding window sum. ~4.5x speedup for n=100k.
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

    # Helper function to calculate average
    def calc_average(bp_sum, tr_sum):
        return bp_sum / tr_sum if tr_sum != 0 else 0.0

    start_idx = slow + drift - 1

    if start_idx >= n:
        return uo

    # Initialize sums for the first window
    bp_sum_fast = np.sum(bp[start_idx - fast + 1:start_idx + 1])
    tr_sum_fast = np.sum(tr[start_idx - fast + 1:start_idx + 1])

    bp_sum_medium = np.sum(bp[start_idx - medium + 1:start_idx + 1])
    tr_sum_medium = np.sum(tr[start_idx - medium + 1:start_idx + 1])

    bp_sum_slow = np.sum(bp[start_idx - slow + 1:start_idx + 1])
    tr_sum_slow = np.sum(tr[start_idx - slow + 1:start_idx + 1])

    # Calculate UO for the first window
    avg_fast = calc_average(bp_sum_fast, tr_sum_fast)
    avg_medium = calc_average(bp_sum_medium, tr_sum_medium)
    avg_slow = calc_average(bp_sum_slow, tr_sum_slow)

    uo[start_idx] = 100 * ((avg_fast * fast_w) + (avg_medium * medium_w) + (avg_slow * slow_w)) / (
            fast_w + medium_w + slow_w)

    # Use sliding window for the rest
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
    """
    Know Sure Thing (KST) indicator - Optimized single-pass implementation.

    Computes ROC for 4 periods, smooths each with SMA, and combines with weights.
    Uses sliding window sums to calculate SMAs and ROCs on-the-fly, avoiding
    intermediate array allocations.

    Args:
        close: Close prices
        roc1_length: First ROC period (default 5)
        roc2_length: Second ROC period (default 10)
        roc3_length: Third ROC period (default 15)
        roc4_length: Fourth ROC period (default 20)
        sma1_length: SMA period for first ROC (default 3)
        sma2_length: SMA period for second ROC (default 5)
        sma3_length: SMA period for third ROC (default 7)
        sma4_length: SMA period for fourth ROC (default 9)

    Returns:
        KST values (weighted sum of smoothed ROCs)
    """
    n = len(close)
    kst = np.full(n, np.nan)

    # Calculate validity start indices for each component
    # A component is valid when we have enough data for ROC + SMA window
    start_idx1 = roc1_length + sma1_length - 1
    start_idx2 = roc2_length + sma2_length - 1
    start_idx3 = roc3_length + sma3_length - 1
    start_idx4 = roc4_length + sma4_length - 1

    # KST is valid when all components are valid
    valid_start = max(start_idx1, start_idx2, start_idx3, start_idx4)

    # Running sums for SMAs
    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0
    sum4 = 0.0

    # Minimum index to start processing to avoid negative indexing
    min_roc_len = min(roc1_length, roc2_length, roc3_length, roc4_length)

    for i in range(min_roc_len, n):
        # Component 1
        if i >= roc1_length:
            roc = ((close[i] / close[i - roc1_length]) - 1) * 100
            sum1 += roc
            if i >= roc1_length + sma1_length:
                old_roc = ((close[i - sma1_length] / close[i - sma1_length - roc1_length]) - 1) * 100
                sum1 -= old_roc

        # Component 2
        if i >= roc2_length:
            roc = ((close[i] / close[i - roc2_length]) - 1) * 100
            sum2 += roc
            if i >= roc2_length + sma2_length:
                old_roc = ((close[i - sma2_length] / close[i - sma2_length - roc2_length]) - 1) * 100
                sum2 -= old_roc

        # Component 3
        if i >= roc3_length:
            roc = ((close[i] / close[i - roc3_length]) - 1) * 100
            sum3 += roc
            if i >= roc3_length + sma3_length:
                old_roc = ((close[i - sma3_length] / close[i - sma3_length - roc3_length]) - 1) * 100
                sum3 -= old_roc

        # Component 4
        if i >= roc4_length:
            roc = ((close[i] / close[i - roc4_length]) - 1) * 100
            sum4 += roc
            if i >= roc4_length + sma4_length:
                old_roc = ((close[i - sma4_length] / close[i - sma4_length - roc4_length]) - 1) * 100
                sum4 -= old_roc

        # Calculate KST if all components are valid
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
    """
    Ultimate Oscillator (UO) - Simple interface using config object or dictionary.

    Ultimate Oscillator using three timeframes: 7, 14 and 28 periods.

    Sources:
        https://www.tradingview.com/wiki/Ultimate_Oscillator_(UO)

    Calculation:
        Default Inputs:
            fast=7, medium=14, slow=28, fast_w=4.0, medium_w=2.0, slow_w=1.0
        SUM = Summation
        BP = Buying Pressure = close - low(min(low, previous_close))
        TR = True Range = high(max(high, previous_close)) - low(min(low, previous_close))
        Average7 = SUM(BP, 7) / SUM(TR, 7)
        Average14 = SUM(BP, 14) / SUM(TR, 14)
        Average28 = SUM(BP, 28) / SUM(TR, 28)

        UO = 100 * (4 * Average7 + 2 * Average14 + Average28) / (4 + 2 + 1)

    Args:
        high (pd.Series): Series of 'high's
        low (pd.Series): Series of 'low's
        close (pd.Series): Series of 'close's
        config (Any): Configuration object or dict containing all parameters

    Returns:
        pd.Series: New feature generated.
    """
    if isinstance(config, dict):
        return _uo_numba(
            high, low, close,
            config['fast'], config['medium'], config['slow'],
            config['fast_w'], config['medium_w'], config['slow_w'],
            config['drift']
        )
    else:
        return _uo_numba(
            high, low, close,
            config.fast, config.medium, config.slow,
            config.fast_w, config.medium_w, config.slow_w,
            config.drift
        )
