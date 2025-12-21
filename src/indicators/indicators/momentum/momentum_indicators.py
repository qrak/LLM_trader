from typing import Tuple
from dataclasses import dataclass

import numpy as np
from numba import njit

from src.indicators.indicators.overlap import ema_numba, ewma_numba


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
        high_max = np.max(high[i - period_k + 1:i + 1])
        low_min = np.min(low[i - period_k + 1:i + 1])

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

    for i in range(length - 1, n):
        highest_high = np.max(high[i - length + 1: i + 1])
        lowest_low = np.min(low[i - length + 1: i + 1])

        if highest_high != lowest_low:
            williams_r[i] = ((highest_high - close[i]) / (highest_high - lowest_low)) * -100

    return williams_r

@njit(cache=True)
def tsi_numba(close, long_length, short_length):
    n = len(close)
    tsi = np.full(n, np.nan)
    m = np.zeros(n)
    abs_m = np.zeros(n)
    ema1 = np.zeros(n)
    abs_ema1 = np.zeros(n)
    ema2 = np.zeros(n)
    abs_ema2 = np.zeros(n)

    for i in range(1, n):
        m[i] = close[i] - close[i - 1]
        abs_m[i] = abs(m[i])

    alpha_long = 2 / (long_length + 1)
    alpha_short = 2 / (short_length + 1)

    ema1[long_length] = np.mean(m[1:long_length + 1])
    abs_ema1[long_length] = np.mean(abs_m[1:long_length + 1])

    for i in range(long_length + 1, n):
        ema1[i] = ((m[i] - ema1[i - 1]) * alpha_long) + ema1[i - 1]
        abs_ema1[i] = ((abs_m[i] - abs_ema1[i - 1]) * alpha_long) + abs_ema1[i - 1]

    ema2[long_length + short_length - 1] = np.mean(ema1[long_length:long_length + short_length])
    abs_ema2[long_length + short_length - 1] = np.mean(abs_ema1[long_length:long_length + short_length])

    for i in range(long_length + short_length, n):
        ema2[i] = ((ema1[i] - ema2[i - 1]) * alpha_short) + ema2[i - 1]
        abs_ema2[i] = ((abs_ema1[i] - abs_ema2[i - 1]) * alpha_short) + abs_ema2[i - 1]

    for i in range(long_length + short_length, n):
        if abs_ema2[i] != 0:
            tsi[i] = (ema2[i] / abs_ema2[i]) * 100
        else:
            tsi[i] = tsi[i - 1]

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
    roc_long = ((close - np.roll(close, wl1)) / np.roll(close, wl1)) * 100
    roc_short = ((close - np.roll(close, wl2)) / np.roll(close, wl2)) * 100
    coppock_arr = roc_long + roc_short
    ewma_coppock = ewma_numba(coppock_arr, wma_length)
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
        return bp_sum / tr_sum if tr_sum != 0 else 0

    for i in range(slow + drift - 1, n):
        bp_sum_fast = np.sum(bp[i - fast + 1:i + 1])
        tr_sum_fast = np.sum(tr[i - fast + 1:i + 1])
        bp_sum_medium = np.sum(bp[i - medium + 1:i + 1])
        tr_sum_medium = np.sum(tr[i - medium + 1:i + 1])
        bp_sum_slow = np.sum(bp[i - slow + 1:i + 1])
        tr_sum_slow = np.sum(tr[i - slow + 1:i + 1])

        avg_fast = calc_average(bp_sum_fast, tr_sum_fast)
        avg_medium = calc_average(bp_sum_medium, tr_sum_medium)
        avg_slow = calc_average(bp_sum_slow, tr_sum_slow)

        uo[i] = 100 * ((avg_fast * fast_w) + (avg_medium * medium_w) + (avg_slow * slow_w)) / (
                fast_w + medium_w + slow_w)

    return uo


def uo_numba(
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray, 
    config: UltimateOscillatorConfig
) -> np.ndarray:
    """
    Ultimate Oscillator (UO) - Simple interface using config object.

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
        config (UltimateOscillatorConfig): Configuration object containing all parameters

    Returns:
        pd.Series: New feature generated.
    """
    return _uo_numba(
        high, low, close, 
        config.fast, config.medium, config.slow,
        config.fast_w, config.medium_w, config.slow_w,
        config.drift
    )