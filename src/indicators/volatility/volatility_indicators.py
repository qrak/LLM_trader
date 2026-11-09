
import math

import numpy as np
from numba import njit

from src.indicators.overlap import ema_numba, sma_numba
from src.indicators.statistical import stdev_numba
from src.indicators.support_resistance import support_resistance_numba


@njit(cache=True)
def atr_numba(high, low, close, length=14, mamode="rma", percent=False):

    n = len(high)
    atr = np.full(n, np.nan)

    if n < length:
        return atr

    tr = np.zeros(n)

    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    if mamode == "ema":
        atr[length - 1] = np.mean(tr[0:length])
        alpha = 2 / (length + 1)
        for i in range(length, n):
            atr[i] = (1 - alpha) * atr[i - 1] + alpha * tr[i]

    elif mamode == "sma":
        sum_tr = 0.0
        for i in range(length):
            sum_tr += tr[i]
        atr[length - 1] = sum_tr / length
        for i in range(length, n):
            sum_tr += tr[i] - tr[i - length]
            atr[i] = sum_tr / length

    elif mamode == "wma":
        weight_sum = length * (length + 1) / 2
        current_sum = 0.0
        current_numerator = 0.0
        for j in range(length):
            val = tr[j]
            current_sum += val
            current_numerator += val * (j + 1)
        atr[length - 1] = current_numerator / weight_sum
        for i in range(length, n):
            current_numerator += length * tr[i] - current_sum
            current_sum += tr[i] - tr[i - length]
            atr[i] = current_numerator / weight_sum

    else:
        atr[length - 1] = np.mean(tr[0:length])
        for i in range(length, n):
            atr[i] = (atr[i - 1] * (length - 1) + tr[i]) / length

    if percent:
        atr[length - 1:] *= 100 / close[length - 1:]

    return atr


@njit(cache=True)
def bollinger_bands_numba(close, length, num_std_dev):
    middle_band = sma_numba(close, length)
    std_dev = stdev_numba(close, length, ddof=0)

    n = len(close)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)

    for i in range(length - 1, n):
        dev = std_dev[i] * num_std_dev
        upper_band[i] = middle_band[i] + dev
        lower_band[i] = middle_band[i] - dev

    return upper_band, middle_band, lower_band


@njit(cache=True)
def chandelier_exit_numba(high, low, close, length, multiplier, mamode="rma"):
    n = len(close)
    atr_values = atr_numba(high, low, close, length, mamode)

    chandelier_exit_long = np.empty(n, dtype=np.float64)
    chandelier_exit_short = np.empty(n, dtype=np.float64)

    chandelier_exit_long[:] = np.nan
    chandelier_exit_short[:] = np.nan

    for i in range(length, n):
        if not math.isnan(atr_values[i]):
            period_high = np.max(high[i - length + 1:i + 1])
            period_low = np.min(low[i - length + 1:i + 1])

            chandelier_exit_long[i] = period_high - atr_values[i] * multiplier
            chandelier_exit_short[i] = period_low + atr_values[i] * multiplier

    return chandelier_exit_long, chandelier_exit_short


@njit(cache=True)
def donchian_channels_numba(high, low, length=20):

    lower_channel, upper_channel = support_resistance_numba(high, low, length)
    middle_channel = (upper_channel + lower_channel) / 2.0

    return upper_channel, middle_channel, lower_channel


@njit(cache=True)
def keltner_channels_numba(high, low, close, length=20, multiplier=2.0, mamode="ema", atr_mamode="ema"):

    n = len(close)

    if mamode == "sma":
        middle = sma_numba(close, length)
    else:
        middle = ema_numba(close, length)

    atr_val = atr_numba(high, low, close, length, mamode=atr_mamode)

    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    for i in range(n):
        if not math.isnan(middle[i]) and not math.isnan(atr_val[i]):
            upper[i] = middle[i] + multiplier * atr_val[i]
            lower[i] = middle[i] - multiplier * atr_val[i]

    return upper, middle, lower


@njit(cache=True)
def choppiness_index_numba(high, low, close, length=14):

    n = len(high)
    ci = np.full(n, np.nan)

    if n <= length:
        return ci

    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )

    current_tr_sum = 0.0
    for i in range(1, length + 1):
        current_tr_sum += tr[i]

    for i in range(length, n):
        period_high = high[i - length + 1]
        period_low = low[i - length + 1]
        for j in range(i - length + 2, i + 1):
            period_high = max(period_high, high[j])
            period_low = min(period_low, low[j])

        range_hl = period_high - period_low

        if range_hl > 0 and current_tr_sum > 0:
            ci[i] = 100.0 * np.log10(current_tr_sum / range_hl) / np.log10(length)
        else:
            ci[i] = 50.0

        if i < n - 1:
            current_tr_sum += tr[i + 1] - tr[i - length + 1]

    return ci
