import numpy as np
from numba import njit


@njit(cache=True)
def atr_numba(high, low, close, length=14, mamode='rma', percent=False):
    n = len(high)
    atr = np.empty(n)
    tr = np.empty(n)

    atr[:length] = np.nan
    tr[0] = 0

    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    if np.any(np.isnan(tr)) or np.any(np.isnan(high)) or np.any(np.isnan(low)) or np.any(np.isnan(close)):
        return np.full(n, np.inf)

    if mamode == 'ema':
        atr[length - 1] = np.mean(tr[1:length])
        alpha = 2 / (length + 1)
        for i in range(length, n):
            atr[i] = (1 - alpha) * atr[i - 1] + alpha * tr[i]
    elif mamode == 'sma':
        sum_tr = np.sum(tr[1:length])
        for i in range(length, n):
            atr[i] = sum_tr / length
            sum_tr += tr[i] - tr[i - length + 1]
    elif mamode == 'wma':
        weights = np.arange(1, length + 1).astype(np.float64)
        weight_sum = np.sum(weights)
        for i in range(length, n):
            atr[i] = np.dot(tr[i - length + 1:i + 1], weights) / weight_sum
    else:
        atr[length - 1] = np.mean(tr[1:length])
        for i in range(length, n):
            atr[i] = (atr[i - 1] * (length - 1) + tr[i]) / length

    if percent:
        atr[length:] *= 100 / close[length:]

    return atr

@njit(cache=True)
def bollinger_bands_numba(close, length, num_std_dev):
    n = len(close)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    middle_band = np.full(n, np.nan)

    for i in range(length - 1, n):
        window = close[i - length + 1:i + 1]
        mean = np.mean(window)
        std = np.sqrt(np.sum((window - mean) ** 2) / (len(window) - 0))
        upper_band[i] = mean + (std * num_std_dev)
        middle_band[i] = mean
        lower_band[i] = mean - (std * num_std_dev)

    return upper_band, middle_band, lower_band

@njit(cache=True)
def chandelier_exit_numba(high, low, close, length, multiplier, mamode='rma'):
    n = len(close)
    atr_values = atr_numba(high, low, close, length, mamode)

    chandelier_exit_long = np.empty(n, dtype=np.float64)
    chandelier_exit_short = np.empty(n, dtype=np.float64)

    chandelier_exit_long[0:length] = 0
    chandelier_exit_short[0:length] = 0

    for i in range(length - 1, n):
        period_high = np.max(high[i - length + 1:i + 1])
        period_low = np.min(low[i - length + 1:i + 1])

        # Calculate the chandelier exits
        chandelier_exit_long[i] = period_high - atr_values[i] * multiplier
        chandelier_exit_short[i] = period_low + atr_values[i] * multiplier

    return chandelier_exit_long, chandelier_exit_short


@njit(cache=True)
def ebsw_numba(close, length=40, bars=10):
    n = len(close)
    ebsw = np.full(n, np.nan)

    last_close = last_hp = 0
    filter_hist = np.zeros(2)

    for i in range(length, n):
        alpha1 = (1 - np.sin(np.pi * 360 / length)) / np.cos(np.pi * 360 / length)
        hp = 0.5 * (1 + alpha1) * (close[i] - last_close) + alpha1 * last_hp

        a1 = np.exp(-np.sqrt(2) * np.pi / bars)
        b1 = 2 * a1 * np.cos(np.sqrt(2) * np.pi * 180 / bars)
        c2 = b1
        c3 = -1 * a1 * a1
        c1 = 1 - c2 - c3
        filt = c1 * (hp + last_hp) / 2 + c2 * filter_hist[1] + c3 * filter_hist[0]

        wave = (filt + filter_hist[1] + filter_hist[0]) / 3
        pwr = (filt * filt + filter_hist[1] * filter_hist[1] + filter_hist[0] * filter_hist[0]) / 3

        wave = wave / np.sqrt(pwr)

        filter_hist[0] = filter_hist[1]
        filter_hist[1] = filt
        last_hp = hp
        last_close = close[i]
        ebsw[i] = wave

    return ebsw


@njit(cache=True)
def vhf_numba(close, length=28, drift=1):
    n = len(close)
    if n < length:
        return np.full(n, np.nan)

    vhf = np.full(n, np.nan)

    for i in range(length - 1 + drift, n):
        hcp = np.max(close[i - length + 1:i + 1:drift])
        lcp = np.min(close[i - length + 1:i + 1:drift])

        sliced_close = close[i - length + 1:i + 1:drift]

        # Manually compute the differences
        diff = np.abs(sliced_close[1:] - sliced_close[:-1])
        sum_diff = np.sum(diff)

        # Handle division by zero
        if sum_diff != 0:
            vhf[i] = np.abs(hcp - lcp) / sum_diff
        else:
            vhf[i] = 0

    return vhf
