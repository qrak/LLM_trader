import numpy as np
from numba import njit
from src.indicators.overlap import sma_numba
from src.indicators.statistical import stdev_numba


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
    # Replaced custom sliding window implementation with optimized shared utilities
    # to adhere to DRY principles and ensure consistent performance across indicators.
    middle_band = sma_numba(close, length)
    std_dev = stdev_numba(close, length)

    n = len(close)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)

    for i in range(length - 1, n):
        dev = std_dev[i] * num_std_dev
        upper_band[i] = middle_band[i] + dev
        lower_band[i] = middle_band[i] - dev

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


@njit(cache=True)
def donchian_channels_numba(high, low, length=20):
    """Calculate Donchian Channels
    
    Args:
        high: High prices array
        low: Low prices array  
        length: Period for calculation (default: 20)
        
    Returns:
        Tuple of (upper_channel, middle_channel, lower_channel)
    """
    n = len(high)
    
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    middle_channel = np.full(n, np.nan)
    
    # Calculate rolling max and min
    for i in range(length - 1, n):
        upper_channel[i] = np.max(high[i - length + 1:i + 1])
        lower_channel[i] = np.min(low[i - length + 1:i + 1])
        middle_channel[i] = (upper_channel[i] + lower_channel[i]) / 2.0
        
    return upper_channel, middle_channel, lower_channel


@njit(cache=True)
def keltner_channels_numba(high, low, close, length=20, multiplier=2.0, mamode='ema'):
    """Calculate Keltner Channels"""
    n = len(close)
    
    # Calculate middle line (EMA of close prices)
    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    # Calculate ATR
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    
    # Calculate EMA for middle line
    alpha = 2.0 / (length + 1)
    middle[length - 1] = np.mean(close[:length])
    for i in range(length, n):
        middle[i] = alpha * close[i] + (1 - alpha) * middle[i - 1]
    
    # Calculate ATR EMA
    atr_ema = np.full(n, np.nan)
    atr_ema[length - 1] = np.mean(tr[1:length])
    for i in range(length, n):
        atr_ema[i] = alpha * tr[i] + (1 - alpha) * atr_ema[i - 1]
    
    # Calculate upper and lower bands
    for i in range(length - 1, n):
        if not np.isnan(middle[i]) and not np.isnan(atr_ema[i]):
            upper[i] = middle[i] + multiplier * atr_ema[i]
            lower[i] = middle[i] - multiplier * atr_ema[i]
    
    return upper, middle, lower


@njit(cache=True)
def choppiness_index_numba(high, low, close, length=14):
    """Calculate Choppiness Index (CI)
    
    The Choppiness Index measures market choppiness vs trending behavior.
    - Values > 61.8: Market is choppy/ranging (low directional movement)
    - Values < 38.2: Market is trending (strong directional movement)
    - Values between: Transitional state
    
    Args:
        high: High prices array
        low: Low prices array
        close: Close prices array
        length: Period for calculation (default: 14)
        
    Returns:
        Choppiness Index array (0-100 scale)
    """
    n = len(high)
    ci = np.full(n, np.nan)
    
    if n <= length:
        return ci

    # Pre-calculate True Range for the whole series
    # tr[0] is 0 because there is no previous close, and the original loop skipped j=0
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )

    # Initial sum for the first window (at i = length)
    # The window indices are from 1 to length (inclusive)
    current_tr_sum = 0.0
    for i in range(1, length + 1):
        current_tr_sum += tr[i]

    for i in range(length, n):
        # Calculate highest high and lowest low over the period
        # Note: Optimization opportunity: monotonic queue for O(1) min/max
        period_high = np.max(high[i - length + 1:i + 1])
        period_low = np.min(low[i - length + 1:i + 1])
        
        # Calculate Choppiness Index
        # CI = 100 * log10(sum(TR) / (highest_high - lowest_low)) / log10(length)
        range_hl = period_high - period_low
        
        if range_hl > 0 and current_tr_sum > 0:
            ci[i] = 100.0 * np.log10(current_tr_sum / range_hl) / np.log10(length)
        else:
            ci[i] = 50.0  # Neutral value when range is zero

        # Update sum for next iteration
        if i < n - 1:
            current_tr_sum += tr[i + 1] - tr[i - length + 1]
    
    return ci
