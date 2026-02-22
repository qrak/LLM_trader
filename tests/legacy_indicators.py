import numpy as np
from numba import njit

@njit(cache=True)
def entropy_numba_legacy(close, length=10, base=2.0):
    n = len(close)
    entropy = np.full(n, np.nan)
    log_base = np.log(base)  # precompute log base

    for i in range(length, n):
        # LEGACY BEHAVIOR: i-length to i (excludes i) -> Lag of 1
        total = np.sum(close[i - length:i])
        p = close[i - length:i] / total
        ent = -np.sum(p * np.log(p) / log_base)
        entropy[i] = ent

    return entropy

@njit(cache=True)
def linreg_numba_legacy(close, length=14, r=False):
    n = len(close)
    linreg = np.full(n, np.nan)

    # Pre-compute constant sums (x = 1, 2, ..., length)
    x_sum = 0.5 * length * (length + 1)
    x2_sum = x_sum * (2 * length + 1) / 3
    divisor = length * x2_sum - x_sum * x_sum

    # Initialize first window
    if n >= length:
        x = np.arange(1, length + 1)
        series = close[0:length]
        y_sum = np.sum(series)
        xy_sum = np.sum(x * series)

        m = (length * xy_sum - x_sum * y_sum) / divisor

        if r:
            y2_sum = np.sum(series * series)
            rn = length * xy_sum - x_sum * y_sum
            rd = np.sqrt(divisor * (length * y2_sum - y_sum * y_sum))
            linreg[length - 1] = rn / rd if rd != 0 else 0.0
        else:
            linreg[length - 1] = m

        # Slide window for remaining positions
        for i in range(length, n):
            y_old = close[i - length]
            y_new = close[i]
            
            # Update sums: CRITICAL ORDER - update xy_sum before y_sum
            xy_sum = xy_sum + length * y_new - y_sum
            y_sum = y_sum - y_old + y_new

            m = (length * xy_sum - x_sum * y_sum) / divisor

            if r:
                # Recalculate y2_sum for correlation coefficient
                series = close[i - length + 1:i + 1]
                y2_sum = np.sum(series * series)
                rn = length * xy_sum - x_sum * y_sum
                rd = np.sqrt(divisor * (length * y2_sum - y_sum * y_sum))
                linreg[i] = rn / rd if rd != 0 else 0.0
            else:
                linreg[i] = m


    return linreg

@njit(cache=True)
def atr_wma_numba_legacy(high, low, close, length=14):
    n = len(high)
    atr = np.empty(n)
    tr = np.empty(n)
    atr[:length] = np.nan
    tr[0] = 0

    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    # Legacy WMA was likely checking loop O(N*L) or unoptimized
    # Simulating simple WMA loop
    weight_sum = length * (length + 1) / 2
    
    for i in range(length, n):
        # Window: tr[i-length+1] ... tr[i]
        wma_sum = 0.0
        for j in range(length):
            # weight 1..length
            idx = i - length + 1 + j
            wma_sum += tr[idx] * (j + 1)
            
        atr[i] = wma_sum / weight_sum
        
    return atr

@njit(cache=True)
def twap_numba_legacy(high, low, close, length):
    n = len(high)
    twap = np.full(n, np.nan)

    # Legacy O(N*L) implementation
    for i in range(length - 1, n):
        window_sum = 0.0
        for j in range(length):
            idx = i - length + 1 + j
            tp = (high[idx] + low[idx] + close[idx]) / 3
            window_sum += tp
        twap[i] = window_sum / length

    return twap


@njit(cache=True)
def choppiness_index_numba_legacy(high, low, close, length=14):
    # Legacy O(N*L) implementation (inner loop for TR sum)
    n = len(high)
    ci = np.full(n, np.nan)
    
    if n <= length:
        return ci

    for i in range(length, n):
        # Calculate True Range for the period (O(L))
        true_range_sum = 0.0
        for j in range(i - length + 1, i + 1):
            if j > 0:
                tr = max(
                    high[j] - low[j],
                    abs(high[j] - close[j - 1]),
                    abs(low[j] - close[j - 1])
                )
                true_range_sum += tr
        
        period_high = np.max(high[i - length + 1:i + 1])
        period_low = np.min(low[i - length + 1:i + 1])
        
        range_hl = period_high - period_low
        
        if range_hl > 0 and true_range_sum > 0:
            ci[i] = 100.0 * np.log10(true_range_sum / range_hl) / np.log10(length)
        else:
            ci[i] = 50.0
    
    return ci

@njit(cache=True)
def bollinger_bands_numba_legacy(close, length, num_std_dev):
    n = len(close)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    middle_band = np.full(n, np.nan)

    for i in range(length - 1, n):
        window = close[i - length + 1:i + 1]
        mean = np.mean(window)
        std = np.sqrt(np.sum((window - mean) ** 2) / len(window))
        
        upper_band[i] = mean + (std * num_std_dev)
        middle_band[i] = mean
        lower_band[i] = mean - (std * num_std_dev)

    return upper_band, middle_band, lower_band

@njit(cache=True)
def variance_numba_legacy(close, length=30, ddof=1):
    n = len(close)
    variance_values = np.full(n, np.nan)

    if n < length:
        return variance_values

    for i in range(length - 1, n):
        window = close[i - length + 1:i + 1] - offset  # noqa: F821
        mean = np.mean(window)
        # var = sum((x - mean)^2) / (N - ddof)
        var_sum = np.sum((window - mean) ** 2)
        variance_values[i] = var_sum / (length - ddof)

    return variance_values

@njit(cache=True)
def uo_numba_legacy(high, low, close, fast=7, medium=14, slow=28, fast_w=4.0, medium_w=2.0, slow_w=1.0, drift=1):
    n = len(high)
    uo = np.full(n, np.nan)
    bp = np.zeros(n)
    tr = np.zeros(n)

    for i in range(drift, n):
        pc = close[i - drift]
        bp[i] = close[i] - min(low[i], pc)
        tr[i] = max(high[i], pc) - min(low[i], pc)

    for i in range(slow + drift - 1, n):
        # Naive sums O(L)
        bp_sum_fast = np.sum(bp[i - fast + 1:i + 1])
        tr_sum_fast = np.sum(tr[i - fast + 1:i + 1])
        bp_sum_medium = np.sum(bp[i - medium + 1:i + 1])
        tr_sum_medium = np.sum(tr[i - medium + 1:i + 1])
        bp_sum_slow = np.sum(bp[i - slow + 1:i + 1])
        tr_sum_slow = np.sum(tr[i - slow + 1:i + 1])

        avg_fast = bp_sum_fast / tr_sum_fast if tr_sum_fast != 0 else 0
        avg_medium = bp_sum_medium / tr_sum_medium if tr_sum_medium != 0 else 0
        avg_slow = bp_sum_slow / tr_sum_slow if tr_sum_slow != 0 else 0

        uo[i] = 100 * ((avg_fast * fast_w) + (avg_medium * medium_w) + (avg_slow * slow_w)) / (
                fast_w + medium_w + slow_w)

    return uo

@njit(cache=True)
def skew_numba_legacy(close, length=30):
    n = len(close)
    skew_values = np.full(n, np.nan)

    for i in range(length - 1, n):
        window = close[i - length + 1:i + 1]
        mean = np.sum(window) / length
        std_dev = np.sqrt(np.sum((window - mean) ** 2) / length)

        skew_sum = np.sum(((window - mean) / std_dev) ** 3)
        skew_values[i] = ((length * (length + 1)) / ((length - 1) * (length - 2) * (length - 3))) * skew_sum

    return skew_values
