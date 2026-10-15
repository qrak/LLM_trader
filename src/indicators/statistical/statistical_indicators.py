"""Statistical Indicators module.

Provides functionality for indicators.statistical.statistical_indicators.py.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def kurtosis_numba(arr, length):
    """O(N) Sample Kurtosis using sliding window raw moments."""
    n = len(arr)
    kurtosis_values = np.full(n, np.nan)

    if n < length or length <= 3:
        return kurtosis_values

    length_reciprocal = 1.0 / length

    n_val = float(length)
    factor1 = (n_val + 1) * (n_val - 1) / (n_val * (n_val - 2) * (n_val - 3))
    factor2 = 3 * (n_val - 1) ** 2 / ((n_val - 2) * (n_val - 3))

    offset = arr[0]
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0

    for i in range(length):
        val = arr[i] - offset
        val2 = val * val
        s1 += val
        s2 += val2
        s3 += val2 * val
        s4 += val2 * val2

    mean = s1 * length_reciprocal
    sum_sq_diff = s2 - length * mean * mean
    if sum_sq_diff < 0:
        sum_sq_diff = 0.0

    variance = sum_sq_diff * length_reciprocal

    if variance > 1e-20:
        sum_fourth_diff = s4 - 4 * mean * s3 + 6 * mean * mean * s2 - 3 * length * mean * mean * mean * mean
        kurtosis_sum = sum_fourth_diff / (variance * variance)
        kurtosis = factor1 * kurtosis_sum - factor2
        kurtosis_values[length - 1] = kurtosis

    for i in range(length, n):
        old_val = arr[i - length] - offset
        new_val = arr[i] - offset

        if i % 1000 == 0:
            offset = arr[i - length + 1]
            window = arr[i - length + 1: i + 1] - offset
            s1 = 0.0
            s2 = 0.0
            s3 = 0.0
            s4 = 0.0
            for j in range(length):
                v = window[j]
                v2 = v * v
                s1 += v
                s2 += v2
                s3 += v2 * v
                s4 += v2 * v2
        else:
            old_val2 = old_val * old_val
            new_val2 = new_val * new_val

            s1 += new_val - old_val
            s2 += new_val2 - old_val2
            s3 += new_val2 * new_val - old_val2 * old_val
            s4 += new_val2 * new_val2 - old_val2 * old_val2

        mean = s1 * length_reciprocal
        sum_sq_diff = s2 - length * mean * mean
        if sum_sq_diff < 0:
            sum_sq_diff = 0.0

        variance = sum_sq_diff * length_reciprocal

        if variance > 1e-20:
            sum_fourth_diff = s4 - 4 * mean * s3 + 6 * mean * mean * s2 - 3 * length * mean * mean * mean * mean
            kurtosis_sum = sum_fourth_diff / (variance * variance)
            kurtosis = factor1 * kurtosis_sum - factor2
            kurtosis_values[i] = kurtosis

    return kurtosis_values

@njit(cache=True)
def skew_numba(close, length=30):
    """O(N) Sample Skewness using sliding window raw moments."""
    n = len(close)
    skew_values = np.full(n, np.nan)

    if n < length:
        return skew_values

    sqrt_n_n1 = np.sqrt(length * (length - 1))
    bias_factor = sqrt_n_n1 / (length * (length - 2))

    offset = close[0]
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0

    for i in range(length):
        val = close[i] - offset
        s1 += val
        s2 += val * val
        s3 += val * val * val

    mean = s1 / length
    sum_sq_diff = max(0.0, s2 - length * mean * mean)
    std_dev = np.sqrt(sum_sq_diff / length)
    sum_cubed_diff = s3 - 3 * mean * s2 + 2 * length * mean * mean * mean

    if std_dev > 1e-10:
        skew_values[length-1] = bias_factor * (sum_cubed_diff / (std_dev * std_dev * std_dev))
    else:
        skew_values[length-1] = 0.0

    for i in range(length, n):
        old_val = close[i - length] - offset
        new_val = close[i] - offset

        if i % 1000 == 0:
            window = close[i - length + 1 : i + 1] - offset
            s1 = np.sum(window)
            s2 = np.sum(window * window)
            s3 = np.sum(window * window * window)
        else:
            s1 = s1 - old_val + new_val
            s2 = s2 - old_val * old_val + new_val * new_val
            s3 = s3 - old_val * old_val * old_val + new_val * new_val * new_val

        mean = s1 / length
        sum_sq_diff = max(0.0, s2 - length * mean * mean)
        std_dev = np.sqrt(sum_sq_diff / length)
        sum_cubed_diff = s3 - 3 * mean * s2 + 2 * length * mean * mean * mean

        if std_dev > 1e-10:
            skew_values[i] = bias_factor * (sum_cubed_diff / (std_dev * std_dev * std_dev))
        else:
            skew_values[i] = 0.0

    return skew_values

@njit(cache=True)
def stdev_numba(close, length=30, ddof=1):
    variance_values = variance_numba(close, length, ddof)
    return np.sqrt(variance_values)

@njit(cache=True)
def variance_numba(close, length=30, ddof=1):
    """O(N) Sample Variance using sliding window sums."""
    n = len(close)
    variance_values = np.full(n, np.nan)

    if n < length:
        return variance_values

    offset = close[0]
    sum_x = 0.0
    sum_x2 = 0.0

    for i in range(length):
        val = close[i] - offset
        sum_x += val
        sum_x2 += val * val

    var = (sum_x2 - (sum_x * sum_x) / length) / (length - ddof)
    variance_values[length - 1] = max(0.0, var)

    for i in range(length, n):
        old_val = close[i - length] - offset
        new_val = close[i] - offset

        if i % 1000 == 0:
            window = close[i - length + 1 : i + 1] - offset
            sum_x = np.sum(window)
            sum_x2 = np.sum(window * window)
        else:
            sum_x += new_val - old_val
            sum_x2 += new_val * new_val - old_val * old_val

        var = (sum_x2 - (sum_x * sum_x) / length) / (length - ddof)
        variance_values[i] = max(0.0, var)

    return variance_values

@njit(cache=True)
def zscore_numba(close, length=30, std=1.0):
    """Rolling Z-Score (O(N) optimized)."""
    n = len(close)
    zscore_values = np.full(n, np.nan)

    if n < length:
        return zscore_values

    offset = close[0]

    sum_x = 0.0
    sum_x2 = 0.0

    for i in range(length):
        val = close[i] - offset
        sum_x += val
        sum_x2 += val * val

    if length > 0:
        mean = sum_x / length
        var = (sum_x2 - (sum_x * sum_x) / length) / (length - 1)
        stdev = np.sqrt(max(0.0, var))
        if stdev > 1e-10:
            zscore_values[length - 1] = ((close[length - 1] - offset) - mean) / (std * stdev)
        else:
            zscore_values[length - 1] = 0.0

    for i in range(length, n):
        old_val = close[i - length] - offset
        new_val = close[i] - offset

        if i % 1000 == 0:
            window = close[i - length + 1 : i + 1] - offset
            sum_x = np.sum(window)
            sum_x2 = np.sum(window * window)
        else:
            sum_x += new_val - old_val
            sum_x2 += new_val * new_val - old_val * old_val

        mean = sum_x / length
        var = (sum_x2 - (sum_x * sum_x) / length) / (length - 1)
        stdev = np.sqrt(max(0.0, var))

        if stdev > 1e-10:
            zscore_values[i] = ((close[i] - offset) - mean) / (std * stdev)
        else:
            zscore_values[i] = 0.0

    return zscore_values

@njit(cache=True)
def quantile_numba(close, length=30, q=0.5):
    n = len(close)
    quantile_values = np.full(n, np.nan)

    for i in range(length - 1, n):
        window = close[i - length + 1:i + 1]
        quantile_values[i] = np.quantile(window, q)

    return quantile_values

@njit(cache=True)
def entropy_numba(close, length=10, base=2.0):
    """
    O(N) Shannon entropy (default base 2) of the window's normalized prices
    p_i = x_i / sum(x): H = -sum(p_i * log(p_i)); a flat window yields
    log2(length) and a constant price scale factor cancels out.
    """
    n = len(close)
    entropy = np.full(n, np.nan)

    safe_close = np.copy(close)
    for k in range(n):
        safe_close[k] = max(1e-10, safe_close[k])

    x_ln_x = safe_close * np.log(safe_close)
    sum_x = 0.0
    sum_x_ln_x = 0.0

    if n >= length:
        for i in range(length):
            sum_x += safe_close[i]
            sum_x_ln_x += x_ln_x[i]

        term = np.log(sum_x) - (sum_x_ln_x / sum_x)
        entropy[length - 1] = max(0.0, term / np.log(base))

        for i in range(length, n):
            old_idx = i - length
            new_idx = i

            if i % 1000 == 0:
                 window = safe_close[i - length + 1 : i + 1]
                 window_ln = x_ln_x[i - length + 1 : i + 1]
                 sum_x = np.sum(window)
                 sum_x_ln_x = np.sum(window_ln)
            else:
                diff_x = safe_close[new_idx] - safe_close[old_idx]
                diff_x_ln_x = x_ln_x[new_idx] - x_ln_x[old_idx]
                sum_x += diff_x
                sum_x_ln_x += diff_x_ln_x

            if sum_x > 0:
                term = np.log(sum_x) - (sum_x_ln_x / sum_x)
                entropy[i] = max(0.0, term / np.log(base))
            else:
                entropy[i] = 0.0

    return entropy

@njit(cache=True)
def hurst_numba(ts: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """
    Hurst exponent from the log-log slope of the structure function
    tau(lag) = sqrt(mean((ts[j] - ts[j - lag])**2)) over lags 2..max_lag - 1.

    Expanding-window estimate: each value uses only the bars up to that index
    (first finite at max_lag + 2); ~0.5 random walk, >0.5 trending, <0.5
    mean-reverting.
    """
    n = len(ts)
    hurst_values = np.full(n, np.nan, dtype=np.float64)

    sum_diff_sq = np.zeros(max_lag)

    for i in range(n):
        for lag in range(2, max_lag):
            if i >= lag:
                diff = ts[i] - ts[i - lag]
                sum_diff_sq[lag] += diff * diff

        if i >= max_lag + 2:
            lags = np.arange(2, max_lag)
            tau = np.zeros(len(lags))

            for j, lag in enumerate(lags):
                count = i - lag + 1
                if count > 0:
                    tau[j] = np.sqrt(sum_diff_sq[lag] / count)
                else:
                    tau[j] = 0.0

            non_zero_tau = tau[tau > 0]
            if len(non_zero_tau) < 2:
                continue

            log_lags = np.log(lags[tau > 0])
            log_tau = np.log(non_zero_tau)

            n_points = len(log_lags)
            sum_xy = np.sum(log_lags * log_tau)
            sum_x = np.sum(log_lags)
            sum_y = np.sum(log_tau)
            sum_x2 = np.sum(log_lags ** 2)

            denominator = n_points * sum_x2 - sum_x ** 2
            if denominator != 0:
                slope = (n_points * sum_xy - sum_x * sum_y) / denominator
                hurst_values[i] = slope

    return hurst_values

@njit(cache=True)
def linreg_numba(close, length=14, r=False):
    """O(N) Linear Regression (slope or Pearson correlation)."""
    n = len(close)
    linreg = np.full(n, np.nan)

    x = np.arange(1, length + 1)
    x_sum = np.sum(x)
    x2_sum = np.sum(x * x)
    divisor = length * x2_sum - x_sum * x_sum

    offset = close[0]

    if n >= length:
        series = close[0:length] - offset
        y_sum = np.sum(series)
        xy_sum = np.sum(x * series)
        y2_sum = 0.0

        if r:
            y2_sum = np.sum(series * series)
            rn = length * xy_sum - x_sum * y_sum
            term1 = length * y2_sum - y_sum * y_sum
            rd = np.sqrt(divisor * max(0.0, term1))
            linreg[length - 1] = rn / rd if rd != 0 else 0.0
        else:
            m = (length * xy_sum - x_sum * y_sum) / divisor
            linreg[length - 1] = m

        for i in range(length, n):
            y_old = close[i - length] - offset
            y_new = close[i] - offset

            if i % 100 == 0:
                series = close[i - length + 1 : i + 1] - offset
                y_sum = np.sum(series)
                xy_sum = np.sum(x * series)
                if r:
                     y2_sum = np.sum(series * series)
            else:
                xy_sum = xy_sum - y_sum + length * y_new
                y_sum = y_sum - y_old + y_new
                if r:
                    y2_sum = y2_sum - y_old * y_old + y_new * y_new

            if r:
                rn = length * xy_sum - x_sum * y_sum
                term1 = length * y2_sum - y_sum * y_sum
                rd = np.sqrt(divisor * max(0.0, term1))
                linreg[i] = rn / rd if rd != 0 else 0.0
            else:
                m = (length * xy_sum - x_sum * y_sum) / divisor
                linreg[i] = m

    return linreg
