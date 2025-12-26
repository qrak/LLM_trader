import numpy as np
from numba import njit
from .utils import (
    f_ess, f_hp,
    calculate_correlation_matrix,
    calculate_spectral_components,
    smooth_power_spectrum,
    calculate_dominant_cycle
)

@njit(cache=True)
def apa_adaptive_eot_numba(closeprices, q1_=0.8, q2_=0.4, minlen=10, maxlen=48, avelen=3):
    masterdom = _auto_dom_imp(closeprices, minlen, maxlen, avelen)
    dcout = max(minlen, min(maxlen, int(round(masterdom))))

    qup = _eot(closeprices, dcout, q1_)
    qdn = _eot(closeprices, dcout, q2_)

    return qup, qdn

@njit(cache=True)
def _eot(closeprices, lpperiod, k):
    n = len(closeprices)
    pk = np.zeros(n)
    filt = f_ess(f_hp(closeprices, lpperiod), lpperiod)
    x = np.zeros(n)
    q = np.zeros(n)

    for i in range(1, n):
        pk[i] = np.maximum(abs(filt[i]), 0.99 * pk[i - 1])
        x[i] = filt[i] / pk[i] if pk[i] != 0 else 0
        q[i] = (x[i] + k) / (k * x[i] + 1) if x[i] != 0 else np.nan

    return q

@njit(cache=True)
def _auto_dom_imp(source, minlen, maxlen, avelen):
    """Simplified auto-dominant impulse using extracted utilities"""
    # Apply filtering
    filt = f_ess(f_hp(source, maxlen), minlen)
    
    # Calculate correlation matrix
    corr = calculate_correlation_matrix(filt, maxlen, avelen)
    
    # Calculate spectral components
    sqsum = calculate_spectral_components(corr, minlen, maxlen, avelen)
    
    # Smooth power spectrum
    r1 = smooth_power_spectrum(sqsum, minlen, maxlen)
    
    # Calculate dominant cycle
    return calculate_dominant_cycle(r1, minlen, maxlen, avelen)

@njit(cache=True)
def kurtosis_numba(arr, length):
    n = len(arr)
    kurtosis_values = np.full(n, np.nan)
    length_reciprocal = 1.0 / length

    for i in range(length - 1, n):
        window = arr[i - length + 1:i + 1]
        mean = np.sum(window) * length_reciprocal
        variance = np.sum((window - mean) ** 2) * length_reciprocal
        std_dev = np.sqrt(variance)

        kurtosis_sum = np.sum(((window - mean) / std_dev) ** 4)
        kurtosis_constant = (length * (length + 1)) / ((length - 1) * (length - 2) * (length - 3))
        kurtosis = kurtosis_constant * kurtosis_sum
        kurtosis -= 3 * (length - 1) / ((length - 2) * (length - 3))

        kurtosis_values[i] = kurtosis

    return kurtosis_values

@njit(cache=True)
def skew_numba(close, length=30):
    n = len(close)
    skew_values = np.full(n, np.nan)

    for i in range(length - 1, n):
        window = close[i - length + 1:i + 1]
        mean = np.sum(window) / length
        std_dev = np.sqrt(np.sum((window - mean) ** 2) / length)

        skew_sum = np.sum(((window - mean) / std_dev) ** 3)
        skew_values[i] = ((length * (length + 1)) / ((length - 1) * (length - 2) * (length - 3))) * skew_sum

    return skew_values

@njit(cache=True)
def stdev_numba(close, length=30, ddof=1):
    variance_values = variance_numba(close, length, ddof)
    return np.sqrt(variance_values)

@njit(cache=True)
def variance_numba(close, length=30, ddof=1):
    n = len(close)
    variance_values = np.full(n, np.nan)

    for i in range(length - 1, n):
        window = close[i - length + 1:i + 1]
        mean = np.sum(window) / length
        variance_values[i] = np.sum((window - mean) ** 2) / (length - ddof)

    return variance_values

@njit(cache=True)
def zscore_numba(close, length=30, std=1.0):
    n = len(close)
    zscore_values = np.full(n, np.nan)

    for i in range(length - 1, n):
        window = close[i - length + 1:i + 1]
        mean = np.sum(window) / length
        variance = np.sum((window - mean) ** 2) / (length - 1)
        stdev = np.sqrt(variance)

        if i >= length:
            zscore_values[i] = (close[i] - mean) / (std * stdev)

    return zscore_values

@njit(cache=True)
def mad_numba(close, length=30):
    n = len(close)
    mad_values = np.full(n, np.nan)

    for i in range(length - 1, n):
        window = close[i - length + 1:i + 1]
        mean = np.mean(window)
        mad_values[i] = np.mean(np.abs(window - mean))

    return mad_values

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
    n = len(close)
    entropy = np.full(n, np.nan)
    log_base = np.log(base)  # precompute log base

    for i in range(length, n):
        total = np.sum(close[i - length:i])
        p = close[i - length:i] / total
        ent = -np.sum(p * np.log(p) / log_base)
        entropy[i] = ent

    return entropy

@njit(cache=True)
def hurst_numba(ts: np.ndarray, max_lag: int = 20) -> np.ndarray:
    n = len(ts)
    hurst_values = np.full(n, np.nan, dtype=np.float64)
    
    # Start from index where we have enough data
    for i in range(max_lag + 2, n):
        # Use expanding window up to current position
        window = ts[:i+1]
        lags = np.arange(2, max_lag)
        tau = np.zeros(len(lags))

        # Calculate tau for each lag
        for j, lag in enumerate(lags):
            sum_diff_sq = 0.0
            count = 0
            for idx in range(lag, len(window)):
                diff = window[idx] - window[idx - lag]
                sum_diff_sq += diff * diff
                count += 1
            if count > 0:
                tau[j] = np.sqrt(sum_diff_sq / count)
            else:
                tau[j] = 0.0

        # Filter out zero values
        non_zero_tau = tau[tau > 0]
        if len(non_zero_tau) < 2:
            continue  # Skip calculation if not enough data

        # Calculate slope using valid values
        log_lags = np.log(lags[tau > 0])
        log_tau = np.log(non_zero_tau)
        
        # Linear regression using method of moments
        n_points = len(log_lags)
        sum_xy = np.sum(log_lags * log_tau)
        sum_x = np.sum(log_lags)
        sum_y = np.sum(log_tau)
        sum_x2 = np.sum(log_lags ** 2)
        
        denominator = n_points * sum_x2 - sum_x ** 2
        if denominator == 0:
            continue
            
        slope = (n_points * sum_xy - sum_x * sum_y) / denominator
        hurst_values[i] = slope

    return hurst_values

@njit(cache=True)
def linreg_numba(close, length=14, r=False):
    n = len(close)
    linreg = np.full(n, np.nan)

    x = np.arange(1, length + 1)
    x_sum = 0.5 * length * (length + 1)
    x2_sum = x_sum * (2 * length + 1) / 3
    divisor = length * x2_sum - x_sum * x_sum

    for i in range(length - 1, n):
        series = close[i - length + 1:i + 1]
        y_sum = np.sum(series)
        xy_sum = np.sum(x * series)

        m = (length * xy_sum - x_sum * y_sum) / divisor

        if r:
            y2_sum = np.sum(series * series)
            rn = length * xy_sum - x_sum * y_sum
            rd = np.sqrt(divisor * (length * y2_sum - y_sum * y_sum))
            linreg[i] = rn / rd
        else:
            linreg[i] = m

    return linreg

@njit(cache=True)
def calculate_eot_numba(close_prices, period=21, q1=0.8, q2=0.4):
    angle = 0.707 * 2 * np.pi / 100
    alpha1 = (np.cos(angle) + np.sin(angle) - 1) / np.cos(angle)
    a1 = np.exp(-1.414 * np.pi / period)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    n = len(close_prices)
    hp = np.zeros(n)
    filt = np.zeros(n)
    pk = np.zeros(n)
    x = np.zeros(n)
    quotient1 = np.full(n, np.nan)
    quotient2 = np.full(n, np.nan)

    for i in range(period, n):
        hp[i] = (1 - alpha1 / 2) ** 2 * (close_prices[i] - 2 * close_prices[i - 1] + close_prices[i - 2]) + \
                2 * (1 - alpha1) * hp[i - 1] - (1 - alpha1) ** 2 * hp[i - 2]
        filt[i] = c1 * (hp[i] + hp[i - 1]) / 2 + c2 * filt[i - 1] + c3 * filt[i - 2]
        if abs(filt[i]) > 0.991 * pk[i - 1]:
            pk[i] = abs(filt[i])
        else:
            pk[i] = 0.991 * pk[i - 1]
        x[i] = filt[i] / pk[i] if pk[i] != 0 else 0
        quotient1[i] = (x[i] + q1) / (q1 * x[i] + 1) if x[i] != 0 else np.nan
        quotient2[i] = (x[i] + q2) / (q2 * x[i] + 1) if x[i] != 0 else np.nan

    return quotient1, quotient2
