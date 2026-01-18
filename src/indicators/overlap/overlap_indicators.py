import numpy as np
from numba import njit

@njit(cache=True)
def ewma_numba(data, span):
    alpha = 2 / (span + 1)
    out = np.empty_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out

@njit(cache=True)
def ema_numba(arr, length):
    ema_arr = np.empty_like(arr)
    multiplier = 2 / (length + 1)

    first_valid = 0
    while first_valid < len(arr) and np.isnan(arr[first_valid]):
        first_valid += 1

    if first_valid == len(arr):
        return np.full_like(arr, np.nan)

    ema_arr[first_valid] = arr[first_valid]

    for i in range(first_valid + 1, len(arr)):
        if np.isnan(arr[i]):
            ema_arr[i] = ema_arr[i - 1]
        else:
            ema_arr[i] = ((arr[i] - ema_arr[i - 1]) * multiplier) + ema_arr[i - 1]

    ema_arr[:first_valid] = np.nan

    return ema_arr


@njit(cache=True)
def sma_numba(arr, length):
    n = len(arr)
    sma_values = np.full(n, np.nan, dtype=np.float64)

    if n < length:
        return sma_values

    # Check for NaNs in the first window
    # If using sliding window, NaNs can propagate if not careful.
    # Numba's sum propagates NaNs correctly (returns NaN).
    # But removing a NaN from sum is tricky (NaN - NaN is NaN).
    # For performance, we assume clean data or let NaN propagate.
    # If we need to handle NaNs robustly like pandas (ignoring them), it's O(N*L) or complex O(N).
    # Standard technical analysis libraries usually propagate NaN or assume filled data.
    # The existing implementation assumes standard behavior.

    window_sum = 0.0
    for i in range(length):
        window_sum += arr[i]

    sma_values[length - 1] = window_sum / length

    for i in range(length, n):
        window_sum += arr[i] - arr[i - length]
        sma_values[i] = window_sum / length

    return sma_values
