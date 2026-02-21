import numpy as np
import time
import pytest
from src.indicators.statistical.statistical_indicators import hurst_numba

def hurst_reference(ts, max_lag=20):
    n = len(ts)
    hurst_values = np.full(n, np.nan)

    for i in range(max_lag + 2, n):
        # Window is ts[:i+1]
        # We need to calculate tau for lags 2..max_lag
        lags = np.arange(2, max_lag)
        tau = np.zeros(len(lags))

        for j, lag in enumerate(lags):
            # Calculate mean of squared differences
            # diffs = ts[lag:i+1] - ts[0:i+1-lag]
            # But wait, let's match exact indexing of original:
            # for idx in range(lag, len(window)):
            #    diff = window[idx] - window[idx - lag]
            # window[idx] is ts[idx]
            # So it is ts[lag] - ts[0], ..., ts[i] - ts[i-lag]

            diffs = ts[lag:i+1] - ts[0:i+1-lag]
            sum_sq = np.sum(diffs**2)
            count = len(diffs)

            if count > 0:
                tau[j] = np.sqrt(sum_sq / count)
            else:
                tau[j] = 0

        # Regression
        valid = tau > 0
        if np.sum(valid) < 2:
            continue

        x = np.log(lags[valid])
        y = np.log(tau[valid])

        # Manual regression to match method of moments exactly
        n_points = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x**2)

        denom = n_points * sum_x2 - sum_x**2
        if denom == 0:
            continue

        slope = (n_points * sum_xy - sum_x * sum_y) / denom
        hurst_values[i] = slope

    return hurst_values

def test_hurst_correctness():
    """Verify that the optimized Hurst calculation matches the reference implementation."""
    print("Verifying Hurst calculation...")
    np.random.seed(42)
    N = 500
    data = np.random.randn(N)

    print("Running reference implementation...")
    start = time.time()
    ref_values = hurst_reference(data)
    print(f"Reference took {time.time() - start:.4f}s")

    print("Running numba implementation...")
    # Run once to compile
    hurst_numba(np.random.randn(100))
    start = time.time()
    numba_values = hurst_numba(data)
    print(f"Numba took {time.time() - start:.4f}s")

    # Compare
    # NaNs should match
    mask = ~np.isnan(ref_values)

    # Check if NaN mask matches
    nan_mismatch = np.sum(mask != ~np.isnan(numba_values))
    if nan_mismatch > 0:
        pytest.fail(f"NaN mask mismatch at {nan_mismatch} positions")

    diff = np.abs(ref_values[mask] - numba_values[mask])
    max_diff = np.max(diff) if len(diff) > 0 else 0
    print(f"Max difference: {max_diff:.8e}")

    assert max_diff < 1e-10, f"Implementations differ by {max_diff:.8e}"

if __name__ == "__main__":
    test_hurst_correctness()
