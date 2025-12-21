import numpy as np
from numba import njit


@njit(cache=True)
def fit_trendline_numba(indices: np.ndarray, values: np.ndarray) -> tuple:
    n = len(indices)
    if n < 2:
        return 0.0, 0.0
    
    x = indices.astype(np.float64)
    y = values.astype(np.float64)
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = 0.0
    denominator = 0.0
    
    for i in range(n):
        x_diff = x[i] - x_mean
        numerator += x_diff * (y[i] - y_mean)
        denominator += x_diff * x_diff
    
    if denominator == 0:
        return 0.0, y_mean
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    return slope, intercept


@njit(cache=True)
def parallel_trendlines_numba(swing_high_indices: np.ndarray, swing_high_values: np.ndarray,
                              swing_low_indices: np.ndarray, swing_low_values: np.ndarray,
                              slope_tolerance: float = 0.1) -> bool:
    if len(swing_high_indices) < 2 or len(swing_low_indices) < 2:
        return False
    
    slope_upper, _ = fit_trendline_numba(swing_high_indices, swing_high_values)
    slope_lower, _ = fit_trendline_numba(swing_low_indices, swing_low_values)
    
    if max(abs(slope_upper), abs(slope_lower)) == 0:
        return False
    
    slope_diff = abs(slope_upper - slope_lower) / max(abs(slope_upper), abs(slope_lower))
    
    return slope_diff < slope_tolerance


@njit(cache=True)
def converging_trendlines_numba(swing_high_indices: np.ndarray, swing_high_values: np.ndarray,
                                swing_low_indices: np.ndarray, swing_low_values: np.ndarray,
                                min_convergence: float = 0.1) -> bool:
    if len(swing_high_indices) < 2 or len(swing_low_indices) < 2:
        return False
    
    slope_upper, _ = fit_trendline_numba(swing_high_indices, swing_high_values)
    slope_lower, _ = fit_trendline_numba(swing_low_indices, swing_low_values)
    
    slope_diff = abs(slope_upper - slope_lower)
    
    return slope_diff >= min_convergence
