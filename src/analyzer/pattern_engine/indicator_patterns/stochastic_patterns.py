"""
Stochastic Oscillator Patterns - Pure NumPy/Numba implementation

Detects oversold/overbought conditions and %K/%D crossovers.
Excellent momentum indicator for identifying reversal points.
"""

import math

import numpy as np
from numba import njit


@njit
def detect_stoch_oversold_numba(stoch_k: np.ndarray, threshold: float = 20.0) -> tuple:
    """
    Detect oversold Stochastic condition.

    Stochastic below 20 indicates oversold - potential bullish reversal.
    Returns:
        (is_oversold: bool, periods_ago: int, stoch_value: float)
    """
    if len(stoch_k) == 0:
        return False, 0, 0.0

    current_value = stoch_k[-1]

    if math.isnan(current_value):
        return False, 0, 0.0

    if current_value < threshold:
        periods_ago = 0
        for i in range(len(stoch_k) - 1, -1, -1):
            if math.isnan(stoch_k[i]) or stoch_k[i] >= threshold:
                break
            periods_ago = len(stoch_k) - i - 1

        return True, periods_ago, float(current_value)

    return False, 0, 0.0


@njit
def detect_stoch_overbought_numba(stoch_k: np.ndarray, threshold: float = 80.0) -> tuple:
    """
    Detect overbought Stochastic condition.

    Stochastic above 80 indicates overbought - potential bearish reversal.
    Returns:
        (is_overbought: bool, periods_ago: int, stoch_value: float)
    """
    if len(stoch_k) == 0:
        return False, 0, 0.0

    current_value = stoch_k[-1]

    if math.isnan(current_value):
        return False, 0, 0.0

    if current_value > threshold:
        periods_ago = 0
        for i in range(len(stoch_k) - 1, -1, -1):
            if math.isnan(stoch_k[i]) or stoch_k[i] <= threshold:
                break
            periods_ago = len(stoch_k) - i - 1

        return True, periods_ago, float(current_value)

    return False, 0, 0.0


@njit
def _detect_stochastic_crossover_numba(stoch_k: np.ndarray, stoch_d: np.ndarray,
                                       is_bullish: bool, threshold: float) -> tuple:
    """
    Generic stochastic crossover detection (helper function).

    Scans ENTIRE array for the most recent crossover event.
    Returns:
        (found: bool, periods_ago: int, k_value: float, d_value: float, is_in_zone: bool)
    """
    if len(stoch_k) < 2 or len(stoch_d) < 2:
        return False, 0, 0.0, 0.0, False

    for i in range(1, len(stoch_k)):
        idx = len(stoch_k) - i - 1

        if math.isnan(stoch_k[idx]) or math.isnan(stoch_k[idx + 1]):
            continue
        if math.isnan(stoch_d[idx]) or math.isnan(stoch_d[idx + 1]):
            continue

        if is_bullish:
            was_below = stoch_k[idx] <= stoch_d[idx]
            now_above = stoch_k[idx + 1] > stoch_d[idx + 1]
            crossover = was_below and now_above
        else:
            was_above = stoch_k[idx] >= stoch_d[idx]
            now_below = stoch_k[idx + 1] < stoch_d[idx + 1]
            crossover = was_above and now_below

        if crossover:
            k_val = float(stoch_k[idx + 1])
            d_val = float(stoch_d[idx + 1])

            k_pre = float(stoch_k[idx])
            d_pre = float(stoch_d[idx])

            if is_bullish:
                in_zone = k_pre < threshold or d_pre < threshold
            else:
                in_zone = k_pre > threshold or d_pre > threshold

            return True, i - 1, k_val, d_val, in_zone

    return False, 0, 0.0, 0.0, False


@njit
def detect_stoch_bullish_crossover_numba(stoch_k: np.ndarray, stoch_d: np.ndarray, oversold_threshold: float = 30.0) -> tuple:
    """
    Detect bullish Stochastic crossover: %K crosses above %D while in oversold territory.

    Strong bullish signal when occurring below 30.
    Returns:
        (found: bool, periods_ago: int, k_value: float, d_value: float, is_in_oversold: bool)
    """
    return _detect_stochastic_crossover_numba(stoch_k, stoch_d, True, oversold_threshold)


@njit
def detect_stoch_bearish_crossover_numba(stoch_k: np.ndarray, stoch_d: np.ndarray, overbought_threshold: float = 70.0) -> tuple:
    """
    Detect bearish Stochastic crossover: %K crosses below %D while in overbought territory.

    Strong bearish signal when occurring above 70.
    Returns:
        (found: bool, periods_ago: int, k_value: float, d_value: float, is_in_overbought: bool)
    """
    return _detect_stochastic_crossover_numba(stoch_k, stoch_d, False, overbought_threshold)

