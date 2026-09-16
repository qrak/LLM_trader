"""
Moving Average Crossover Patterns - Pure NumPy/Numba implementation

Detects Golden Cross, Death Cross, and short-term MA crossovers.
Classic long-term trend reversal signals.
"""

import math

import numpy as np
from numba import njit


@njit
def _detect_ma_crossover_numba(sma_50: np.ndarray, sma_200: np.ndarray, is_bullish: bool) -> tuple:
    """
    Generic MA crossover detection (helper function).

    Scans ENTIRE array for the most recent crossover event.
    Returns:
        (found: bool, periods_ago: int, sma_50_value: float, sma_200_value: float)
    """
    if len(sma_50) < 2 or len(sma_200) < 2:
        return False, 0, 0.0, 0.0

    for i in range(1, len(sma_50)):
        idx = len(sma_50) - i - 1

        if math.isnan(sma_50[idx]) or math.isnan(sma_50[idx + 1]):
            continue
        if math.isnan(sma_200[idx]) or math.isnan(sma_200[idx + 1]):
            continue

        if is_bullish:
            was_below = sma_50[idx] <= sma_200[idx]
            now_above = sma_50[idx + 1] > sma_200[idx + 1]
            crossover = was_below and now_above
        else:
            was_above = sma_50[idx] >= sma_200[idx]
            now_below = sma_50[idx + 1] < sma_200[idx + 1]
            crossover = was_above and now_below

        if crossover:
            return True, i - 1, float(sma_50[idx + 1]), float(sma_200[idx + 1])

    return False, 0, 0.0, 0.0


@njit
def detect_golden_cross_numba(sma_50: np.ndarray, sma_200: np.ndarray) -> tuple:
    """
    Detect Golden Cross: 50 SMA crosses above 200 SMA.

    Strong bullish signal indicating potential long-term uptrend.
    Returns:
        (found: bool, periods_ago: int, sma_50_value: float, sma_200_value: float)
    """
    return _detect_ma_crossover_numba(sma_50, sma_200, True)


@njit
def detect_death_cross_numba(sma_50: np.ndarray, sma_200: np.ndarray) -> tuple:
    """
    Detect Death Cross: 50 SMA crosses below 200 SMA.

    Strong bearish signal indicating potential long-term downtrend.
    Returns:
        (found: bool, periods_ago: int, sma_50_value: float, sma_200_value: float)
    """
    return _detect_ma_crossover_numba(sma_50, sma_200, False)


@njit
def detect_short_term_crossover_numba(sma_20: np.ndarray, sma_50: np.ndarray) -> tuple:
    """
    Detect short-term MA crossover: 20 SMA crosses 50 SMA.

    Scans ENTIRE array for the most recent crossover event.
    Medium-term trend signal for swing trading.
    Returns:
        (found: bool, is_bullish: bool, periods_ago: int, sma_20_value: float, sma_50_value: float)
    """
    if len(sma_20) < 2 or len(sma_50) < 2:
        return False, False, 0, 0.0, 0.0

    for i in range(1, len(sma_20)):
        idx = len(sma_20) - i - 1

        if math.isnan(sma_20[idx]) or math.isnan(sma_20[idx + 1]):
            continue
        if math.isnan(sma_50[idx]) or math.isnan(sma_50[idx + 1]):
            continue

        was_below = sma_20[idx] <= sma_50[idx]
        now_above = sma_20[idx + 1] > sma_50[idx + 1]

        if was_below and now_above:
            return True, True, i - 1, float(sma_20[idx + 1]), float(sma_50[idx + 1])

        was_above = sma_20[idx] >= sma_50[idx]
        now_below = sma_20[idx + 1] < sma_50[idx + 1]

        if was_above and now_below:
            return True, False, i - 1, float(sma_20[idx + 1]), float(sma_50[idx + 1])

    return False, False, 0, 0.0, 0.0

