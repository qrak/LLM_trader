"""Shared directional risk/reward arithmetic for proposals and execution."""

import math

from numba import njit


@njit(cache=True)
def risk_reward_ratio_numba(entry: float, stop: float, target: float, side: int) -> float:
    if (not math.isfinite(entry) or not math.isfinite(stop) or not math.isfinite(target)
            or entry <= 0 or stop <= 0 or target <= 0):
        return math.nan
    if side == 1 and stop < entry < target:
        return (target - entry) / (entry - stop)
    if side == -1 and target < entry < stop:
        return (entry - target) / (stop - entry)
    return math.nan
