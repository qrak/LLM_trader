"""Directional risk/reward kernel shared by the analysis and execution paths."""

import math

import pytest

from src.indicators.price.risk_reward import risk_reward_ratio_numba


@pytest.mark.parametrize(("entry", "stop", "target", "side", "expected"), [
    (84380.0, 84750.0, 83726.0, -1, 654 / 370),
    (100.0, 90.0, 125.0, 1, 2.5),
    (100.0, 100.0, 125.0, 1, math.nan),
    (100.0, 110.0, 125.0, 1, math.nan),
    (100.0, 90.0, 105.0, -1, math.nan),
    (math.nan, 90.0, 125.0, 1, math.nan),
    (100.0, math.inf, 80.0, -1, math.nan),
    (0.0, 90.0, 125.0, 1, math.nan),
])
def test_risk_reward_kernel(entry, stop, target, side, expected):
    actual = risk_reward_ratio_numba(entry, stop, target, side)
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual == pytest.approx(expected)
    assert risk_reward_ratio_numba.nopython_signatures
