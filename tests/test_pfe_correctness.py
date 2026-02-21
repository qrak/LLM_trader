
import numpy as np
import pytest
from src.indicators.trend.trend_indicators import pfe_numba

def test_pfe_straight_line_uptrend():
    """
    Test PFE on a perfect straight line uptrend.
    Expected PFE should be 100.
    """
    # Create a straight line: 10, 11, 12, ... 29. Length 20.
    close = np.arange(10, 30, dtype=np.float64)
    n = 5 # Period 5 intervals.
    m = 1 # Smoothing length 1 (no smoothing basically, just current value).

    result = pfe_numba(close, n, m)

    # Expected: 100.
    assert result[-1] == pytest.approx(100.0, abs=1e-5), f"Expected 100.0, got {result[-1]}"

def test_pfe_straight_line_downtrend():
    """
    Test PFE on a perfect straight line downtrend.
    Expected PFE should be -100.
    """
    # Create a straight line down: 30, 29, ... 11.
    close = np.arange(30, 10, -1, dtype=np.float64)
    n = 5
    m = 1

    result = pfe_numba(close, n, m)

    # Expected: -100.
    assert result[-1] == pytest.approx(-100.0, abs=1e-5), f"Expected -100.0, got {result[-1]}"

def test_pfe_flat_line():
    """
    Test PFE on a flat line.
    Expected PFE should be 0 because trend is 0.
    """
    close = np.full(20, 10.0, dtype=np.float64)
    n = 5
    m = 1

    result = pfe_numba(close, n, m)

    # If dy=0, sign is 0, so result is 0.
    assert result[-1] == 0.0
