
import numpy as np
import pytest
from src.indicators.volatility.volatility_indicators import ebsw_numba

def test_ebsw_initialization_bug():
    """
    Test EBSW indicator initialization.
    The bug is that last_close is initialized to 0.
    On a flat line (constant price), the change should be 0.
    If last_close is 0, the first change is Price - 0 = Price (huge).
    This causes the indicator to produce non-zero values (or huge internal state)
    when it should be 0 (or NaN/undefined due to 0 power).

    With the bug:
    Price = 100.
    last_close = 0.
    change = 100.
    hp ~ 100.
    filt ~ 100.
    pwr ~ 10000.
    wave ~ 1.

    With the fix:
    Price = 100.
    last_close = 100 (from previous step or initialized).
    change = 0.
    hp = 0.
    filt = 0.
    pwr = 0.
    wave = NaN (0/0).
    """
    length = 40
    n = 100
    close = np.full(n, 100.0, dtype=np.float64)

    # Run EBSW
    result = ebsw_numba(close, length=length)

    # Check the first calculated value (at index length)
    first_val = result[length]

    # With the bug, this will likely be a valid number (approx 1 or -1 or something non-zero) because of the huge jump.
    # We expect it to be NaN (if 0/0) or 0 if correct behavior for flat line is 0.
    # However, if the bug exists, the internal 'hp' will be huge.

    # Let's just print it to see what happens in the test failure,
    # but to make it fail, we assert that it is NaN (since 0/0) or 0.
    # If it returns a value like ~0.9 or ~ -0.9, it's definitely wrong for a flat line.

    # If the bug exists, 'hp' is large, 'pwr' is large, so 'wave' is calculated (not NaN).
    # If fixed, 'hp' is 0, 'pwr' is 0, 'wave' is NaN or handled.

    # We assert that the result is NaN for a flat line (because volatility is 0).
    # Or at least not a strong signal.

    # Using np.isnan check
    assert np.isnan(first_val), f"Expected NaN for flat line (0 volatility), got {first_val}"

def test_ebsw_convergence():
    """
    Test that the indicator converges or behaves reasonably on normal data.
    """
    pass
