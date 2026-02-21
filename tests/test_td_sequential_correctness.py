
import numpy as np
import pytest
from src.indicators.trend.trend_indicators import td_sequential_numba

def test_td_sequential_correctness():
    """
    Test TD Sequential logic correctness.

    Sequence logic:
    - If close[i] > close[i-4]: +1 (Bullish Setup Count)
    - If close[i] < close[i-4]: -1 (Bearish Setup Count)
    - Resets if condition flips.

    The test constructs a scenario where:
    Indices 0-3: Initialization (can't compare back 4 periods).
    Indices 4-7: Bullish condition met (15>10, 16>11, 17>12, 18>13). Expected: 1, 2, 3, 4.
    Index 8: Bearish flip (14 < 15). Expected: -1.
    Index 9: Bullish flip (20 > 16). Expected: 1.
    """

    # Synthetic data
    # i=0..3: 10, 11, 12, 13
    # i=4: 15 (vs 10) -> Bullish
    # i=5: 16 (vs 11) -> Bullish
    # i=6: 17 (vs 12) -> Bullish
    # i=7: 18 (vs 13) -> Bullish
    # i=8: 14 (vs 15) -> Bearish (14 < 15)
    # i=9: 20 (vs 16) -> Bullish (20 > 16)

    closes = np.array([
        10, 11, 12, 13, # 0-3
        15, 16, 17, 18, # 4-7
        14, # 8
        20  # 9
    ], dtype=np.float64)

    # Expected results
    # 0-3: 0 (or NaN, implementation dependent, current impl returns 0/NaN)
    # 4: 1
    # 5: 2
    # 6: 3
    # 7: 4
    # 8: -1 (Flip!)
    # 9: 1 (Flip!)

    result = td_sequential_numba(closes, length=9)

    # Check index 8 (Bearish flip)
    # The buggy implementation returns 4 here because it looks back and sees 4 bullish candles in the window.
    # The correct implementation should see that close[8] < close[4] and return -1.
    assert result[8] == -1, f"Index 8 should be -1 (Bearish Flip), got {result[8]}"

    # Check index 9 (Bullish flip)
    assert result[9] == 1, f"Index 9 should be 1 (Bullish Flip), got {result[9]}"

    # Check sequence 4-7
    assert result[4] == 1
    assert result[5] == 2
    assert result[6] == 3
    assert result[7] == 4

if __name__ == "__main__":
    pytest.main([__file__])
