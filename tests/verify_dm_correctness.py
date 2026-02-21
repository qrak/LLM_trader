import numpy as np
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.indicators.trend.trend_calculation_utils import calculate_directional_movement

def verify_dm():
    print("VERIFYING DIRECTIONAL MOVEMENT CORRECTNESS")

    # Test Case 1: Bug Reproduction
    # high[i] - high[i-1] is negative but greater than low[i-1] - low[i]
    # high: 12 -> 10 (diff -2)
    # low: 11 -> 8 (diff 3, so low[i-1]-low[i] = 3). Wait.
    # low[i-1] - low[i] means (Previous Low - Current Low).
    # If low drops from 11 to 8, then 11 - 8 = 3. Positive.
    # If high drops from 12 to 10, then 10 - 12 = -2. Negative.
    # -2 is NOT > 3. So dm_pos should be 0.

    # I need a case where low[i-1] - low[i] is MORE negative than high[i] - high[i-1].
    # So low must INCREASE significantly.
    # high: 12 -> 10 (diff -2)
    # low: 8 -> 15 (diff 8 - 15 = -7).
    # -2 > -7 is True.
    # Current code: dm_pos = -2.
    # Correct code: dm_pos = max(-2, 0) = 0.

    high = np.array([12.0, 10.0])
    low = np.array([8.0, 15.0])

    dm_pos, dm_neg = calculate_directional_movement(high, low)

    print(f"\nTest Case 1: Negative High Diff > More Negative Low Diff")
    print(f"High: {high}")
    print(f"Low: {low}")
    print(f"High Diff: {high[1] - high[0]}")
    print(f"Low Diff (prev-curr): {low[0] - low[1]}")
    print(f"Calculated +DM: {dm_pos[1]}")

    if dm_pos[1] < 0:
        print(">> FAIL: +DM is negative! This is the bug. ❌")
    elif dm_pos[1] == 0:
        print(">> PASS: +DM is 0. ✅")
    else:
        print(f">> FAIL: Unexpected +DM value: {dm_pos[1]} ❌")

    # Test Case 2: Standard Up Move
    # high: 10 -> 15 (diff 5)
    # low: 8 -> 9 (diff 8 - 9 = -1)
    # 5 > -1. dm_pos = 5.

    high2 = np.array([10.0, 15.0])
    low2 = np.array([8.0, 9.0])

    dm_pos2, dm_neg2 = calculate_directional_movement(high2, low2)

    print(f"\nTest Case 2: Standard Up Move")
    print(f"Calculated +DM: {dm_pos2[1]}")

    if dm_pos2[1] == 5.0:
        print(">> PASS: +DM is correct (5.0). ✅")
    else:
        print(f">> FAIL: +DM should be 5.0, got {dm_pos2[1]} ❌")

if __name__ == "__main__":
    verify_dm()
