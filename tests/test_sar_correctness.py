import numpy as np
import pytest
from src.indicators.trend.trend_indicators import parabolic_sar_numba

def test_parabolic_sar_lookahead_bullish():
    # Setup scenario for Bullish Trend
    # 0: High=10, Low=8
    # 1: High=12, Low=11  -> Trend Bullish initiated. SAR[0]=8.
    # 2: High=13, Low=12
    # 3: High=14, Low=13

    high = np.array([10.0, 12.0, 13.0, 14.0], dtype=np.float64)
    low = np.array([8.0, 11.0, 12.0, 13.0], dtype=np.float64)

    # Calculate SAR with step=0.02
    sar = parabolic_sar_numba(high, low, step=0.02, max_step=0.2)

    # i=0: Init. Trend=1 (Bullish). SAR[0]=8. EP[0]=10. AF[0]=0.02.
    # i=1: new_sar = 8 + 0.02 * (10 - 8) = 8.04.
    #      low[1]=11 > 8.04. Continue Bullish.
    #      Constraint (Wilder): min(8.04, low[0], low[-1] (invalid)) -> min(8.04, low[0]).
    #      Usually for first step (i=1), constraint is just previous low (low[0]).
    #      Wait. Wilder says SAR[n+1] <= Low[n], Low[n-1].
    #      If n=0 (first period), n-1 is invalid. So just Low[0].
    #      Code uses min(8.04, low[0], low[1]). low[1]=11. 8.04 < 11.
    #      So SAR[1] = 8. Correct.
    #      EP update: high[1]=12 > 10. EP[1]=12. AF[1]=0.04.

    # i=2: new_sar = 8 + 0.04 * (12 - 8) = 8 + 0.16 = 8.16.
    #      low[2]=12 > 8.16. Continue.
    #      Constraint (Wilder): min(8.16, low[1], low[0]) = min(8.16, 11, 8) = 8.
    #      Current Incorrect Code: min(8.16, low[1], low[2]) = min(8.16, 11, 12) = 8.16.

    assert sar[2] == 8.0, f"SAR[2] should be 8.0 (constrained by low[0]), but got {sar[2]}"

    # i=3:
    #      If SAR[2] was 8.0 (Correct):
    #      new_sar = 8.0 + 0.04 * (13 - 8.0) (Wait. EP update at i=2: high[2]=13 > 12. EP=13. AF=0.06).
    #      Actually at i=2, if we use 8.0 as SAR[2].
    #      EP update logic depends on high[i] > EP[i-1].
    #      high[2]=13 > 12. So EP[2]=13. AF[2]=0.06.
    #      SAR calculation for i=3 uses SAR[2], EP[2], AF[2].
    #      Wait. The update uses values from i-1.
    #      So SAR[3] uses SAR[2], EP[2], AF[2]? No.
    #      Code: `new_sar = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])`.
    #      So SAR[3] uses SAR[2], EP[2], AF[2].
    #      Wait. `af[i-1]` is AF stored at end of `i-1`.
    #      At i=2:
    #      Calculate SAR[2] using i-1 (i=1 values).
    #      Update EP, AF for i=2 based on high[2].
    #      So at i=3, we use updated AF from i=2.
    #      So calculation for SAR[3]:
    #      new_sar = 8.0 + 0.06 * (13 - 8.0) = 8.0 + 0.06 * 5 = 8.3.
    #      Constraint: min(8.3, low[2], low[1]) = min(8.3, 12, 11) = 8.3.
    #      So SAR[3] = 8.3.

    #      If SAR[2] was 8.16 (Incorrect):
    #      new_sar = 8.16 + 0.06 * (13 - 8.16) = 8.16 + 0.06 * 4.84 = 8.16 + 0.2904 = 8.4504.
    #      Constraint: min(8.4504, 12, 13) = 8.4504.
    #      So SAR[3] = 8.4504.

    # We can check SAR[3] too.
    # But SAR[2] is the direct proof.

def test_parabolic_sar_lookahead_bearish():
    # Setup scenario for Bearish Trend
    # 0: High=10, Low=8
    # 1: High=9, Low=7. Trend Bearish initiated (if logic allows).
    #      Wait. Logic: trend = -1 if high[0] > low[1]. 10 > 7. Yes. Bearish.
    #      SAR[0] = max(high[:2]) = max(10, 9) = 10. EP[0] = low[0] = 8. AF[0] = 0.02.

    high = np.array([10.0, 9.0, 8.0, 7.0], dtype=np.float64)
    low = np.array([8.0, 7.0, 6.0, 5.0], dtype=np.float64)

    sar = parabolic_sar_numba(high, low, step=0.02, max_step=0.2)

    # i=0: SAR[0]=10. EP[0]=8. AF[0]=0.02. Trend=-1.

    # i=1:
    # new_sar = 10 + 0.02 * (8 - 10) = 10 - 0.04 = 9.96.
    # Check: high[1]=9 < 9.96. Continue Bearish.
    # Constraint (Wilder): max(9.96, high[0], high[-1]). -> max(9.96, high[0]) = max(9.96, 10) = 10.
    # Code: max(9.96, high[0], high[1]) = max(9.96, 10, 9) = 10.
    # SAR[1] = 10. Correct.
    # EP update: low[1]=7 < 8. EP[1]=7. AF[1]=0.04.

    # i=2:
    # new_sar = 10 + 0.04 * (7 - 10) = 10 - 0.12 = 9.88.
    # Check: high[2]=8 < 9.88. Continue.
    # Constraint (Wilder): max(9.88, high[1], high[0]) = max(9.88, 9, 10) = 10.
    # Current Incorrect Code: max(9.88, high[1], high[2]) = max(9.88, 9, 8) = 9.88.

    assert sar[2] == 10.0, f"SAR[2] should be 10.0 (constrained by high[0]), but got {sar[2]}"

def test_parabolic_sar_premature_reversal_bug():
    """
    Test for bug where SAR prematurely reverses when new_sar crosses price
    but constrained_sar does not.
    """
    # Period 0: Start Bullish.
    # high[0]=90, low[1]=91. -> trend=1 (Bullish).
    # Period 1: Spike High to increase EP.

    high = np.array([90.0, 150.0, 155.0, 160.0], dtype=np.float64)
    low = np.array([80.0, 91.0, 90.0, 95.0], dtype=np.float64)

    # Run with step=0.2 to accelerate SAR
    sar = parabolic_sar_numba(high, low, step=0.2, max_step=0.2)

    # Check that trend did NOT reverse at index 2
    # If reversed, SAR jumps to EP (which is high)
    # If not reversed, SAR stays low (constrained by previous lows)

    assert sar[2] < 100, f"Premature reversal detected at index 2. SAR jumped to {sar[2]}"
    assert sar[2] == 80.0, f"SAR[2] should be 80.0 (constrained), got {sar[2]}"
