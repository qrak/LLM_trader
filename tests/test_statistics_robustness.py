import sys
import unittest
import numpy as np
import os

# Add src to path if running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from src.trading.statistics_calculator import StatisticsCalculator

class TestStatisticsRobustness(unittest.TestCase):
    """
    Tests ensuring robustness of statistical calculations and trade extraction logic.
    """

    def test_sortino_ratio_standard_definition(self):
        """
        Verify that Sortino Ratio uses the total number of periods (N)
        in the downside deviation calculation, not just the number of negative periods.

        Scenario:
        Returns: [+10%, -10%]
        Mean Return: 0%
        Target Return: 0%

        Downside Deviation (Standard):
        Sum of squared negative returns (assuming 0 for positive) / Total N
        = (0^2 + (-0.1)^2) / 2
        = 0.01 / 2 = 0.005
        DD = sqrt(0.005) ~= 0.07071

        Sortino Ratio = (Mean - Target) / DD
        = 0 / 0.07071 = 0.0

        Wait, if Mean is 0, Sortino is 0 regardless of DD.
        Let's use a case with positive Mean.

        Scenario 2:
        Returns: [+20%, -10%]
        Mean Return: (0.2 - 0.1) / 2 = 0.05 (5%)
        Target Return: 0%

        Downside Deviation (Standard):
        Sum of squared downside deviations / Total N
        = (0^2 + (-0.1)^2) / 2
        = 0.01 / 2 = 0.005
        DD = sqrt(0.005) ~= 0.07071

        Sortino Ratio = 0.05 / 0.07071 ~= 0.7071

        Current (Incorrect) Calculation:
        DD = sqrt(mean(negative_returns^2))
        = sqrt((-0.1)^2) = 0.1
        Sortino = 0.05 / 0.1 = 0.5
        """
        returns = np.array([0.2, -0.1])

        # Expected Standard Sortino
        mean_return = np.mean(returns) # 0.05
        # downside_returns = [0, -0.1]
        # sum_sq = 0 + 0.01 = 0.01
        # mean_sq = 0.01 / 2 = 0.005
        # dd = sqrt(0.005) ~= 0.0707106
        expected_dd = np.sqrt(0.005)
        expected_sortino = mean_return / expected_dd # ~ 0.7071

        calculated_sortino = StatisticsCalculator._calculate_sortino_ratio(returns)

        # We assert roughly equal to standard definition
        self.assertAlmostEqual(calculated_sortino, expected_sortino, places=2,
                               msg=f"Sortino Ratio should be ~{expected_sortino:.4f} but got {calculated_sortino:.4f}")


if __name__ == '__main__':
    unittest.main()
