import sys
import unittest
import numpy as np
import os

# Add src to path if running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



# Now import the class under test
from src.trading.statistics_calculator import StatisticsCalculator

class TestStatisticsCalculatorCoverage(unittest.TestCase):
    """
    Unit tests for StatisticsCalculator focusing on edge cases,
    mathematical correctness, and boundary conditions.
    """

    def test_calculate_sharpe_ratio_edge_cases(self):
        """Test Sharpe Ratio calculation with various inputs."""
        # Case 1: Empty input
        self.assertEqual(StatisticsCalculator._calculate_sharpe_ratio(np.array([])), 0.0)

        # Case 2: Single element (std dev is undefined/zero)
        self.assertEqual(StatisticsCalculator._calculate_sharpe_ratio(np.array([0.1])), 0.0)

        # Case 3: Zero standard deviation (constant returns)
        # Mean = 0.1, Std = 0 -> Avoid division by zero
        returns = np.array([0.1, 0.1, 0.1])
        self.assertEqual(StatisticsCalculator._calculate_sharpe_ratio(returns), 0.0)

        # Case 4: Normal calculation
        # Returns: 10%, -10%. Mean=0, Std=0.1. Sharpe=0
        returns = np.array([0.1, -0.1])
        self.assertEqual(StatisticsCalculator._calculate_sharpe_ratio(returns), 0.0)

        # Returns: 10%, 20%. Mean=0.15, Std=0.05. Sharpe=(0.15)/0.05 = 3.0
        returns = np.array([0.1, 0.2])
        self.assertAlmostEqual(StatisticsCalculator._calculate_sharpe_ratio(returns), 3.0)

    def test_calculate_sortino_ratio_edge_cases(self):
        """Test Sortino Ratio calculation with various inputs."""
        # Case 1: Empty input
        self.assertEqual(StatisticsCalculator._calculate_sortino_ratio(np.array([])), 0.0)

        # Case 2: All positive returns (downside deviation is zero)
        # Should return inf if mean > 0, else 0
        returns = np.array([0.1, 0.2, 0.3])
        self.assertEqual(StatisticsCalculator._calculate_sortino_ratio(returns), float('inf'))

        # Case 3: All negative returns
        returns = np.array([-0.1, -0.2])
        # Mean = -0.15
        # Downside Deviation calculation: sqrt(mean(negative_returns^2))
        # (-0.1)^2 = 0.01, (-0.2)^2 = 0.04. Mean = 0.025. Sqrt = 0.158113...
        # Sortino = -0.15 / 0.158113... ~= -0.948...
        expected = -0.15 / np.sqrt(np.mean(returns**2))
        self.assertAlmostEqual(StatisticsCalculator._calculate_sortino_ratio(returns), round(expected, 2))

        # Case 4: Zero downside deviation but negative mean (e.g. all zeros)
        returns = np.array([0.0, 0.0])
        self.assertEqual(StatisticsCalculator._calculate_sortino_ratio(returns), 0.0)

    def test_calculate_profit_factor_edge_cases(self):
        """Test Profit Factor calculation."""
        # Case 1: No trades
        self.assertEqual(StatisticsCalculator._calculate_profit_factor(np.array([])), 0.0)

        # Case 2: Only profits (Zero loss)
        pnl = np.array([100.0, 50.0])
        self.assertEqual(StatisticsCalculator._calculate_profit_factor(pnl), float('inf'))

        # Case 3: Only losses
        pnl = np.array([-50.0, -50.0])
        self.assertEqual(StatisticsCalculator._calculate_profit_factor(pnl), 0.0)

        # Case 4: Mixed
        pnl = np.array([100.0, -50.0])
        # Gross Profit = 100, Gross Loss = 50. PF = 2.0
        self.assertEqual(StatisticsCalculator._calculate_profit_factor(pnl), 2.0)

    def test_calculate_drawdowns_edge_cases(self):
        """Test Drawdown calculations."""
        # Case 1: Not enough data
        self.assertEqual(StatisticsCalculator._calculate_drawdowns(np.array([100.0])), (0.0, 0.0))

        # Case 2: Monotonically increasing (No drawdown)
        equity = np.array([100.0, 110.0, 120.0])
        max_dd, avg_dd = StatisticsCalculator._calculate_drawdowns(equity)
        self.assertEqual(max_dd, 0.0)
        self.assertEqual(avg_dd, 0.0)

        # Case 3: Simple Drawdown
        # 100 -> 120 -> 60 -> 120
        # Peak at 120. Drop to 60 is -50%.
        equity = np.array([100.0, 120.0, 60.0, 120.0])
        max_dd, avg_dd = StatisticsCalculator._calculate_drawdowns(equity)
        self.assertEqual(max_dd, -50.0)
        # Drawdowns: 0, 0, -50%, 0
        # Avg of negative drawdowns: -50.0
        self.assertEqual(avg_dd, -50.0)

    def test_calculate_from_history_empty(self):
        """Test public API with empty history."""
        stats = StatisticsCalculator.calculate_from_history([])
        self.assertEqual(stats.total_trades, 0)
        self.assertEqual(stats.win_rate, 0.0)
        self.assertEqual(stats.sharpe_ratio, 0.0)

    def test_calculate_from_history_mixed_actions(self):
        """Test history with ignored actions (e.g. deposits)."""
        history = [
            {"action": "DEPOSIT", "amount": 1000},
            {"action": "BUY", "price": 100, "quantity": 1, "timestamp": "2023-01-01"},
            {"action": "CLOSE_LONG", "price": 110, "quantity": 1, "timestamp": "2023-01-02"},
            {"action": "WITHDRAW", "amount": 500}
        ]
        stats = StatisticsCalculator.calculate_from_history(history, initial_capital=1000.0)
        self.assertEqual(stats.total_trades, 1)
        self.assertAlmostEqual(stats.total_pnl_quote, 10.0)
        self.assertEqual(stats.winning_trades, 1)

if __name__ == '__main__':
    unittest.main()
