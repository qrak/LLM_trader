import unittest
import numpy as np
from datetime import datetime
from src.utils.data_utils import get_last_valid_value, get_last_n_valid, safe_array_to_scalar
from src.trading.statistics_calculator import StatisticsCalculator, TradingStatistics
from src.analyzer.market_metrics_calculator import MarketMetricsCalculator
from src.notifiers.base_notifier import BaseNotifier
from src.trading.brain import TradingBrainService
from collections import Counter

from unittest.mock import MagicMock

class TestRefactoring(unittest.TestCase):

    def setUp(self):
        self.mock_logger = MagicMock()

    def test_array_utils(self):
        # ... (unchanged)
        # Test get_last_valid_value
        arr_with_none = [1, 2, None, 4, None]
        self.assertEqual(get_last_valid_value(arr_with_none), 4)
        self.assertIsNone(get_last_valid_value([None, None]))
        self.assertIsNone(get_last_valid_value([]))
        
        # Test get_last_n_valid
        # Convert numpy array to list for comparison to avoid ValueError
        self.assertEqual(get_last_n_valid(arr_with_none, 2).tolist(), [2.0, 4.0])
        self.assertEqual(get_last_n_valid(arr_with_none, 5).tolist(), [1.0, 2.0, 4.0])
        
        # Test safe_array_to_scalar
        np_arr = np.array([10, 20, 30])
        self.assertEqual(safe_array_to_scalar(np_arr, -1), 30)
        self.assertIsNone(safe_array_to_scalar(None, -1))
        self.assertIsNone(safe_array_to_scalar([], -1))

    def test_statistics_calculator_numpy(self):
        # Create mock trade history
        history = [
            {"action": "BUY", "price": 100, "quantity": 1, "pnl_pct": 0, "pnl_quote": 0},
            {"action": "CLOSE_LONG", "price": 110, "quantity": 1, "pnl_pct": 10.0, "pnl_quote": 10.0}, # Win
            {"action": "BUY", "price": 100, "quantity": 1, "pnl_pct": 0, "pnl_quote": 0},
            {"action": "CLOSE_LONG", "price": 90, "quantity": 1, "pnl_pct": -10.0, "pnl_quote": -10.0}, # Loss
        ]
        
        stats = StatisticsCalculator.calculate_from_history(history, initial_capital=1000)
        
        self.assertEqual(stats.total_trades, 2)
        self.assertEqual(stats.winning_trades, 1)
        self.assertEqual(stats.losing_trades, 1)
        self.assertEqual(stats.win_rate, 50.0)
        self.assertEqual(stats.total_pnl_quote, 0.0)
        
        # Test Sharpe/Sortino internal logic (mocking returns directly for clarity)
        returns = np.array([0.05, 0.02, -0.01, 0.03, -0.02])
        # Manually calculating expected mean/std for Sharpe check
        # Mean: 0.014, Std: ~0.0265
        sharpe = StatisticsCalculator._calculate_sharpe_ratio(returns)
        self.assertTrue(isinstance(sharpe, float))
        
        # Test Profit Factor
        pnl_amounts = np.array([100, -50, 200, -100])
        pf = StatisticsCalculator._calculate_profit_factor(pnl_amounts)
        self.assertEqual(pf, 2.0) # (100+200) / (50+100) = 300 / 150 = 2.0

    def test_market_metrics_calculator_numpy(self):
        calc = MarketMetricsCalculator(self.mock_logger)
        # Mock data (list of dictionaries)
        data = [
            {"close": 100, "high": 105, "low": 95, "volume": 1000},
            {"close": 110, "high": 115, "low": 105, "volume": 1200},
            {"close": 105, "high": 110, "low": 100, "volume": 1100}
        ]
        
        metrics = calc._calculate_basic_metrics(data, "test_period")
        
        self.assertEqual(metrics["highest_price"], 115)
        self.assertEqual(metrics["lowest_price"], 95)
        self.assertEqual(metrics["total_volume"], 3300)
        self.assertAlmostEqual(metrics["avg_price"], 105.0)
        
    def test_base_notifier_styling(self):
        green, emoji_up = BaseNotifier.get_pnl_styling(5.0)
        self.assertEqual(green, 'green')
        self.assertEqual(emoji_up, '📈')
        
        red, emoji_down = BaseNotifier.get_pnl_styling(-2.0)
        self.assertEqual(red, 'red')
        self.assertEqual(emoji_down, '📉')
        
        grey, emoji_flat = BaseNotifier.get_pnl_styling(0.0)
        self.assertEqual(grey, 'grey')
        self.assertEqual(emoji_flat, '➡️')

    def test_brain_counter_logic(self):
        # Mocking the _count_patterns logic locally since it's an internal helper
        experiences = [
            {"metadata": {"type": "A"}},
            {"metadata": {"type": "B"}},
            {"metadata": {"type": "A"}},
        ]
        
        key_builder = lambda m: m["type"]
        counter = Counter(key_builder(exp["metadata"]) for exp in experiences)
        
        self.assertEqual(counter["A"], 2)
        self.assertEqual(counter["B"], 1)

if __name__ == '__main__':
    unittest.main()
