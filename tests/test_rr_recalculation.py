"""
Tests for the R/R recalculation fix (Audit Fix C1).
Verifies that AnalysisResultProcessor._recalculate_risk_reward overwrites
hallucinated R/R ratios with correct calculations.
"""
import unittest
from unittest.mock import MagicMock
from src.analyzer.analysis_result_processor import AnalysisResultProcessor


class TestRiskRewardRecalculation(unittest.TestCase):
    """Tests for _recalculate_risk_reward in AnalysisResultProcessor."""

    def setUp(self):
        self.mock_model_manager = MagicMock()
        self.mock_logger = MagicMock()
        self.processor = AnalysisResultProcessor(
            model_manager=self.mock_model_manager,
            logger=self.mock_logger,
        )

    def test_buy_signal_rr_recalculated_from_entry(self):
        """BUY signal: R/R should use entry_price as reference."""
        analysis = {
            "signal": "BUY",
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "take_profit": 115.0,
            "risk_reward_ratio": 9.99,  # Hallucinated value
        }
        self.processor._recalculate_risk_reward(analysis)
        # risk = |100 - 95| = 5, reward = |115 - 100| = 15, R/R = 3.0
        self.assertAlmostEqual(analysis["risk_reward_ratio"], 3.0)

    def test_update_signal_uses_current_price(self):
        """UPDATE signal: R/R should use current_price (from context), not entry."""
        # Simulate the exact audit scenario: entry=66462.83, current=67536.99
        mock_context = MagicMock()
        mock_context.current_price = 67536.99
        self.processor.context = mock_context

        analysis = {
            "signal": "UPDATE",
            "entry_price": 66462.83,
            "stop_loss": 66750.0,
            "take_profit": 69376.10,
            "risk_reward_ratio": 3.73,  # The hallucinated value from the audit
        }
        self.processor._recalculate_risk_reward(analysis)
        # risk = |67536.99 - 66750.0| = 786.99
        # reward = |69376.10 - 67536.99| = 1839.11
        # R/R = 1839.11 / 786.99 = 2.34
        self.assertAlmostEqual(analysis["risk_reward_ratio"], 2.34, places=1)

    def test_update_without_context_falls_back_to_entry(self):
        """UPDATE signal without context: falls back to entry_price."""
        self.processor.context = None
        analysis = {
            "signal": "UPDATE",
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "take_profit": 115.0,
            "risk_reward_ratio": 0.5,
        }
        self.processor._recalculate_risk_reward(analysis)
        self.assertAlmostEqual(analysis["risk_reward_ratio"], 3.0)

    def test_sell_signal_rr_recalculated(self):
        """SELL (SHORT) signal: R/R calculation uses absolute distances."""
        analysis = {
            "signal": "SELL",
            "entry_price": 100.0,
            "stop_loss": 105.0,
            "take_profit": 85.0,
            "risk_reward_ratio": 1.0,  # Wrong
        }
        self.processor._recalculate_risk_reward(analysis)
        # risk = |100 - 105| = 5, reward = |85 - 100| = 15, R/R = 3.0
        self.assertAlmostEqual(analysis["risk_reward_ratio"], 3.0)

    def test_missing_fields_skips_recalculation(self):
        """Missing entry/SL/TP fields should skip recalculation."""
        analysis = {
            "signal": "BUY",
            "entry_price": 100.0,
            "risk_reward_ratio": 5.0,
        }
        self.processor._recalculate_risk_reward(analysis)
        self.assertEqual(analysis["risk_reward_ratio"], 5.0)  # Unchanged

    def test_zero_risk_skips_division(self):
        """When risk = 0 (SL == ref_price), no division by zero."""
        analysis = {
            "signal": "BUY",
            "entry_price": 100.0,
            "stop_loss": 100.0,  # Zero risk
            "take_profit": 110.0,
            "risk_reward_ratio": 2.0,
        }
        self.processor._recalculate_risk_reward(analysis)
        self.assertEqual(analysis["risk_reward_ratio"], 2.0)  # Unchanged

    def test_logs_warning_on_significant_deviation(self):
        """Should log a warning when correction differs by >0.1."""
        analysis = {
            "signal": "BUY",
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "take_profit": 115.0,
            "risk_reward_ratio": 9.99,  # Way off
        }
        self.processor._recalculate_risk_reward(analysis)
        self.mock_logger.warning.assert_called_once()

    def test_no_warning_on_small_deviation(self):
        """Should NOT log when correction is within 0.1 tolerance."""
        analysis = {
            "signal": "BUY",
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "take_profit": 115.0,
            "risk_reward_ratio": 2.95,  # Close enough to 3.0
        }
        self.processor._recalculate_risk_reward(analysis)
        self.mock_logger.warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
