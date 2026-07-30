"""Tests for RiskProfileSelector."""

import pytest
from unittest.mock import MagicMock

from src.trading.risk_profile_selector import RiskProfile, RiskProfileSelector


class TestRiskProfileSelector:
    """Tests for the adaptive risk profile selector."""

    @pytest.fixture
    def mock_vm(self):
        vm = MagicMock()
        vm.trade_count = 0
        vm.compute_confidence_stats.return_value = {
            "HIGH": {"win_rate": 0.5, "trade_count": 0},
            "MEDIUM": {"win_rate": 0.5, "trade_count": 0},
            "LOW": {"win_rate": 0.5, "trade_count": 0},
        }
        return vm

    @pytest.fixture
    def selector(self, mock_vm):
        return RiskProfileSelector(mock_vm)

    def test_not_enough_data_returns_neutral(self, selector):
        """With < 5 trades, always return NEUTRAL."""
        selector._vm.trade_count = 3
        result = selector.select_profile()
        assert result == RiskProfile.NEUTRAL

    def test_high_volatility_forces_conservative(self, selector):
        """ATR > 4% should FORCE conservative regardless of other factors."""
        selector._vm.trade_count = 10
        selector._vm.compute_confidence_stats.return_value = {
            "HIGH": {"win_rate": 0.80, "trade_count": 8},  # Great stats
        }
        result = selector.select_profile(atr_percentage=6.0)
        assert result == RiskProfile.CONSERVATIVE

    def test_deep_drawdown_forces_conservative(self, selector):
        """PNL below -10% forces conservative."""
        selector._vm.trade_count = 10
        result = selector.select_profile(current_pnl_pct=-15.0, atr_percentage=2.0)
        assert result == RiskProfile.CONSERVATIVE

    def test_strong_high_confidence_triggers_aggressive(self, selector):
        """65%+ HIGH win rate with moderate ATR → Aggressive."""
        selector._vm.trade_count = 10
        selector._vm.compute_confidence_stats.return_value = {
            "HIGH": {"win_rate": 0.70, "trade_count": 7},
            "MEDIUM": {"win_rate": 0.50, "trade_count": 0},
            "LOW": {"win_rate": 0.50, "trade_count": 0},
        }
        result = selector.select_profile(atr_percentage=2.0)
        assert result == RiskProfile.AGGRESSIVE

    def test_weak_high_confidence_triggers_conservative(self, selector):
        """Below 45% HIGH win rate with enough samples → Conservative."""
        selector._vm.trade_count = 10
        selector._vm.compute_confidence_stats.return_value = {
            "HIGH": {"win_rate": 0.35, "trade_count": 5},
        }
        result = selector.select_profile(atr_percentage=2.0)
        assert result == RiskProfile.CONSERVATIVE

    def test_losing_streak_conservative(self, selector):
        """2+ consecutive losses as most recent trades → conservative."""
        selector._vm.trade_count = 10
        selector._vm.compute_confidence_stats.return_value = {
            "HIGH": {"win_rate": 0.60, "trade_count": 5},
        }
        # Most recent trade is a loss, preceded by another loss
        recent = [{"pnl": 200}, {"pnl": -100}, {"pnl": -50}]
        result = selector.select_profile(atr_percentage=2.0, recent_trades=recent)
        assert result == RiskProfile.CONSERVATIVE

    def test_winning_streak_aggressive(self, selector):
        """3+ consecutive wins as most recent trades → aggressive."""
        selector._vm.trade_count = 10
        selector._vm.compute_confidence_stats.return_value = {
            "HIGH": {"win_rate": 0.60, "trade_count": 5},
        }
        # Most recent 3 trades are wins
        recent = [{"pnl": -30}, {"pnl": 100}, {"pnl": 200}, {"pnl": 50}]
        result = selector.select_profile(atr_percentage=2.0, recent_trades=recent)
        assert result == RiskProfile.AGGRESSIVE

    def test_compute_streak_mixed(self, selector):
        # Most recent 2 are wins, preceded by a loss → streak = +2
        recent = [{"pnl": -50}, {"pnl": 100}, {"pnl": 200}]
        assert selector._compute_streak(recent) == 2

    def test_compute_streak_mixed_loss_end(self, selector):
        # Most recent 2 are losses, preceded by a win → streak = -2
        recent = [{"pnl": 100}, {"pnl": -30}, {"pnl": -50}]
        assert selector._compute_streak(recent) == -2

    def test_compute_streak_all_losses(self, selector):
        recent = [{"pnl": -30}, {"pnl": -50}, {"pnl": -10}]
        assert selector._compute_streak(recent) == -3

    def test_compute_streak_empty(self, selector):
        assert selector._compute_streak([]) == 0

    def test_sl_multiplier_by_profile(self, selector):
        assert selector.get_profile_sl_multiplier(RiskProfile.AGGRESSIVE) == 1.5
        assert selector.get_profile_sl_multiplier(RiskProfile.NEUTRAL) == 2.0
        assert selector.get_profile_sl_multiplier(RiskProfile.CONSERVATIVE) == 2.5

    def test_tp_multiplier_by_profile(self, selector):
        assert selector.get_profile_tp_multiplier(RiskProfile.AGGRESSIVE) == 3.0
        assert selector.get_profile_tp_multiplier(RiskProfile.NEUTRAL) == 4.0
        assert selector.get_profile_tp_multiplier(RiskProfile.CONSERVATIVE) == 5.0

    def test_position_size_cap_by_profile(self, selector):
        assert selector.get_profile_position_size_cap(RiskProfile.AGGRESSIVE) == 0.10
        assert selector.get_profile_position_size_cap(RiskProfile.NEUTRAL) == 0.08
        assert selector.get_profile_position_size_cap(RiskProfile.CONSERVATIVE) == 0.05

    def test_profitable_bias_pushes_aggressive(self, selector):
        """High P&L% pushes toward aggressive."""
        selector._vm.trade_count = 10
        selector._vm.compute_confidence_stats.return_value = {
            "HIGH": {"win_rate": 0.55, "trade_count": 5},
        }
        result = selector.select_profile(atr_percentage=2.0, current_pnl_pct=15.0)
        assert result == RiskProfile.AGGRESSIVE

    def test_default_conditions_return_neutral(self, selector):
        """Standard conditions: enough trades, moderate stats, moderate ATR."""
        selector._vm.trade_count = 10
        selector._vm.compute_confidence_stats.return_value = {
            "HIGH": {"win_rate": 0.55, "trade_count": 5},
        }
        result = selector.select_profile(atr_percentage=2.0, current_pnl_pct=0.0)
        assert result == RiskProfile.NEUTRAL

    def test_build_profile_context(self, selector):
        ctx = selector.build_profile_context(
            RiskProfile.AGGRESSIVE, "test reason"
        )
        assert "AGGRESSIVE" in ctx
        assert "test reason" in ctx
        assert "1.5" in ctx
