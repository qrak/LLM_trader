"""Tests for EVFrameworkFormatter."""

from unittest.mock import MagicMock

import pytest

from src.analyzer.formatters.ev_formatter import EVFrameworkFormatter
from src.trading.regime_risk_profile import RegimeRiskProfile, RegimeRiskProfileSelector


class TestEVFrameworkFormatter:
    """Tests for the Expected Value framework formatter."""

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.DEMO_QUOTE_CAPITAL = 10000.0
        config.TRANSACTION_FEE_PERCENT = 0.00075
        return config

    @pytest.fixture
    def formatter(self, mock_config):
        return EVFrameworkFormatter(mock_config)

    def test_starting_capital(self, formatter):
        assert formatter.starting_capital == 10000.0

    def test_fee_percent(self, formatter):
        assert formatter.fee_percent == 0.00075

    def test_standard_position_pct_matches_neutral_profile_cap(self):
        # The fee example must stay pinned to the trading-side NEUTRAL cap.
        assert EVFrameworkFormatter.STANDARD_POSITION_PCT == pytest.approx(
            RegimeRiskProfileSelector.get_position_size_cap(RegimeRiskProfile.NEUTRAL)
        )

    def test_round_trip_fee_scales_with_position_notional(self, formatter):
        # 0.075% per side → 0.15% round trip of the POSITION, not of the capital.
        assert formatter.round_trip_fee(800.0) == pytest.approx(1.20)
        assert formatter.round_trip_fee(480.0) == pytest.approx(0.72)

    def test_build_ev_framework_section_positive_pnl(self, formatter):
        section = formatter.build_ev_framework_section(current_capital=10500.0)
        assert "EXPECTED VALUE FRAMEWORK" in section
        assert "$10,000.00" in section
        assert "$10,500.00" in section
        assert "$+500.00" in section    # format is $+XXX.XX
        assert "+5.00%" in section
        assert "EV = P" in section

    def test_build_ev_framework_section_negative_pnl(self, formatter):
        section = formatter.build_ev_framework_section(current_capital=9500.0)
        assert "$-500.00" in section
        assert "-5.00%" in section

    def test_build_ev_framework_section_zero_change(self, formatter):
        section = formatter.build_ev_framework_section(current_capital=10000.0)
        assert "$+0.00" in section
        assert "+0.00%" in section

    def test_ev_section_fee_is_position_based(self, formatter):
        section = formatter.build_ev_framework_section(current_capital=10000.0)
        assert "0.150% of the position size" in section
        assert "$1.20" in section   # standard 8% position ($800) round-trip fee
        assert "$1.80" in section   # 1.5× fee threshold
        assert "$7.50" not in section  # the old capital-based fee must stay gone
