"""Tests for EVFrameworkFormatter."""

import pytest
from unittest.mock import MagicMock

from src.analyzer.formatters.ev_formatter import EVFrameworkFormatter


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

    def test_build_ev_quick_section(self, formatter):
        quick = formatter.build_ev_quick_section(current_capital=10250.0)
        assert "PORTFOLIO STATUS" in quick
        assert "10,000" in quick
        assert "10,250" in quick
        assert "+250" in quick

    def test_fee_round_trip_cost(self, formatter):
        # Fee per full round-trip: 0.075% of 10000 = $7.50
        assert formatter.starting_capital * formatter.fee_percent == pytest.approx(7.50)
