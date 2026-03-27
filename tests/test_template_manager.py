"""Tests for template_manager.py changes: system prompt and response template."""
from unittest.mock import MagicMock
import pytest

from src.analyzer.prompts.template_manager import TemplateManager


def _make_manager(**overrides):
    """Create a TemplateManager with mocked config."""
    config = MagicMock()
    defaults = dict(config=config, logger=MagicMock(), timeframe_validator=MagicMock())
    defaults.update(overrides)
    return TemplateManager(**defaults)


# ── build_system_prompt ──────────────────────────────────────────


class TestBuildSystemPrompt:
    """Tests for findings implemented in build_system_prompt."""

    def setup_method(self):
        self.mgr = _make_manager()

    def test_golden_cross_terminology(self):
        prompt = self.mgr.build_system_prompt("BTC/USDT")
        assert "Golden Cross" in prompt
        assert "Death Cross" in prompt
        assert "50 SMA crosses ABOVE 200 SMA" in prompt

    def test_soft_stops_mentioned(self):
        prompt = self.mgr.build_system_prompt("BTC/USDT")
        assert "SOFT STOPS" in prompt
        assert "candle CLOSE" in prompt

    def test_temporal_context_with_last_analysis_time(self):
        prompt = self.mgr.build_system_prompt("BTC/USDT", last_analysis_time="2025-12-26 14:30:00")
        assert "Temporal Context" in prompt
        assert "2025-12-26 14:30:00" in prompt

    def test_no_temporal_without_last_analysis_time(self):
        prompt = self.mgr.build_system_prompt("BTC/USDT")
        assert "Temporal Context" not in prompt

    def test_indicator_delta_alert_injected(self):
        alert = "⚠️ SIGNIFICANT DATA SHIFT: 4 indicators changed"
        prompt = self.mgr.build_system_prompt(
            "BTC/USDT",
            previous_response="Some prior analysis text",
            indicator_delta_alert=alert,
        )
        assert alert in prompt

    def test_no_alert_when_empty(self):
        prompt = self.mgr.build_system_prompt(
            "BTC/USDT",
            previous_response="Some prior analysis text",
            indicator_delta_alert="",
        )
        assert "SIGNIFICANT DATA SHIFT" not in prompt

    def test_previous_response_stripping(self):
        """JSON block should be stripped from previous response."""
        prev = 'Some reasoning text\n```json\n{"signal": "BUY"}\n```'
        prompt = self.mgr.build_system_prompt("BTC/USDT", previous_response=prev)
        assert "Some reasoning text" in prompt
        assert '"signal"' not in prompt

    def test_performance_context_included(self):
        prompt = self.mgr.build_system_prompt("BTC/USDT", performance_context="Win Rate: 60%")
        assert "Win Rate: 60%" in prompt
        assert "Profit Maximization Strategy" in prompt

    def test_brain_context_included(self):
        prompt = self.mgr.build_system_prompt("BTC/USDT", brain_context="Brain insights here")
        assert "Brain insights here" in prompt

    def test_deterministic_time_check_with_previous(self):
        prompt = self.mgr.build_system_prompt("BTC/USDT", previous_response="test analysis")
        assert "DETERMINISTIC TIME CHECK" in prompt
        assert "Relevance Window" in prompt


# ── build_response_template ──────────────────────────────────────


class TestBuildResponseTemplate:
    """Tests for dynamic thresholds in build_response_template."""

    def setup_method(self):
        self.mgr = _make_manager()

    def test_default_thresholds(self):
        tmpl = self.mgr.build_response_template()
        assert "ADX < 20" in tmpl  # adx_weak default
        assert "ADX >= 25" in tmpl or "ADX >= 25" in tmpl  # adx_strong default

    def test_custom_thresholds_injected(self):
        thresholds = {
            "adx_strong_threshold": 30,
            "adx_weak_threshold": 18,
            "min_rr_recommended": 2.5,
            "avg_sl_pct": 3.0,
            "confidence_threshold": 75,
            "min_confluences_weak": 5,
            "min_confluences_standard": 4,
            "trade_count": 0,
            "learned_keys": [],
        }
        tmpl = self.mgr.build_response_template(dynamic_thresholds=thresholds)
        assert "ADX >= 30" in tmpl
        assert "ADX < 18" in tmpl
        assert "2.5:1" in tmpl  # min_rr
        assert "3.0%" in tmpl  # avg_sl

    def test_threshold_origin_with_brain_data(self):
        thresholds = {
            "trade_count": 50,
            "learned_keys": ["min_rr_recommended", "adx_strong_threshold"],
            "adx_strong_threshold": 28,
            "min_rr_recommended": 2.2,
        }
        tmpl = self.mgr.build_response_template(dynamic_thresholds=thresholds)
        assert "THRESHOLD ORIGIN" in tmpl
        assert "brain-learned from 50 closed trades" in tmpl
        assert "min_rr=2.2" in tmpl
        assert "adx_strong=28" in tmpl

    def test_threshold_origin_no_learned_keys(self):
        """When trade_count > 0 but no learned_keys match the listed subset, origin note still appears."""
        thresholds = {
            "trade_count": 5,
            "learned_keys": ["some_unrelated_key"],
        }
        tmpl = self.mgr.build_response_template(dynamic_thresholds=thresholds)
        assert "THRESHOLD ORIGIN" in tmpl
        assert "industry-standard defaults" in tmpl

    def test_safe_mae_line_with_data(self):
        thresholds = {
            "safe_mae_pct": 0.02,
            "trade_count": 20,
            "learned_keys": [],
        }
        tmpl = self.mgr.build_response_template(dynamic_thresholds=thresholds)
        assert "Safe Drawdown" in tmpl
        assert "2.00%" in tmpl

    def test_safe_mae_line_insufficient(self):
        thresholds = {
            "safe_mae_pct": 0,
            "trade_count": 5,
            "learned_keys": [],
        }
        tmpl = self.mgr.build_response_template(dynamic_thresholds=thresholds)
        assert "Insufficient trade data" in tmpl

    def test_position_sizing_formula(self):
        tmpl = self.mgr.build_response_template()
        assert "POSITION SIZING FORMULA" in tmpl
        assert "Base size = confidence / 100" in tmpl

    def test_confluence_scoring(self):
        tmpl = self.mgr.build_response_template()
        assert "CONFLUENCE SCORING" in tmpl
        assert "trend_alignment" in tmpl

    def test_macro_timeframe_conflict(self):
        tmpl = self.mgr.build_response_template()
        assert "MACRO TIMEFRAME CONFLICT" in tmpl
        assert "365D" in tmpl

    def test_rr_calculation_mandatory(self):
        tmpl = self.mgr.build_response_template()
        assert "R/R CALCULATION (MANDATORY" in tmpl
