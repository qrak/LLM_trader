"""Tests for template_manager.py changes: system prompt and response template."""
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

from src.analyzer.prompts.template_manager import TemplateManager
from src.utils.timeframe_validator import TimeframeValidator


def _make_manager(**overrides):
    """Create a TemplateManager with mocked config."""
    config = SimpleNamespace(
        STOP_LOSS_TYPE="soft",
        STOP_LOSS_CHECK_INTERVAL="1h",
        TAKE_PROFIT_TYPE="soft",
        TAKE_PROFIT_CHECK_INTERVAL="1h",
    )
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

    def test_soft_exits_mentioned(self):
        prompt = self.mgr.build_system_prompt("BTC/USDT")
        assert "Stop loss: SOFT" in prompt
        assert "Take profit: SOFT" in prompt
        assert "candle CLOSE" in prompt

    def test_hard_exit_intervals_mentioned(self):
        config = SimpleNamespace(
            STOP_LOSS_TYPE="hard",
            STOP_LOSS_CHECK_INTERVAL="5m",
            TAKE_PROFIT_TYPE="hard",
            TAKE_PROFIT_CHECK_INTERVAL="15m",
        )
        mgr = _make_manager(config=config)

        prompt = mgr.build_system_prompt("BTC/USDT", timeframe="1h")

        assert "Stop loss: HARD bot-side interval check on live ticker every 5m" in prompt
        assert "Take profit: HARD bot-side interval check on live ticker every 15m" in prompt

    def test_mixed_exit_modes_are_explicit(self):
        config = SimpleNamespace(
            STOP_LOSS_TYPE="hard",
            STOP_LOSS_CHECK_INTERVAL="5m",
            TAKE_PROFIT_TYPE="soft",
            TAKE_PROFIT_CHECK_INTERVAL="15m",
        )
        mgr = _make_manager(config=config)

        prompt = mgr.build_system_prompt("BTC/USDT", timeframe="4h")

        assert "Stop loss: HARD bot-side interval check on live ticker every 5m" in prompt
        assert "Take profit: SOFT, evaluated only at 4h candle CLOSE" in prompt

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

    def test_profitable_stop_loss_guidance_is_explicit(self):
        prompt = self.mgr.build_system_prompt("BTC/USDT", performance_context="Recent trades available")
        assert "profit-protecting stop" in prompt
        assert "loss-cutting stop" in prompt
        assert "profitable stop-loss exits" in prompt

    def test_brain_context_included(self):
        prompt = self.mgr.build_system_prompt("BTC/USDT", brain_context="Brain insights here")
        assert "Brain insights here" in prompt

    def test_deterministic_time_check_with_previous(self):
        prompt = self.mgr.build_system_prompt("BTC/USDT", previous_response="test analysis")
        assert "DETERMINISTIC TIME CHECK" in prompt
        assert "Relevance Window" in prompt

    @pytest.mark.parametrize(
        ("timeframe", "expected_window"),
        [
            ("5m", 10),
            ("15m", 30),
            ("30m", 60),
        ]
    )
    def test_relevance_window_uses_sub_hour_timeframes(self, timeframe, expected_window):
        mgr = _make_manager(timeframe_validator=TimeframeValidator)

        prompt = mgr.build_system_prompt(
            "BTC/USDT",
            timeframe=timeframe,
            previous_response="Previous reasoning text",
        )

        assert f"Window: {expected_window} minutes" in prompt


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

    def test_response_template_reasoning_continuity_guidance(self):
        """Response template reasoning field should guide for vector DB continuity data."""
        tmpl = self.mgr.build_response_template()
        reasoning_idx = tmpl.find('"reasoning":')
        assert reasoning_idx != -1
        reasoning_context = tmpl[reasoning_idx : reasoning_idx + 400]
        assert "invalidation" in reasoning_context.lower()
        assert "watch" in reasoning_context.lower()
        assert "thesis" in reasoning_context.lower()
        assert "regime" in reasoning_context.lower() or "trend" in reasoning_context.lower()


# ── Previous response JSON snapshot ──────────────────────────────


class TestPreviousResponseSnapshot:
    """Tests for structured decision snapshot in PREVIOUS ANALYSIS CONTEXT."""

    def setup_method(self):
        self.mgr = _make_manager()

    def _full_prev(self, **overrides) -> str:
        analysis = {
            "signal": "BUY",
            "confidence": 80,
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "take_profit": 115.0,
            "position_size": 0.07,
            "risk_reward_ratio": 3.0,
            "trend": {
                "direction": "BULLISH",
                "strength_4h": 70,
                "strength_daily": 55,
                "timeframe_alignment": "ALIGNED",
            },
            "confluence_factors": {"trend_alignment": 80, "momentum_strength": 75},
            "key_levels": {"support": [95.0, 90.0], "resistance": [115.0, 120.0]},
            "reasoning": "Strong breakout above resistance.",
        }
        analysis.update(overrides)
        import json
        return (
            "1) MARKET STRUCTURE: Bullish.\n"
            "```json\n"
            + json.dumps({"analysis": analysis})
            + "\n```"
        )

    def test_analysis_json_creates_snapshot(self):
        """Full analysis JSON should produce a structured decision snapshot."""
        prompt = self.mgr.build_system_prompt("BTC/USDT", previous_response=self._full_prev())
        assert "Prior decision snapshot:" in prompt
        assert "Signal: BUY" in prompt
        assert "confidence: 80" in prompt
        assert "Entry: 100.0" in prompt
        assert "SL: 95.0" in prompt
        assert "TP: 115.0" in prompt
        assert "R/R: 3.0" in prompt
        assert "Trend: BULLISH" in prompt
        assert "alignment: ALIGNED" in prompt
        assert "Thesis: Strong breakout above resistance." in prompt

    def test_snapshot_excludes_raw_json_block(self):
        """Raw JSON key strings must not appear verbatim in the prompt."""
        prev = (
            "Some text.\n"
            "```json\n"
            '{"analysis": {"signal": "HOLD", "confidence": 60}}\n'
            "```"
        )
        prompt = self.mgr.build_system_prompt("BTC/USDT", previous_response=prev)
        assert '"signal"' not in prompt
        assert "Signal: HOLD" in prompt

    def test_json_only_response_still_creates_context(self):
        """A JSON-only previous response (no narrative) still produces a context section."""
        import json
        prev = (
            "```json\n"
            + json.dumps({"analysis": {"signal": "SELL", "confidence": 75, "entry_price": 200.0,
                                        "stop_loss": 210.0, "take_profit": 170.0}})
            + "\n```"
        )
        prompt = self.mgr.build_system_prompt("BTC/USDT", previous_response=prev)
        assert "PREVIOUS ANALYSIS CONTEXT" in prompt
        assert "Prior decision snapshot:" in prompt
        assert "Signal: SELL" in prompt
        assert "DETERMINISTIC TIME CHECK" in prompt

    def test_malformed_json_falls_back_to_text(self):
        """Malformed JSON block falls back gracefully to text-only context."""
        prev = "My reasoning text.\n```json\n{this is not valid json\n```"
        prompt = self.mgr.build_system_prompt("BTC/USDT", previous_response=prev)
        assert "PREVIOUS ANALYSIS CONTEXT" in prompt
        assert "My reasoning text." in prompt
        assert "Prior decision snapshot:" not in prompt

    def test_snapshot_key_levels_capped_at_two(self):
        """Key levels snapshot shows at most 2 support and 2 resistance levels."""
        import json
        prev = (
            "```json\n"
            + json.dumps({"analysis": {
                "signal": "BUY",
                "key_levels": {
                    "support": [90.0, 85.0, 80.0],
                    "resistance": [110.0, 115.0, 120.0],
                },
            }})
            + "\n```"
        )
        prompt = self.mgr.build_system_prompt("BTC/USDT", previous_response=prev)
        assert "80.0" not in prompt  # third support level must be excluded
        assert "120.0" not in prompt  # third resistance level must be excluded
        assert "90.0" in prompt
        assert "110.0" in prompt

    def test_snapshot_and_narrative_both_present(self):
        """When response has both narrative and JSON, both appear in the prompt."""
        prompt = self.mgr.build_system_prompt("BTC/USDT", previous_response=self._full_prev())
        assert "Prior decision snapshot:" in prompt
        assert "Your last analysis reasoning (for continuity):" in prompt
        assert "1) MARKET STRUCTURE: Bullish." in prompt

    def test_no_analysis_wrapper_falls_back_to_text(self):
        """JSON without 'analysis' wrapper does not create a snapshot, text still shown."""
        prev = "Some reasoning text\n```json\n{\"signal\": \"BUY\"}\n```"
        prompt = self.mgr.build_system_prompt("BTC/USDT", previous_response=prev)
        assert "Some reasoning text" in prompt
        assert "Prior decision snapshot:" not in prompt
        assert '"signal"' not in prompt
