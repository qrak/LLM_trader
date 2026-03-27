"""Tests for brain.py changes: classify_adx_label integration in _build_rich_context_string."""
from unittest.mock import MagicMock, patch
import pytest

from src.trading.brain import TradingBrainService


def _make_brain():
    """Create a TradingBrainService with mocked dependencies."""
    logger = MagicMock()
    persistence = MagicMock()
    vector_memory = MagicMock()
    vector_memory.trade_count = 0
    vector_memory.get_relevant_rules.return_value = []
    return TradingBrainService(logger=logger, persistence=persistence, vector_memory=vector_memory)


# ── _build_rich_context_string ──────────────────────────────────


class TestBuildRichContextString:
    """Verify _build_rich_context_string uses classify_adx_label."""

    def setup_method(self):
        self.brain = _make_brain()

    def test_high_adx_label(self):
        ctx = self.brain._build_rich_context_string(adx=30)
        assert "High ADX" in ctx

    def test_low_adx_label(self):
        ctx = self.brain._build_rich_context_string(adx=10)
        assert "Low ADX" in ctx

    def test_medium_adx_label(self):
        ctx = self.brain._build_rich_context_string(adx=22)
        assert "Medium ADX" in ctx

    def test_contains_trend_direction(self):
        ctx = self.brain._build_rich_context_string(trend_direction="BULLISH", adx=25)
        assert "BULLISH" in ctx

    def test_contains_volatility(self):
        ctx = self.brain._build_rich_context_string(adx=25, volatility_level="HIGH")
        assert "HIGH Volatility" in ctx

    def test_rsi_included_when_not_neutral(self):
        ctx = self.brain._build_rich_context_string(adx=25, rsi_level="OVERBOUGHT")
        assert "RSI OVERBOUGHT" in ctx

    def test_rsi_excluded_when_neutral(self):
        ctx = self.brain._build_rich_context_string(adx=25, rsi_level="NEUTRAL")
        assert "RSI" not in ctx

    def test_weekend_flag(self):
        ctx = self.brain._build_rich_context_string(adx=25, is_weekend=True)
        assert "Weekend Low Volume" in ctx

    def test_separator_is_plus(self):
        ctx = self.brain._build_rich_context_string(adx=25)
        assert " + " in ctx

    def test_macd_included_when_not_neutral(self):
        ctx = self.brain._build_rich_context_string(adx=25, macd_signal="BULLISH")
        assert "MACD BULLISH" in ctx

    def test_volume_included_when_not_normal(self):
        ctx = self.brain._build_rich_context_string(adx=25, volume_state="ACCUMULATION")
        assert "Volume ACCUMULATION" in ctx

    def test_bb_included_when_not_middle(self):
        ctx = self.brain._build_rich_context_string(adx=25, bb_position="UPPER")
        assert "Price at BB UPPER" in ctx

    def test_sentiment_included(self):
        ctx = self.brain._build_rich_context_string(adx=25, market_sentiment="EXTREME_FEAR")
        assert "Sentiment EXTREME_FEAR" in ctx

    def test_order_book_included(self):
        ctx = self.brain._build_rich_context_string(adx=25, order_book_bias="BUY_PRESSURE")
        assert "OrderBook BUY_PRESSURE" in ctx


# ── get_vector_context ──────────────────────────────────────────


class TestGetVectorContext:
    """Verify get_vector_context calls _build_rich_context_string and forwards to vector_memory."""

    def setup_method(self):
        self.brain = _make_brain()
        # Stub the two methods get_vector_context calls after building the query
        self.brain.vector_memory.get_context_for_prompt.return_value = "some context"
        self.brain.vector_memory.get_stats_for_context.return_value = {
            "total_trades": 0, "win_rate": 0, "avg_pnl": 0
        }

    def test_calls_vector_memory_get_context(self):
        self.brain.get_vector_context(adx=30, trend_direction="BULLISH")
        self.brain.vector_memory.get_context_for_prompt.assert_called_once()

    def test_query_contains_adx_label(self):
        self.brain.get_vector_context(adx=30, trend_direction="BULLISH")
        call_args = self.brain.vector_memory.get_context_for_prompt.call_args
        query_str = call_args[0][0]  # first positional arg is context_query
        assert "High ADX" in query_str


# ── get_dynamic_thresholds ──────────────────────────────────────


class TestGetDynamicThresholds:
    """Verify thresholds structure returned by get_dynamic_thresholds."""

    def setup_method(self):
        self.brain = _make_brain()
        self.brain.vector_memory.compute_optimal_thresholds.return_value = {}

    def test_returns_all_expected_keys(self):
        t = self.brain.get_dynamic_thresholds()
        expected = {
            "adx_strong_threshold", "avg_sl_pct", "min_rr_recommended",
            "confidence_threshold", "safe_mae_pct",
            "adx_weak_threshold", "min_confluences_weak", "min_confluences_standard",
            "position_reduce_mixed", "position_reduce_divergent", "min_position_size",
            "rr_borderline_min", "rr_strong_setup",
            "trade_count", "learned_keys",
        }
        assert expected.issubset(set(t.keys()))

    def test_defaults_when_empty(self):
        t = self.brain.get_dynamic_thresholds()
        assert t["adx_strong_threshold"] == 25
        assert t["min_rr_recommended"] == 2.0
        assert t["confidence_threshold"] == 70


class TestReflectionRuleFormatting:
    """Verify reflection-generated rule text preserves full ADX bucket labels."""

    def test_trigger_reflection_preserves_high_adx_label(self):
        brain = _make_brain()

        win_metas = [
            {
                "outcome": "WIN",
                "market_regime": "BULLISH",
                "adx_at_entry": 30,
                "direction": "LONG",
            }
            for _ in range(10)
        ]

        brain.vector_memory._get_trade_metadatas.return_value = win_metas

        brain._trigger_reflection()

        assert brain.vector_memory.store_semantic_rule.called
        rule_text = brain.vector_memory.store_semantic_rule.call_args.kwargs["rule_text"]
        assert "with High ADX." in rule_text

