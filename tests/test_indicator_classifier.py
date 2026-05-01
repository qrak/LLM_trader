"""Tests for indicator_classifier — classify_adx_label and classify_rsi_label.

These two functions are the single source of truth for ADX/RSI labeling,
used by brain.py, dashboard/routers/brain.py, vector_memory.py, and
build_context_string_from_technical_data.
"""
from types import SimpleNamespace

import pytest

from src.utils.indicator_classifier import (
    classify_adx_label,
    classify_rsi_label,
    classify_rsi_level,
    classify_trend_direction,
    classify_volatility_level,
    classify_macd_signal,
    classify_volume_state,
    classify_bb_position,
    classify_market_sentiment,
    classify_order_book_bias,
    build_exit_execution_context,
    build_exit_execution_context_from_config,
    build_context_string_from_technical_data,
    build_query_document_from_technical_data,
    format_exit_execution_context,
)


# ── classify_adx_label ──────────────────────────────────────────


class TestClassifyAdxLabel:
    """classify_adx_label(adx: float) → str"""

    def test_high_adx(self):
        assert classify_adx_label(30) == "High ADX"

    def test_high_adx_boundary(self):
        assert classify_adx_label(25) == "High ADX"

    def test_medium_adx(self):
        assert classify_adx_label(22) == "Medium ADX"

    def test_medium_adx_lower_boundary(self):
        assert classify_adx_label(20) == "Medium ADX"

    def test_low_adx(self):
        assert classify_adx_label(15) == "Low ADX"

    def test_low_adx_boundary(self):
        assert classify_adx_label(19.99) == "Low ADX"

    def test_zero_adx(self):
        assert classify_adx_label(0) == "Low ADX"

    def test_very_high_adx(self):
        assert classify_adx_label(80) == "High ADX"


# ── classify_rsi_label ──────────────────────────────────────────


class TestClassifyRsiLabel:
    """classify_rsi_label(rsi: float) → str"""

    def test_overbought(self):
        assert classify_rsi_label(75) == "OVERBOUGHT"

    def test_overbought_boundary(self):
        assert classify_rsi_label(70) == "OVERBOUGHT"

    def test_strong(self):
        assert classify_rsi_label(65) == "STRONG"

    def test_strong_boundary(self):
        assert classify_rsi_label(60) == "STRONG"

    def test_neutral(self):
        assert classify_rsi_label(50) == "NEUTRAL"

    def test_weak(self):
        assert classify_rsi_label(35) == "WEAK"

    def test_weak_boundary(self):
        assert classify_rsi_label(40) == "WEAK"

    def test_oversold(self):
        assert classify_rsi_label(25) == "OVERSOLD"

    def test_oversold_boundary(self):
        assert classify_rsi_label(30) == "OVERSOLD"


class TestClassifyRsiLevelDelegation:
    """classify_rsi_level should delegate to classify_rsi_label."""

    def test_delegates_rsi_from_dict(self):
        td = {"rsi": 75}
        assert classify_rsi_level(td) == "OVERBOUGHT"

    def test_default_rsi_returns_neutral(self):
        assert classify_rsi_level({}) == "NEUTRAL"

    def test_exact_match_with_label_fn(self):
        for rsi in [10, 30, 40, 50, 60, 70, 90]:
            td = {"rsi": rsi}
            assert classify_rsi_level(td) == classify_rsi_label(rsi)


# ── Other classifiers (existing behavior coverage) ──────────────


class TestClassifyTrendDirection:

    def test_bullish(self):
        assert classify_trend_direction({"plus_di": 30, "minus_di": 10}) == "BULLISH"

    def test_bearish(self):
        assert classify_trend_direction({"plus_di": 10, "minus_di": 30}) == "BEARISH"

    def test_neutral_within_threshold(self):
        assert classify_trend_direction({"plus_di": 20, "minus_di": 18}) == "NEUTRAL"

    def test_default_values(self):
        assert classify_trend_direction({}) == "NEUTRAL"


class TestClassifyVolatilityLevel:

    def test_high(self):
        assert classify_volatility_level({"atr_percent": 5.0}) == "HIGH"

    def test_low(self):
        assert classify_volatility_level({"atr_percent": 1.0}) == "LOW"

    def test_medium(self):
        assert classify_volatility_level({"atr_percent": 2.0}) == "MEDIUM"


class TestClassifyMacdSignal:

    def test_bullish(self):
        assert classify_macd_signal({"macd_line": 1.5, "macd_signal": 0.5}) == "BULLISH"

    def test_bearish(self):
        assert classify_macd_signal({"macd_line": -0.5, "macd_signal": 0.5}) == "BEARISH"

    def test_neutral_when_missing(self):
        assert classify_macd_signal({}) == "NEUTRAL"


class TestClassifyVolumeState:

    def test_accumulation(self):
        assert classify_volume_state({"obv_slope": 1.0}) == "ACCUMULATION"

    def test_distribution(self):
        assert classify_volume_state({"obv_slope": -1.0}) == "DISTRIBUTION"

    def test_normal(self):
        assert classify_volume_state({"obv_slope": 0.2}) == "NORMAL"


class TestClassifyBbPosition:

    def test_upper(self):
        assert classify_bb_position({"bb_upper": 100, "bb_lower": 80}, 100) == "UPPER"

    def test_lower(self):
        assert classify_bb_position({"bb_upper": 100, "bb_lower": 80}, 80) == "LOWER"

    def test_middle(self):
        assert classify_bb_position({"bb_upper": 100, "bb_lower": 80}, 90) == "MIDDLE"


class TestClassifyMarketSentiment:

    def test_extreme_fear(self):
        assert classify_market_sentiment({"fear_greed_index": 20}) == "EXTREME_FEAR"

    def test_extreme_greed(self):
        assert classify_market_sentiment({"fear_greed_index": 80}) == "EXTREME_GREED"

    def test_neutral_default(self):
        assert classify_market_sentiment(None) == "NEUTRAL"


class TestClassifyOrderBookBias:

    def test_buy_pressure(self):
        assert classify_order_book_bias({"order_book": {"imbalance": 0.3}}) == "BUY_PRESSURE"

    def test_sell_pressure(self):
        assert classify_order_book_bias({"order_book": {"imbalance": -0.3}}) == "SELL_PRESSURE"

    def test_balanced_default(self):
        assert classify_order_book_bias(None) == "BALANCED"


# ── build_context_string_from_technical_data ─────────────────


class TestBuildContextString:
    """Verify that build_context_string uses classify_adx_label."""

    def test_uses_classify_adx_label(self):
        td = {"adx": 30, "plus_di": 30, "minus_di": 10, "atr_percent": 2.0}
        ctx = build_context_string_from_technical_data(td)
        assert "High ADX" in ctx

    def test_low_adx_appears(self):
        td = {"adx": 10, "plus_di": 10, "minus_di": 10, "atr_percent": 2.0}
        ctx = build_context_string_from_technical_data(td)
        assert "Low ADX" in ctx

    def test_rsi_label_appears_when_not_neutral(self):
        td = {"adx": 25, "rsi": 75, "atr_percent": 2.0}
        ctx = build_context_string_from_technical_data(td)
        assert "RSI OVERBOUGHT" in ctx

    def test_rsi_absent_when_neutral(self):
        td = {"adx": 25, "rsi": 50, "atr_percent": 2.0}
        ctx = build_context_string_from_technical_data(td)
        assert "RSI" not in ctx

    def test_weekend_flag(self):
        td = {"adx": 25, "atr_percent": 2.0}
        ctx = build_context_string_from_technical_data(td, is_weekend=True)
        assert "Weekend Low Volume" in ctx

    def test_separator_is_plus(self):
        td = {"adx": 30, "plus_di": 30, "minus_di": 10, "atr_percent": 2.0}
        ctx = build_context_string_from_technical_data(td)
        assert " + " in ctx

    def test_exit_execution_context_is_included(self):
        td = {"adx": 30, "plus_di": 30, "minus_di": 10, "atr_percent": 2.0}
        exit_execution = build_exit_execution_context(
            stop_loss_type="hard",
            stop_loss_check_interval="15m",
            take_profit_type="hard",
            take_profit_check_interval="15m",
        )

        ctx = build_context_string_from_technical_data(td, exit_execution_context=exit_execution)

        assert "Exit Execution: SL hard/15m | TP hard/15m" in ctx

    def test_unknown_exit_execution_context_is_omitted_by_default(self):
        assert format_exit_execution_context(build_exit_execution_context()) == ""

    def test_exit_execution_config_falls_back_when_attributes_are_missing(self):
        context = build_exit_execution_context_from_config(SimpleNamespace(), timeframe="4h")

        assert context == {
            "stop_loss_type": "unknown",
            "stop_loss_check_interval": "4h",
            "take_profit_type": "unknown",
            "take_profit_check_interval": "4h",
        }


class TestBuildQueryDocument:
    """Verify enriched query documents carry the same risk execution phrase."""

    def test_exit_execution_context_is_in_structure(self):
        td = {"adx": 30, "rsi": 61, "plus_di": 30, "minus_di": 10, "atr_percent": 2.0}
        exit_execution = build_exit_execution_context(
            stop_loss_type="hard",
            stop_loss_check_interval="15m",
            take_profit_type="soft",
            take_profit_check_interval="4h",
        )

        query = build_query_document_from_technical_data(td, exit_execution_context=exit_execution)

        assert "Indicators: ADX=30.0 (High ADX)" in query
        assert "Structure: Exit Execution: SL hard/15m | TP soft/4h" in query
