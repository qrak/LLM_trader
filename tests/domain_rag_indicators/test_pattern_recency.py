

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.analyzer.formatters.technical_formatter import TechnicalFormatter
from src.analyzer.pattern_quality_scorer import PatternQualityScorer
from src.utils.pattern_recency import (
    filter_recent_patterns,
    is_pattern_recent,
    recency_threshold,
    recent_history_window,
    staleness_threshold,
)

TOTAL_CANDLES = 999
LAST_INDEX = TOTAL_CANDLES - 1


def macd_pattern(index: int) -> dict:

    age = LAST_INDEX - index
    label = "now" if age == 0 else ("1 period ago" if age == 1 else f"{age} periods ago")
    return {
        "type": "macd_bullish_crossover",
        "description": f"MACD bullish crossover {label}",
        "index": index,
        "confidence": 70,
        "details": {"is_bullish": True, "periods_ago": age},
    }


@pytest.fixture
def technical_formatter(format_utils):
    return TechnicalFormatter(technical_calculator=MagicMock(), format_utils=format_utils)


@pytest.mark.parametrize(
    ("category", "expected_bars"),
    [("macd", 10), ("rsi", 10), ("stochastic", 10), ("ma_crossover", 50), ("volatility", 5), ("divergence", 20)],
)
def test_staleness_window_is_the_category_target_in_bars(category, expected_bars):
    assert staleness_threshold(category, "4h") == expected_bars


def test_unknown_category_keeps_the_default_window_and_a_missing_frame_is_refused():
    assert staleness_threshold("unknown", "4h") == 10
    with pytest.raises(ValueError, match="timeframe is required"):
        staleness_threshold("macd", None)


def test_recency_threshold_is_the_stricter_of_time_and_history_windows():
    assert recent_history_window("macd", TOTAL_CANDLES) == 149
    assert recency_threshold("macd", "4h", TOTAL_CANDLES) == 10
    assert recent_history_window("ma_crossover", 60) == 18
    assert recency_threshold("ma_crossover", "4h", 60) == 18


def test_is_pattern_recent_boundaries():
    assert is_pattern_recent(995, TOTAL_CANDLES, "macd", "4h") is True
    assert is_pattern_recent(988, TOTAL_CANDLES, "macd", "4h") is True
    assert is_pattern_recent(987, TOTAL_CANDLES, "macd", "4h") is False
    assert is_pattern_recent(300, TOTAL_CANDLES, "macd", "4h") is False
    assert is_pattern_recent(None, TOTAL_CANDLES, "macd", "4h") is True
    assert is_pattern_recent(300, None, "macd", "4h") is True


def test_filter_recent_patterns_drops_stale_entries_and_keeps_shape():
    patterns = {"macd": [macd_pattern(300), macd_pattern(997)], "rsi": [macd_pattern(400)]}

    filtered = filter_recent_patterns(patterns, TOTAL_CANDLES, "4h")

    assert list(filtered) == ["macd", "rsi"]
    assert [p["index"] for p in filtered["macd"]] == [997]
    assert filtered["rsi"] == []


def test_stale_crossover_never_reaches_the_prompt(technical_formatter):
    context = MagicMock()
    context.ohlcv_candles = np.zeros((TOTAL_CANDLES, 6))
    context.technical_patterns = {"macd": [macd_pattern(300)]}

    assert technical_formatter._format_patterns_section(context, "4h") == ""


def test_recent_crossover_still_reaches_the_prompt(technical_formatter):
    context = MagicMock()
    context.ohlcv_candles = np.zeros((TOTAL_CANDLES, 6))
    context.technical_patterns = {"macd": [macd_pattern(994)]}

    section = technical_formatter._format_patterns_section(context, "4h")

    assert "MACD bull × 4 bars ago" in section


def test_quality_score_counts_the_same_patterns_the_prompt_shows():

    scorer = PatternQualityScorer()
    stale = {"macd": [macd_pattern(300)] * 3}

    unfiltered = scorer.score(patterns=stale, tech_data={})
    filtered = scorer.score(patterns=filter_recent_patterns(stale, TOTAL_CANDLES, "4h"), tech_data={})

    assert unfiltered.quantity_score > filtered.quantity_score
    assert filtered.quantity_score == 0.0
