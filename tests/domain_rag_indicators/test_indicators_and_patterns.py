"""Dense domain tests for indicator classification, pattern scoring and the pattern engine.

Four suites were folded into this module: the indicator_classifier utilities
(label tables, context/query strings, exit-execution normalization), the
PatternQualityScorer components, the numba indicator-pattern detectors and the
candlestick chart builder. Value-only variants became rows of parametrized
matrices, and every value recorded from the pre-refactor implementation stays
asserted: a difference means production behaviour changed, not that a test is stale.
"""

from __future__ import annotations

import io
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.analyzer.pattern_engine.chart_generator import ChartGenerator
from src.analyzer.pattern_engine.indicator_patterns.divergence_patterns import (
    detect_bearish_divergence_numba,
    detect_bullish_divergence_numba,
)
from src.analyzer.pattern_engine.indicator_patterns.rsi_patterns import (
    detect_rsi_overbought_numba,
    detect_rsi_oversold_numba,
)
from src.analyzer.pattern_engine.indicator_patterns.stochastic_patterns import (
    detect_stoch_overbought_numba,
    detect_stoch_oversold_numba,
)
from src.analyzer.pattern_engine.indicator_patterns.volume_patterns import (
    detect_climax_volume_numba,
    detect_volume_dryup_numba,
    detect_volume_spike_numba,
)
from src.analyzer.pattern_quality_scorer import (
    QUALITY_DISCREPANCY_THRESHOLD,
    WEIGHT_INDICATOR_ALIGNMENT,
    WEIGHT_PATTERN_CONFIRMATION,
    WEIGHT_PATTERN_QUANTITY,
    WEIGHT_PATTERN_RECENCY,
    PatternQualityScorer,
    QualityScore,
)
from src.indicators.trend.sar_utils import (
    get_initial_sar_state,
    initialize_sar_arrays,
    update_bearish_sar,
    update_bullish_sar,
)
from src.trading.data_models import ExitExecutionContext, MarketSnapshot
from src.utils.indicator_classifier import (
    build_context_string_from_classified_values,
    build_context_string_from_technical_data,
    build_exit_execution_context,
    build_exit_execution_context_from_config,
    build_query_document_from_classified_values,
    build_query_document_from_technical_data,
    classify_adx_label,
    classify_bb_position,
    classify_macd_signal,
    classify_market_sentiment,
    classify_order_book_bias,
    classify_rsi_label,
    classify_rsi_level,
    classify_trend_direction,
    classify_volatility_level,
    classify_volume_state,
    format_exit_execution_context,
    resolve_scalar,
)
from tests.conftest import make_config

REPO_ROOT = Path(__file__).resolve().parents[2]
NAN = float("nan")

EXIT_HARD_15M = build_exit_execution_context(
    stop_loss_type="hard",
    stop_loss_check_interval="15m",
    take_profit_type="hard",
    take_profit_check_interval="15m",
)
EXIT_HARD_SOFT = build_exit_execution_context(
    stop_loss_type="hard",
    stop_loss_check_interval="15m",
    take_profit_type="soft",
    take_profit_check_interval="4h",
)


def candles(*values: float) -> np.ndarray:
    """Float64 series from the given values."""
    return np.array([float(value) for value in values], dtype=float)


def flat(value: float, count: int) -> np.ndarray:
    """Constant series of *count* copies of *value*."""
    return np.full(count, value, dtype=float)


@pytest.mark.parametrize(
    ("adx", "expected"),
    [
        (30, "High ADX"),
        (25, "High ADX"),
        (22, "Medium ADX"),
        (20, "Medium ADX"),
        (19.99, "Low ADX"),
        (15, "Low ADX"),
        (0, "Low ADX"),
        (80, "High ADX"),
        (NAN, "Medium ADX"),
        (float("inf"), "High ADX"),
    ],
    ids=["30", "25", "22", "20", "19.99", "15", "0", "80", "nan", "inf"],
)
def test_adx_label_follows_its_threshold_table(adx, expected):
    assert classify_adx_label(adx) == expected


@pytest.mark.parametrize(
    ("rsi", "expected"),
    [
        (75, "OVERBOUGHT"),
        (70, "OVERBOUGHT"),
        (69.9, "STRONG"),
        (65, "STRONG"),
        (60, "STRONG"),
        (59.9, "NEUTRAL"),
        (50, "NEUTRAL"),
        (40.5, "NEUTRAL"),
        (40, "WEAK"),
        (35, "WEAK"),
        (30.1, "WEAK"),
        (30, "OVERSOLD"),
        (25, "OVERSOLD"),
        (NAN, "NEUTRAL"),
        (float("inf"), "OVERBOUGHT"),
    ],
    ids=[
        "75", "70", "69.9", "65", "60", "59.9", "50", "40.5",
        "40", "35", "30.1", "30", "25", "nan", "inf",
    ],
)
def test_rsi_label_follows_its_threshold_table(rsi, expected):
    assert classify_rsi_label(rsi) == expected


DICT_CLASSIFIER_CASES: list[tuple[str, Any, dict[str, Any] | None, str]] = [
    ("trend-bullish", classify_trend_direction, {"plus_di": 30, "minus_di": 10}, "BULLISH"),
    ("trend-bearish", classify_trend_direction, {"plus_di": 10, "minus_di": 30}, "BEARISH"),
    ("trend-inside-threshold", classify_trend_direction, {"plus_di": 20, "minus_di": 18}, "NEUTRAL"),
    ("trend-defaults", classify_trend_direction, {}, "NEUTRAL"),
    ("trend-nan", classify_trend_direction, {"plus_di": NAN, "minus_di": 10}, "NEUTRAL"),
    ("trend-inf", classify_trend_direction, {"plus_di": float("inf"), "minus_di": 10}, "BULLISH"),
    ("volatility-high", classify_volatility_level, {"atr_percent": 5.0}, "HIGH"),
    ("volatility-medium", classify_volatility_level, {"atr_percent": 2.0}, "MEDIUM"),
    ("volatility-low", classify_volatility_level, {"atr_percent": 1.0}, "LOW"),
    ("volatility-defaults", classify_volatility_level, {}, "MEDIUM"),
    ("volatility-nan", classify_volatility_level, {"atr_percent": NAN}, "MEDIUM"),
    ("macd-bullish", classify_macd_signal, {"macd_line": 1.5, "macd_signal": 0.5}, "BULLISH"),
    ("macd-bearish", classify_macd_signal, {"macd_line": -0.5, "macd_signal": 0.5}, "BEARISH"),
    ("macd-missing", classify_macd_signal, {}, "NEUTRAL"),
    ("macd-nan", classify_macd_signal, {"macd_line": NAN, "macd_signal": 1.0}, "NEUTRAL"),
    ("volume-accumulation", classify_volume_state, {"obv_slope": 1.0}, "ACCUMULATION"),
    ("volume-distribution", classify_volume_state, {"obv_slope": -1.0}, "DISTRIBUTION"),
    ("volume-normal", classify_volume_state, {"obv_slope": 0.2}, "NORMAL"),
    ("volume-nan", classify_volume_state, {"obv_slope": NAN}, "NORMAL"),
    ("sentiment-extreme-fear", classify_market_sentiment, {"fear_greed_index": 20}, "EXTREME_FEAR"),
    ("sentiment-fear-boundary", classify_market_sentiment, {"fear_greed_index": 25}, "EXTREME_FEAR"),
    ("sentiment-fear", classify_market_sentiment, {"fear_greed_index": 45}, "FEAR"),
    ("sentiment-greed", classify_market_sentiment, {"fear_greed_index": 55}, "GREED"),
    ("sentiment-extreme-greed", classify_market_sentiment, {"fear_greed_index": 80}, "EXTREME_GREED"),
    ("sentiment-missing", classify_market_sentiment, None, "NEUTRAL"),
    ("sentiment-nan", classify_market_sentiment, {"fear_greed_index": NAN}, "NEUTRAL"),
    ("orderbook-buy", classify_order_book_bias, {"order_book": {"imbalance": 0.3}}, "BUY_PRESSURE"),
    ("orderbook-sell", classify_order_book_bias, {"order_book": {"imbalance": -0.3}}, "SELL_PRESSURE"),
    ("orderbook-threshold", classify_order_book_bias, {"order_book": {"imbalance": 0.1}}, "BALANCED"),
    ("orderbook-empty", classify_order_book_bias, {"order_book": {}}, "BALANCED"),
    ("orderbook-missing", classify_order_book_bias, None, "BALANCED"),
    ("orderbook-nan", classify_order_book_bias, {"order_book": {"imbalance": NAN}}, "BALANCED"),
]


@pytest.mark.parametrize(
    ("classifier", "technical_data", "expected"),
    [case[1:] for case in DICT_CLASSIFIER_CASES],
    ids=[case[0] for case in DICT_CLASSIFIER_CASES],
)
def test_dict_classifiers_map_raw_indicators_to_labels(classifier, technical_data, expected):
    assert classifier(technical_data) == expected


@pytest.mark.parametrize(
    ("technical_data", "current_price", "expected"),
    [
        ({"bb_upper": 100, "bb_lower": 80}, 100, "UPPER"),
        ({"bb_upper": 100, "bb_lower": 80}, 99.0, "UPPER"),
        ({"bb_upper": 100, "bb_lower": 80}, 98.99, "MIDDLE"),
        ({"bb_upper": 100, "bb_lower": 80}, 90, "MIDDLE"),
        ({"bb_upper": 100, "bb_lower": 80}, 80.81, "MIDDLE"),
        ({"bb_upper": 100, "bb_lower": 80}, 80.8, "LOWER"),
        ({"bb_upper": 100, "bb_lower": 80}, 80, "LOWER"),
        ({"bb_upper": 100, "bb_lower": 80}, 0, "MIDDLE"),
        ({"bb_upper": 100, "bb_lower": 80}, None, "MIDDLE"),
        ({}, 100, "MIDDLE"),
    ],
    ids=[
        "at-upper", "upper-ratio-edge", "just-inside-upper", "middle",
        "just-inside-lower", "lower-ratio-edge", "at-lower", "zero-price",
        "missing-price", "missing-bands",
    ],
)
def test_bb_position_maps_price_to_band(technical_data, current_price, expected):
    assert classify_bb_position(technical_data, current_price) == expected


def test_rsi_level_reads_the_dict_and_delegates_to_the_label_function():
    assert classify_rsi_level({"rsi": 75}) == "OVERBOUGHT"
    assert classify_rsi_level({}) == "NEUTRAL"
    assert [classify_rsi_level({"rsi": rsi}) for rsi in (10, 30, 40, 50, 60, 70, 90)] == [
        classify_rsi_label(rsi) for rsi in (10, 30, 40, 50, 60, 70, 90)
    ]


@pytest.mark.parametrize(
    ("technical_data", "kwargs", "expected_tokens", "absent_tokens"),
    [
        (
            {"adx": 30, "plus_di": 30, "minus_di": 10, "atr_percent": 2.0},
            {},
            ["BULLISH", "High ADX", "MEDIUM Volatility", " + "],
            ["RSI", "MACD", "Volume", "BB", "Sentiment", "OrderBook", "Exit Execution"],
        ),
        (
            {"adx": 10, "plus_di": 10, "minus_di": 10, "atr_percent": 2.0},
            {},
            ["NEUTRAL + Low ADX + MEDIUM Volatility"],
            [],
        ),
        (
            {"adx": 25, "rsi": 75, "atr_percent": 2.0},
            {},
            ["RSI OVERBOUGHT"],
            ["Opportunity"],
        ),
        (
            {"adx": 25, "rsi": 50, "atr_percent": 2.0},
            {},
            ["NEUTRAL + High ADX + MEDIUM Volatility"],
            ["RSI"],
        ),
        (
            {"adx": 25, "atr_percent": 2.0},
            {"is_weekend": True},
            ["Weekend Low Volume"],
            [],
        ),
        (
            {"adx": 30, "plus_di": 30, "minus_di": 10, "atr_percent": 2.0},
            {"exit_execution_context": EXIT_HARD_15M},
            ["Exit Execution: SL hard/15m | TP hard/15m"],
            [],
        ),
    ],
    ids=[
        "active-signals", "low-adx", "rsi-overbought",
        "rsi-neutral-omitted", "weekend", "exit-execution-context",
    ],
)
def test_context_string_reports_only_active_signals(
    technical_data, kwargs, expected_tokens, absent_tokens
):
    context = build_context_string_from_technical_data(technical_data, **kwargs)

    for token in expected_tokens:
        assert token in context
    for token in absent_tokens:
        assert token not in context


SNAPSHOT = MarketSnapshot(
    trend_direction="BULLISH",
    adx=30.0,
    rsi=61.0,
    volatility_level="MEDIUM",
    rsi_level="STRONG",
    exit_execution_context=EXIT_HARD_SOFT,
    atr_percentage=2.0,
    mfi=0.0,
    cmf=0.0,
)
SIGNAL_TECHNICAL_DATA = {"adx": 30, "rsi": 61, "plus_di": 30, "minus_di": 10, "atr_percent": 2.0}
SIGNAL_CONTEXT = (
    "BULLISH + High ADX + MEDIUM Volatility + RSI STRONG + "
    "Exit Execution: SL hard/15m | TP soft/4h"
)
SIGNAL_QUERY = (
    f"{SIGNAL_CONTEXT} Indicators: ADX=30.0 (High ADX) | RSI=61.0 (STRONG) | Vol=MEDIUM | "
    "MACD=NEUTRAL | BB=MIDDLE | RSI=STRONG Structure: "
    "Exit Execution: SL hard/15m | TP soft/4h | MFI=0.0 | CMF=+0.000"
)


def test_classified_snapshot_path_matches_the_technical_data_path():
    assert (
        build_context_string_from_technical_data(
            SIGNAL_TECHNICAL_DATA, exit_execution_context=EXIT_HARD_SOFT
        )
        == SIGNAL_CONTEXT
    )
    assert build_context_string_from_classified_values(SNAPSHOT) == SIGNAL_CONTEXT
    assert (
        build_query_document_from_technical_data(
            SIGNAL_TECHNICAL_DATA, exit_execution_context=EXIT_HARD_SOFT
        )
        == SIGNAL_QUERY
    )
    assert build_query_document_from_classified_values(SNAPSHOT) == SIGNAL_QUERY


def test_blank_indicators_and_a_default_snapshot_yield_the_neutral_baseline():
    assert build_context_string_from_technical_data({}) == "NEUTRAL + Low ADX + MEDIUM Volatility"
    assert (
        build_context_string_from_classified_values(MarketSnapshot())
        == "NEUTRAL + Low ADX + MEDIUM Volatility"
    )


def test_exit_execution_context_normalizes_case_whitespace_and_unknown_types():
    assert build_exit_execution_context() == ExitExecutionContext()
    assert format_exit_execution_context(build_exit_execution_context()) == ""
    assert build_exit_execution_context(
        stop_loss_type="HARD ",
        take_profit_check_interval=" 4H ",
    ) == ExitExecutionContext(stop_loss_type="hard", take_profit_check_interval="4h")
    assert build_exit_execution_context(
        stop_loss_type="trailing",
        stop_loss_check_interval="15m",
    ) == ExitExecutionContext(stop_loss_check_interval="15m")


def test_exit_execution_context_from_config_uses_the_timeframe_fallbacks(config):
    assert build_exit_execution_context_from_config(config, timeframe="4h") == ExitExecutionContext(
        stop_loss_type="soft",
        stop_loss_check_interval="1h",
        take_profit_type="soft",
        take_profit_check_interval="1h",
    )

    blank_intervals = make_config(
        STOP_LOSS_TYPE="unknown",
        STOP_LOSS_CHECK_INTERVAL="",
        TAKE_PROFIT_TYPE="unknown",
        TAKE_PROFIT_CHECK_INTERVAL="",
    )

    assert build_exit_execution_context_from_config(
        blank_intervals, timeframe="4h"
    ) == ExitExecutionContext(
        stop_loss_type="unknown",
        stop_loss_check_interval="4h",
        take_profit_type="unknown",
        take_profit_check_interval="4h",
    )
    assert build_exit_execution_context_from_config(blank_intervals, timeframe="") == (
        ExitExecutionContext()
    )


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [
        (None, 0.0, 0.0),
        (None, 7.5, 7.5),
        (3, 0.0, 3.0),
        (2.5, 0.0, 2.5),
        ("1.25", 0.0, 1.25),
        ("abc", 0.0, 0.0),
        ("abc", 9.0, 9.0),
        ([1.0, 2.0, 3.0], 0.0, 3.0),
        ([], 0.0, 0.0),
        (np.array([5.0, 6.0]), 0.0, 6.0),
        (np.array(5.0), 0.0, 5.0),
    ],
    ids=[
        "none", "none-with-default", "int", "float", "numeric-string",
        "non-numeric-string", "non-numeric-string-with-default", "list",
        "empty-list", "array", "zero-dim-array",
    ],
)
def test_resolve_scalar_reduces_any_value_to_a_scalar(value, default, expected):
    assert resolve_scalar(value, default) == expected


def test_resolve_scalar_passes_nan_through_and_raises_on_a_mapping():
    assert math.isnan(resolve_scalar(NAN, 7.5))

    with pytest.raises(KeyError):
        resolve_scalar({"confidence": 0.5})


def test_indicator_classifier_imports_without_pulling_the_trading_data_models():
    probe = (
        "import sys;"
        "import src.utils.indicator_classifier as classifier;"
        "assert 'src.trading.data_models' not in sys.modules;"
        "print(classifier.classify_adx_label(30));"
        "print(classifier.build_exit_execution_context().stop_loss_type)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        check=True,
    )

    assert completed.stdout.split() == ["High", "ADX", "unknown"]


@pytest.fixture
def scorer() -> PatternQualityScorer:
    return PatternQualityScorer()


def chart_patterns(*name_and_index: tuple[str, int]) -> dict[str, list[dict[str, Any]]]:
    """Patterns dict in the shape the detector emits: chart list of dicts."""
    return {"chart": [{"name": name, "bar_index": index} for name, index in name_and_index]}


STRONG_BULLISH = chart_patterns(
    ("bullish_engulfing", 98),
    ("hammer", 96),
    ("golden_cross", 94),
    ("rsi_bullish_divergence", 95),
)
MIXED = chart_patterns(("bullish_engulfing", 95), ("bearish_engulfing", 94), ("doji", 93))
HAMMER_DOJI = chart_patterns(("hammer", 95), ("doji", 94))


@pytest.mark.parametrize(
    ("pattern_name", "expected"),
    [
        ("bullish_engulfing", "bullish"),
        ("bearish_engulfing", "bearish"),
        ("hammer", "bullish"),
        ("shooting_star", "bearish"),
        ("doji", "neutral"),
        ("golden_cross", "bullish"),
        ("death_cross", "bearish"),
        ("double_bottom", "bullish"),
        ("head_and_shoulders", "bearish"),
        ("unknown_mystery_pattern", "neutral"),
        ("BULLISH_ENGULFING", "bullish"),
        ("bullish engulfing", "bullish"),
        ("bullish_engulfing_pattern", "bullish"),
        ("Hammer", "bullish"),
    ],
)
def test_pattern_direction_classification_table(scorer, pattern_name, expected):
    assert scorer._classify_direction(pattern_name) == expected


@pytest.mark.parametrize(
    ("patterns", "expected_names"),
    [
        ({"chart": [{"name": "hammer"}, {"name": "doji"}]}, ["hammer", "doji"]),
        (
            {"indicator": [{"type": "rsi_oversold"}, {"type": "macd_bullish_cross"}]},
            ["rsi_oversold", "macd_bullish_cross"],
        ),
        (
            {"chart": [{"name": "hammer"}, {"name": "doji"}], "indicator": [{"pattern": "golden_cross"}]},
            ["hammer", "doji", "golden_cross"],
        ),
        ({"chart": [{"type": "bullish_engulfing"}]}, ["bullish_engulfing"]),
        ({"chart": [{"name": "", "pattern": "hammer"}]}, ["hammer"]),
        ({"chart": [{"name": "hammer", "type": "doji"}]}, ["hammer"]),
        ({"chart": [{"bar_index": 3}]}, []),
        ({}, []),
        (None, []),
    ],
    ids=[
        "name-keys", "type-keys", "mixed-keys", "single-type",
        "empty-name-falls-through", "name-wins-over-type", "entry-without-name",
        "empty-dict", "none",
    ],
)
def test_pattern_name_extraction_flattens_every_supported_shape(scorer, patterns, expected_names):
    assert scorer._extract_pattern_names(patterns) == expected_names


@pytest.mark.parametrize(
    ("pattern_count", "expected"),
    [(0, 0.0), (1, 35.0), (2, 55.0), (3, 70.0), (4, 85.0), (5, 100.0), (20, 100.0)],
    ids=["0", "1", "2", "3", "4", "5", "20"],
)
def test_quantity_scoring_table(scorer, pattern_count, expected):
    assert scorer._score_quantity(pattern_count) == expected


@pytest.mark.parametrize(
    ("bullish_count", "bearish_count", "expected"),
    [
        (5, 0, 100.0),
        (9, 1, 100.0),
        (4, 1, 80.0),
        (8, 2, 80.0),
        (3, 2, 50.0),
        (7, 3, 50.0),
        (4, 6, 50.0),
        (2, 2, 30.0),
        (1, 1, 30.0),
        (0, 0, 0.0),
        (0, 4, 100.0),
    ],
    ids=[
        "all-bullish", "ratio-0.9", "mostly-bullish", "ratio-0.8", "slight-majority",
        "ratio-0.7", "bearish-majority", "even-split", "single-each",
        "no-directional", "all-bearish",
    ],
)
def test_confirmation_scoring_table(scorer, bullish_count, bearish_count, expected):
    assert scorer._score_confirmation(bullish_count, bearish_count) == expected


OLD_PATTERNS = dict(
    chart_patterns(("hammer", 5), ("doji", 12)),
    reference=[{"name": "spinning_top", "bar_index": 100}],
)


@pytest.mark.parametrize(
    ("patterns", "expected"),
    [
        ({"chart": [{"name": "hammer"}]}, 50.0),
        ({"chart": [{"name": "doji", "bar_index": 0}]}, 50.0),
        ({"chart": [{"name": "doji", "bar_index": "abc"}]}, 50.0),
        (chart_patterns(("hammer", 98), ("doji", 99), ("bullish_engulfing", 97)), 98.556999),
        (OLD_PATTERNS, 12.857143),
        (chart_patterns(("hammer", 10), ("bullish_engulfing", 95)), 36.090226),
    ],
    ids=[
        "no-index", "zero-index-is-falsy", "unparsable-index",
        "all-recent", "old-patterns", "mixed-recency",
    ],
)
def test_recency_scoring_table(scorer, patterns, expected):
    assert scorer._score_recency(patterns) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    ("tech_data", "direction", "expected"),
    [
        ({"adx": 40, "rsi": 55}, "bullish", 100.0),
        ({"adx": 40, "rsi": 70}, "bullish", 100.0),
        ({"adx": 40, "rsi": 70.001}, "bullish", 65.0),
        ({"adx": 20, "rsi": 25}, "bullish", 45.0),
        ({"adx": 40, "rsi": 45}, "bearish", 100.0),
        ({"adx": 40, "rsi": 30}, "bearish", 100.0),
        ({"adx": 40, "rsi": 29.999}, "bearish", 65.0),
        ({"adx": 20, "rsi": 75}, "bearish", 45.0),
        ({"adx": 40, "rsi": 30}, "bullish", 65.0),
        ({"adx": 40, "rsi": 55}, "neutral", 25.0),
        ({"adx": 15, "rsi": 50}, "bullish", 50.0),
        ({}, "bullish", 50.0),
        ({"adx": -5, "rsi": 50}, "bullish", 50.0),
        ({"adx": "abc", "rsi": "abc"}, "bullish", 50.0),
        ({"adx": NAN, "rsi": NAN}, "bullish", 50.0),
        ({"adx": float("inf"), "rsi": float("inf")}, "bullish", 50.0),
    ],
    ids=[
        "bullish-aligned", "rsi-upper-edge", "rsi-just-outside-edge", "bullish-oversold-reversal",
        "bearish-aligned", "rsi-lower-edge", "rsi-just-outside-lower-edge", "bearish-overbought",
        "bearish-rsi-with-bullish-direction", "neutral-direction", "low-adx-no-trend",
        "missing-data", "negative-adx", "non-numeric-values", "nan-values", "inf-values",
    ],
)
def test_indicator_alignment_table(scorer, tech_data, direction, expected):
    """Missing, non-numeric, negative and non-finite inputs all land on the neutral 50 (F6)."""
    assert scorer._score_indicator_alignment(tech_data, direction) == expected


@pytest.mark.parametrize(
    ("patterns", "tech_data", "llm_quality", "expected"),
    [
        pytest.param(
            STRONG_BULLISH,
            {"adx": 40, "rsi": 55},
            None,
            (94.844023, 85.0, 100.0, 96.720117, 100.0, "strong", True, 0),
            id="strong-bullish",
        ),
        pytest.param(
            MIXED,
            {"adx": 30, "rsi": 50},
            None,
            (54.699248, 70.0, 30.0, 98.496241, 25.0, "moderate", True, 0),
            id="mixed-patterns",
        ),
        pytest.param(
            {},
            {"adx": 35.0, "rsi": 55.0},
            None,
            (15.0, 0.0, 0.0, 50.0, 25.0, "negligible", True, 0),
            id="no-patterns",
        ),
        pytest.param(
            chart_patterns(("hammer", 95)),
            {},
            None,
            (70.5, 35.0, 100.0, 100.0, 50.0, "moderate", True, 0),
            id="patterns-without-tech-data",
        ),
        pytest.param(
            None,
            None,
            None,
            (15.0, 0.0, 0.0, 50.0, 25.0, "negligible", True, 0),
            id="both-none",
        ),
        pytest.param(
            HAMMER_DOJI,
            {"adx": 35, "rsi": 55},
            65,
            (83.349624, 55.0, 100.0, 99.248120, 85.0, "strong", True, 0),
            id="llm-quality-in-agreement",
        ),
        pytest.param(
            HAMMER_DOJI,
            {"adx": 35, "rsi": 55},
            95.0,
            (83.349624, 55.0, 100.0, 99.248120, 85.0, "strong", True, 0),
            id="llm-quality-within-tolerance",
        ),
        pytest.param(
            HAMMER_DOJI,
            {"adx": 35, "rsi": 55},
            5.0,
            (83.349624, 55.0, 100.0, 99.248120, 85.0, "strong", False, 1),
            id="llm-quality-discrepancy",
        ),
        pytest.param(
            {},
            {},
            "not_a_number",
            (15.0, 0.0, 0.0, 50.0, 25.0, "negligible", True, 0),
            id="llm-quality-not-a-number",
        ),
        pytest.param(
            {},
            {},
            150.0,
            (15.0, 0.0, 0.0, 50.0, 25.0, "negligible", True, 0),
            id="llm-quality-above-range",
        ),
        pytest.param(
            {},
            {},
            -5.0,
            (15.0, 0.0, 0.0, 50.0, 25.0, "negligible", True, 0),
            id="llm-quality-negative",
        ),
    ],
)
def test_score_combines_components_and_validates_llm_quality(
    scorer, patterns, tech_data, llm_quality, expected
):
    quality = scorer.score(patterns=patterns, tech_data=tech_data, llm_quality=llm_quality)

    assert quality.overall == pytest.approx(expected[0], abs=1e-6)
    assert quality.quantity_score == expected[1]
    assert quality.confirmation_score == expected[2]
    assert quality.recency_score == pytest.approx(expected[3], abs=1e-6)
    assert quality.indicator_score == expected[4]
    assert quality.label == expected[5]
    assert quality.passed is expected[6]
    assert len(quality.discrepancies) == expected[7]


def test_llm_quality_discrepancy_rounds_both_scores_into_the_message(scorer):
    """Both scores are rounded once, then the delta is computed from the rounded values:
    overall 77.5 renders as ``78`` and 78 - 5 = 73 (F7 in ``docs/SRC_AUDIT_FINDINGS.md``)."""
    quality = scorer.score(
        patterns=chart_patterns(("hammer", 95)),
        tech_data={"adx": 35, "rsi": 55},
        llm_quality=5.0,
    )

    assert quality.discrepancies == [
        "Pattern quality: LLM reported 5, computed 78 (strong). Delta=73 > 25."
    ]


def test_overwrite_llm_quality_replaces_the_score_and_attaches_the_validation_block(scorer):
    analysis: dict[str, Any] = {"pattern_quality": 99}
    quality = scorer.score(patterns={}, tech_data={})

    result = scorer.overwrite_llm_quality(analysis, quality)

    assert result is analysis
    assert result["pattern_quality"] == 15
    assert result["_pattern_validation"] == {
        "overall": 15,
        "quantity_score": 0.0,
        "confirmation_score": 0.0,
        "recency_score": 50.0,
        "indicator_score": 25.0,
        "label": "negligible",
        "discrepancies": [],
        "passed": True,
    }


def test_quality_score_defaults_labels_and_serialization():
    default = QualityScore()

    assert (default.overall, default.passed, default.label) == (0.0, True, "negligible")
    assert default.to_dict() == {
        "overall": 0,
        "quantity_score": 0.0,
        "confirmation_score": 0.0,
        "recency_score": 0.0,
        "indicator_score": 0.0,
        "label": "negligible",
        "discrepancies": [],
        "passed": True,
    }
    assert [
        QualityScore(overall=overall).label
        for overall in (75.0, 74.999, 50.0, 49.999, 25.0, 24.999)
    ] == ["strong", "moderate", "moderate", "weak", "weak", "negligible"]


def test_scoring_weights_and_discrepancy_threshold_are_normalized():
    weights = (
        WEIGHT_PATTERN_QUANTITY,
        WEIGHT_PATTERN_CONFIRMATION,
        WEIGHT_PATTERN_RECENCY,
        WEIGHT_INDICATOR_ALIGNMENT,
    )

    assert sum(weights) == pytest.approx(1.0, abs=0.001)
    assert QUALITY_DISCREPANCY_THRESHOLD == 25.0


BASELINE: dict[str, list] = json.loads(r"""
{
 "rsi_oversold::empty": [false, -1, 0.0],
 "rsi_overbought::empty": [false, -1, 0.0],
 "rsi_oversold::mid": [false, -1, 50.0],
 "rsi_overbought::mid": [false, -1, 50.0],
 "rsi_oversold::oversold_run": [true, 0, 22.0],
 "rsi_overbought::oversold_run": [false, -1, 22.0],
 "rsi_oversold::at_threshold": [false, -1, 30.0],
 "rsi_overbought::at_threshold": [false, -1, 30.0],
 "rsi_oversold::touch_below_once": [false, -1, 45.0],
 "rsi_overbought::touch_below_once": [false, -1, 45.0],
 "rsi_oversold::overbought_run": [false, -1, 80.0],
 "rsi_overbought::overbought_run": [true, 0, 80.0],
 "rsi_oversold::near_flat": [true, 0, 29.6],
 "rsi_overbought::near_flat": [false, -1, 29.6],
 "rsi_oversold::min_periods_3": [true, 0, 23.0],
 "rsi_overbought::min_periods_3": [true, 0, 77.0],
 "stoch_oversold::empty": [false, 0, 0.0],
 "stoch_overbought::empty": [false, 0, 0.0],
 "stoch_oversold::nan_last": [false, 0, 0.0],
 "stoch_overbought::nan_last": [false, 0, 0.0],
 "stoch_oversold::low": [true, 0, 15.0],
 "stoch_overbought::low": [false, 0, 0.0],
 "stoch_oversold::boundary_low": [false, 0, 0.0],
 "stoch_overbought::boundary_low": [false, 0, 0.0],
 "stoch_oversold::high": [false, 0, 0.0],
 "stoch_overbought::high": [true, 0, 85.0],
 "stoch_oversold::boundary_high": [false, 0, 0.0],
 "stoch_overbought::boundary_high": [false, 0, 0.0],
 "volume_spike::short": [false, 0.0, 0.0, 0.0],
 "volume_dryup::short": [false, 0.0, 0.0, 0.0],
 "volume_climax::short": [false, 0.0, 0.0, 0.0],
 "volume_spike::nan_last": [false, 0.0, 0.0, 0.0],
 "volume_dryup::nan_last": [false, 0.0, 0.0, 0.0],
 "volume_climax::nan_last": [false, 0.0, 0.0, 0.0],
 "volume_spike::flat_spike": [true, 300.0, 100.0, 3.0],
 "volume_dryup::flat_spike": [false, 300.0, 100.0, 3.0],
 "volume_climax::flat_spike": [false, 0.0, 0.0, 0.0],
 "volume_spike::flat_no_spike": [false, 150.0, 100.0, 1.5],
 "volume_dryup::flat_no_spike": [false, 150.0, 100.0, 1.5],
 "volume_climax::flat_no_spike": [false, 0.0, 0.0, 0.0],
 "volume_spike::flat_dryup": [false, 30.0, 100.0, 0.3],
 "volume_dryup::flat_dryup": [true, 30.0, 100.0, 0.3],
 "volume_climax::flat_dryup": [false, 0.0, 0.0, 0.0],
 "volume_spike::flat_no_dryup": [false, 80.0, 100.0, 0.8],
 "volume_dryup::flat_no_dryup": [false, 80.0, 100.0, 0.8],
 "volume_climax::flat_no_dryup": [false, 0.0, 0.0, 0.0],
 "volume_spike::climax": [true, 400.0, 100.0, 4.0],
 "volume_dryup::climax": [false, 400.0, 100.0, 4.0],
 "volume_climax::climax": [true, 400.0, 100.0, 4.0],
 "volume_spike::negative_avg": [false, 0.0, 0.0, 0.0],
 "volume_dryup::negative_avg": [false, 0.0, 0.0, 0.0],
 "volume_climax::negative_avg": [false, 0.0, 0.0, 0.0],
 "divergence_bullish::short": [false, -1, -1, 0.0, 0.0, 0.0, 0.0],
 "divergence_bearish::short": [false, -1, -1, 0.0, 0.0, 0.0, 0.0],
 "divergence_bullish::flat": [false, -1, -1, 0.0, 0.0, 0.0, 0.0],
 "divergence_bearish::flat": [false, -1, -1, 0.0, 0.0, 0.0, 0.0],
 "divergence_bullish::trend_down": [false, -1, -1, 0.0, 0.0, 0.0, 0.0],
 "divergence_bearish::trend_down": [false, -1, -1, 0.0, 0.0, 0.0, 0.0],
 "divergence_bullish::seeded_7": [false, -1, -1, 0.0, 0.0, 0.0, 0.0],
 "divergence_bearish::seeded_7": [false, -1, -1, 0.0, 0.0, 0.0, 0.0],
 "divergence_bullish::seeded_11": [false, -1, -1, 0.0, 0.0, 0.0, 0.0],
 "divergence_bearish::seeded_11": [false, -1, -1, 0.0, 0.0, 0.0, 0.0],
 "sar_bullish::i1": [1, 99.02, 102.0, 0.04],
 "sar_bearish::i1": [-1, 199.98, 101.0, 0.04],
 "sar_bullish::i2": [-1, NaN, 102.0, 0.02],
 "sar_bearish::i2": [1, NaN, 103.0, 0.02],
 "sar_bullish::i3": [-1, NaN, 101.0, 0.02],
 "sar_bearish::i3": [1, NaN, 102.0, 0.02],
 "sar_bullish::i5": [-1, NaN, 104.0, 0.02],
 "sar_bearish::i5": [1, NaN, 105.0, 0.02],
 "sar_initial_state": [1.0, 100.0, 101.0, 0.02],
 "divergence_bullish::bull_true": [true, 15, 30, 95.0, 90.0, 30.0, 40.0],
 "divergence_bearish::bull_true": [false, -1, -1, 0.0, 0.0, 0.0, 0.0],
 "divergence_bullish::bull_false": [false, -1, -1, 0.0, 0.0, 0.0, 0.0],
 "divergence_bearish::bull_false": [false, -1, -1, 0.0, 0.0, 0.0, 0.0],
 "divergence_bullish::bear_true": [false, -1, -1, 0.0, 0.0, 0.0, 0.0],
 "divergence_bearish::bear_true": [true, 15, 30, 105.0, 110.0, 70.0, 60.0],
 "divergence_bullish::bear_false": [false, -1, -1, 0.0, 0.0, 0.0, 0.0],
 "divergence_bearish::bear_false": [false, -1, -1, 0.0, 0.0, 0.0, 0.0]
}
""")


def matches_recorded(actual: Any, case: str) -> bool:
    """Compare a detector result with the recorded baseline, treating NaN as equal."""
    expected = BASELINE[case]
    return len(actual) == len(expected) and all(
        (value == recorded) or (math.isnan(value) and math.isnan(recorded))
        for value, recorded in zip(actual, expected)
    )


RSI_MID = flat(50.0, 10)
RSI_DOWN = candles(60, 55, 40, 28, 26, 24, 22)
RSI_AT_THRESHOLD = candles(60, 40, 30.0)
RSI_TOUCHED_ONCE = candles(60, 29.0, 45.0)
RSI_UP = candles(40, 50, 65, 72, 75, 78, 80)
RSI_NEAR_FLAT = candles(29.9, 29.8, 29.7, 29.6)

RSI_CASES: dict[str, tuple[np.ndarray, tuple, np.ndarray, tuple]] = {
    "empty": (candles(), (), candles(), ()),
    "mid": (RSI_MID, (), RSI_MID, ()),
    "oversold_run": (RSI_DOWN, (), RSI_DOWN, ()),
    "at_threshold": (RSI_AT_THRESHOLD, (), RSI_AT_THRESHOLD, ()),
    "touch_below_once": (RSI_TOUCHED_ONCE, (), RSI_TOUCHED_ONCE, ()),
    "overbought_run": (RSI_UP, (), RSI_UP, ()),
    "near_flat": (RSI_NEAR_FLAT, (), RSI_NEAR_FLAT, ()),
    "min_periods_3": (candles(40, 25, 24, 23), (30.0, 3), candles(60, 75, 76, 77), (70.0, 3)),
}

STOCH_CASES: dict[str, np.ndarray] = {
    "empty": candles(),
    "nan_last": candles(50, 40, NAN),
    "low": candles(60, 40, 15),
    "boundary_low": candles(60, 40, 20.0),
    "high": candles(40, 60, 85),
    "boundary_high": candles(40, 60, 80.0),
}

VOLUME_CASES: dict[str, np.ndarray] = {
    "short": candles(100, 110),
    "nan_last": candles(*([100.0] * 25), NAN),
    "flat_spike": candles(*([100.0] * 25), 300.0),
    "flat_no_spike": candles(*([100.0] * 25), 150.0),
    "flat_dryup": candles(*([100.0] * 25), 30.0),
    "flat_no_dryup": candles(*([100.0] * 25), 80.0),
    "climax": candles(*([100.0] * 55), 400.0),
    "negative_avg": candles(*([-5.0] * 25), 100.0),
}


def interpolated(anchors: list[tuple[int, float]], length: int = 45) -> np.ndarray:
    """Piecewise-linear series through the anchor points of the crafted cases."""
    xs = [anchor[0] for anchor in anchors]
    ys = [anchor[1] for anchor in anchors]
    return np.interp(np.arange(length), xs, ys)


def seeded_pair(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic price/indicator pair drawn sequentially from one RandomState."""
    rng = np.random.RandomState(seed)
    return np.round(rng.uniform(90, 110, 60), 3), np.round(rng.uniform(20, 80, 60), 3)


DIVERGENCE_CASES: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "short": (candles(*range(8)), candles(*range(8))),
    "flat": (flat(100.0, 40), flat(50.0, 40)),
    "trend_down": (
        candles(*[100 - i for i in range(40)]),
        candles(*[60 - i for i in range(40)]),
    ),
    "seeded_7": seeded_pair(7),
    "seeded_11": seeded_pair(11),
    "bull_true": (
        interpolated([(0, 110), (15, 95), (22, 100), (30, 90), (44, 104)]),
        interpolated([(0, 60), (15, 30), (22, 55), (30, 40), (44, 70)]),
    ),
    "bull_false": (
        interpolated([(0, 110), (15, 95), (22, 100), (30, 90), (44, 104)]),
        interpolated([(0, 60), (15, 45), (22, 55), (30, 35), (44, 70)]),
    ),
    "bear_true": (
        interpolated([(0, 90), (15, 105), (22, 100), (30, 110), (44, 96)]),
        interpolated([(0, 40), (15, 70), (22, 45), (30, 60), (44, 30)]),
    ),
    "bear_false": (
        interpolated([(0, 90), (15, 105), (22, 100), (30, 110), (44, 96)]),
        interpolated([(0, 40), (15, 60), (22, 45), (30, 75), (44, 30)]),
    ),
}

SAR_HIGH = candles(101, 102, 103, 102, 104, 105)
SAR_LOW = candles(100, 101, 102, 101, 103, 104)
SAR_INDICES = (1, 2, 3, 5)


def seeded_sar(start_sar: float, start_ep: float, start_af: float) -> tuple[np.ndarray, ...]:
    """Fresh SAR state arrays seeded with the recorded starting values."""
    sar, ep, af = initialize_sar_arrays(len(SAR_HIGH))
    sar[0] = start_sar
    ep[0] = start_ep
    af[0] = start_af
    return sar, ep, af


@pytest.mark.parametrize("vector", list(RSI_CASES), ids=list(RSI_CASES))
def test_rsi_threshold_detectors_match_recorded_baseline(vector):
    oversold_series, oversold_args, overbought_series, overbought_args = RSI_CASES[vector]

    assert matches_recorded(
        detect_rsi_oversold_numba(oversold_series, *oversold_args),
        f"rsi_oversold::{vector}",
    )
    assert matches_recorded(
        detect_rsi_overbought_numba(overbought_series, *overbought_args),
        f"rsi_overbought::{vector}",
    )


@pytest.mark.parametrize("vector", list(STOCH_CASES), ids=list(STOCH_CASES))
def test_stochastic_threshold_detectors_match_recorded_baseline(vector):
    series = STOCH_CASES[vector]

    assert matches_recorded(detect_stoch_oversold_numba(series), f"stoch_oversold::{vector}")
    assert matches_recorded(detect_stoch_overbought_numba(series), f"stoch_overbought::{vector}")


@pytest.mark.parametrize("vector", list(VOLUME_CASES), ids=list(VOLUME_CASES))
def test_volume_detectors_match_recorded_baseline(vector):
    series = VOLUME_CASES[vector]

    assert matches_recorded(detect_volume_spike_numba(series), f"volume_spike::{vector}")
    assert matches_recorded(detect_volume_dryup_numba(series), f"volume_dryup::{vector}")
    assert matches_recorded(detect_climax_volume_numba(series), f"volume_climax::{vector}")


@pytest.mark.parametrize("vector", list(DIVERGENCE_CASES), ids=list(DIVERGENCE_CASES))
def test_divergence_detectors_match_recorded_baseline(vector):
    prices, indicator = DIVERGENCE_CASES[vector]

    assert matches_recorded(
        detect_bullish_divergence_numba(prices, indicator),
        f"divergence_bullish::{vector}",
    )
    assert matches_recorded(
        detect_bearish_divergence_numba(prices, indicator),
        f"divergence_bearish::{vector}",
    )


@pytest.mark.parametrize("index", SAR_INDICES)
def test_sar_updates_match_recorded_baseline(index):
    sar, ep, af = seeded_sar(99.0, 100.0, 0.02)
    bullish = update_bullish_sar(index, SAR_HIGH, SAR_LOW, sar, ep, af, 0.02, 0.2)

    assert matches_recorded(
        (
            bullish,
            round(float(sar[index]), 6),
            round(float(ep[index]), 6),
            round(float(af[index]), 6),
        ),
        f"sar_bullish::i{index}",
    )

    sar, ep, af = seeded_sar(200.0, 199.0, 0.02)
    bearish = update_bearish_sar(index, SAR_HIGH, SAR_LOW, sar, ep, af, 0.02, 0.2)

    assert matches_recorded(
        (
            bearish,
            round(float(sar[index]), 6),
            round(float(ep[index]), 6),
            round(float(af[index]), 6),
        ),
        f"sar_bearish::i{index}",
    )


def test_initial_sar_state_matches_recorded_baseline():
    assert matches_recorded(
        tuple(round(float(value), 6) for value in get_initial_sar_state(SAR_HIGH, SAR_LOW, 0.02)),
        "sar_initial_state",
    )


def test_every_recorded_detector_case_is_covered_by_the_matrices():
    registered = (
        {f"rsi_oversold::{vector}" for vector in RSI_CASES}
        | {f"rsi_overbought::{vector}" for vector in RSI_CASES}
        | {f"stoch_oversold::{vector}" for vector in STOCH_CASES}
        | {f"stoch_overbought::{vector}" for vector in STOCH_CASES}
        | {f"volume_spike::{vector}" for vector in VOLUME_CASES}
        | {f"volume_dryup::{vector}" for vector in VOLUME_CASES}
        | {f"volume_climax::{vector}" for vector in VOLUME_CASES}
        | {f"divergence_bullish::{vector}" for vector in DIVERGENCE_CASES}
        | {f"divergence_bearish::{vector}" for vector in DIVERGENCE_CASES}
        | {f"sar_bullish::i{index}" for index in SAR_INDICES}
        | {f"sar_bearish::i{index}" for index in SAR_INDICES}
        | {"sar_initial_state"}
    )

    assert registered == set(BASELINE)


def test_all_nan_series_are_guarded_or_propagate_through_the_detectors():
    nan_rsi = flat(NAN, 5)

    oversold = detect_rsi_oversold_numba(nan_rsi)
    overbought = detect_rsi_overbought_numba(nan_rsi)

    assert (not oversold[0], oversold[1], math.isnan(oversold[2])) == (True, -1, True)
    assert (not overbought[0], overbought[1], math.isnan(overbought[2])) == (True, -1, True)
    assert detect_stoch_oversold_numba(nan_rsi) == (False, 0, 0.0)
    assert detect_stoch_overbought_numba(nan_rsi) == (False, 0, 0.0)
    assert detect_volume_spike_numba(flat(NAN, 30)) == (False, 0.0, 0.0, 0.0)

    nan_series = flat(NAN, 40)

    assert detect_bullish_divergence_numba(nan_series, nan_series) == (
        False, -1, -1, 0.0, 0.0, 0.0, 0.0,
    )
    assert detect_bearish_divergence_numba(nan_series, nan_series) == (
        False, -1, -1, 0.0, 0.0, 0.0, 0.0,
    )


def test_constant_zero_and_negative_volume_series_fail_the_average_guard():
    zeros = flat(0.0, 30)

    assert detect_volume_spike_numba(zeros) == (False, 0.0, 0.0, 0.0)
    assert detect_volume_dryup_numba(zeros) == (False, 0.0, 0.0, 0.0)
    assert detect_climax_volume_numba(zeros) == (False, 0.0, 0.0, 0.0)
    assert detect_volume_spike_numba(flat(-5.0, 30)) == (False, 0.0, 0.0, 0.0)


def test_volume_ratios_trigger_exactly_at_their_thresholds():
    assert detect_volume_spike_numba(candles(*([100.0] * 20), 250.0)) == (True, 250.0, 100.0, 2.5)
    assert detect_volume_spike_numba(candles(*([100.0] * 20), 249.9)) == (False, 249.9, 100.0, 2.499)
    assert detect_volume_dryup_numba(candles(*([100.0] * 20), 50.0)) == (True, 50.0, 100.0, 0.5)
    assert detect_volume_dryup_numba(candles(*([100.0] * 20), 50.1)) == (False, 50.1, 100.0, 0.501)


def test_volume_detectors_need_lookback_plus_one_candles():
    assert detect_volume_spike_numba(flat(100.0, 20)) == (False, 0.0, 0.0, 0.0)
    assert detect_volume_spike_numba(candles(*([100.0] * 20), 300.0)) == (True, 300.0, 100.0, 3.0)
    assert detect_climax_volume_numba(candles(*([100.0] * 49), 400.0)) == (False, 0.0, 0.0, 0.0)
    assert detect_climax_volume_numba(candles(*([100.0] * 50), 400.0)) == (True, 400.0, 100.0, 4.0)


def test_threshold_breaches_are_strict_at_the_boundary():
    assert detect_rsi_oversold_numba(candles(60, 40, 30.0)) == (False, -1, 30.0)
    assert detect_rsi_oversold_numba(candles(60, 40, 29.99)) == (True, 0, 29.99)
    assert detect_rsi_overbought_numba(candles(40, 60, 70.0)) == (False, -1, 70.0)
    assert detect_rsi_overbought_numba(candles(40, 60, 70.01)) == (True, 0, 70.01)
    assert detect_stoch_oversold_numba(candles(60, 40, 20.0)) == (False, 0, 0.0)
    assert detect_stoch_oversold_numba(candles(60, 40, 19.99)) == (True, 0, 19.99)
    assert detect_stoch_overbought_numba(candles(40, 60, 80.0)) == (False, 0, 0.0)
    assert detect_stoch_overbought_numba(candles(40, 60, 80.01)) == (True, 0, 80.01)


def test_divergence_needs_ten_points_and_rejects_constant_series():
    nine = candles(*range(9))

    assert detect_bullish_divergence_numba(nine, nine) == (False, -1, -1, 0.0, 0.0, 0.0, 0.0)
    assert detect_bearish_divergence_numba(nine, nine) == (False, -1, -1, 0.0, 0.0, 0.0, 0.0)
    assert detect_bearish_divergence_numba(flat(100.0, 21), flat(50.0, 21)) == (
        False, -1, -1, 0.0, 0.0, 0.0, 0.0,
    )


def test_sar_acceleration_factor_is_clamped_at_the_max_step():
    sar, ep, af = seeded_sar(99.0, 100.0, 0.2)

    trends = [
        update_bullish_sar(index, SAR_HIGH, SAR_LOW, sar, ep, af, 0.02, 0.2)
        for index in range(1, len(SAR_HIGH))
    ]

    assert trends == [1, 1, 1, 1, 1]
    assert [round(float(value), 6) for value in af] == [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    assert [round(float(value), 6) for value in ep] == [100.0, 102.0, 103.0, 103.0, 104.0, 105.0]
    assert [round(float(value), 6) for value in sar] == [99.0, 99.2, 99.76, 100.408, 100.9264, 101.0]


def test_initial_sar_state_on_a_flat_series_stays_bullish_at_the_price():
    flat_series = flat(100.0, 6)

    assert get_initial_sar_state(flat_series, flat_series, 0.02) == (1, 100.0, 100.0, 0.02)


CHART_BASELINE: dict[str, dict[str, Any]] = json.loads(r"""
{
 "full": {
  "annotation_texts_head": ["104.50", "95.51", "95.51", "104.50", "104.50", "MAX: 104.50"],
  "annotations": 20,
  "height": 1080,
  "shapes": 14,
  "trace_names": ["Price", "SMA 50", "SMA 200", "RSI (14)", "Volume", "CMF (20)", "OBV"],
  "trace_types": ["candlestick", "scatter", "scatter", "scatter", "bar", "scatter", "scatter"],
  "traces": 7,
  "width": 1920,
  "xaxis_count": 4,
  "yaxis_count": 5,
  "yaxis_tickformats": [".2f"]
 },
 "no_history": {
  "annotation_texts_head": ["104.50", "95.51", "95.51", "104.50", "104.50", "MAX: 104.50"],
  "annotations": 17,
  "height": 1080,
  "shapes": 6,
  "trace_names": ["Price", "Volume"],
  "trace_types": ["candlestick", "bar"],
  "traces": 2,
  "width": 1920,
  "xaxis_count": 4,
  "yaxis_count": 5,
  "yaxis_tickformats": [".2f"]
 },
 "explicit_timestamps": {
  "annotation_texts_head": ["104.50", "95.51", "95.51", "104.50", "104.50", "MAX: 104.50"],
  "annotations": 20,
  "height": 1080,
  "shapes": 14,
  "trace_names": ["Price", "SMA 50", "SMA 200", "RSI (14)", "Volume", "CMF (20)", "OBV"],
  "trace_types": ["candlestick", "scatter", "scatter", "scatter", "bar", "scatter", "scatter"],
  "traces": 7,
  "width": 1920,
  "xaxis_count": 4,
  "yaxis_count": 5,
  "yaxis_tickformats": [".2f"]
 },
 "short_series": {
  "annotation_texts_head": [
   "MAX: 104.50",
   "MIN: 98.50",
   "$100",
   "2K",
   "2K",
   "<span style='color:#ff8c00'>\u2501</span> SMA 50 (Short-term trend)<br><span style='color:#9932cc'>\u2501</span> SMA 200 (Long-term trend)<br><b>Golden Cross:</b> SMA50 crosses above SMA200 = Bullish<br><b>Death Cross:</b> SMA50 crosses below SMA200 = Bearish"
  ],
  "annotations": 8,
  "height": 900,
  "shapes": 6,
  "trace_names": ["Price", "SMA 50", "SMA 200", "RSI (14)", "Volume", "CMF (20)", "OBV"],
  "trace_types": ["candlestick", "scatter", "scatter", "scatter", "bar", "scatter", "scatter"],
  "traces": 7,
  "width": 1600,
  "xaxis_count": 4,
  "yaxis_count": 5,
  "yaxis_tickformats": [".2f"]
 }
}
""")


def ohlcv_window(count: int = 60, base: float = 100.0) -> np.ndarray:
    """Synthetic OHLCV matrix in the column order the chart builder expects."""
    rows = []
    start = datetime(2026, 3, 1, tzinfo=timezone.utc)
    for index in range(count):
        open_price = base + np.sin(index / 5.0) * 3.0
        close_price = base + np.sin((index + 1) / 5.0) * 3.0
        high = max(open_price, close_price) + 1.5
        low = min(open_price, close_price) - 1.5
        volume = 1000.0 + (index % 7) * 250.0
        stamp = int((start + timedelta(hours=index)).timestamp() * 1000)
        rows.append([stamp, open_price, high, low, close_price, volume])
    return np.array(rows, dtype=float)


def indicator_history(count: int = 60) -> dict[str, np.ndarray]:
    """Technical-history series matching the synthetic candle window."""
    index = np.arange(count, dtype=float)
    return {
        "rsi": 30.0 + 40.0 * np.abs(np.sin(index / 6.0)),
        "sma_50": 100.0 + np.sin(index / 9.0),
        "sma_200": 99.0 + np.cos(index / 11.0),
        "cmf": np.sin(index / 4.0) * 0.2,
        "obv": np.cumsum(np.sin(index / 3.0) * 120.0),
    }


def explicit_timestamps(count: int = 60) -> list[datetime]:
    """Explicit hourly candle timestamps starting at the series origin."""
    start = datetime(2026, 3, 1, tzinfo=timezone.utc)
    return [start + timedelta(hours=index) for index in range(count)]


def digest(fig: Any) -> dict[str, Any]:
    """Structural fingerprint of a figure: traces, annotations, shapes and axes."""
    layout_keys = fig.layout.to_plotly_json()
    return {
        "traces": len(fig.data),
        "trace_types": [trace.type for trace in fig.data],
        "trace_names": [trace.name for trace in fig.data],
        "annotations": len(fig.layout.annotations or []),
        "annotation_texts_head": [a.text for a in (fig.layout.annotations or [])[:6]],
        "shapes": len(fig.layout.shapes or []),
        "height": fig.layout.height,
        "width": fig.layout.width,
        "xaxis_count": len([key for key in layout_keys if key.startswith("xaxis")]),
        "yaxis_count": len([key for key in layout_keys if key.startswith("yaxis")]),
        "yaxis_tickformats": [
            fig.layout[key].tickformat
            for key in sorted(layout_keys)
            if key.startswith("yaxis") and fig.layout[key].tickformat
        ],
    }


CHART_CASES: dict[str, dict[str, Any]] = {
    "full": {
        "ohlcv": ohlcv_window(),
        "pair_symbol": "BTC/USDC",
        "timeframe": "4h",
        "height": 1080,
        "width": 1920,
        "timestamps": None,
        "technical_history": indicator_history(),
    },
    "no_history": {
        "ohlcv": ohlcv_window(),
        "pair_symbol": "BTC/USDC",
        "timeframe": "4h",
        "height": 1080,
        "width": 1920,
        "timestamps": None,
        "technical_history": None,
    },
    "explicit_timestamps": {
        "ohlcv": ohlcv_window(),
        "pair_symbol": "BTC/USDC",
        "timeframe": "4h",
        "height": 1080,
        "width": 1920,
        "timestamps": explicit_timestamps(),
        "technical_history": indicator_history(),
    },
    "short_series": {
        "ohlcv": ohlcv_window(8),
        "pair_symbol": "ETH/USDC",
        "timeframe": "1h",
        "height": 900,
        "width": 1600,
        "timestamps": None,
        "technical_history": indicator_history(8),
    },
}

PNG_BYTES = b"\x89PNG\r\n\x1a\nstub-image-payload"


@pytest.fixture
def generator(config) -> ChartGenerator:
    return ChartGenerator(config=config)


def stub_image_export(
    monkeypatch: pytest.MonkeyPatch,
    generator: ChartGenerator,
    payload: bytes = PNG_BYTES,
    error: Exception | None = None,
) -> list[tuple[str, int, int, int]]:
    """Replace the kaleido export with a canned payload, recording every attempt."""
    attempts: list[tuple[str, int, int, int]] = []

    def export(fig: Any, img_format: str, width: int, height: int, scale: int, timeout: int = 30) -> bytes:
        attempts.append((img_format, width, height, scale))
        if error is not None:
            raise error
        return payload

    monkeypatch.setattr(generator, "_image_export_with_timeout", export)
    return attempts


@pytest.mark.parametrize("case", list(CHART_CASES), ids=list(CHART_CASES))
def test_candlestick_chart_structure_matches_recorded_baseline(generator, case):
    fig = generator._create_simple_candlestick_chart(**CHART_CASES[case])

    assert digest(fig) == CHART_BASELINE[case]


def test_single_candle_chart_keeps_every_panel_without_an_x_range(generator):
    fig = generator._create_simple_candlestick_chart(
        ohlcv_window(1), "BTC/USDC", "1h", 900, 1600, None, indicator_history(1)
    )

    assert [trace.name for trace in fig.data] == [
        "Price", "SMA 50", "SMA 200", "RSI (14)", "Volume", "CMF (20)", "OBV",
    ]
    assert len(fig.layout.annotations) == 7
    assert len(fig.layout.shapes) == 6
    assert fig.layout.xaxis.range is None


def test_chart_truncates_the_window_to_the_configured_candle_limit(generator):
    fig = generator._create_simple_candlestick_chart(
        ohlcv_window(250), "BTC/USDC", "1h", 1080, 1920, None, indicator_history(250)
    )

    assert len(fig.data[0].close) == 200
    assert len(fig.data[1].y) == 200
    assert "Last 200 Closed Candles" in fig.layout.title.text


def test_nan_last_close_is_formatted_as_na_in_the_title(generator):
    window = ohlcv_window()
    window[-1, 4] = NAN

    fig = generator._create_simple_candlestick_chart(window, "BTC/USDC", "1h", 1080, 1920, None, None)

    assert "Price: N/A" in fig.layout.title.text
    assert [trace.name for trace in fig.data] == ["Price", "Volume"]


def test_empty_candle_window_raises_before_any_figure_is_built(generator):
    with pytest.raises(IndexError):
        generator._create_simple_candlestick_chart(
            np.empty((0, 6)), "BTC/USDC", "1h", 1080, 1920, None, None
        )


async def test_create_chart_image_wraps_the_export_payload_in_a_rewound_buffer(
    generator, monkeypatch
):
    attempts = stub_image_export(monkeypatch, generator)

    buffer = await generator.create_chart_image(
        ohlcv=ohlcv_window(),
        technical_history=indicator_history(),
        pair_symbol="BTC/USDC",
        timeframe="4h",
        height=400,
        width=600,
    )

    assert type(buffer) is io.BytesIO
    assert buffer.getvalue() == PNG_BYTES
    assert buffer.tell() == 0
    assert attempts == [("png", 600, 400, 1)]


async def test_create_chart_image_writes_the_export_payload_to_disk(generator, monkeypatch, tmp_path):
    stub_image_export(monkeypatch, generator)
    target = tmp_path / "chart.png"

    written = await generator.create_chart_image(
        ohlcv=ohlcv_window(),
        technical_history=indicator_history(),
        pair_symbol="BTC/USDC",
        timeframe="4h",
        height=400,
        width=600,
        save_to_disk=True,
        output_path=str(target),
    )

    assert written == str(target)
    assert target.read_bytes() == PNG_BYTES


async def test_image_export_retries_then_raises_the_last_error(generator, monkeypatch):
    attempts = stub_image_export(monkeypatch, generator, error=TimeoutError("kaleido hung"))
    fig = generator._create_simple_candlestick_chart(
        ohlcv_window(8), "ETH/USDC", "1h", 400, 600, None, None
    )

    with pytest.raises(TimeoutError, match="kaleido hung"):
        await generator._retry_image_export(fig, "png", 600, 400, 1, max_retries=2, timeout=1)

    assert len(attempts) == 2
