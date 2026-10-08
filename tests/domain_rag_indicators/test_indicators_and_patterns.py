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
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.analyzer.market_metrics_calculator import MarketMetricsCalculator
from src.analyzer.pattern_engine.chart_generator import ChartGenerator
from src.analyzer.pattern_engine.indicator_patterns.divergence_patterns import (
    _find_local_extrema_numba,
    detect_bearish_divergence_numba,
    detect_bullish_divergence_numba,
)
from src.analyzer.pattern_engine.indicator_patterns.indicator_pattern_engine import (
    IndicatorPatternEngine,
)
from src.analyzer.pattern_engine.indicator_patterns.rsi_patterns import (
    detect_rsi_overbought_numba,
    detect_rsi_oversold_numba,
)
from src.analyzer.pattern_engine.indicator_patterns.stochastic_patterns import (
    detect_stoch_overbought_numba,
    detect_stoch_oversold_numba,
)
from src.analyzer.pattern_engine.indicator_patterns.volatility_patterns import (
    detect_bb_squeeze_numba,
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
from src.indicators.base.technical_indicators import TechnicalIndicators
from src.indicators.momentum.momentum_indicators import (
    coppock_curve_numba,
    kst_numba,
    macd_numba,
    ppo_numba,
    rmi_numba,
    roc_numba,
    rsi_numba,
    stochastic_numba,
    williams_r_numba,
)
from src.indicators.overlap.overlap_indicators import ema_numba
from src.indicators.sentiment.sentiment_indicators import (
    _calculate_macd_window,
    _calculate_mfi_window,
    _calculate_rsi_window,
)
from src.indicators.statistical.statistical_indicators import (
    entropy_numba,
    hurst_numba,
    kurtosis_numba,
    skew_numba,
    stdev_numba,
    variance_numba,
    zscore_numba,
)
from src.indicators.support_resistance.support_resistance_indicators import (
    fibonacci_pivot_points_numba,
    pivot_points_numba,
)
from src.indicators.trend.sar_utils import (
    get_initial_sar_state,
    initialize_sar_arrays,
    update_bearish_sar,
    update_bullish_sar,
)
from src.indicators.trend.trend_indicators import (
    adx_numba,
    ichimoku_cloud_numba,
    parabolic_sar_numba,
    supertrend_numba,
    td_setup_numba,
    trix_numba,
)
from src.indicators.volatility.volatility_indicators import (
    atr_numba,
    chandelier_exit_numba,
    donchian_channels_numba,
    ebsw_numba,
    keltner_channels_numba,
)
from src.indicators.volume.volume_indicators import (
    ad_line_numba,
    cci_numba,
    chaikin_money_flow_numba,
    force_index_numba,
    mfi_numba,
    net_flow_ratio_numba,
    obv_numba,
    pvt_numba,
    rolling_vwap_numba,
    twap_numba,
)
from src.trading.data_models import ExitExecutionContext, MarketSnapshot
from src.utils.data_utils import get_last_valid_value
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
from tests.conftest import make_config, null_logger

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
    ("volume-accumulation", classify_volume_state, {"net_flow_ratio": 1.0}, "ACCUMULATION"),
    ("volume-distribution", classify_volume_state, {"net_flow_ratio": -1.0}, "DISTRIBUTION"),
    ("volume-normal", classify_volume_state, {"net_flow_ratio": 0.2}, "NORMAL"),
    ("volume-nan", classify_volume_state, {"net_flow_ratio": NAN}, "NORMAL"),
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


def band(widths):
    """Turn band widths into (upper, lower) arrays sharing one middle."""
    widths = np.asarray(widths, dtype=float)
    return 100.0 + widths / 2, 100.0 - widths / 2


def test_bb_squeeze_is_judged_against_120_candles():
    """A series shorter than the window cannot be called a squeeze."""
    upper, lower = band(np.full(60, 1000.0))

    assert detect_bb_squeeze_numba(upper, lower) == (False, 0.0, 0.0, 0.0)


def test_bb_squeeze_flags_only_the_low_end_of_the_width_range():
    widening = np.concatenate([np.linspace(500.0, 1000.0, 119), [2000.0]])
    tightest = np.concatenate([np.linspace(2000.0, 1000.0, 119), [400.0]])

    found, current_width, _, width_percentile = detect_bb_squeeze_numba(*band(widening))
    assert (found, current_width) == (False, 2000.0)
    assert width_percentile > 90.0  # widest band in the window: not a squeeze

    found, current_width, percentile_width, width_percentile = detect_bb_squeeze_numba(*band(tightest))
    assert (found, current_width, width_percentile) == (True, 400.0, 0.0)
    assert current_width <= percentile_width


def test_bb_squeeze_confidence_stays_inside_zero_to_hundred():
    """Regression: the old formula scaled a price width and returned -395982."""
    upper, lower = band(np.concatenate([np.linspace(2000.0, 1000.0, 119), [400.0]]))

    volatility = IndicatorPatternEngine().detect_patterns(
        {"bb_upper": upper, "bb_lower": lower}
    )["volatility"]
    squeeze = [pattern for pattern in volatility if pattern["type"] == "bb_squeeze"]

    assert len(squeeze) == 1
    assert squeeze[0]["confidence"] == 100
    assert squeeze[0]["details"]["width_percentile"] == 0.0


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


def test_sample_excess_kurtosis_matches_adjusted_moments():
    prices = np.linspace(1.0, 80.0, 80) + np.sin(np.arange(80)) * 3
    actual = kurtosis_numba(prices, 20)
    expected = np.full(len(prices), np.nan)
    for i in range(19, len(prices)):
        window = prices[i - 19:i + 1]
        centered = window - np.mean(window)
        second = np.mean(centered ** 2)
        fourth = np.mean(centered ** 4)
        excess = fourth / second ** 2 - 3
        expected[i] = 19 / (18 * 17) * (21 * excess + 6)
    np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_coppock_uses_weighted_moving_average_after_both_roc_windows():
    prices = 100 + np.cumsum(np.random.default_rng(7).normal(0, 2, 120))
    actual = coppock_curve_numba(prices, 14, 11, 10)
    roc14 = (prices[14:] / prices[:-14] - 1) * 100
    roc11 = (prices[11:] / prices[:-11] - 1) * 100
    combined = roc14 + roc11[3:]
    expected = np.array([np.dot(combined[i - 9:i + 1], np.arange(1, 11)) / 55
                         for i in range(9, len(combined))])
    assert np.isnan(actual[:23]).all()
    np.testing.assert_allclose(actual[23:], expected)


def test_td_setup_nine_is_reported_only_on_completion():
    prices = np.concatenate((np.arange(1.0, 40.0), np.arange(39.0, 0.0, -1)))
    actual = td_setup_numba(prices, 9)
    assert np.count_nonzero(actual == 9) == 1
    assert np.count_nonzero(actual == -9) == 1
    assert np.all(actual[13:39] == 0)


def test_volume_indicators_drop_empty_windows_and_restart_on_fresh_volume():
    rng = np.random.default_rng(11)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 405)))
    high = prices * (1 + np.abs(rng.normal(0, 0.004, 405)))
    low = prices * (1 - np.abs(rng.normal(0, 0.004, 405)))
    volume = rng.lognormal(10, 0.5, 405)
    volume[360:390] = 0.0

    mfi = mfi_numba(high, low, prices, volume, 14)
    cmf = chaikin_money_flow_numba(high, low, prices, volume, 20)
    vwap = rolling_vwap_numba(high, low, prices, volume, 20)
    typical = (high + low + prices) / 3
    assert vwap[300] == pytest.approx(np.average(typical[281:301], weights=volume[281:301]))
    flow = (2 * prices - high - low) / (high - low) * volume
    assert cmf[300] == pytest.approx(np.sum(flow[281:301]) / np.sum(volume[281:301]))
    up = np.diff(typical[286:301]) > 0
    down = np.diff(typical[286:301]) < 0
    positive = np.sum(typical[287:301][up] * volume[287:301][up])
    negative = np.sum(typical[287:301][down] * volume[287:301][down])
    assert mfi[300] == pytest.approx(100 * positive / (positive + negative))
    assert np.isnan(mfi[383]) and np.isnan(cmf[383]) and np.isnan(vwap[383])
    assert np.isfinite(mfi[-1]) and 0 <= mfi[-1] <= 100
    assert np.isfinite(cmf[-1]) and -1 <= cmf[-1] <= 1
    assert vwap[-1] == pytest.approx(np.average(typical[-20:], weights=volume[-20:]))


def test_rsi_without_a_full_window_returns_all_nan():
    prices = candles(*np.arange(1.0, 15.0).tolist())
    assert len(prices) == 14

    assert np.isnan(rsi_numba(prices, 14)).all()
    assert np.isnan(rsi_numba(prices[:-1], 14)).all()

    first_window = rsi_numba(candles(*np.arange(1.0, 16.0).tolist()), 14)
    assert np.isnan(first_window[:14]).all()
    assert np.isfinite(first_window[14])


def test_rsi_flat_series_is_neutral_and_directional_runs_hit_the_rails():
    neutral = rsi_numba(flat(100.0, 30), 14)
    assert np.all(neutral[14:] == pytest.approx(50.0))

    rising = rsi_numba(np.arange(1.0, 31.0), 14)
    assert np.all(rising[14:] == pytest.approx(100.0))

    falling = rsi_numba(np.arange(30.0, 0.0, -1.0), 14)
    assert np.all(falling[14:] == pytest.approx(0.0))


def test_rsi_matches_an_independent_wilder_recursion():
    prices = 100 + np.cumsum(np.random.default_rng(3).normal(0, 1.5, 90))
    actual = rsi_numba(prices, 14)

    diffs = np.diff(prices)
    gains = np.maximum(diffs, 0.0)
    losses = np.maximum(-diffs, 0.0)
    expected = np.full(len(prices), np.nan)
    avg_gain = gains[:14].mean()
    avg_loss = losses[:14].mean()
    expected[14] = 100 - 100 / (1 + avg_gain / avg_loss)
    for i in range(14, len(gains)):
        avg_gain = (avg_gain * 13 + gains[i]) / 14
        avg_loss = (avg_loss * 13 + losses[i]) / 14
        expected[i + 1] = 100 - 100 / (1 + avg_gain / avg_loss)

    np.testing.assert_allclose(actual, expected)


def test_rmi_with_unit_momentum_equals_rsi_and_wilder_smooths_momentum():
    prices = 100 + np.cumsum(np.random.default_rng(5).normal(0, 2, 140))
    np.testing.assert_allclose(rmi_numba(prices, 14, 1), rsi_numba(prices, 14))

    actual = rmi_numba(prices, 20, 5)
    momentum = prices[5:] - prices[:-5]
    up = np.maximum(momentum, 0.0)
    down = np.maximum(-momentum, 0.0)
    expected = np.full(len(prices), np.nan)
    avg_up = up[:20].mean()
    avg_down = down[:20].mean()
    expected[24] = 100 - 100 / (1 + avg_up / avg_down)
    for i in range(20, len(momentum)):
        avg_up = (avg_up * 19 + up[i]) / 20
        avg_down = (avg_down * 19 + down[i]) / 20
        expected[i + 5] = 100 - 100 / (1 + avg_up / avg_down)

    np.testing.assert_allclose(actual, expected)


def test_rmi_flat_series_is_neutral():
    neutral = rmi_numba(flat(100.0, 60), 20, 5)
    assert np.isnan(neutral[:24]).all()
    assert np.all(neutral[24:] == pytest.approx(50.0))


def test_atr_percent_scales_the_first_valid_row_and_short_input_stays_nan():
    highs = 110.0 + np.arange(20.0)
    lows = 90.0 + np.arange(20.0)
    closes = 100.0 + np.arange(20.0)

    for mode in ("rma", "sma", "wma", "ema"):
        absolute = atr_numba(highs, lows, closes, 14, mode)
        percent = atr_numba(highs, lows, closes, 14, mode, percent=True)
        assert np.isnan(absolute[:13]).all()
        assert np.isfinite(absolute[13])
        np.testing.assert_allclose(percent[13:], absolute[13:] * 100 / closes[13:])

    short = atr_numba(highs[:10], lows[:10], closes[:10], 14)
    assert np.isnan(short).all()


def test_net_flow_ratio_is_prefix_invariant_and_matches_signed_volume_balance():
    rng = np.random.default_rng(13)
    closes = 100 + np.cumsum(rng.normal(0, 1.0, 60))
    volumes = rng.lognormal(8, 0.4, 60)
    actual = net_flow_ratio_numba(closes, volumes, 10)

    expected = np.full(len(closes), np.nan)
    for i in range(10, len(closes)):
        signs = np.sign(np.diff(closes[i - 10:i + 1]))
        window_volume = volumes[i - 9:i + 1]
        expected[i] = np.dot(signs, window_volume) / window_volume.sum()
    np.testing.assert_allclose(actual, expected)

    prefixed = net_flow_ratio_numba(
        np.concatenate((np.full(30, 50.0), closes)),
        np.concatenate((np.full(30, 123.0), volumes)),
        10,
    )
    np.testing.assert_allclose(prefixed[40:], actual[10:])


def test_net_flow_ratio_rails_neutral_flat_and_zero_volume():
    up = net_flow_ratio_numba(np.arange(1.0, 40.0), flat(5.0, 39), 10)
    assert np.all(up[10:] == pytest.approx(1.0))

    down = net_flow_ratio_numba(np.arange(40.0, 1.0, -1.0), flat(5.0, 39), 10)
    assert np.all(down[10:] == pytest.approx(-1.0))

    still = net_flow_ratio_numba(flat(100.0, 30), flat(5.0, 30), 10)
    assert np.all(still[10:] == pytest.approx(0.0))

    no_trades = net_flow_ratio_numba(np.arange(1.0, 30.0), flat(0.0, 29), 10)
    assert np.isnan(no_trades[10:]).all()


def test_period_metrics_report_an_obv_flow_ratio_instead_of_a_level_percent():
    calculator = MarketMetricsCalculator(null_logger())
    context = SimpleNamespace(
        technical_history={"obv": [100.0, 100.0, 100.0, 130.0], "rsi": [50.0, 50.0, 50.0, 60.0]}
    )

    changes = calculator._calculate_indicator_changes_for_period(context, -4, -1, 200.0)

    assert changes["obv_change"] == pytest.approx(30.0)
    assert changes["obv_flow_ratio"] == pytest.approx(0.15)
    assert "obv_change_pct" not in changes
    assert changes["rsi_change_pct"] == pytest.approx(20.0)


def test_td_setup_progresses_resets_on_equality_and_flips_direction():
    rising = td_setup_numba(np.arange(1.0, 21.0), 9)
    assert np.all(np.isnan(rising[:4]))
    assert list(rising[4:13]) == [float(count) for count in range(1, 10)]
    assert np.all(rising[13:] == 0)

    crafted = candles(10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 15.0, 12.0, 10.0)
    counts = td_setup_numba(crafted, 9)
    assert list(counts[4:9]) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert counts[9] == 0.0
    assert counts[10] == -1.0
    assert counts[11] == -2.0


def _ema_reference(values: np.ndarray, length: int) -> np.ndarray:
    """SMA-seeded EMA over the first full window of finite values (TA-Lib style)."""
    ema = np.full(len(values), np.nan)
    alpha = 2 / (length + 1)
    start = length - 1
    while start < len(values) and not np.isfinite(values[start - length + 1:start + 1]).all():
        start += 1
    if start >= len(values):
        return ema
    ema[start] = values[start - length + 1:start + 1].mean()
    for i in range(start + 1, len(values)):
        ema[i] = values[i] * alpha + ema[i - 1] * (1 - alpha)
    return ema


def test_macd_warm_up_boundaries_and_mature_values_match_the_reference():
    prices = 100 + np.cumsum(np.random.default_rng(17).normal(0, 1.0, 300))
    line, signal, histogram = macd_numba(prices)

    assert np.isnan(line[:25]).all() and np.isfinite(line[25])
    assert np.isnan(signal[:26]).all() and np.isfinite(signal[26])
    assert np.isnan(histogram[:26]).all() and np.isfinite(histogram[26])

    reference_line = _ema_reference(prices, 12) - _ema_reference(prices, 26)
    reference_signal = np.full(len(prices), np.nan)
    reference_signal[33] = reference_line[25:34].mean()
    for i in range(34, len(prices)):
        reference_signal[i] = reference_line[i] * 0.2 + reference_signal[i - 1] * 0.8

    np.testing.assert_allclose(line[-1], reference_line[-1], atol=1e-6)
    np.testing.assert_allclose(signal[-1], reference_signal[-1], atol=1e-6)
    np.testing.assert_allclose(histogram[-1], (reference_line - reference_signal)[-1], atol=1e-6)


def test_trix_warm_up_boundary_and_mature_values_match_the_reference():
    prices = 100 + np.cumsum(np.random.default_rng(17).normal(0, 1.0, 300))
    trix = trix_numba(prices, 20, 100, 1)

    assert np.isnan(trix[:21]).all() and np.isfinite(trix[21])

    smooth2 = _ema_reference(_ema_reference(prices, 20), 20)
    smooth3 = _ema_reference(smooth2, 20)
    reference = np.full(len(prices), np.nan)
    for i in range(58, len(prices)):
        reference[i] = 100 * (smooth3[i] - smooth3[i - 1]) / smooth3[i - 1]

    np.testing.assert_allclose(trix[-40:], reference[-40:], atol=1e-6)


def test_ebsw_wave_follows_the_degrees_to_radians_reference():
    prices = 100 + np.cumsum(np.random.default_rng(29).normal(0, 1.0, 160))
    actual = ebsw_numba(prices, 40, 10)

    angle = 2 * np.pi / 40
    alpha1 = (1 - np.sin(angle)) / np.cos(angle)
    a1 = np.exp(-np.sqrt(2) * np.pi / 10)
    c2 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / 10)
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    expected = np.full(len(prices), np.nan)
    last_close = prices[39]
    last_hp = 0.0
    history = [0.0, 0.0]
    for i in range(40, len(prices)):
        hp = 0.5 * (1 + alpha1) * (prices[i] - last_close) + alpha1 * last_hp
        filtered = c1 * (hp + last_hp) / 2 + c2 * history[1] + c3 * history[0]
        wave = (filtered + history[1] + history[0]) / 3
        power = (filtered ** 2 + history[1] ** 2 + history[0] ** 2) / 3
        expected[i] = wave / np.sqrt(power) if power > 0 else np.nan
        history[0] = history[1]
        history[1] = filtered
        last_hp = hp
        last_close = prices[i]

    assert np.isnan(actual[:40]).all()
    assert np.isfinite(actual[40])
    np.testing.assert_allclose(actual, expected)


def test_entropy_is_scale_invariant_and_matches_normalized_price_entropy():
    rng = np.random.default_rng(21)
    prices = 100 + 5 * np.sin(np.arange(60) / 3) + rng.normal(0, 0.5, 60)
    actual = entropy_numba(prices, 20)

    expected = np.full(len(prices), np.nan)
    for i in range(19, len(prices)):
        window = prices[i - 19:i + 1]
        probabilities = window / window.sum()
        expected[i] = -np.sum(probabilities * np.log2(probabilities))

    assert np.isnan(actual[:19]).all()
    np.testing.assert_allclose(actual, expected, atol=1e-9)
    np.testing.assert_allclose(entropy_numba(prices * 1000.0, 20), actual, atol=1e-9)


def test_entropy_flat_window_peaks_at_log2_length():
    flat_entropy = entropy_numba(flat(100.0, 60), 20)
    assert np.isnan(flat_entropy[:19]).all()
    assert np.all(flat_entropy[19:] == pytest.approx(math.log2(20)))


def test_hurst_expanding_estimate_scores_known_series_without_future_data():
    rng = np.random.default_rng(4)
    walk = 100 + np.cumsum(rng.normal(0, 1.0, 1000))
    trend = 100 + np.cumsum(0.5 + rng.normal(0, 0.2, 1000))
    mean_reverting = np.zeros(1000)
    mean_reverting[0] = 100.0
    for i in range(1, 1000):
        mean_reverting[i] = 100 + 0.9 * (mean_reverting[i - 1] - 100) + rng.normal(0, 0.5)

    walk_hurst = hurst_numba(walk, 20)
    assert np.isnan(walk_hurst[:22]).all() and np.isfinite(walk_hurst[22])
    assert walk_hurst[-1] == pytest.approx(0.5, abs=0.15)
    assert hurst_numba(trend, 20)[-1] > 0.9
    assert hurst_numba(mean_reverting, 20)[-1] < 0.5

    truncated = hurst_numba(walk[:600], 20)
    np.testing.assert_allclose(walk_hurst[:600], truncated, equal_nan=True)

    shorter = hurst_numba(walk[:400], 20)
    assert abs(shorter[-1] - walk_hurst[-1]) > 1e-3


def test_local_extrema_ignore_nan_candidates_and_nan_neighbours():
    warm_up = np.concatenate(
        (np.full(15, np.nan), candles(1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0))
    )
    maxima_idx, maxima_values = _find_local_extrema_numba(warm_up, 2, True)
    assert list(maxima_idx) == [17, 21]
    assert np.isfinite(maxima_values).all()

    hole = candles(1.0, 3.0, NAN, 3.0, 1.0, 5.0, 1.0, 3.0, 1.0)
    hole_idx, hole_values = _find_local_extrema_numba(hole, 1, True)
    assert list(hole_idx) == [5, 7]
    assert np.isfinite(hole_values).all()


def test_divergence_detectors_never_report_nan_indicator_values():
    rng = np.random.default_rng(31)
    prices = 100 + np.cumsum(rng.normal(0, 1.0, 800))
    macd_line = np.concatenate((np.full(25, np.nan), rng.normal(0, 2.0, 775)))

    for detector in (detect_bullish_divergence_numba, detect_bearish_divergence_numba):
        found, first_idx, second_idx, first_price, second_price, first_value, second_value = (
            detector(prices, macd_line)
        )
        if found:
            assert first_idx > 24 and second_idx > 24
            assert np.isfinite([first_price, second_price, first_value, second_value]).all()


def _synthetic_ohlcv(count: int) -> np.ndarray:
    rng = np.random.default_rng(23)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, count)))
    high = close * (1 + np.abs(rng.normal(0, 0.004, count)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, count)))
    open_ = np.empty(count)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    volume = rng.lognormal(10, 0.5, count)
    return np.column_stack([open_, high, low, close, volume])


OHLCV_INDICATOR_CALLS = [
    ("rsi", lambda ti: ti.rsi(length=14)),
    ("mfi", lambda ti: ti.mfi(length=14)),
    ("cmf", lambda ti: ti.chaikin_money_flow(length=20)),
    ("net_flow_ratio", lambda ti: ti.net_flow_ratio(lookback=10)),
    ("obv", lambda ti: ti.obv(length=20)),
    ("vwap", lambda ti: ti.rolling_vwap(length=20)),
    ("atr", lambda ti: ti.atr(length=20)),
    ("cci", lambda ti: ti.cci(length=14)),
    ("williams_r", lambda ti: ti.williams_r(length=14)),
    ("stochastic_k", lambda ti: ti.stochastic(period_k=14, smooth_k=3, period_d=3)[0]),
    ("adx", lambda ti: ti.adx(length=14)[0]),
]


@pytest.mark.parametrize(
    "name,compute", OHLCV_INDICATOR_CALLS, ids=[case[0] for case in OHLCV_INDICATOR_CALLS]
)
def test_active_indicators_do_not_look_ahead_of_the_last_candle(name, compute):
    ohlcv = _synthetic_ohlcv(240)
    full = TechnicalIndicators()
    full.get_data(ohlcv)
    truncated = TechnicalIndicators()
    truncated.get_data(ohlcv[:180])

    full_series = np.asarray(compute(full))
    cut_series = np.asarray(compute(truncated))

    window = slice(40, 180)
    usable = np.isfinite(full_series[window]) & np.isfinite(cut_series[window])
    assert usable.sum() >= 100
    np.testing.assert_allclose(
        full_series[window][usable], cut_series[window][usable], rtol=1e-9, atol=1e-12
    )

def test_pivot_points_follow_the_classic_floor_trader_ladder():
    highs = candles(10.0, 12.0, 14.0)
    lows = candles(8.0, 10.0, 12.0)
    closes = candles(9.0, 11.0, 13.0)

    pivot, r1, r2, r3, r4, s1, s2, s3, s4 = pivot_points_numba(highs, lows, closes)

    for series in (pivot, r1, r2, r3, r4, s1, s2, s3, s4):
        assert np.isnan(series[0])

    assert pivot[1] == pytest.approx(9.0)
    assert r1[1] == pytest.approx(10.0)
    assert r2[1] == pytest.approx(11.0)
    assert r3[1] == pytest.approx(13.0)
    assert r4[1] == pytest.approx(15.0)
    assert s1[1] == pytest.approx(8.0)
    assert s2[1] == pytest.approx(7.0)
    assert s3[1] == pytest.approx(5.0)
    assert s4[1] == pytest.approx(3.0)
    assert r4[1] > r3[1] > r2[1] > r1[1] > pivot[1] > s1[1] > s2[1] > s3[1] > s4[1]


def test_fibonacci_pivots_use_the_documented_ratios():
    highs = candles(10.0, 12.0)
    lows = candles(8.0, 10.0)
    closes = candles(9.0, 11.0)

    pivot, r1, r2, r3, s1, s2, s3 = fibonacci_pivot_points_numba(highs, lows, closes)

    assert np.isnan(pivot[0])
    assert pivot[1] == pytest.approx(9.0)
    assert r1[1] == pytest.approx(9.0 + 0.382 * 2.0)
    assert r2[1] == pytest.approx(9.0 + 0.618 * 2.0)
    assert r3[1] == pytest.approx(11.0)
    assert s1[1] == pytest.approx(9.0 - 0.382 * 2.0)
    assert s2[1] == pytest.approx(9.0 - 0.618 * 2.0)
    assert s3[1] == pytest.approx(7.0)


def test_donchian_channels_are_rolling_extremes_and_survive_nan_gaps():
    highs = candles(1.0, 5.0, 3.0, 4.0, 2.0, NAN, 6.0, 7.0, 8.0)
    lows = candles(1.0, 4.0, 2.0, 3.0, 1.0, 0.5, 5.0, 6.0, 7.0)

    upper, middle, lower = donchian_channels_numba(highs, lows, 3)

    for i in (2, 3, 4, 8):
        assert middle[i] == pytest.approx((upper[i] + lower[i]) / 2.0)

    assert upper[2] == pytest.approx(5.0) and lower[2] == pytest.approx(1.0)
    assert upper[3] == pytest.approx(5.0) and lower[3] == pytest.approx(2.0)
    assert upper[4] == pytest.approx(4.0) and lower[4] == pytest.approx(1.0)
    assert lower[5] == pytest.approx(0.5)

    assert np.isnan(upper[5]) and np.isnan(upper[6]) and np.isnan(upper[7])
    assert upper[8] == pytest.approx(8.0) and lower[8] == pytest.approx(5.0)


def test_cci_matches_the_typical_price_mad_formula():
    highs = 10.0 + np.arange(1.0, 31.0)
    lows = 8.0 + np.arange(1.0, 31.0)
    closes = 9.0 + np.arange(1.0, 31.0)

    actual = cci_numba(highs, lows, closes, 14)

    expected = np.full(30, np.nan)
    for i in range(13, 30):
        typical = (highs[i - 13:i + 1] + lows[i - 13:i + 1] + closes[i - 13:i + 1]) / 3.0
        mean_tp = typical.mean()
        mad = np.abs(typical - mean_tp).mean()
        expected[i] = (typical[-1] - mean_tp) / (0.015 * mad)

    np.testing.assert_allclose(actual[13:], expected[13:])
    assert np.isnan(actual[:13]).all()
    assert np.all(cci_numba(flat(10.0, 20), flat(8.0, 20), flat(9.0, 20), 14)[13:] == 0.0)


def test_williams_r_matches_the_negative_range_position_formula():
    highs = candles(10.0, 12.0, 11.0, 13.0, 12.0, 15.0)
    lows = candles(8.0, 9.0, 7.0, 10.0, 9.0, 12.0)
    closes = candles(9.0, 11.0, 10.0, 12.0, 11.0, 14.0)

    actual = williams_r_numba(highs, lows, closes, 4)

    expected = np.full(6, np.nan)
    for i in range(3, 6):
        highest_high = highs[i - 3:i + 1].max()
        lowest_low = lows[i - 3:i + 1].min()
        expected[i] = ((highest_high - closes[i]) / (highest_high - lowest_low)) * -100.0

    np.testing.assert_allclose(actual, expected, equal_nan=True)
    flat_range = williams_r_numba(flat(5.0, 10), flat(5.0, 10), flat(5.0, 10), 4)
    assert np.isnan(flat_range[3:]).all()


def test_roc_is_a_simple_percentage_change():
    actual = roc_numba(candles(100.0, 110.0, 99.0, 108.9), 1)
    assert np.isnan(actual[0])
    np.testing.assert_allclose(actual[1:], [10.0, -10.0, 10.0])
    np.testing.assert_allclose(
        roc_numba(candles(100.0, 110.0, 121.0, 133.1), 2),
        [NAN, NAN, 21.0, 21.0],
        equal_nan=True,
    )


def test_statistical_windows_match_sample_moments():
    rng = np.random.default_rng(12)
    prices = 100 + np.cumsum(rng.normal(0, 1.0, 90))
    length = 30

    variance = variance_numba(prices, length)
    stdev = stdev_numba(prices, length)
    zscore = zscore_numba(prices, length)
    skew = skew_numba(prices, length)

    for i in (29, 45, 89):
        window = prices[i - 29:i + 1]
        centered = window - window.mean()
        sample_variance = (centered ** 2).sum() / (length - 1)
        sample_stdev = np.sqrt(sample_variance)
        assert variance[i] == pytest.approx(sample_variance)
        assert stdev[i] == pytest.approx(sample_stdev)
        assert zscore[i] == pytest.approx(centered[-1] / sample_stdev)
        m2 = (centered ** 2).mean()
        m3 = (centered ** 3).mean()
        adjusted = np.sqrt(length * (length - 1)) / (length - 2) * m3 / m2 ** 1.5
        assert skew[i] == pytest.approx(adjusted)

    assert np.isnan(variance[:29]).all() and np.isnan(zscore[:29]).all()

    constant = flat(100.0, 60)
    assert np.all(variance_numba(constant, length)[29:] == pytest.approx(0.0))
    assert np.all(zscore_numba(constant, length)[29:] == pytest.approx(0.0))
    assert np.all(skew_numba(constant, length)[29:] == pytest.approx(0.0))


def test_ichimoku_leading_spans_stay_candle_aligned():
    """The cloud is shifted back in origin, never extended past the last candle.

    span[t] is the cloud drawn at bar t, computed from bar t - displacement, so
    the arrays carry exactly as many slots as the OHLCV series; the projected
    cloud a chart draws ahead of the last candle is not stored at all.
    """
    n = 120
    high = 100.0 + np.arange(n, dtype=float)
    low = high - 5.0

    conversion, base, span_a, span_b = ichimoku_cloud_numba(high, low, 9, 26, 52, 26)

    assert len(span_a) == len(span_b) == len(high) == n
    assert span_a[-1] == pytest.approx((conversion[n - 27] + base[n - 27]) / 2)
    assert span_a[-1] != pytest.approx((conversion[-1] + base[-1]) / 2)
    assert get_last_valid_value(span_a) == pytest.approx(span_a[-1])

    # Undefined slots sit at the head: span A needs the 26-bar base line
    # (25 + displacement), span B additionally the 52-bar mid line (51 + 26).
    assert np.isnan(span_a[:51]).all() and np.isfinite(span_a[51:]).all()
    assert np.isnan(span_b[:77]).all() and np.isfinite(span_b[77:]).all()


def test_ichimoku_span_b_stays_empty_without_enough_history():
    short = np.arange(60, dtype=float) + 100.0

    _, _, span_a, span_b = ichimoku_cloud_numba(short, short - 1.0, 9, 26, 52, 26)

    assert get_last_valid_value(span_a) is not None
    assert get_last_valid_value(span_b) is None


def test_sentiment_rsi_window_matches_the_shared_flat_window_convention():
    assert rsi_numba(flat(100.0, 40), 14)[-1] == pytest.approx(50.0)

    window = _calculate_rsi_window(flat(100.0, 40), 14)
    assert window[13] == pytest.approx(50.0)
    assert window[-1] == pytest.approx(50.0)

    rising = candles(*[100.0 + index for index in range(40)])
    assert _calculate_rsi_window(rising, 14)[-1] == pytest.approx(100.0)


def test_sentiment_macd_window_agrees_with_the_shared_macd_after_warmup():
    rng = np.random.default_rng(7)
    close = 100.0 + rng.normal(0.0, 1.0, 400).cumsum()

    shared_line, _, shared_hist = macd_numba(close, 12, 26, 9)
    window_line, _, window_hist = _calculate_macd_window(close, 12, 26, 9)

    # Both lines start at slow - 1 = 25, but the signal/histogram start later in
    # the window copy (slow + signal - 2 = 33) than in the shared one (26).
    assert np.isnan(shared_line[:25]).all() and np.isfinite(shared_line[25:]).all()
    assert np.isnan(window_line[:25]).all() and np.isfinite(window_line[25:]).all()
    assert np.isnan(shared_hist[:26]).all() and np.isfinite(shared_hist[26:]).all()
    assert np.isnan(window_hist[:33]).all() and np.isfinite(window_hist[33:]).all()

    # Only the warm-up seed differs; mature bars agree.
    np.testing.assert_allclose(window_line[-50:], shared_line[-50:], atol=1e-9)
    np.testing.assert_allclose(window_hist[-50:], shared_hist[-50:], atol=1e-9)


def test_sentiment_mfi_window_is_nan_without_traded_volume():
    n = 40
    close = 100.0 + np.arange(n, dtype=float)
    high, low = close + 1.0, close - 1.0

    untraded = _calculate_mfi_window(high, low, close, np.zeros(n), 14)
    traded = _calculate_mfi_window(high, low, close, np.full(n, 10.0), 14)

    assert np.isnan(untraded[14:]).all()
    assert np.isfinite(traded[14:]).all()
    assert traded[-1] == pytest.approx(100.0)  # rising prices: inflow only


def test_obv_accumulates_signed_volume():
    close = candles(10.0, 11.0, 10.5, 10.5, 12.0, 11.0)
    volume = candles(100.0, 200.0, 300.0, 400.0, 500.0, 600.0)

    obv = obv_numba(close, volume, 3)

    np.testing.assert_allclose(obv, [NAN, NAN, 300.0, 300.0, 800.0, 200.0], equal_nan=True)


def test_pvt_sums_the_relative_price_change_weighted_by_volume():
    close = candles(100.0, 110.0, 99.0, 108.9)
    volume = candles(10.0, 20.0, 30.0, 40.0)

    pvt = pvt_numba(close, volume, 2)

    roc = (np.asarray(close[1:]) / np.asarray(close[:-1]) - 1.0) * np.asarray(volume[1:])
    np.testing.assert_allclose(pvt, [NAN] + list(np.cumsum(roc)), equal_nan=True)


def test_ad_line_accumulates_the_money_flow_multiplier():
    high = candles(11.0, 12.0, 10.0, 10.0)
    low = candles(9.0, 10.0, 8.0, 10.0)
    close = candles(9.5, 11.5, 9.0, 10.0)
    volume = candles(100.0, 200.0, 300.0, 400.0)

    ad_line = ad_line_numba(high, low, close, volume)

    expected = [0.0]
    for i in range(1, len(close)):
        if high[i] == low[i]:
            expected.append(expected[-1])
        else:
            multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            expected.append(expected[-1] + multiplier * volume[i])
    np.testing.assert_allclose(ad_line, expected)


def test_force_index_is_the_smoothed_close_change_times_volume():
    close = candles(100.0, 102.0, 101.0, 105.0, 104.0, 107.0, 109.0, 108.0)
    volume = candles(*[10.0] * 8)

    raw = np.zeros(len(close))
    for i in range(1, len(close)):
        raw[i] = (close[i] - close[i - 1]) * volume[i]

    np.testing.assert_allclose(force_index_numba(close, volume, 3), ema_numba(raw, 3))


def test_twap_averages_the_typical_price_over_a_rolling_window():
    high = candles(11.0, 12.0, 13.0, 14.0, 15.0)
    low = candles(9.0, 10.0, 11.0, 12.0, 13.0)
    close = candles(10.0, 11.0, 12.0, 13.0, 14.0)

    twap = twap_numba(high, low, close, 3)

    typical = (np.asarray(high) + np.asarray(low) + np.asarray(close)) / 3
    expected = [NAN, NAN] + [typical[i - 2:i + 1].mean() for i in range(2, 5)]
    np.testing.assert_allclose(twap, expected, equal_nan=True)


def _wave(length: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic trending-and-wobbling OHLC triple for the reference tests."""
    steps = np.arange(length, dtype=float)
    base = 100.0 + steps * 0.4 + 3.0 * np.sin(steps / 3.0)
    high = base + 1.0
    low = base - 1.0
    close = base + 0.25 * np.cos(steps / 2.0)
    return high, low, close


def test_adx_and_directional_indicators_follow_wilder_smoothing():
    length = 14
    high, low, close = _wave()

    adx, pdi, ndi = adx_numba(high, low, close, length)

    true_range = np.zeros(len(close))
    dm_pos = np.zeros(len(close))
    dm_neg = np.zeros(len(close))
    for i in range(1, len(close)):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        dm_pos[i] = up_move if up_move > down_move and up_move > 0 else 0.0
        dm_neg[i] = down_move if down_move > up_move and down_move > 0 else 0.0
        true_range[i] = max(
            high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])
        )

    def wilder(values: np.ndarray) -> np.ndarray:
        smoothed = np.full(len(values), NAN)
        smoothed[length] = values[1:length + 1].sum()
        for index in range(length + 1, len(values)):
            smoothed[index] = smoothed[index - 1] * (1.0 - 1.0 / length) + values[index]
        return smoothed

    tr14 = wilder(true_range)
    dm_pos14 = wilder(dm_pos)
    dm_neg14 = wilder(dm_neg)
    expected_pdi = 100.0 * dm_pos14 / tr14
    expected_ndi = 100.0 * dm_neg14 / tr14
    expected_dx = 100.0 * np.abs(expected_pdi - expected_ndi) / (expected_pdi + expected_ndi)

    expected_adx = np.full(len(close), NAN)
    expected_adx[length * 2 - 1] = expected_dx[length:length * 2].mean()
    for index in range(length * 2, len(close)):
        expected_adx[index] = (
            expected_adx[index - 1] * (length - 1) + expected_dx[index]
        ) / length

    np.testing.assert_allclose(pdi[length:], expected_pdi[length:], rtol=1e-10)
    np.testing.assert_allclose(ndi[length:], expected_ndi[length:], rtol=1e-10)
    np.testing.assert_allclose(adx[length * 2 - 1:], expected_adx[length * 2 - 1:], rtol=1e-10)


def test_keltner_channels_wrap_the_middle_line_two_atr_wide():
    length = 20
    high, low, close = _wave()

    upper, middle, lower = keltner_channels_numba(high, low, close, length, 2.0, "ema")

    expected_middle = ema_numba(close, length)
    atr = atr_numba(high, low, close, length, mamode="ema")

    np.testing.assert_allclose(middle, expected_middle, equal_nan=True)
    np.testing.assert_allclose(upper, expected_middle + 2.0 * atr, equal_nan=True)
    np.testing.assert_allclose(lower, expected_middle - 2.0 * atr, equal_nan=True)


def test_chandelier_exit_is_the_window_extreme_offset_by_atr():
    length = 5
    multiplier = 1.5
    high, low, close = _wave()

    long_exit, short_exit = chandelier_exit_numba(high, low, close, length, multiplier)

    atr = atr_numba(high, low, close, length, "rma")
    expected_long = np.full(len(close), NAN)
    expected_short = np.full(len(close), NAN)
    for i in range(length, len(close)):
        if not np.isnan(atr[i]):
            expected_long[i] = np.max(high[i - length + 1:i + 1]) - atr[i] * multiplier
            expected_short[i] = np.min(low[i - length + 1:i + 1]) + atr[i] * multiplier

    np.testing.assert_allclose(long_exit, expected_long, equal_nan=True)
    np.testing.assert_allclose(short_exit, expected_short, equal_nan=True)


def test_ppo_is_the_ema_spread_over_the_slow_ema():
    fast_length = 3
    slow_length = 5
    _, _, close = _wave()

    ppo = ppo_numba(close, fast_length, slow_length)

    fast_ema = ema_numba(close, fast_length)
    slow_ema = ema_numba(close, slow_length)
    expected = np.full(len(close), NAN)
    for i in range(slow_length - 1, len(close)):
        if slow_ema[i] != 0:
            expected[i] = ((fast_ema[i] - slow_ema[i]) / slow_ema[i]) * 100.0

    np.testing.assert_allclose(ppo, expected, equal_nan=True, rtol=1e-12)


def test_kst_weights_four_smoothed_roc_windows():
    roc_lengths = (5, 10, 15, 20)
    sma_lengths = (3, 5, 7, 9)
    _, _, close = _wave(length=60)

    kst = kst_numba(close, *roc_lengths, *sma_lengths)

    first_valid = max(roc + sma - 1 for roc, sma in zip(roc_lengths, sma_lengths))
    expected = np.full(len(close), NAN)
    for i in range(first_valid, len(close)):
        total = 0.0
        for weight, (roc_length, sma_length) in enumerate(zip(roc_lengths, sma_lengths), start=1):
            window = [
                (close[i - offset] / close[i - offset - roc_length] - 1.0) * 100.0
                for offset in range(sma_length)
            ]
            total += weight * (sum(window) / sma_length)
        expected[i] = total

    np.testing.assert_allclose(kst, expected, equal_nan=True, rtol=1e-10)


def test_supertrend_line_tracks_the_adjusted_bands():
    length = 3
    multiplier = 1.0
    steps = np.arange(40, dtype=float)
    close = candles(*list(100.0 + 12.0 * np.sin(steps / 5.0) + steps * 0.2))
    high = close + 1.0
    low = close - 1.0

    trend_line, direction = supertrend_numba(high, low, close, length, multiplier)

    atr = atr_numba(high, low, close, length)
    midpoint = (high + low) / 2.0
    upper = np.array(midpoint + multiplier * atr)
    lower = np.array(midpoint - multiplier * atr)
    expected_direction = np.ones(len(close))
    expected_trend = np.full(len(close), NAN)
    expected_trend[0] = lower[0]

    for i in range(1, len(close)):
        if close[i - 1] <= upper[i - 1]:
            upper[i] = min(upper[i], upper[i - 1])
        if close[i - 1] >= lower[i - 1]:
            lower[i] = max(lower[i], lower[i - 1])

        if close[i] > upper[i - 1]:
            expected_direction[i] = 1
        elif close[i] < lower[i - 1]:
            expected_direction[i] = -1
        else:
            expected_direction[i] = expected_direction[i - 1]

        expected_trend[i] = lower[i] if expected_direction[i] == 1 else upper[i]

    assert -1 in expected_direction
    np.testing.assert_allclose(direction, expected_direction)
    np.testing.assert_allclose(trend_line, expected_trend, equal_nan=True)


def test_parabolic_sar_accelerates_with_the_extension_of_the_move():
    high = candles(10.0, 11.0, 12.0, 13.0, 14.0, 15.0)
    low = candles(9.0, 10.0, 11.0, 12.0, 13.0, 14.0)

    sar = parabolic_sar_numba(high, low)

    np.testing.assert_allclose(sar, [9.0, 9.0, 9.0, 9.18, 9.4856, 9.93704], rtol=1e-12)


def test_stochastic_matches_smoothed_percent_k_and_d():
    n = 12
    close = candles(*[100.0 + index for index in range(n)])
    high = np.asarray(close) + 2.0
    low = np.asarray(close) - 2.0

    k_values, d_values = stochastic_numba(high, low, close, 5, 3, 3)

    raw_k = np.full(n, NAN)
    for i in range(4, n):
        window_high = high[i - 4:i + 1].max()
        window_low = low[i - 4:i + 1].min()
        raw_k[i] = 100 * (close[i] - window_low) / (window_high - window_low)

    smoothed = [raw_k[i - 2:i + 1].mean() for i in range(6, n)]
    np.testing.assert_allclose(k_values[6:], smoothed)
    np.testing.assert_allclose(d_values[8:], [np.mean(smoothed[j - 2:j + 1]) for j in range(2, len(smoothed))])
    assert np.isnan(k_values[:6]).all() and np.isnan(d_values[:8]).all()
