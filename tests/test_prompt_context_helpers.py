"""Tests for prompt context section helpers."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.analyzer.prompts.prompt_builder import PromptBuilder
from src.utils.timeframe_validator import TimeframeValidator


def _make_builder(timeframe="1h", timeframe_validator=None):
    """Build a minimal PromptBuilder with mocked dependencies."""
    market_formatter = MagicMock()
    market_formatter.period_formatter = MagicMock()
    market_formatter.format_coin_details_section.return_value = ""

    return PromptBuilder(
        timeframe=timeframe,
        config=MagicMock(MODEL_VERBOSITY="medium"),
        format_utils=MagicMock(),
        overview_formatter=MagicMock(),
        long_term_formatter=MagicMock(),
        technical_formatter=MagicMock(),
        market_formatter=market_formatter,
        template_manager=MagicMock(),
        timeframe_validator=timeframe_validator,
    )


class TestSubHourTimeframes:
    """Verify prompt context calculations understand sub-hour timeframes."""

    @pytest.mark.parametrize(
        ("timeframe", "expected_minutes"),
        [
            ("5m", 5),
            ("15m", 15),
            ("30m", 30),
        ],
    )
    def test_trading_context_reports_sub_hour_candle_minutes(self, timeframe, expected_minutes):
        builder = _make_builder(timeframe=timeframe, timeframe_validator=TimeframeValidator)
        context = MagicMock(symbol="BTC/USDT", current_price=50000)

        result = builder.build_trading_context(context)

        assert f"Primary Timeframe: {timeframe} ({expected_minutes} min/candle)" in result
        assert "Next Candle Close" in result

    def test_market_data_summary_uses_5m_period_candle_counts(self):
        builder = _make_builder(timeframe="5m", timeframe_validator=TimeframeValidator)
        builder.format_utils.fmt.side_effect = lambda value: f"{value:.2f}"
        ohlcv_candles = np.array([
            [index * 300000, 100 + index, 101 + index, 99 + index, 100 + index, 1000]
            for index in range(100)
        ], dtype=np.float64)

        result = builder.build_market_data_section(ohlcv_candles)

        assert "Multi-Timeframe Price Summary (Based on 5m candles):" in result
        assert "4h:" in result
        assert "12h:" not in result


class TestResolveIndicatorValue:
    """Extract scalar floats from varying input types."""

    def test_none_returns_none(self):
        assert PromptBuilder._resolve_indicator_value(None) is None

    def test_scalar_float(self):
        assert PromptBuilder._resolve_indicator_value(42.5) == 42.5

    def test_scalar_int(self):
        assert PromptBuilder._resolve_indicator_value(10) == 10.0

    def test_list_returns_last(self):
        assert PromptBuilder._resolve_indicator_value([1.0, 2.0, 3.0]) == 3.0

    def test_tuple_returns_last(self):
        assert PromptBuilder._resolve_indicator_value((4.0, 5.0)) == 5.0

    def test_numpy_array_returns_last(self):
        arr = np.array([10.0, 20.0, 30.0])
        assert PromptBuilder._resolve_indicator_value(arr) == 30.0

    def test_empty_list_returns_none(self):
        assert PromptBuilder._resolve_indicator_value([]) is None

    def test_empty_tuple_returns_none(self):
        assert PromptBuilder._resolve_indicator_value(()) is None

    def test_empty_numpy_array_returns_none(self):
        assert PromptBuilder._resolve_indicator_value(np.array([])) is None

    def test_string_numeric(self):
        assert PromptBuilder._resolve_indicator_value("3.14") == 3.14

    def test_string_non_numeric_returns_none(self):
        assert PromptBuilder._resolve_indicator_value("abc") is None

    def test_list_with_none_last_element(self):
        assert PromptBuilder._resolve_indicator_value([1.0, None]) is None