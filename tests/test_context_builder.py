"""Tests for context_builder changes: _resolve_indicator_value and compute_indicator_delta_alert."""
import numpy as np
import pytest

from src.analyzer.prompts.context_builder import ContextBuilder


# ── Minimal ContextBuilder for static / pure methods ────────────


def _make_builder():
    """Build a minimal ContextBuilder with mocked dependencies."""
    from unittest.mock import MagicMock

    return ContextBuilder(
        timeframe="1h",
        format_utils=MagicMock(),
        market_formatter=MagicMock(),
        period_formatter=MagicMock(),
        long_term_formatter=MagicMock(),
    )


# ── _resolve_indicator_value ─────────────────────────────────────


class TestResolveIndicatorValue:
    """Static method: extract scalar float from varying input types."""

    def test_none_returns_none(self):
        assert ContextBuilder._resolve_indicator_value(None) is None

    def test_scalar_float(self):
        assert ContextBuilder._resolve_indicator_value(42.5) == 42.5

    def test_scalar_int(self):
        assert ContextBuilder._resolve_indicator_value(10) == 10.0

    def test_list_returns_last(self):
        assert ContextBuilder._resolve_indicator_value([1.0, 2.0, 3.0]) == 3.0

    def test_tuple_returns_last(self):
        assert ContextBuilder._resolve_indicator_value((4.0, 5.0)) == 5.0

    def test_numpy_array_returns_last(self):
        arr = np.array([10.0, 20.0, 30.0])
        assert ContextBuilder._resolve_indicator_value(arr) == 30.0

    def test_empty_list_returns_none(self):
        assert ContextBuilder._resolve_indicator_value([]) is None

    def test_empty_tuple_returns_none(self):
        assert ContextBuilder._resolve_indicator_value(()) is None

    def test_empty_numpy_array_returns_none(self):
        assert ContextBuilder._resolve_indicator_value(np.array([])) is None

    def test_string_numeric(self):
        assert ContextBuilder._resolve_indicator_value("3.14") == 3.14

    def test_string_non_numeric_returns_none(self):
        assert ContextBuilder._resolve_indicator_value("abc") is None

    def test_list_with_none_last_element(self):
        # [1.0, None] — last element is None → should be None
        assert ContextBuilder._resolve_indicator_value([1.0, None]) is None


# ── compute_indicator_delta_alert ────────────────────────────────


class TestComputeIndicatorDeltaAlert:
    """Verify the anchoring-prevention delta alert fires correctly."""

    def setup_method(self):
        self.builder = _make_builder()

    def test_empty_previous_returns_empty(self):
        assert self.builder.compute_indicator_delta_alert({}, {"rsi": 50}) == ""

    def test_empty_current_returns_empty(self):
        assert self.builder.compute_indicator_delta_alert({"rsi": 50}, {}) == ""

    def test_none_inputs_return_empty(self):
        assert self.builder.compute_indicator_delta_alert(None, None) == ""

    def test_no_alert_below_threshold(self):
        """Small changes (<20%) should NOT trigger alert."""
        prev = {"rsi": 50, "adx": 25, "mfi": 50, "stoch_k": 50}
        curr = {"rsi": 52, "adx": 26, "mfi": 51, "stoch_k": 51}
        assert self.builder.compute_indicator_delta_alert(prev, curr) == ""

    def test_alert_fires_with_big_changes(self):
        """When >= min_count indicators change > threshold%, alert should fire."""
        prev = {"rsi": 30, "adx": 15, "mfi": 20, "stoch_k": 25, "williams_r": -80}
        curr = {"rsi": 60, "adx": 35, "mfi": 50, "stoch_k": 55, "williams_r": -30}
        result = self.builder.compute_indicator_delta_alert(prev, curr, change_threshold=20.0, min_count=3)
        assert "SIGNIFICANT DATA SHIFT" in result
        assert "Re-derive conclusions" in result

    def test_alert_respects_min_count(self):
        """If fewer than min_count indicators changed, no alert."""
        prev = {"rsi": 30, "adx": 25}
        curr = {"rsi": 60, "adx": 50}
        # Only 2 changed, min_count=3 → no alert
        result = self.builder.compute_indicator_delta_alert(prev, curr, change_threshold=20.0, min_count=3)
        assert result == ""

    def test_alert_respects_threshold(self):
        """Changes below threshold should not be counted."""
        prev = {"rsi": 50, "adx": 25, "mfi": 50}
        curr = {"rsi": 55, "adx": 27, "mfi": 53}  # ~10% change, below 20%
        result = self.builder.compute_indicator_delta_alert(prev, curr, change_threshold=20.0, min_count=2)
        assert result == ""

    def test_zero_previous_skipped(self):
        """Indicators with prev close to 0 should be skipped to avoid div-by-zero."""
        prev = {"rsi": 0.00001, "adx": 0.00001, "mfi": 0.00001}
        curr = {"rsi": 80, "adx": 40, "mfi": 60}
        # abs(prev) <= 0.0001, so these are skipped
        result = self.builder.compute_indicator_delta_alert(prev, curr, min_count=1)
        assert result == ""

    def test_array_values_handled(self):
        """Indicator values stored as arrays should be resolved before comparison."""
        prev = {"rsi": [40, 45, 50], "adx": [20, 22, 25], "mfi": [30, 35, 40]}
        curr = {"rsi": [70, 75, 80], "adx": [40, 45, 50], "mfi": [60, 65, 70]}
        result = self.builder.compute_indicator_delta_alert(prev, curr, change_threshold=20.0, min_count=3)
        assert "SIGNIFICANT DATA SHIFT" in result

    def test_max_five_changes_in_output(self):
        """Alert should list at most 5 indicator changes."""
        prev = {
            "rsi": 20, "adx": 10, "mfi": 15, "stoch_k": 20,
            "williams_r": -90, "obv_slope": 5, "cmf": 0.1, "macd_hist": 1.0,
            "roc_14": 2, "bb_percent_b": 0.1
        }
        curr = {
            "rsi": 80, "adx": 50, "mfi": 60, "stoch_k": 80,
            "williams_r": -10, "obv_slope": 30, "cmf": 0.5, "macd_hist": 5.0,
            "roc_14": 20, "bb_percent_b": 0.9
        }
        result = self.builder.compute_indicator_delta_alert(prev, curr, change_threshold=20.0, min_count=1)
        assert "SIGNIFICANT DATA SHIFT" in result
        # Count commas — at most 4 (i.e., 5 items)
        changes_text = result.split("(")[1].split(")")[0]  # extract "RSI +x%, ..."
        assert changes_text.count(",") <= 4
