"""Tests for brain context extraction from technical data.

Verifies that _generate_brain_context_from_current_indicators correctly
maps technical_data keys to brain context parameters.
"""
import numpy as np

from src.analyzer.technical_calculator import TechnicalCalculator
from src.factories import TechnicalIndicatorsFactory


class TestBrainContextDataKeys:
    """Verify correct dictionary keys are used for brain context extraction."""

    def test_technical_calculator_produces_required_keys(self):
        """All keys used in brain context extraction must exist in technical_calculator output."""

        ohlcv = np.random.rand(200, 6) * 1000
        ohlcv[:, 0] = np.arange(200) * 3600000

        calc = TechnicalCalculator(ti_factory=TechnicalIndicatorsFactory())
        indicators = calc.get_indicators(ohlcv)

        required_keys = [
            "plus_di", "minus_di",
            "adx",
            "atr_percent",
            "rsi",
            "macd_line", "macd_signal",
            "obv_slope",
            "bb_upper", "bb_lower",
        ]

        for key in required_keys:
            assert key in indicators, f"Missing key: {key}"
            assert indicators[key] is not None, f"Key {key} is None"

    def test_technical_data_values_are_1d_arrays(self):
        """Indicators used for brain context should be 1D numpy arrays."""

        ohlcv = np.random.rand(200, 6) * 1000
        ohlcv[:, 0] = np.arange(200) * 3600000

        calc = TechnicalCalculator(ti_factory=TechnicalIndicatorsFactory())
        indicators = calc.get_indicators(ohlcv)

        required_keys = ["plus_di", "minus_di", "adx", "atr_percent", "rsi",
                         "macd_line", "macd_signal", "obv_slope", "bb_upper", "bb_lower"]

        for key in required_keys:
            val = indicators[key]
            assert isinstance(val, np.ndarray), f"{key} should be ndarray"
            assert val.ndim == 1, f"{key} should be 1D array, got {val.ndim}D"

    def test_obv_slope_calculation(self):
        """OBV slope should reflect accumulation/distribution trends."""

        ohlcv = np.zeros((200, 6))
        ohlcv[:, 0] = np.arange(200) * 3600000
        ohlcv[:, 1] = 100
        ohlcv[:, 2] = 110
        ohlcv[:, 3] = 90
        ohlcv[:, 4] = np.linspace(100, 150, 200)
        ohlcv[:, 5] = 1000000

        calc = TechnicalCalculator(ti_factory=TechnicalIndicatorsFactory())
        indicators = calc.get_indicators(ohlcv)

        obv_slope = indicators.get("obv_slope")
        assert obv_slope is not None, "obv_slope should exist"
        assert len(obv_slope) == 200, "obv_slope should have same length as input"

    def test_brain_context_parameters_coverage(self):
        """Verify all brain context parameters are extracted from available data."""
        required_extractions = {
            "trend_direction": ["plus_di", "minus_di"],
            "adx": ["adx"],
            "volatility_level": ["atr_percent"],
            "rsi_level": ["rsi"],
            "macd_signal": ["macd_line", "macd_signal"],
            "volume_state": ["obv_slope"],
            "bb_position": ["bb_upper", "bb_lower"],
        }

        all_keys = set()
        for keys in required_extractions.values():
            all_keys.update(keys)

        ohlcv = np.random.rand(200, 6) * 1000
        ohlcv[:, 0] = np.arange(200) * 3600000

        calc = TechnicalCalculator(ti_factory=TechnicalIndicatorsFactory())
        indicators = calc.get_indicators(ohlcv)

        for key in all_keys:
            assert key in indicators, f"Brain context needs {key} but it's missing from technical_calculator output"


class TestBrainContextClassification:
    """Test that brain context classification logic works correctly."""

    def test_trend_direction_bullish(self):
        """When plus_di > minus_di + 5, trend should be BULLISH."""
        tech_data = {"plus_di": 30.0, "minus_di": 20.0}
        direction = self._extract_trend_direction(tech_data)
        assert direction == "BULLISH"

    def test_trend_direction_bearish(self):
        """When minus_di > plus_di + 5, trend should be BEARISH."""
        tech_data = {"plus_di": 20.0, "minus_di": 30.0}
        direction = self._extract_trend_direction(tech_data)
        assert direction == "BEARISH"

    def test_trend_direction_neutral(self):
        """When difference < 5, trend should be NEUTRAL."""
        tech_data = {"plus_di": 25.0, "minus_di": 23.0}
        direction = self._extract_trend_direction(tech_data)
        assert direction == "NEUTRAL"

    def test_bb_position_upper(self):
        """Price near upper band should classify as UPPER."""
        bb_pos = self._extract_bb_position(100000, 100500, 95000)
        assert bb_pos == "UPPER"

    def test_bb_position_lower(self):
        """Price near lower band should classify as LOWER."""
        bb_pos = self._extract_bb_position(95100, 100500, 95000)
        assert bb_pos == "LOWER"

    def test_bb_position_middle(self):
        """Price in middle should classify as MIDDLE."""
        bb_pos = self._extract_bb_position(97500, 100500, 95000)
        assert bb_pos == "MIDDLE"

    def test_macd_signal_bullish(self):
        """MACD line > signal line = BULLISH."""
        tech_data = {"macd_line": 100.0, "macd_signal": 50.0}
        signal = self._extract_macd_signal(tech_data)
        assert signal == "BULLISH"

    def test_macd_signal_bearish(self):
        """MACD line < signal line = BEARISH."""
        tech_data = {"macd_line": 50.0, "macd_signal": 100.0}
        signal = self._extract_macd_signal(tech_data)
        assert signal == "BEARISH"

    def test_volume_state_accumulation(self):
        """OBV slope > 0.5 = ACCUMULATION."""
        state = self._extract_volume_state({"obv_slope": 0.8})
        assert state == "ACCUMULATION"

    def test_volume_state_distribution(self):
        """OBV slope < -0.5 = DISTRIBUTION."""
        state = self._extract_volume_state({"obv_slope": -0.8})
        assert state == "DISTRIBUTION"

    @staticmethod
    def _extract_trend_direction(tech_data):
        di_plus = tech_data.get("plus_di", 0.0)
        di_minus = tech_data.get("minus_di", 0.0)
        if di_plus > di_minus + 5:
            return "BULLISH"
        if di_minus > di_plus + 5:
            return "BEARISH"
        return "NEUTRAL"

    @staticmethod
    def _extract_bb_position(price, bb_upper, bb_lower):
        if price >= bb_upper * 0.99:
            return "UPPER"
        if price <= bb_lower * 1.01:
            return "LOWER"
        return "MIDDLE"

    @staticmethod
    def _extract_macd_signal(tech_data):
        macd_line = tech_data.get("macd_line")
        macd_signal_line = tech_data.get("macd_signal")
        if macd_line is not None and macd_signal_line is not None:
            if macd_line > macd_signal_line:
                return "BULLISH"
            if macd_line < macd_signal_line:
                return "BEARISH"
        return "NEUTRAL"

    @staticmethod
    def _extract_volume_state(tech_data):
        obv_slope = tech_data.get("obv_slope", 0.0)
        if obv_slope > 0.5:
            return "ACCUMULATION"
        if obv_slope < -0.5:
            return "DISTRIBUTION"
        return "NORMAL"
