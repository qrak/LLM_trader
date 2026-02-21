"""
Tests for TimeframeValidator in src/utils/timeframe_validator.py.
"""
import pytest
from src.utils.timeframe_validator import TimeframeValidator

class TestTimeframeValidator:
    """Tests for the TimeframeValidator class."""

    def test_validate_valid_timeframes(self):
        """Test validation of supported timeframes."""
        valid_timeframes = ['1h', '2h', '4h', '6h', '8h', '12h', '1d', '1w']
        for tf in valid_timeframes:
            assert TimeframeValidator.validate(tf) is True

    def test_validate_invalid_timeframes(self):
        """Test validation of unsupported timeframes."""
        invalid_timeframes = ['30m', '1m', '2d', '1M', 'invalid', '']
        for tf in invalid_timeframes:
            assert TimeframeValidator.validate(tf) is False

    def test_to_minutes_valid(self):
        """Test conversion of valid timeframes to minutes."""
        assert TimeframeValidator.to_minutes('1h') == 60
        assert TimeframeValidator.to_minutes('2h') == 120
        assert TimeframeValidator.to_minutes('4h') == 240
        assert TimeframeValidator.to_minutes('1d') == 1440
        assert TimeframeValidator.to_minutes('1w') == 10080

    def test_to_minutes_invalid(self):
        """Test that invalid timeframes raise ValueError."""
        with pytest.raises(ValueError, match="Unrecognized timeframe"):
            TimeframeValidator.to_minutes('30m')

    def test_parse_period_to_minutes_valid(self):
        """Test parsing of period strings to minutes."""
        # Hours
        assert TimeframeValidator.parse_period_to_minutes('1h') == 60
        assert TimeframeValidator.parse_period_to_minutes('24h') == 1440
        # Days
        assert TimeframeValidator.parse_period_to_minutes('1d') == 1440
        assert TimeframeValidator.parse_period_to_minutes('7d') == 10080
        assert TimeframeValidator.parse_period_to_minutes('30d') == 43200
        # Case insensitivity
        assert TimeframeValidator.parse_period_to_minutes('1H') == 60
        assert TimeframeValidator.parse_period_to_minutes('1D') == 1440

    def test_parse_period_to_minutes_invalid(self):
        """Test that invalid period formats raise ValueError."""
        # Minutes (not supported by implementation regex)
        with pytest.raises(ValueError, match="Invalid period format"):
            TimeframeValidator.parse_period_to_minutes('30m')

        # Invalid units
        with pytest.raises(ValueError, match="Invalid period format"):
            TimeframeValidator.parse_period_to_minutes('1w')  # 'w' not in regex [hd]

        # Invalid format
        with pytest.raises(ValueError, match="Invalid period format"):
            TimeframeValidator.parse_period_to_minutes('invalid')

        with pytest.raises(ValueError, match="Invalid period format"):
            TimeframeValidator.parse_period_to_minutes('')

    def test_calculate_period_candles(self):
        """Test calculation of required candles for a period."""
        # 1 day in 1h candles = 24
        assert TimeframeValidator.calculate_period_candles('1h', '24h') == 24
        assert TimeframeValidator.calculate_period_candles('1h', '1d') == 24

        # 1 day in 4h candles = 6
        assert TimeframeValidator.calculate_period_candles('4h', '1d') == 6

        # 7 days in 1d candles = 7
        assert TimeframeValidator.calculate_period_candles('1d', '7d') == 7

        # 30 days in 4h candles = 30 * 6 = 180
        assert TimeframeValidator.calculate_period_candles('4h', '30d') == 180

    def test_to_cryptocompare_format(self):
        """Test conversion to CryptoCompare API format."""
        # Hourly
        ep, mult = TimeframeValidator.to_cryptocompare_format('1h')
        assert ep == 'hour' and mult == 1

        ep, mult = TimeframeValidator.to_cryptocompare_format('4h')
        assert ep == 'hour' and mult == 4

        # Daily
        ep, mult = TimeframeValidator.to_cryptocompare_format('1d')
        assert ep == 'day' and mult == 1

        # Unsupported
        with pytest.raises(ValueError, match="not supported by CryptoCompare"):
            TimeframeValidator.to_cryptocompare_format('1w') # Assuming 1w is not in CRYPTOCOMPARE_FORMAT based on file read

    def test_calculate_next_candle_time(self):
        """Test calculation of next candle start time."""
        # 1h candle
        # 1600000000000 is divisible by 3600000?
        # 1600000000000 / 3600000 = 444444.44...
        # Next boundary should be 444445 * 3600000 = 1600002000000
        current_ms = 1600000000000
        expected_next = 1600002000000
        assert TimeframeValidator.calculate_next_candle_time(current_ms, '1h') == expected_next

        # Exact boundary
        # If current time is exactly at boundary, next candle is +interval
        # Implementation: ((current // interval) + 1) * interval
        # If current == interval * N, then (N+1) * interval. Correct.
        boundary_ms = 1600002000000
        expected_next_boundary = 1600002000000 + 3600000 # +1h
        assert TimeframeValidator.calculate_next_candle_time(boundary_ms, '1h') == expected_next_boundary

    def test_is_same_candle(self):
        """Test detection of timestamps within the same candle."""
        tf = '1h'
        interval_ms = 3600000
        start_ms = 1000 * interval_ms # 1000th hour

        t1 = start_ms + 100 # Just after start
        t2 = start_ms + interval_ms - 100 # Just before end
        t3 = start_ms + interval_ms + 100 # Just after end (next candle)

        assert TimeframeValidator.is_same_candle(t1, t2, tf) is True
        assert TimeframeValidator.is_same_candle(t1, t3, tf) is False
        assert TimeframeValidator.is_same_candle(t2, t3, tf) is False

    def test_validate_and_normalize(self):
        """Test validation and normalization."""
        assert TimeframeValidator.validate_and_normalize('1H') == '1h'
        assert TimeframeValidator.validate_and_normalize('4h') == '4h'

        with pytest.raises(ValueError, match="Unsupported timeframe"):
            TimeframeValidator.validate_and_normalize('30m')

    def test_get_candle_limit_for_days(self):
        """Test candle limit calculation."""
        # 1 day of 1h candles = 24
        assert TimeframeValidator.get_candle_limit_for_days('1h', 1) == 24
        # 30 days of 4h candles = 30 * 6 = 180
        assert TimeframeValidator.get_candle_limit_for_days('4h', 30) == 180
