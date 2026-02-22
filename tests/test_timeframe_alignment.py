
from datetime import datetime, timezone
from src.utils.timeframe_validator import TimeframeValidator

class TestTimeframeAlignment:
    """Tests for TimeframeValidator alignment logic."""

    def test_weekly_alignment_monday_start(self):
        """Test that weekly timeframe aligns to Monday 00:00 UTC."""
        # 2023-12-25 was a Monday. 00:00 UTC. Timestamp: 1703462400000
        monday_ts = 1703462400000

        # 2023-12-28 was a Thursday. 00:00 UTC. Timestamp: 1703721600000
        thursday_ts = 1703721600000

        # Calculate next candle time for 1w from Monday
        # Should be next Monday (2024-01-01)
        next_candle_monday = TimeframeValidator.calculate_next_candle_time(monday_ts, '1w')
        dt_next_monday = datetime.fromtimestamp(next_candle_monday / 1000, tz=timezone.utc)

        # 2024-01-01 00:00:00 UTC
        expected_next_monday = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        assert dt_next_monday == expected_next_monday, \
            f"Expected next candle from Monday to be {expected_next_monday}, got {dt_next_monday}"

        # Calculate next candle time for 1w from Thursday
        # Should be NEXT Monday (2024-01-01), NOT next Thursday
        next_candle_thursday = TimeframeValidator.calculate_next_candle_time(thursday_ts, '1w')
        dt_next_thursday = datetime.fromtimestamp(next_candle_thursday / 1000, tz=timezone.utc)

        assert dt_next_thursday == expected_next_monday, \
            f"Expected next candle from Thursday to be {expected_next_monday}, got {dt_next_thursday}"

    def test_is_same_candle_weekly(self):
        """Test is_same_candle for weekly timeframe crossing Monday boundary."""
        # Sunday 2023-12-31 23:59:59 UTC
        sunday_end = datetime(2023, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp() * 1000

        # Monday 2024-01-01 00:00:01 UTC
        monday_start = datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc).timestamp() * 1000

        # These should be in DIFFERENT candles if Monday is start
        # But with Thursday alignment, they might be in same candle (Thu-Thu covers Sun-Mon)
        assert TimeframeValidator.is_same_candle(sunday_end, monday_start, '1w') is False, \
            "Sunday end and Monday start should be in different candles for 1w"

    def test_daily_alignment_unchanged(self):
        """Test that daily alignment (00:00 UTC) remains correct."""
        # 2023-12-25 12:00:00 UTC
        ts = datetime(2023, 12, 25, 12, 0, 0, tzinfo=timezone.utc).timestamp() * 1000

        # Next candle should be 2023-12-26 00:00:00 UTC
        expected = datetime(2023, 12, 26, 0, 0, 0, tzinfo=timezone.utc)

        next_candle = TimeframeValidator.calculate_next_candle_time(ts, '1d')
        dt_next = datetime.fromtimestamp(next_candle / 1000, tz=timezone.utc)

        assert dt_next == expected

    def test_4h_alignment_unchanged(self):
        """Test that 4h alignment remains correct."""
        # 2023-12-25 01:00:00 UTC
        ts = datetime(2023, 12, 25, 1, 0, 0, tzinfo=timezone.utc).timestamp() * 1000

        # Next candle should be 04:00:00 UTC
        expected = datetime(2023, 12, 25, 4, 0, 0, tzinfo=timezone.utc)

        next_candle = TimeframeValidator.calculate_next_candle_time(ts, '4h')
        dt_next = datetime.fromtimestamp(next_candle / 1000, tz=timezone.utc)

        assert dt_next == expected
