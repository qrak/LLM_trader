"""
Timeframe validation and conversion utilities.

This module provides comprehensive timeframe validation, conversion between formats,
and compatibility checking for various APIs and exchanges.

Supported timeframe range: 5m (minimum) to 1w (maximum)
"""

from typing import Optional
import re


class TimeframeValidator:
    """Validates and manages timeframe configurations"""
    SUPPORTED_TIMEFRAMES = ['5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1w']
    # Time constants
    MINUTES_IN_HOUR = 60
    MINUTES_IN_DAY = 1440
    MINUTES_IN_WEEK = 10080

    MS_IN_MINUTE = 60 * 1000
    MS_IN_HOUR = 60 * 60 * 1000
    MS_IN_DAY = 24 * 60 * 60 * 1000

    # Alignment constants
    # Offset to align weekly candles to Monday.
    # Unix Epoch (1970-01-01) was Thursday.
    # Monday (1970-01-05) is +4 days from Epoch.
    MONDAY_ALIGNMENT_OFFSET_DAYS = 4

    TIMEFRAME_MINUTES = {
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': MINUTES_IN_HOUR,
        '2h': 2 * MINUTES_IN_HOUR,
        '4h': 4 * MINUTES_IN_HOUR,
        '6h': 6 * MINUTES_IN_HOUR,
        '8h': 8 * MINUTES_IN_HOUR,
        '12h': 12 * MINUTES_IN_HOUR,
        '1d': MINUTES_IN_DAY,
        '1w': MINUTES_IN_WEEK
    }
    CCXT_STANDARD_TIMEFRAMES = ['5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1w']

    @classmethod
    def validate(cls, timeframe: str) -> bool:
        """
        Check if timeframe is fully supported.

        Args:
            timeframe: Timeframe string (e.g., "5m", "1h", "4h", "1d")

        Returns:
            bool: True if timeframe is supported, False otherwise
        """
        return timeframe in cls.SUPPORTED_TIMEFRAMES

    @classmethod
    def to_minutes(cls, timeframe: str) -> int:
        """
        Convert timeframe to minutes.

        Args:
            timeframe: Timeframe string (e.g., "5m", "1h", "4h", "1d")

        Returns:
            int: Number of minutes in the timeframe

        Raises:
            ValueError: If timeframe is not recognized
        """
        if timeframe not in cls.TIMEFRAME_MINUTES:
            raise ValueError(f"Unrecognized timeframe: {timeframe}")
        return cls.TIMEFRAME_MINUTES[timeframe]

    @classmethod
    def parse_period_to_minutes(cls, period: str) -> int:
        """
        Parse period string to minutes. Supports both timeframes and arbitrary periods.

        Args:
            period: Period string (e.g., "5m", "1h", "4h", "24h", "7d", "30d")

        Returns:
            int: Number of minutes in the period

        Raises:
            ValueError: If period format is invalid
        """
        match = re.match(r'^(\d+)([mhdw])$', period.lower())
        if not match:
            raise ValueError(f"Invalid period format: {period}")

        value = int(match.group(1))
        unit = match.group(2)

        if unit == 'm':
            return value
        if unit == 'h':
            return value * cls.MINUTES_IN_HOUR
        if unit == 'd':
            return value * cls.MINUTES_IN_DAY
        if unit == 'w':
            return value * cls.MINUTES_IN_WEEK
        raise ValueError(f"Invalid period unit: {unit}")

    @classmethod
    def calculate_period_candles(cls, base_timeframe: str, target_period: str) -> int:
        """
        Calculate how many candles are needed for a target period.

        Args:
            base_timeframe: The base timeframe (e.g., "5m", "1h", "4h")
            target_period: The target period (e.g., "24h", "7d", "30d")

        Returns:
            int: Number of candles needed

        Example:
            >>> calculate_period_candles("4h", "24h")
            6  # 6 four-hour candles = 24 hours
        """
        base_mins = cls.to_minutes(base_timeframe)
        target_mins = cls.parse_period_to_minutes(target_period)
        return target_mins // base_mins

    @classmethod
    def is_ccxt_compatible(cls, timeframe: str, exchange_name: Optional[str] = None) -> bool:
        """
        Check if timeframe is compatible with CCXT exchanges.

        Args:
            timeframe: Timeframe string (e.g., "5m", "1h", "4h")
            exchange_name: Optional specific exchange name for exact validation

        Returns:
            bool: True if timeframe is likely supported

        Note:
            For exact validation with a specific exchange, pass the exchange instance
            to check its .timeframes property directly.
        """
        # Basic check for standard timeframes
        if timeframe in cls.CCXT_STANDARD_TIMEFRAMES:
            return True

        # Could be extended to check specific exchange support
        # via exchange.timeframes property if exchange instance is available
        return False

    @classmethod
    def get_candle_limit_for_days(cls, timeframe: str, target_days: int = 30) -> int:
        """
        Calculate how many candles are needed to cover a target number of days.

        Args:
            timeframe: The timeframe (e.g., "5m", "1h", "4h", "1d")
            target_days: Number of days to cover (default: 30)

        Returns:
            int: Number of candles needed

        Example:
            >>> get_candle_limit_for_days("4h", 30)
            180  # 6 candles per day * 30 days
        """
        timeframe_minutes = cls.to_minutes(timeframe)
        target_minutes = target_days * cls.MINUTES_IN_DAY
        return target_minutes // timeframe_minutes

    @classmethod
    def calculate_coverage_days(cls, timeframe: str, candle_count: int) -> float:
        """
        Calculate how many days of market history a candle count represents.

        Args:
            timeframe: The timeframe (e.g., "5m", "1h", "4h", "1d")
            candle_count: Number of closed candles available

        Returns:
            float: Approximate number of days covered by the candles
        """
        timeframe_minutes = cls.to_minutes(timeframe)
        return (candle_count * timeframe_minutes) / cls.MINUTES_IN_DAY

    @classmethod
    def validate_and_normalize(cls, timeframe: str) -> str:
        """
        Validate and normalize timeframe string.

        Handles case variations and returns normalized form.

        Args:
            timeframe: Timeframe string (e.g., "5M", "1H", "4h", "1D")

        Returns:
            str: Normalized timeframe (lowercase)

        Raises:
            ValueError: If timeframe is invalid or unsupported
        """
        normalized = timeframe.lower()

        if not cls.validate(normalized):
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. "
                f"Supported: {', '.join(cls.SUPPORTED_TIMEFRAMES)}"
            )

        return normalized

    @classmethod
    def _get_alignment_offset(cls, timeframe: str) -> int:
        """
        Get alignment offset in milliseconds for timeframes that don't align with Unix Epoch (Thursday).

        Weekly candles ('1w') are aligned to start on Monday 00:00 UTC.
        Since the Unix Epoch (1970-01-01) was a Thursday, we need an offset to shift
        the reference point to the next Monday (1970-01-05).

        Args:
            timeframe: Timeframe string

        Returns:
            int: Offset in milliseconds required to align the timeframe
        """
        if timeframe == '1w':
            return cls.MONDAY_ALIGNMENT_OFFSET_DAYS * cls.MS_IN_DAY
        return 0

    @classmethod
    def calculate_next_candle_time(cls, current_time_ms: int, timeframe: str) -> int:
        """
        Calculate the start time of the next candle for a given timeframe.

        Args:
            current_time_ms: Current timestamp in milliseconds
            timeframe: Timeframe string (e.g., "5m", "1h", "4h", "1d")

        Returns:
            int: Next candle start time in milliseconds

        Example:
            >>> calculate_next_candle_time(1704067200000, "4h")
            1704081600000  # Next 4h candle boundary
        """
        interval_minutes = cls.to_minutes(timeframe)
        interval_ms = interval_minutes * cls.MS_IN_MINUTE
        offset = cls._get_alignment_offset(timeframe)

        # Calculate next candle boundary with offset
        # (time - offset) aligns to 0-based index relative to alignment point
        aligned_time = current_time_ms - offset
        next_index = (aligned_time // interval_ms) + 1
        next_candle_ms = (next_index * interval_ms) + offset

        return next_candle_ms

    @classmethod
    def is_same_candle(cls, time1_ms: int, time2_ms: int, timeframe: str) -> bool:
        """
        Check if two timestamps fall within the same candle period.

        Args:
            time1_ms: First timestamp in milliseconds
            time2_ms: Second timestamp in milliseconds
            timeframe: Timeframe string (e.g., "5m", "1h", "4h", "1d")

        Returns:
            bool: True if both timestamps are in the same candle period
        """
        interval_minutes = cls.to_minutes(timeframe)
        interval_ms = interval_minutes * cls.MS_IN_MINUTE
        offset = cls._get_alignment_offset(timeframe)

        candle1 = (time1_ms - offset) // interval_ms
        candle2 = (time2_ms - offset) // interval_ms

        return candle1 == candle2
