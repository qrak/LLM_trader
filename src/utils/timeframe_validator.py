

import re


class TimeframeValidator:
    """Validates and manages timeframe configurations"""
    SUPPORTED_TIMEFRAMES = ["5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1w"]  # noqa: RUF012
    MINUTES_IN_HOUR = 60
    MINUTES_IN_DAY = 1440
    MINUTES_IN_WEEK = 10080

    MS_IN_MINUTE = 60 * 1000
    MS_IN_DAY = 24 * 60 * 60 * 1000

    MONDAY_ALIGNMENT_OFFSET_DAYS = 4

    TIMEFRAME_MINUTES = {  # noqa: RUF012
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": MINUTES_IN_HOUR,
        "2h": 2 * MINUTES_IN_HOUR,
        "4h": 4 * MINUTES_IN_HOUR,
        "6h": 6 * MINUTES_IN_HOUR,
        "8h": 8 * MINUTES_IN_HOUR,
        "12h": 12 * MINUTES_IN_HOUR,
        "1d": MINUTES_IN_DAY,
        "1w": MINUTES_IN_WEEK
    }
    CCXT_STANDARD_TIMEFRAMES = ["5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1w"]  # noqa: RUF012

    @classmethod
    def validate(cls, timeframe: str) -> bool:

        return timeframe in cls.SUPPORTED_TIMEFRAMES

    @classmethod
    def to_minutes(cls, timeframe: str) -> int:

        if timeframe not in cls.TIMEFRAME_MINUTES:
            raise ValueError(f"Unrecognized timeframe: {timeframe}")
        return cls.TIMEFRAME_MINUTES[timeframe]

    @classmethod
    def parse_period_to_minutes(cls, period: str) -> int:

        match = re.match(r"^(\d+)([mhdw])$", period.lower())
        if not match:
            raise ValueError(f"Invalid period format: {period}")

        value = int(match.group(1))
        unit = match.group(2)

        if unit == "m":
            return value
        if unit == "h":
            return value * cls.MINUTES_IN_HOUR
        if unit == "d":
            return value * cls.MINUTES_IN_DAY
        if unit == "w":
            return value * cls.MINUTES_IN_WEEK
        raise ValueError(f"Invalid period unit: {unit}")

    @classmethod
    def calculate_period_candles(cls, base_timeframe: str, target_period: str) -> int:

        base_mins = cls.to_minutes(base_timeframe)
        target_mins = cls.parse_period_to_minutes(target_period)
        return target_mins // base_mins

    @classmethod
    def is_ccxt_compatible(cls, timeframe: str, _exchange_name: str | None = None) -> bool:

        if timeframe in cls.CCXT_STANDARD_TIMEFRAMES:  # noqa: SIM103
            return True

        return False

    @classmethod
    def get_candle_limit_for_days(cls, timeframe: str, target_days: int = 30) -> int:

        timeframe_minutes = cls.to_minutes(timeframe)
        target_minutes = target_days * cls.MINUTES_IN_DAY
        return target_minutes // timeframe_minutes

    @classmethod
    def calculate_coverage_days(cls, timeframe: str, candle_count: int) -> float:

        timeframe_minutes = cls.to_minutes(timeframe)
        return (candle_count * timeframe_minutes) / cls.MINUTES_IN_DAY

    @classmethod
    def validate_and_normalize(cls, timeframe: str) -> str:

        normalized = timeframe.lower()

        if not cls.validate(normalized):
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. "
                f"Supported: {', '.join(cls.SUPPORTED_TIMEFRAMES)}"
            )

        return normalized

    @classmethod
    def _get_alignment_offset(cls, timeframe: str) -> int:

        if timeframe == "1w":
            return cls.MONDAY_ALIGNMENT_OFFSET_DAYS * cls.MS_IN_DAY
        return 0

    @classmethod
    def calculate_next_candle_time(cls, current_time_ms: int, timeframe: str) -> int:

        interval_minutes = cls.to_minutes(timeframe)
        interval_ms = interval_minutes * cls.MS_IN_MINUTE
        offset = cls._get_alignment_offset(timeframe)

        aligned_time = current_time_ms - offset
        next_index = (aligned_time // interval_ms) + 1
        next_candle_ms = (next_index * interval_ms) + offset

        return next_candle_ms

    @classmethod
    def is_same_candle(cls, time1_ms: int, time2_ms: int, timeframe: str) -> bool:

        interval_minutes = cls.to_minutes(timeframe)
        interval_ms = interval_minutes * cls.MS_IN_MINUTE
        offset = cls._get_alignment_offset(timeframe)

        candle1 = (time1_ms - offset) // interval_ms
        candle2 = (time2_ms - offset) // interval_ms

        return candle1 == candle2

