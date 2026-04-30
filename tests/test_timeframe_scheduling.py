from datetime import datetime, timezone
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from src.app import CryptoTradingBot
from src.utils.timeframe_validator import TimeframeValidator


class NaiveUtcDateTime(datetime):
    def astimezone(self, tz=None):
        raise AssertionError("naive UTC timestamps must not be reinterpreted as local time")


def test_4h_next_candle_stays_on_utc_boundaries_across_warsaw_dst():
    warsaw = ZoneInfo("Europe/Warsaw")

    before_dst = datetime(2026, 3, 29, 1, 30, tzinfo=warsaw)
    after_dst = datetime(2026, 3, 29, 6, 30, tzinfo=warsaw)

    next_before = TimeframeValidator.calculate_next_candle_time(int(before_dst.timestamp() * 1000), "4h")
    next_after = TimeframeValidator.calculate_next_candle_time(int(after_dst.timestamp() * 1000), "4h")

    assert datetime.fromtimestamp(next_before / 1000, timezone.utc) == datetime(2026, 3, 29, 4, 0, tzinfo=timezone.utc)
    assert datetime.fromtimestamp(next_after / 1000, timezone.utc) == datetime(2026, 3, 29, 8, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("timeframe", "expected_minutes"),
    [
        ("5m", 5),
        ("15m", 15),
        ("30m", 30),
    ]
)
def test_sub_hour_timeframes_are_supported(timeframe, expected_minutes):
    assert TimeframeValidator.validate(timeframe) is True
    assert TimeframeValidator.is_ccxt_compatible(timeframe) is True
    assert TimeframeValidator.to_minutes(timeframe) == expected_minutes
    assert TimeframeValidator.validate_and_normalize(timeframe.upper()) == timeframe


@pytest.mark.parametrize(
    ("timeframe", "expected_next"),
    [
        ("5m", datetime(2026, 4, 30, 13, 45, tzinfo=timezone.utc)),
        ("15m", datetime(2026, 4, 30, 13, 45, tzinfo=timezone.utc)),
        ("30m", datetime(2026, 4, 30, 14, 0, tzinfo=timezone.utc)),
    ]
)
def test_sub_hour_next_candle_boundaries_align_to_utc(timeframe, expected_next):
    current_time = datetime(2026, 4, 30, 13, 43, 28, tzinfo=timezone.utc)

    next_candle = TimeframeValidator.calculate_next_candle_time(
        int(current_time.timestamp() * 1000),
        timeframe
    )

    assert datetime.fromtimestamp(next_candle / 1000, timezone.utc) == expected_next


def test_unsupported_timeframe_is_still_rejected():
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        TimeframeValidator.validate_and_normalize("3m")

    with pytest.raises(ValueError, match="Unrecognized timeframe"):
        TimeframeValidator.to_minutes("3m")


def test_get_formatted_last_analysis_time_treats_naive_timestamp_as_utc():
    bot = CryptoTradingBot.__new__(CryptoTradingBot)
    bot.persistence = MagicMock()
    bot.persistence.get_last_analysis_time.return_value = NaiveUtcDateTime(2026, 4, 14, 4, 0, 48)

    assert bot._get_formatted_last_analysis_time() == "2026-04-14 04:00:48"


def test_format_utc_and_local_shows_warsaw_summer_time():
    formatted = CryptoTradingBot._format_utc_and_local(
        datetime(2026, 4, 14, 8, 0, tzinfo=timezone.utc),
        ZoneInfo("Europe/Warsaw")
    )

    assert formatted == "2026-04-14 08:00:00 UTC / 2026-04-14 10:00:00 CEST"