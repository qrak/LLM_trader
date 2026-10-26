"""One clock for every operator-facing timestamp.

The trading maths stays on UTC candles; these tests pin down the *text*: an explicit
offset on every rendered instant, a day folder that follows the same zone as the
lines inside it, and the DST hour that repeats on the wall clock without repeating
in the log.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from src.logger.logger import Logger, _write_fallback_crash
from src.utils import time_utils
from src.utils.time_utils import (
    day_stamp,
    format_offset,
    local_stamp,
    reset_timezone_cache,
    resolve_timezone,
    utc_and_local,
    utc_stamp,
)


@pytest.fixture
def warsaw(monkeypatch):
    """Pin the configured display zone to Europe/Warsaw for the duration of a test."""
    monkeypatch.setattr(time_utils, "configured_timezone_name", lambda: "Europe/Warsaw")
    reset_timezone_cache()
    yield ZoneInfo("Europe/Warsaw")
    reset_timezone_cache()


def test_a_zone_name_follows_the_iana_dst_rules():
    zone = resolve_timezone("Europe/Warsaw")

    assert datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc).astimezone(zone).utcoffset() == timedelta(hours=2)
    assert datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc).astimezone(zone).utcoffset() == timedelta(hours=1)


def test_an_unset_zone_falls_back_to_the_operating_system_clock():
    assert resolve_timezone("").utcoffset(None) == datetime.now().astimezone().utcoffset()


def test_an_unknown_zone_name_fails_loudly():
    with pytest.raises(ValueError, match="Europe/Warsaw"):
        resolve_timezone("Europe/Warsow")


@pytest.mark.parametrize(
    ("zone_name", "moment", "expected"),
    [
        pytest.param("Europe/Warsaw", datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc), "+02:00", id="summer"),
        pytest.param("Europe/Warsaw", datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc), "+01:00", id="winter"),
        pytest.param("Asia/Kolkata", datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc), "+05:30", id="half-hour"),
        pytest.param("America/St_Johns", datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc), "-03:30", id="negative"),
        pytest.param("UTC", datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc), "+00:00", id="utc"),
    ],
)
def test_offsets_are_rendered_as_hh_mm(zone_name, moment, expected):
    """The offset is the whole point: a bare wall time is ambiguous twice a year."""
    assert format_offset(moment.astimezone(ZoneInfo(zone_name))) == expected


def test_the_repeated_autumn_hour_stays_distinguishable():
    """02:30 occurs twice on 25.10.2026; the UTC half of the label tells them apart."""
    summer = utc_and_local(datetime(2026, 10, 25, 0, 30, tzinfo=timezone.utc), ZoneInfo("Europe/Warsaw"))
    winter = utc_and_local(datetime(2026, 10, 25, 1, 30, tzinfo=timezone.utc), ZoneInfo("Europe/Warsaw"))

    assert summer == "2026-10-25 00:30:00 UTC / 2026-10-25 02:30:00 CEST"
    assert winter == "2026-10-25 01:30:00 UTC / 2026-10-25 02:30:00 CET"


def test_the_display_day_follows_the_configured_zone(warsaw):
    """22:30 UTC is already the next day in Warsaw — the folder must say so."""
    assert day_stamp(datetime(2026, 10, 24, 22, 30, tzinfo=timezone.utc)) == "2026_10_25"


def test_a_log_line_carries_the_offset_of_the_configured_zone(tmp_path, warsaw):
    logger = Logger(logger_name="ZoneBot", log_dir=str(tmp_path))
    logger.info("offset check")
    logger.close()

    written = (tmp_path / "ZoneBot" / day_stamp() / "ZoneBot.log").read_text(encoding="utf-8")
    current_offset = format_offset(datetime.now(warsaw))

    assert re.search(
        rf"^\[\d{{2}}\.\d{{2}}\.\d{{4}} \d{{2}}:\d{{2}}:\d{{2}} {re.escape(current_offset)}\] - offset check$",
        written,
        re.MULTILINE,
    )


def test_the_crash_writer_stamps_the_same_clock(tmp_path, monkeypatch, warsaw):
    crash_path = tmp_path / "errors.log"
    monkeypatch.setattr("src.logger.logger._CRASH_LOG_PATH", str(crash_path))

    try:
        raise RuntimeError("zone check")
    except RuntimeError as exc:
        _write_fallback_crash(type(exc), exc, exc.__traceback__)

    first_line = crash_path.read_text(encoding="utf-8").splitlines()[0]
    current_offset = format_offset(datetime.now(warsaw))
    assert re.match(
        rf"^\[\d{{2}}\.\d{{2}}\.\d{{4}} \d{{2}}:\d{{2}}:\d{{2}} {re.escape(current_offset)}\] Unhandled exception$",
        first_line,
    )


def test_local_stamp_of_a_naive_timestamp_is_read_as_utc(warsaw):
    """A naive stored stamp is UTC (the project convention), never the wall clock."""
    assert local_stamp(datetime(2026, 1, 15, 12, 0)) == "15.01.2026 13:00:00 +01:00"


def test_utc_stamp_marks_the_zone_and_handles_none():
    assert utc_stamp(datetime(2026, 4, 14, 4, 0, 48)) == "2026-04-14 04:00:48 UTC"
    assert utc_stamp(None) == "unknown"
