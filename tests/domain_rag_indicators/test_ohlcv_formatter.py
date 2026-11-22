

from datetime import datetime, timedelta, timezone
from typing import ClassVar

import numpy as np
import pytest

from src.analyzer.formatters import OhlcvTableFormatter, truncate_ohlcv_section
from src.utils.format_utils import FormatUtils

START = datetime(2026, 11, 1, tzinfo=timezone.utc)


def candles(count: int, step_hours: int = 4) -> np.ndarray:

    rows = [
        [
            int((START + timedelta(hours=step_hours * index)).timestamp() * 1000),
            60000.0 + index,
            60100.0 + index,
            59900.0 + index,
            60050.0 + index,
            1234.5 + index,
        ]
        for index in range(count)
    ]
    return np.array(rows, dtype=float)


@pytest.fixture
def formatter() -> OhlcvTableFormatter:

    return OhlcvTableFormatter(FormatUtils())


def test_table_keeps_the_last_183_rows_oldest_first(formatter):

    lines = formatter.format_table(candles(200), "4h").splitlines()

    assert lines[0] == "### 4h OHLCV (last 183 closed candles, UTC, oldest first)"
    assert lines[1] == "time,open,high,low,close,volume"
    assert len(lines) == 185
    assert lines[2].startswith("2026-11-03 20:00,")
    assert lines[-1].startswith("2026-12-04 04:00,")


def test_a_short_series_reports_its_own_length(formatter):

    lines = formatter.format_table(candles(5), "4h").splitlines()

    assert lines[0] == "### 4h OHLCV (last 5 closed candles, UTC, oldest first)"
    assert len(lines) == 7


def test_weekly_rows_carry_a_date_only(formatter):

    lines = formatter.format_table(candles(3, step_hours=168), "1w").splitlines()

    assert lines[2].startswith("2026-11-01,")


def test_non_finite_rows_are_dropped(formatter):

    data = candles(4)
    data[1, 4] = np.nan

    lines = formatter.format_table(data, "4h").splitlines()

    assert len(lines) == 5
    assert lines[0] == "### 4h OHLCV (last 3 closed candles, UTC, oldest first)"


def test_missing_or_empty_series_render_nothing(formatter):

    assert formatter.format_table(None, "4h") == ""
    assert formatter.format_table(np.empty((0, 6)), "4h") == ""


def test_section_lists_primary_daily_and_weekly(formatter):

    class Context:
        ohlcv_candles = candles(10)
        long_term_data: ClassVar[dict] = {"data": candles(10, step_hours=24)}
        weekly_ohlcv = candles(10, step_hours=168)

    section = formatter.format_section(Context(), "4h")

    assert section.startswith("## Raw OHLCV\n\n")
    assert [line for line in section.splitlines() if line.startswith("###")] == [
        "### 4h OHLCV (last 10 closed candles, UTC, oldest first)",
        "### 1d OHLCV (last 10 closed candles, UTC, oldest first)",
        "### 1w OHLCV (last 10 closed candles, UTC, oldest first)",
    ]


def test_a_daily_primary_frame_does_not_repeat_the_daily_table(formatter):

    class Context:
        ohlcv_candles = candles(10, step_hours=24)
        long_term_data: ClassVar[dict] = {"data": candles(10, step_hours=24)}
        weekly_ohlcv = candles(10, step_hours=168)

    section = formatter.format_section(Context(), "1d")

    assert [line for line in section.splitlines() if line.startswith("###")] == [
        "### 1d OHLCV (last 10 closed candles, UTC, oldest first)",
        "### 1w OHLCV (last 10 closed candles, UTC, oldest first)",
    ]


def test_a_context_without_any_series_renders_nothing(formatter):

    class Context:
        ohlcv_candles = None
        long_term_data = None
        weekly_ohlcv = None

    assert formatter.format_section(Context(), "4h") == ""


def test_truncation_replaces_only_the_ohlcv_section():

    prompt = "## Market Data\n4h: +1%\n\n## Raw OHLCV\n\n### 4h OHLCV (last 2 closed candles)\ntime,open\n1,2\n\n## Technical (4h)\nRSI 55"

    truncated = truncate_ohlcv_section(prompt)

    assert truncated == "## Market Data\n4h: +1%\n\n## Raw OHLCV\n<truncated>\n## Technical (4h)\nRSI 55"


def test_truncation_handles_a_trailing_ohlcv_section():

    assert truncate_ohlcv_section("## Raw OHLCV\n\n### 4h\ntime,open\n1,2") == "## Raw OHLCV\n<truncated>"


def test_truncation_leaves_a_prompt_without_the_section_untouched():

    assert truncate_ohlcv_section("## Market Data\n4h: +1%") == "## Market Data\n4h: +1%"
