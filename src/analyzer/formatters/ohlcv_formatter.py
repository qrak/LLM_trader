

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

from src.utils.timeframe_validator import TimeframeValidator

if TYPE_CHECKING:
    from src.utils.format_utils import FormatUtils

WEEKLY_TIMEFRAMES = frozenset({"1w", "1W", "1wk", "7d"})
SECTION_START = "## Raw OHLCV"
TRUNCATED = "<truncated>"


def truncate_ohlcv_section(text: str) -> str:

    start = text.find(SECTION_START)
    if start < 0:
        return text
    end = text.find("\n## ", start + len(SECTION_START))
    if end < 0:
        end = len(text)
    return f"{text[:start]}{SECTION_START}\n{TRUNCATED}{text[end:]}"


class OhlcvTableFormatter:
    CANDLES = 183
    WEEKLY_CANDLES = 104

    def __init__(self, format_utils: "FormatUtils") -> None:

        self.format_utils = format_utils

    def format_table(self, ohlcv: np.ndarray | None, timeframe: str) -> str:

        if ohlcv is None or len(ohlcv) == 0:
            return ""
        limit = self.WEEKLY_CANDLES if timeframe in WEEKLY_TIMEFRAMES else self.CANDLES
        window = ohlcv[-limit:] if len(ohlcv) > limit else ohlcv
        finite = window[np.isfinite(window[:, 1:6]).all(axis=1)]
        if len(finite) == 0:
            return ""
        date_only = timeframe in WEEKLY_TIMEFRAMES
        stamp_format = "%Y-%m-%d" if date_only else "%Y-%m-%d %H:%M"
        rows = [
            f"{datetime.fromtimestamp(float(row[0]) / 1000, tz=timezone.utc):{stamp_format}},"
            f"{self.format_utils.fmt(float(row[1]))},"
            f"{self.format_utils.fmt(float(row[2]))},"
            f"{self.format_utils.fmt(float(row[3]))},"
            f"{self.format_utils.fmt(float(row[4]))},"
            f"{float(row[5]):.1f}"
            for row in finite
        ]
        header = (
            f"### {timeframe} OHLCV (last {len(rows)} closed candles, UTC, oldest first)\n"
            "time,open,high,low,close,volume"
        )
        return "\n".join([header, *rows])

    def format_section(self, context: Any, primary_timeframe: str) -> str:

        tables: list[str] = []
        primary = self.format_table(getattr(context, "ohlcv_candles", None), primary_timeframe)
        if primary:
            tables.append(primary)
        long_term = getattr(context, "long_term_data", None) or {}
        if TimeframeValidator.to_minutes(primary_timeframe) < 1440:
            daily = self.format_table(long_term.get("data"), "1d")
            if daily:
                tables.append(daily)
        weekly = self.format_table(getattr(context, "weekly_ohlcv", None), "1w")
        if weekly:
            tables.append(weekly)
        if not tables:
            return ""
        return "## Raw OHLCV\n\n" + "\n\n".join(tables)
