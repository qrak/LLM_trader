
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.utils.data_utils import get_indicator_value

SCIENTIFIC_NOTATION_THRESHOLD = 1e-7
CRYPTO_DUST_THRESHOLD = 1e-5
MICRO_VALUE_THRESHOLD = 1e-4
MILLI_VALUE_THRESHOLD = 1e-3
CENT_VALUE_THRESHOLD = 1e-2
DIME_VALUE_THRESHOLD = 0.1
FULL_PRECISION_THRESHOLD = 10.0

CLEAN_NUMBER_CHARS = ("$", "€", "£", "%", ",")


def timestamps_from_ms_array(timestamps_ms: np.ndarray) -> list[datetime]:

    return pd.to_datetime(timestamps_ms, unit="ms", utc=True).to_pydatetime().tolist()  # type: ignore[reportAttributeAccessIssue]


class FormatUtils:
    """Utility class for formatting technical analysis data and values.

    This class acts as a centralized formatter to ensure consistent numeric representation
    across the dashboard, logs, and AI prompts. It is designed to be used across the
    application without creating circular import dependencies.
    """

    def __init__(self, default_precision: int = 8) -> None:

        self.default_precision = default_precision

    def parse_value(self, value: Any, default: Any = None) -> float:

        if isinstance(value, (int, float)):
            return float(value)

        if not isinstance(value, str):
            return default

        clean = value.strip()

        for char in CLEAN_NUMBER_CHARS:
            clean = clean.replace(char, "")

        try:
            return float(clean)
        except ValueError:
            return default

    def fmt(self, val: float | None, precision: int | None = None) -> str:

        # pylint: disable=too-many-return-statements
        if val is None:
            return "N/A"

        effective_precision = precision if precision is not None else self.default_precision

        if not math.isnan(val):
            abs_val = abs(val)

            if 0 < abs_val < SCIENTIFIC_NOTATION_THRESHOLD:
                return f"{val:.{effective_precision}e}"
            if abs_val < CRYPTO_DUST_THRESHOLD:
                return f"{val:.8f}"
            if abs_val < MICRO_VALUE_THRESHOLD:
                return f"{val:.7f}"
            if abs_val < MILLI_VALUE_THRESHOLD:
                return f"{val:.6f}"
            if abs_val < CENT_VALUE_THRESHOLD:
                return f"{val:.5f}"
            if abs_val < DIME_VALUE_THRESHOLD:
                return f"{val:.4f}"
            if abs_val < FULL_PRECISION_THRESHOLD:
                return f"{val:.{effective_precision}f}"

            return f"{val:.2f}"
        return "N/A"

    def fmt_ta(self, td: dict, key: str, precision: int | None = None, default: str = "N/A") -> str:

        val = get_indicator_value(td, key)
        if isinstance(val, (int, float)) and not math.isnan(val):
            abs_val = abs(val)
            eff_prec = precision if precision is not None else self.default_precision
            if 0 < abs_val < SCIENTIFIC_NOTATION_THRESHOLD:
                return f"{val:.{eff_prec}e}"
            if abs_val < CRYPTO_DUST_THRESHOLD:
                return f"{val:.8f}"
            if abs_val < MICRO_VALUE_THRESHOLD:
                return f"{val:.7f}"
            if abs_val < MILLI_VALUE_THRESHOLD:
                return f"{val:.6f}"
            if abs_val < CENT_VALUE_THRESHOLD:
                return f"{val:.5f}"
            if abs_val < DIME_VALUE_THRESHOLD:
                return f"{val:.4f}"
            if abs_val < FULL_PRECISION_THRESHOLD:
                return f"{val:.{eff_prec}f}"
            return f"{val:.2f}"
        return default

    def format_current_time(self, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:

        return datetime.now(timezone.utc).strftime(format_str)

    def format_timestamp_seconds(self, timestamp_sec: float, format_str: str = "%Y-%m-%d") -> str:

        try:
            dt = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)
            return dt.strftime(format_str)
        except (ValueError, TypeError, OSError):
            return "N/A"

    def format_date_from_timestamp(self, timestamp_sec: float) -> str:

        return self.format_timestamp_seconds(timestamp_sec, "%Y-%m-%d")

    def timestamp_from_iso(self, iso_str: str) -> float:

        try:
            if iso_str.endswith("Z"):
                iso_str = iso_str[:-1] + "+00:00"
            return datetime.fromisoformat(iso_str).timestamp()
        except (ValueError, TypeError, AttributeError):
            return 0.0

    def parse_timestamp(self, timestamp_field) -> float:

        if timestamp_field is None:
            return 0.0
        if isinstance(timestamp_field, (int, float)):
            return float(timestamp_field)
        if isinstance(timestamp_field, str):
            if timestamp_field.isdigit():
                return float(timestamp_field)
            return self.timestamp_from_iso(timestamp_field)
        return 0.0

    def parse_timestamp_ms(self, timestamp_ms: float) -> datetime | None:

        try:
            return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        except (ValueError, TypeError, OSError):
            return None

    def get_supertrend_direction_string(self, direction) -> str:

        if direction > 0:
            return "Bullish"
        if direction < 0:
            return "Bearish"
        return "Neutral"

    def format_bollinger_interpretation(self, td: dict) -> str:

        bb_position = get_indicator_value(td, "bb_position")
        if isinstance(bb_position, (int, float)):
            if bb_position > 0.8:
                return " [Near upper band - possible overbought]"
            if bb_position < 0.2:
                return " [Near lower band - possible oversold]"
            return " [Within normal range]"
        return ""

    def format_cmf_interpretation(self, td: dict) -> str:

        cmf_val = get_indicator_value(td, "cmf")
        if isinstance(cmf_val, (int, float)):
            if cmf_val > 0.1:
                return " [Accumulation phase]"
            if cmf_val < -0.1:
                return " [Distribution phase]"
            return " [Neutral]"
        return ""

