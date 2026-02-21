"""
Consolidated formatting utilities.
Contains all shared formatting functions and utilities.

This module has NO dependencies on analyzer module to avoid circular imports.

Formatting Strategy:
    The module defines several constants to control the number of decimal places
    for numeric display based on the magnitude of the value. This ensures that:
    1. Very small values (e.g., satoshis, meme coins) are not displayed as 0.00
    2. Large values (e.g., BTC price) are not cluttered with unnecessary decimals
    3. Scientific notation is used only for microscopic values (< 1e-7)
"""
from datetime import datetime
from typing import List, Any, Optional

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

# Characters to remove when cleaning number strings
CLEAN_NUMBER_CHARS = ('$', '€', '£', '%', ',')


def timestamps_from_ms_array(timestamps_ms: np.ndarray) -> List[datetime]:
    """Convert numpy array of millisecond timestamps to list of datetime objects.

    Uses pandas for ~10x faster vectorized conversion compared to list comprehension.

    Args:
        timestamps_ms: Numpy array of timestamps in milliseconds

    Returns:
        List of datetime objects
    """
    return pd.to_datetime(timestamps_ms, unit='ms', utc=True).to_pydatetime().tolist()


class FormatUtils:
    """Utility class for formatting technical analysis data and values.

    This class acts as a centralized formatter to ensure consistent numeric representation
    across the dashboard, logs, and AI prompts. It is designed to be used across the
    application without creating circular import dependencies.
    """

    def __init__(self, default_precision: int = 8) -> None:
        """Initialize the formatting utilities.

        Args:
            default_precision: Default decimal places for numeric formatting (default: 8)
        """
        self.default_precision = default_precision

    def parse_value(self, value: Any, default: Any = None) -> float:
        """Parse various numeric formats into a clean float.

        Handles:
        - Currency symbols ($, €, £, etc.)
        - Percentages (%)
        - Commas (1,000.00)
        - Whitespace
        - Strings containing only numbers
        - Already numeric values

        Args:
            value: The input value to parse (str, int, float, or None)
            default: Value to return if parsing fails (defaults to None)

        Returns:
            Float representation of the value, or default if parsing fails
        """
        if isinstance(value, (int, float)):
            return float(value)

        if not isinstance(value, str):
            return default

        # Clean string
        clean = value.strip()

        # Remove common currency/percentage symbols and separators
        for char in CLEAN_NUMBER_CHARS:
            clean = clean.replace(char, '')

        try:
            return float(clean)
        except ValueError:
            return default

    def fmt(self, val: float | None, precision: int | None = None) -> str:
        """Format a value with appropriate precision based on its magnitude.

        Applies adaptive formatting rules defined by module-level constants.

        Args:
            val: Numeric value to format (None returns 'N/A')
            precision: Decimal places for full precision range, defaults to instance default_precision

        Returns:
            Formatted string representation of the value
        """
        # pylint: disable=too-many-return-statements
        if val is None:
            return "N/A"

        effective_precision = precision if precision is not None else self.default_precision

        if not np.isnan(val):
            abs_val = abs(val)

            if 0 < abs_val < SCIENTIFIC_NOTATION_THRESHOLD:
                return f"{val:.{effective_precision}e}"  # Scientific notation for very small values
            if abs_val < CRYPTO_DUST_THRESHOLD:
                return f"{val:.8f}"  # 8 decimal places for small crypto values
            if abs_val < MICRO_VALUE_THRESHOLD:
                return f"{val:.7f}"  # 7 decimal places
            if abs_val < MILLI_VALUE_THRESHOLD:
                return f"{val:.6f}"  # 6 decimal places
            if abs_val < CENT_VALUE_THRESHOLD:
                return f"{val:.5f}"  # 5 decimal places
            if abs_val < DIME_VALUE_THRESHOLD:
                return f"{val:.4f}"  # 4 decimal places
            if abs_val < FULL_PRECISION_THRESHOLD:
                return f"{val:.{effective_precision}f}"  # Respect original precision for indicators

            return f"{val:.2f}"  # 2 decimal places for larger values
        return "N/A"

    def fmt_ta(self, td: dict, key: str, precision: int | None = None, default: str = 'N/A') -> str:
        """Format technical-analysis indicator values.

        Handles Union[float, str] return from get_indicator_value() which uses
        'N/A' string sentinel for missing/invalid indicators.

        Args:
            td: Technical data dictionary
            key: Indicator key to retrieve
            precision: Number of decimal places, defaults to instance default_precision
            default: Default value if indicator not found

        Returns:
            Formatted numeric string or default value
        """
        effective_precision = precision if precision is not None else self.default_precision
        val = get_indicator_value(td, key)
        if isinstance(val, (int, float)) and not np.isnan(val):  # Polymorphic check - legitimate
            return self.fmt(val, effective_precision)
        return default

    def format_timestamp(self, timestamp_ms) -> str:
        """Format a timestamp from milliseconds since epoch to a human-readable string

        Args:
            timestamp_ms: Timestamp in milliseconds

        Returns:
            Human-readable datetime string
        """
        try:
            dt = datetime.fromtimestamp(timestamp_ms / 1000)
            return dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError, OSError):
            return "N/A"

    def format_current_time(self, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Format current time with specified format.

        Args:
            format_str: strftime format string

        Returns:
            Formatted current time string
        """
        return datetime.now().strftime(format_str)

    def format_timestamp_seconds(self, timestamp_sec: float, format_str: str = "%Y-%m-%d") -> str:
        """Format timestamp in seconds (not milliseconds) to human-readable string.

        Args:
            timestamp_sec: Timestamp in seconds since epoch
            format_str: strftime format string

        Returns:
            Formatted datetime string or 'N/A' if invalid
        """
        try:
            dt = datetime.fromtimestamp(timestamp_sec)
            return dt.strftime(format_str)
        except (ValueError, TypeError, OSError):
            return "N/A"

    def format_date_from_timestamp(self, timestamp_sec: float) -> str:
        """Format timestamp to date only (YYYY-MM-DD).

        Args:
            timestamp_sec: Timestamp in seconds since epoch

        Returns:
            Formatted date string or 'N/A' if invalid
        """
        return self.format_timestamp_seconds(timestamp_sec, "%Y-%m-%d")

    def timestamp_from_iso(self, iso_str: str) -> float:
        """Convert ISO format string to Unix timestamp in seconds.

        Args:
            iso_str: ISO format datetime string (supports 'Z' suffix)

        Returns:
            Unix timestamp in seconds, or 0.0 if conversion fails
        """
        try:
            if iso_str.endswith('Z'):
                iso_str = iso_str[:-1] + '+00:00'
            return datetime.fromisoformat(iso_str).timestamp()
        except (ValueError, TypeError, AttributeError):
            return 0.0

    def parse_timestamp(self, timestamp_field) -> float:
        """Universal timestamp parser supporting int/float/str formats.

        Args:
            timestamp_field: Timestamp as int, float, or ISO string

        Returns:
            Unix timestamp in seconds as float, or 0.0 if conversion fails
        """
        if timestamp_field is None:
            return 0.0
        if isinstance(timestamp_field, (int, float)):
            return float(timestamp_field)
        if isinstance(timestamp_field, str):
            if timestamp_field.isdigit():
                return float(timestamp_field)
            return self.timestamp_from_iso(timestamp_field)
        return 0.0

    def parse_timestamp_ms(self, timestamp_ms: float) -> Optional[datetime]:
        """Parse timestamp in milliseconds to datetime object.

        Args:
            timestamp_ms: Timestamp in milliseconds

        Returns:
            datetime object or None if invalid
        """
        try:
            return datetime.fromtimestamp(timestamp_ms / 1000)
        except (ValueError, TypeError, OSError):
            return None

    def get_supertrend_direction_string(self, direction) -> str:
        """Get supertrend direction as string."""
        if direction > 0:
            return 'Bullish'
        if direction < 0:
            return 'Bearish'
        return 'Neutral'

    def format_bollinger_interpretation(self, td: dict) -> str:
        """Format Bollinger Bands interpretation."""
        bb_position = get_indicator_value(td, 'bb_position')
        if isinstance(bb_position, (int, float)):
            if bb_position > 0.8:
                return " [Near upper band - possible overbought]"
            if bb_position < 0.2:
                return " [Near lower band - possible oversold]"
            return " [Within normal range]"
        return ""

    def format_cmf_interpretation(self, td: dict) -> str:
        """Format Chaikin Money Flow interpretation."""
        cmf_val = get_indicator_value(td, 'cmf')
        if isinstance(cmf_val, (int, float)):
            if cmf_val > 0.1:
                return " [Accumulation phase]"
            if cmf_val < -0.1:
                return " [Distribution phase]"
            return " [Neutral]"
        return ""
