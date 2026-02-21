"""
Tests for FormatUtils in src/utils/format_utils.py.
"""
import pytest
import numpy as np
from datetime import datetime
from src.utils.format_utils import FormatUtils

class TestFormatUtils:
    """Tests for the FormatUtils class."""

    @pytest.fixture
    def formatter(self):
        """Fixture providing a FormatUtils instance."""
        return FormatUtils()

    def test_parse_value_numeric(self, formatter):
        """Test parsing of already numeric values."""
        assert formatter.parse_value(10) == 10.0
        assert formatter.parse_value(10.5) == 10.5
        assert formatter.parse_value(-5) == -5.0
        assert formatter.parse_value(0) == 0.0

    def test_parse_value_currency(self, formatter):
        """Test parsing of currency strings."""
        assert formatter.parse_value("$100") == 100.0
        assert formatter.parse_value("€50.50") == 50.5
        assert formatter.parse_value("£1,000") == 1000.0
        assert formatter.parse_value("$-100") == -100.0
        assert formatter.parse_value("-$100") == -100.0
        assert formatter.parse_value("-$1,234.56") == -1234.56

    def test_parse_value_percentage(self, formatter):
        """Test parsing of percentage strings."""
        assert formatter.parse_value("50%") == 50.0
        assert formatter.parse_value("-12.5%") == -12.5
        assert formatter.parse_value("100.00%") == 100.0

    def test_parse_value_commas(self, formatter):
        """Test parsing of strings with commas."""
        assert formatter.parse_value("1,000") == 1000.0
        assert formatter.parse_value("1,234,567.89") == 1234567.89

    def test_parse_value_edge_cases(self, formatter):
        """Test parsing of edge cases and invalid inputs."""
        assert formatter.parse_value(None) is None
        assert formatter.parse_value(None, default=0.0) == 0.0
        assert formatter.parse_value("") is None
        assert formatter.parse_value("   ") is None
        # Clean string becomes empty -> float("") raises ValueError -> returns default
        assert formatter.parse_value("$") is None
        assert formatter.parse_value("abc") is None
        assert formatter.parse_value("1.2.3") is None
        # Non-string inputs
        assert formatter.parse_value([], default=0.0) == 0.0
        assert formatter.parse_value({}, default=0.0) == 0.0

    def test_fmt_precision(self, formatter):
        """Test formatting with adaptive precision."""
        # Extremely small values (< 1e-7) -> scientific notation
        assert "e" in formatter.fmt(1e-8)

        # Small values
        assert formatter.fmt(0.000005) == "0.00000500" # < 1e-5 -> 8 decimals
        assert formatter.fmt(0.00005) == "0.0000500"  # < 1e-4 -> 7 decimals
        assert formatter.fmt(0.0005) == "0.000500"   # < 1e-3 -> 6 decimals
        assert formatter.fmt(0.005) == "0.00500"    # < 1e-2 -> 5 decimals
        assert formatter.fmt(0.05) == "0.0500"     # < 1e-1 -> 4 decimals

        # Medium values (< 10) -> respect precision (default 8)
        assert formatter.fmt(1.23456789) == "1.23456789"

        # Large values (>= 10) -> 2 decimals
        assert formatter.fmt(10.12345) == "10.12"
        assert formatter.fmt(100.12345) == "100.12"

    def test_fmt_edge_cases(self, formatter):
        """Test formatting of edge cases."""
        assert formatter.fmt(np.nan) == "N/A"
        assert formatter.fmt(float('nan')) == "N/A"
        assert formatter.fmt(None) == "N/A"

        # Infinity test - implementation: abs(inf) < 10 is False -> returns "{:.2f}" -> "inf"
        # This confirms current behavior, even if debatable.
        assert formatter.fmt(float('inf')) == "inf"
        assert formatter.fmt(float('-inf')) == "-inf"

    def test_parse_timestamp(self, formatter):
        """Test parsing of timestamps."""
        ts = 1609459200.0  # 2021-01-01 00:00:00 UTC

        # Int/Float input
        assert formatter.parse_timestamp(ts) == ts
        assert formatter.parse_timestamp(int(ts)) == ts

        # String numeric input
        assert formatter.parse_timestamp(str(int(ts))) == ts

        # ISO String input
        # Note: timestamp_from_iso uses datetime.fromisoformat which handles 'Z' if Python 3.11+
        # or standard formats.
        iso_str = "2021-01-01T00:00:00+00:00"
        # datetime.fromisoformat("...").timestamp() returns local time if no timezone info?
        # Wait, if string has timezone, it returns aware datetime.
        # If not, naive.
        # Let's check implementation of timestamp_from_iso:
        # if iso_str.endswith('Z'): iso_str = iso_str[:-1] + '+00:00'
        # return datetime.fromisoformat(iso_str).timestamp()

        # The test environment is UTC? Or local?
        # datetime.timestamp() returns POSIX timestamp (UTC).
        # So it should be consistent.

        assert formatter.parse_timestamp("2021-01-01T00:00:00+00:00") == ts
        assert formatter.parse_timestamp("2021-01-01T00:00:00Z") == ts

        # Invalid inputs
        assert formatter.parse_timestamp(None) == 0.0
        assert formatter.parse_timestamp("invalid") == 0.0
        assert formatter.parse_timestamp([]) == 0.0

    def test_timestamps_from_ms_array(self):
        """Test the standalone function timestamps_from_ms_array."""
        # This function is outside FormatUtils class but in the same module
        from src.utils.format_utils import timestamps_from_ms_array

        ts_ms = np.array([1609459200000, 1609459201000], dtype=np.int64)
        dts = timestamps_from_ms_array(ts_ms)

        assert len(dts) == 2
        assert isinstance(dts[0], datetime)
        assert dts[0].timestamp() == 1609459200.0
        assert dts[1].timestamp() == 1609459201.0
