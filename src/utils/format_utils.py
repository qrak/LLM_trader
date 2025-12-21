"""
Consolidated formatting utilities.
Contains all shared formatting functions and utilities.

This module has NO dependencies on analyzer module to avoid circular imports.
"""
import numpy as np
from datetime import datetime
from typing import Optional

from src.analyzer.data.data_processor import DataProcessor


class FormatUtils:
    """Utility class for formatting technical analysis data and values.
    
    This class is designed to be used across the application without creating
    circular import dependencies. It has no imports from the analyzer module.
    """
    
    def __init__(self, data_processor: DataProcessor):
        """Initialize the formatting utilities with a data processor instance."""
        self.data_processor = data_processor
    
    def fmt(self, val, precision=8):
        """Format a value with appropriate precision based on its magnitude"""
        if isinstance(val, (int, float)) and not np.isnan(val):
            if 0 < abs(val) < 0.0000001:  # Only use scientific notation for extremely small values
                return f"{val:.{precision}e}"  # Scientific notation for very small values
            elif abs(val) < 0.00001:  # SHIB and similar small crypto coins (0.000001 - 0.00001)
                return f"{val:.8f}"  # 8 decimal places for small crypto values
            elif abs(val) < 0.0001:
                return f"{val:.7f}"  # 7 decimal places 
            elif abs(val) < 0.001:
                return f"{val:.6f}"  # 6 decimal places
            elif abs(val) < 0.01:
                return f"{val:.5f}"  # 5 decimal places
            elif abs(val) < 0.1:
                return f"{val:.4f}"  # 4 decimal places
            elif abs(val) < 10:
                return f"{val:.{precision}f}"  # Respect original precision for indicators
            else:
                return f"{val:.2f}"  # 2 decimal places for larger values
        return "N/A"

    def fmt_ta(self, technical_calculator, td: dict, key: str, precision: int = 8, default: str = 'N/A') -> str:
        """Format technical-analysis indicator values.
        
        Centralizes the logic used across all formatter classes.
        
        Args:
            technical_calculator: TechnicalCalculator instance (for compatibility, not used)
            td: Technical data dictionary
            key: Indicator key to retrieve
            precision: Number of decimal places
            default: Default value if indicator not found
            
        Returns:
            Formatted indicator value string
        """
        try:
            val = self.data_processor.get_indicator_value(td, key)
        except Exception:
            return default

        if isinstance(val, (int, float)) and not np.isnan(val):
            return self.fmt(val, precision)
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
            # Handle ISO format with Z suffix
            if iso_str.endswith('Z'):
                iso_str = iso_str[:-1] + '+00:00'
            return datetime.fromisoformat(iso_str).timestamp()
        except (ValueError, TypeError, AttributeError):
            return 0.0
    
    def parse_timestamp_ms(self, timestamp_ms: float) -> datetime:
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

    def is_valid_value(self, value) -> bool:
        """Check if a value is valid for formatting.
        
        Args:
            value: Value to check
            
        Returns:
            bool: True if value is valid number, False otherwise
        """
        return isinstance(value, (int, float)) and not np.isnan(value)

    def format_value(self, value, precision: int = 8) -> str:
        """Format a value with specified precision.
        
        Args:
            value: Value to format
            precision: Number of decimal places
            
        Returns:
            str: Formatted value or 'N/A' if invalid
        """
        if self.is_valid_value(value):
            return self.fmt(value, precision)
        return 'N/A'

    def get_supertrend_direction_string(self, direction) -> str:
        """Get supertrend direction as string."""
        if direction > 0:
            return 'Bullish'
        elif direction < 0:
            return 'Bearish'
        else:
            return 'Neutral'

    def format_bollinger_interpretation(self, technical_calculator, td: dict) -> str:
        """Format Bollinger Bands interpretation."""
        try:
            bb_position = self.data_processor.get_indicator_value(td, 'bb_position')
            if bb_position is not None:
                if bb_position > 0.8:
                    return " [Near upper band - possible overbought]"
                elif bb_position < 0.2:
                    return " [Near lower band - possible oversold]"
                else:
                    return " [Within normal range]"
        except Exception:
            pass
        return ""

    def format_cmf_interpretation(self, technical_calculator, td: dict) -> str:
        """Format Chaikin Money Flow interpretation."""
        try:
            cmf_val = self.data_processor.get_indicator_value(td, 'cmf')
            if cmf_val is not None:
                if cmf_val > 0.1:
                    return " [Accumulation phase]"
                elif cmf_val < -0.1:
                    return " [Distribution phase]"
                else:
                    return " [Neutral]"
        except Exception:
            pass
        return ""

    def _get_timeframe_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 24 * 60
        else:
            return 60  # Default to 1 hour

