"""
Timeframe validation and conversion utilities.

This module provides comprehensive timeframe validation, conversion between formats,
and compatibility checking for various APIs and exchanges.

Supported timeframe range: 1h (minimum) to 1d (maximum)
"""

from typing import Dict, Optional, Tuple


class TimeframeValidator:
    """Validates and manages timeframe configurations"""
    
    # Supported timeframes: 1h (minimum) to 1w (maximum, for macro analysis only)
    SUPPORTED_TIMEFRAMES = ['1h', '2h', '4h', '6h', '8h', '12h', '1d', '1w']
    
    # Timeframe to minutes mapping
    TIMEFRAME_MINUTES = {
        '1h': 60,
        '2h': 120,
        '4h': 240,
        '6h': 360,
        '8h': 480,
        '12h': 720,
        '1d': 1440,
        '1w': 10080
    }
    
    # CryptoCompare API format mapping
    CRYPTOCOMPARE_FORMAT = {
        '1h': 'hour',
        '2h': 'hour',
        '4h': 'hour',
        '6h': 'hour',
        '8h': 'hour',
        '12h': 'hour',
        '1d': 'day'
    }
    
    # CCXT-compatible timeframes (common across major exchanges)
    CCXT_STANDARD_TIMEFRAMES = ['1h', '2h', '4h', '6h', '8h', '12h', '1d', '1w']
    
    @classmethod
    def validate(cls, timeframe: str) -> bool:
        """
        Check if timeframe is fully supported.
        
        Args:
            timeframe: Timeframe string (e.g., "1h", "4h", "1d")
            
        Returns:
            bool: True if timeframe is supported, False otherwise
        """
        return timeframe in cls.SUPPORTED_TIMEFRAMES
    
    @classmethod
    def to_minutes(cls, timeframe: str) -> int:
        """
        Convert timeframe to minutes.

        Args:
            timeframe: Timeframe string (e.g., "1h", "4h", "1d")

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
            period: Period string (e.g., "1h", "4h", "24h", "7d", "30d")
            
        Returns:
            int: Number of minutes in the period
            
        Raises:
            ValueError: If period format is invalid
        """
        import re
        match = re.match(r'^(\d+)([hd])$', period.lower())
        if not match:
            raise ValueError(f"Invalid period format: {period}")
        
        value = int(match.group(1))
        unit = match.group(2)
        
        if unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 1440
        else:
            raise ValueError(f"Invalid period unit: {unit}")
    
    @classmethod
    def calculate_period_candles(cls, base_timeframe: str, target_period: str) -> int:
        """
        Calculate how many candles are needed for a target period.
        
        Args:
            base_timeframe: The base timeframe (e.g., "1h", "4h")
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
    def to_cryptocompare_format(cls, timeframe: str) -> Tuple[str, int]:
        """
        Convert our timeframe to CryptoCompare API format.
        
        CryptoCompare uses endpoints like:
        - /data/v2/histohour for hourly data
        - /data/v2/histoday for daily data
        
        And uses an "aggregate" parameter for multipliers.
        
        Args:
            timeframe: Our timeframe format (e.g., "1h", "4h", "1d")
            
        Returns:
            tuple: (endpoint_type, multiplier)
                - endpoint_type: "hour", "day", etc.
                - multiplier: Number of base units (e.g., 4 for "4h")
            
        Example:
            >>> to_cryptocompare_format("4h")
            ("hour", 4)
            
        Raises:
            ValueError: If timeframe is not supported by CryptoCompare API
        """
        if timeframe not in cls.CRYPTOCOMPARE_FORMAT:
            raise ValueError(
                f"Timeframe {timeframe} not supported by CryptoCompare API. "
                f"Supported: {', '.join(cls.CRYPTOCOMPARE_FORMAT.keys())}"
            )
        
        endpoint = cls.CRYPTOCOMPARE_FORMAT[timeframe]
        
        # Extract multiplier from timeframe
        if 'h' in timeframe:
            multiplier = int(timeframe.replace('h', ''))
        elif 'd' in timeframe:
            multiplier = int(timeframe.replace('d', ''))
        else:
            multiplier = 1
        
        return endpoint, multiplier
    
    @classmethod
    def is_ccxt_compatible(cls, timeframe: str, exchange_name: Optional[str] = None) -> bool:
        """
        Check if timeframe is compatible with CCXT exchanges.
        
        Args:
            timeframe: Timeframe string (e.g., "1h", "4h")
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
            timeframe: The timeframe (e.g., "1h", "4h", "1d")
            target_days: Number of days to cover (default: 30)
            
        Returns:
            int: Number of candles needed
            
        Example:
            >>> get_candle_limit_for_days("4h", 30)
            180  # 6 candles per day * 30 days
        """
        timeframe_minutes = cls.to_minutes(timeframe)
        target_minutes = target_days * 24 * 60
        return target_minutes // timeframe_minutes
    
    @classmethod
    def validate_and_normalize(cls, timeframe: str) -> str:
        """
        Validate and normalize timeframe string.
        
        Handles case variations and returns normalized form.
        
        Args:
            timeframe: Timeframe string (e.g., "1H", "4h", "1D")
            
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
