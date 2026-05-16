"""
Market Data Cache Manager
Handles caching and storage of market overview data.
"""
from datetime import datetime, timedelta, timezone
from typing import Any

from src.logger.logger import Logger
from ..file_handler import RagFileHandler


class MarketDataCache:
    """Handles caching and storage operations for market data."""

    def __init__(self, logger: Logger, file_handler: RagFileHandler):
        self.logger = logger
        self.file_handler = file_handler
        self.current_market_overview: dict[str, Any] | None = None



    def get_current_overview(self) -> dict[str, Any] | None:
        """Get the current market overview data."""
        return self.current_market_overview

    def is_overview_stale(self, max_age_hours: int = 1, normalize_timestamp_func=None) -> bool:
        """Check if the current market overview is stale."""
        if self.current_market_overview is None:
            return True

        timestamp_field = self.current_market_overview.get('published_on',
                                                           self.current_market_overview.get('timestamp', 0))

        if normalize_timestamp_func:
            timestamp = normalize_timestamp_func(timestamp_field)
        else:
            # Fallback simple normalization
            timestamp = float(timestamp_field)

        if timestamp:
            data_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            current_time = datetime.now(timezone.utc)
            return current_time - data_time > timedelta(hours=max_age_hours)

        return True
