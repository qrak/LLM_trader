"""Market data management for fetching and serving market overview."""

from typing import Dict, Any, Optional
from src.logger.logger import Logger
from .file_handler import RagFileHandler
from .market_components import (
    MarketDataFetcher,
    MarketDataProcessor,
    MarketDataCache,
    MarketOverviewBuilder
)


class MarketDataManager:
    """Manages cryptocurrency market overview data and operations."""

    def __init__(
        self,
        logger: Logger,
        file_handler: RagFileHandler,
        coingecko_api=None,
        market_api=None,
        exchange_manager=None,
        unified_parser=None,
        fetcher: Optional[MarketDataFetcher] = None,
        processor: Optional[MarketDataProcessor] = None,
        cache: Optional[MarketDataCache] = None,
        overview_builder: Optional[MarketOverviewBuilder] = None
    ):
        self.logger = logger
        self.file_handler = file_handler
        self.unified_parser = unified_parser

        # Initialize specialized components
        self.fetcher = fetcher
        self.processor = processor
        self.cache = cache or MarketDataCache(logger=logger, file_handler=file_handler)
        self.overview_builder = overview_builder

        self.coingecko_api = coingecko_api
        self.market_api = market_api
        self.exchange_manager = exchange_manager
    @property
    def current_market_overview(self) -> Optional[Dict[str, Any]]:
        """Backward-compatible alias for cache-backed overview state."""
        return self.cache.current_market_overview

    @current_market_overview.setter
    def current_market_overview(self, value: Optional[Dict[str, Any]]) -> None:
        self.cache.current_market_overview = value

    async def fetch_market_overview(self) -> Optional[Dict[str, Any]]:
        """Fetch overall market data from various sources concurrently."""
        try:
            # Use fetcher component to get global data
            coingecko_data = await self.fetcher.fetch_global_market_data()

            # Use processor to extract top coins
            top_coins = self.processor.extract_top_coins(coingecko_data)

            # Use fetcher to get price data
            price_data = await self.fetcher.fetch_price_data(top_coins)

            # Fetch macro data (DefiLlama) - keeping for backward compatibility
            macro_data = await self.fetcher.fetch_macro_data()

            # Fetch comprehensive DeFi fundamentals (aggregated)
            defi_fundamentals = await self.fetcher.fetch_defi_fundamentals()

            # Use overview builder to create final structure
            overview = self.overview_builder.build_overview(coingecko_data, price_data, top_coins)

            if macro_data:
                overview["macro"] = macro_data.model_dump()

            if defi_fundamentals:
                overview["fundamentals"] = defi_fundamentals.model_dump()

            return overview

        except Exception as e:
            self.logger.error("Error fetching market overview: %s", e)
            return None

    async def update_market_overview_if_needed(self, max_age_hours: int = 24) -> bool:
        """Update market overview if needed based on cache staleness policy."""
        normalize_timestamp = None
        if self.unified_parser:
            normalize_timestamp = self.unified_parser.format_utils.parse_timestamp

        should_update = self.cache.is_overview_stale(
            max_age_hours=max_age_hours,
            normalize_timestamp_func=normalize_timestamp,
        )
        if should_update and self.current_market_overview is not None:
            self.logger.debug("Market overview data is older than %s hours, refreshing", max_age_hours)

        if should_update:
            try:
                self.logger.debug("Fetching market overview data")
                market_overview = await self.fetch_market_overview()
                if market_overview:
                    self.cache.current_market_overview = market_overview
                    self.logger.debug("Market overview updated successfully.")
                    return True
                self.logger.warning("No market overview data was available from data sources")
                return False
            except Exception as e:
                self.logger.error("Error fetching market overview: %s", e)
                return False

        return False

    def get_current_overview(self) -> Optional[Dict[str, Any]]:
        """Get the current market overview data."""
        return self.cache.get_current_overview()

    def is_overview_stale(self, max_age_hours: int = 1) -> bool:
        """Check if the current market overview is stale."""
        normalize_timestamp = None
        if self.unified_parser:
            normalize_timestamp = self.unified_parser.format_utils.parse_timestamp
        return self.cache.is_overview_stale(max_age_hours=max_age_hours, normalize_timestamp_func=normalize_timestamp)
