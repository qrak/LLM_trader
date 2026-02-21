"""
Market Data Management Module for RAG Engine

Handles fetching and processing of cryptocurrency market overview data.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
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
        self.cache = cache
        self.overview_builder = overview_builder

        self.coingecko_api = coingecko_api
        self.market_api = market_api
        self.exchange_manager = exchange_manager

        # Market data storage
        self.current_market_overview: Optional[Dict[str, Any]] = None
        self.coingecko_last_update: Optional[datetime] = None


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
            self.logger.error(f"Error fetching market overview: {e}")
            return None

    def _extract_top_coins(self, coingecko_data: Optional[Dict]) -> List[str]:
        """Extract top coins by dominance, excluding stablecoins - delegates to processor."""
        return self.processor.extract_top_coins(coingecko_data)

    def _build_overview_structure(self, overview: Dict, price_data: Optional[Dict], coingecko_data: Optional[Dict]):
        """Build the overview data structure from fetched data."""
        # Process price data
        if price_data and "RAW" in price_data:
            overview["top_coins"] = {}
            for coin, values in price_data["RAW"].items():
                coin_overview = self._process_coin_data(values)
                if coin_overview:
                    overview["top_coins"][coin] = coin_overview

        # Add CoinGecko global market data
        if coingecko_data:
            overview.update(coingecko_data)
            self.coingecko_last_update = datetime.now(timezone.utc)

    def _process_coin_data(self, values: Dict) -> Optional[Dict]:
        """Process individual coin data from price API response."""
        quote_data = None

        # Try to get USD data first, then USDT if USD not available
        if "USD" in values:
            quote_data = values["USD"]
        elif "USDT" in values:
            quote_data = values["USDT"]

        if not quote_data:
            return None

        coin_overview = {
            "price": quote_data.get("PRICE", 0),
            "change24h": quote_data.get("CHANGEPCT24HOUR", 0),
            "volume24h": quote_data.get("VOLUME24HOUR", 0),
            "mcap": quote_data.get("MKTCAP")
        }

        # Add optional data if available
        if "VWAP" in quote_data and quote_data["VWAP"]:
            coin_overview["vwap"] = quote_data["VWAP"]

        if "BID" in quote_data and "ASK" in quote_data:
            coin_overview["bid"] = quote_data["BID"]
            coin_overview["ask"] = quote_data["ASK"]

        return coin_overview

    def _finalize_overview(self, overview: Dict) -> Optional[Dict]:
        """Finalize and validate the market overview."""
        if overview.get("top_coins") or overview.get("market_cap"):
            overview["id"] = "market_overview"
            overview["title"] = "Crypto Market Overview"

            self.logger.debug("Market overview data fetched/processed.")
            return overview
        else:
            self.logger.error("Failed to fetch any market overview data.")
            return None

    async def update_market_overview_if_needed(self, max_age_hours: int = 24) -> bool:
        """Update market overview if needed based on age."""
        should_update = False

        if self.current_market_overview is None:
            should_update = True
        else:
            # Check if market overview is older than max_age_hours
            timestamp_field = self.current_market_overview.get('published_on',
                                                             self.current_market_overview.get('timestamp', 0))
            timestamp = self.unified_parser.format_utils.parse_timestamp(timestamp_field)

            if timestamp:
                data_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                current_time = datetime.now(timezone.utc)

                if current_time - data_time > timedelta(hours=max_age_hours):
                    self.logger.debug(f"Market overview data is older than {max_age_hours} hours, refreshing")
                    should_update = True

        if should_update:
            try:
                self.logger.debug("Fetching market overview data")
                market_overview = await self.fetch_market_overview()
                if market_overview:
                    self.current_market_overview = market_overview
                    self.logger.debug("Market overview updated successfully.")
                    return True
                else:
                    self.logger.warning("No market overview data was available from data sources")
                    return False
            except Exception as e:
                self.logger.error(f"Error fetching market overview: {e}")
                return False

        return False

    def get_current_overview(self) -> Optional[Dict[str, Any]]:
        """Get the current market overview data."""
        return self.current_market_overview

    def is_overview_stale(self, max_age_hours: int = 1) -> bool:
        """Check if the current market overview is stale."""
        if self.current_market_overview is None:
            return True

        timestamp_field = self.current_market_overview.get('published_on',
                                                           self.current_market_overview.get('timestamp', 0))
        timestamp = self.unified_parser.format_utils.parse_timestamp(timestamp_field)

        if timestamp:
            data_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            current_time = datetime.now(timezone.utc)
            return current_time - data_time > timedelta(hours=max_age_hours)

        return True
