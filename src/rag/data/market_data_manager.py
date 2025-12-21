"""
Market Data Management Module for RAG Engine

Handles fetching and processing of cryptocurrency market overview data.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from src.logger.logger import Logger
from src.parsing.unified_parser import UnifiedParser
from .file_handler import RagFileHandler
from .market_components import (
    MarketDataFetcher,
    MarketDataProcessor, 
    MarketDataCache,
    MarketOverviewBuilder
)


class MarketDataManager:
    """Manages cryptocurrency market overview data and operations."""
    
    def __init__(self, logger: Logger, file_handler: RagFileHandler, 
                 coingecko_api=None, cryptocompare_api=None, symbol_manager=None):
        self.logger = logger
        self.file_handler = file_handler
        self.unified_parser = UnifiedParser(logger)
        
        # Initialize specialized components
        self.fetcher = MarketDataFetcher(logger, coingecko_api, symbol_manager)
        self.processor = MarketDataProcessor(logger)
        self.cache = MarketDataCache(logger, file_handler)
        self.overview_builder = MarketOverviewBuilder(logger, self.processor)
        
        self.coingecko_api = coingecko_api
        self.cryptocompare_api = cryptocompare_api
        self.symbol_manager = symbol_manager
        
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
            
            # Use overview builder to create final structure
            overview = self.overview_builder.build_overview(coingecko_data, price_data, top_coins)
            
            return overview
                
        except Exception as e:
            self.logger.error(f"Error fetching market overview: {e}")
            return None
    
    def _extract_top_coins(self, coingecko_data: Optional[Dict]) -> List[str]:
        """Extract top coins by dominance, excluding stablecoins - delegates to processor."""
        return self.processor.extract_top_coins(coingecko_data)
    
    async def _fetch_price_data(self, top_coins: List[str]) -> Optional[Dict]:
        """Fetch price data using CCXT or fallback to CryptoCompare."""
        # Try CCXT first if available
        price_data = await self._try_ccxt_price_data(top_coins)
        
        # Fallback to CryptoCompare
        if not price_data or not price_data.get("RAW"):
            if self.cryptocompare_api:
                self.logger.debug("Falling back to CryptoCompare API for price data")
                price_data = await self.cryptocompare_api.get_multi_price_data(coins=top_coins)
        
        return price_data
    
    async def _try_ccxt_price_data(self, top_coins: List[str]) -> Optional[Dict]:
        """Try to fetch price data using CCXT exchange."""
        if not (self.symbol_manager and self.symbol_manager.exchanges):
            return None
        
        from src.analyzer.data.data_fetcher import DataFetcher
        
        # Select best available exchange
        exchange = self._select_exchange()
        if not exchange:
            return None
        
        try:
            data_fetcher = DataFetcher(exchange=exchange, logger=self.logger)
            symbols = [f"{coin}/USDT" for coin in top_coins]
            self.logger.debug(f"Fetching data for top coins: {symbols}")
            
            price_data = await data_fetcher.fetch_multiple_tickers(symbols)
            self.logger.debug(f"Fetched price data for {len(symbols)} symbols using CCXT")
            return price_data
        except Exception as e:
            self.logger.warning(f"Failed to fetch ticker data via CCXT: {e}")
            return None
    
    def _select_exchange(self):
        """Select the best available exchange for market data."""
        # Prefer Binance if available
        if 'binance' in self.symbol_manager.exchanges:
            self.logger.debug("Using Binance exchange for market data")
            return self.symbol_manager.exchanges['binance']
        
        # Use first available exchange that supports fetch_tickers
        for exchange_id, exch in self.symbol_manager.exchanges.items():
            if exch.has.get('fetchTickers', False):
                self.logger.debug(f"Using {exchange_id} exchange for market data")
                return exch
        
        return None
    
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
            self.coingecko_last_update = datetime.now()
    
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
            timestamp = self.unified_parser.parse_timestamp(timestamp_field)
            
            if timestamp:
                data_time = datetime.fromtimestamp(timestamp)
                current_time = datetime.now()
                
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
        timestamp = self.unified_parser.parse_timestamp(timestamp_field)

        if timestamp:
            data_time = datetime.fromtimestamp(timestamp)
            current_time = datetime.now()
            return current_time - data_time > timedelta(hours=max_age_hours)
        
        return True
