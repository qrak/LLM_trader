"""
Market Data Fetcher
Handles fetching market data from various sources like CoinGecko and exchanges.
"""
from typing import Dict, Optional, List

from src.logger.logger import Logger


class MarketDataFetcher:
    """Handles fetching market data from external APIs."""
    
    def __init__(self, logger: Logger, coingecko_api=None, exchange_manager=None):
        self.logger = logger
        self.coingecko_api = coingecko_api
        self.exchange_manager = exchange_manager
    
    async def fetch_global_market_data(self) -> Optional[Dict]:
        """Fetch global market data from CoinGecko."""
        if not self.coingecko_api:
            self.logger.error("CoinGecko API client not initialized")
            return None
        
        try:
            return await self.coingecko_api.get_global_market_data()
        except Exception as e:
            self.logger.error(f"Error fetching global market data: {e}")
            return None
    
    async def fetch_price_data(self, top_coins: List[str]) -> Optional[Dict]:
        """Fetch price data for top coins using CCXT."""
        try:
            return await self._try_ccxt_price_data(top_coins)
        except Exception as e:
            self.logger.error(f"Error fetching price data: {e}")
            return None
    

    async def _try_ccxt_price_data(self, top_coins: List[str]) -> Optional[Dict]:
        """Try to fetch price data using CCXT exchange."""
        try:
            # This would need access to exchange manager
            # For now, return None as we need proper integration
            return None
        except Exception as e:
            self.logger.error(f"Error fetching CCXT price data: {e}")
            return None
