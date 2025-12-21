"""
Market Data Fetcher
Handles fetching market data from various sources like CoinGecko and exchanges.
"""
from typing import Dict, Optional, List

from src.logger.logger import Logger


class MarketDataFetcher:
    """Handles fetching market data from external APIs."""
    
    def __init__(self, logger: Logger, coingecko_api=None, symbol_manager=None):
        self.logger = logger
        self.coingecko_api = coingecko_api
        self.symbol_manager = symbol_manager
    
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
        """Fetch price data for top coins using available methods."""
        try:
            # Try CoinGecko first
            if self.coingecko_api:
                coingecko_prices = await self._try_coingecko_price_data(top_coins)
                if coingecko_prices:
                    return coingecko_prices
            
            # Fallback to CCXT if available
            return await self._try_ccxt_price_data(top_coins)
        except Exception as e:
            self.logger.error(f"Error fetching price data: {e}")
            return None
    
    async def _try_coingecko_price_data(self, top_coins: List[str]) -> Optional[Dict]:
        """Try to fetch price data from CoinGecko."""
        try:
            if not self.coingecko_api:
                return None
            
            # Convert symbols to CoinGecko coin IDs if symbol manager available
            if self.symbol_manager:
                coin_ids = []
                for symbol in top_coins:
                    coin_id = self.symbol_manager.get_coin_id_from_symbol(symbol)
                    if coin_id:
                        coin_ids.append(coin_id)
                
                if coin_ids:
                    return await self.coingecko_api.get_simple_price(coin_ids, vs_currencies='usd')
            
            return None
        except Exception as e:
            self.logger.error(f"Error fetching CoinGecko price data: {e}")
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
