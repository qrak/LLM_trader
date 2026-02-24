"""
Market Data Fetcher
Handles fetching market data from various sources like CoinGecko and exchanges.
"""
from typing import Dict, Optional, List, TYPE_CHECKING

from src.logger.logger import Logger
from src.platforms.defillama import DefiLlamaClient, MacroMarketData

if TYPE_CHECKING:
    from src.platforms.defillama import DeFiFundamentalsData

class MarketDataFetcher:
    """Handles fetching market data from external APIs."""

    def __init__(self, logger: Logger, coingecko_api=None, exchange_manager=None, market_api=None,
                 defillama_client: Optional[DefiLlamaClient] = None):
        # pylint: disable=too-many-arguments
        """Initialize MarketDataFetcher."""
        self.logger = logger
        self.coingecko_api = coingecko_api
        self.exchange_manager = exchange_manager
        self.market_api = market_api
        self.defillama_client = defillama_client

    async def fetch_global_market_data(self) -> Optional[Dict]:
        """Fetch global market data from CoinGecko."""
        if not self.coingecko_api:
            self.logger.error("CoinGecko API client not initialized")
            return None

        try:
            return await self.coingecko_api.get_global_market_data()
        except Exception as e:
            self.logger.error("Error fetching global market data: %s", e)
            return None

    async def fetch_macro_data(self) -> Optional[MacroMarketData]:
        """Fetch macro market data (Stablecoins, TVL) from DefiLlama."""
        if not self.defillama_client:
            return None

        try:
            return await self.defillama_client.get_macro_overview()
        except Exception as e:
            self.logger.error("Error fetching macro data from DefiLlama: %s", e)
            return None

    async def fetch_defi_fundamentals(self) -> Optional['DeFiFundamentalsData']:
        """Fetch aggregated DeFi fundamentals from DefiLlama and cache results."""
        if not self.defillama_client:
            return None

        try:
            data = await self.defillama_client.get_defi_fundamentals()
            return data

        except Exception as e:
            self.logger.error("Error fetching DeFi fundamentals: %s", e)
            return None

    async def fetch_price_data(self, top_coins: List[str]) -> Optional[Dict]:
        """Fetch price data for top coins using CCXT or fallback to CryptoCompare."""
        price_data = None
        try:
            price_data = await self._try_ccxt_price_data(top_coins)
        except Exception as e:
            self.logger.error("Error fetching CCXT price data: %s", e)

        # Fallback to CryptoCompare
        if not price_data or not price_data.get("RAW"):
            if self.market_api:
                self.logger.debug("Falling back to CryptoCompare API for price data")
                try:
                    price_data = await self.market_api.get_multi_price_data(coins=top_coins)
                except Exception as e:
                    self.logger.error("Error fetching CryptoCompare price data: %s", e)


        return price_data


    async def _try_ccxt_price_data(self, top_coins: List[str]) -> Optional[Dict]:
        """Try to fetch price data using CCXT exchange."""
        if not (self.exchange_manager and self.exchange_manager.exchanges):
            return None

        # Select best available exchange
        exchange = self._select_exchange()
        if not exchange:
            return None

        try:
            # Import DataFetcher here to avoid circular dependencies if it's in analyzer
            from src.analyzer.data_fetcher import DataFetcher  # pylint: disable=import-outside-toplevel

            data_fetcher = DataFetcher(exchange=exchange, logger=self.logger)
            symbols = [f"{coin}/USDT" for coin in top_coins]
            self.logger.debug("Fetching data for top coins: %s", symbols)

            price_data = await data_fetcher.fetch_multiple_tickers(symbols)
            self.logger.debug("Fetched price data for %s symbols using CCXT", len(symbols))
            return price_data
        except Exception as e:
            self.logger.warning("Failed to fetch ticker data via CCXT: %s", e)
            return None

    def _select_exchange(self):
        """Select the best available exchange for market data."""
        # Prefer Binance if available
        if 'binance' in self.exchange_manager.exchanges:
            self.logger.debug("Using Binance exchange for market data")
            return self.exchange_manager.exchanges['binance']

        # Use first available exchange that supports fetch_tickers
        for exchange_id, exch in self.exchange_manager.exchanges.items():
            if exch.has.get('fetchTickers', False):
                self.logger.debug("Using %s exchange for market data", exchange_id)
                return exch

        return None
