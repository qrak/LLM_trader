import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from src.platforms.coingecko import CoinGeckoAPI
from src.platforms.cryptocompare import CryptoCompareAPI
from src.logger.logger import Logger
from ..data.file_handler import RagFileHandler
from src.utils.token_counter import TokenCounter
from ..data.news_manager import NewsManager
from ..data.market_data_manager import MarketDataManager
from ..management.category_manager import CategoryManager
from ..search.index_manager import IndexManager
from .context_builder import ContextBuilder

if TYPE_CHECKING:
    from src.contracts.config import ConfigProtocol


class RagEngine:
    def __init__(
        self,
        logger: Logger,
        token_counter: TokenCounter,
        config: "ConfigProtocol",
        coingecko_api: Optional[CoinGeckoAPI] = None,
        cryptocompare_api: Optional[CryptoCompareAPI] = None,
        symbol_manager=None,
        format_utils=None
    ):
        """Initialize RagEngine with dependencies.
        
        Args:
            logger: Logger instance
            token_counter: TokenCounter instance
            config: ConfigProtocol instance for RAG update intervals
            coingecko_api: CoinGecko API client (optional)
            cryptocompare_api: CryptoCompare API client (optional)
            symbol_manager: Exchange manager (optional)
            format_utils: Format utilities (optional)
        """
        self.logger = logger
        self.config = config
        self.token_counter = token_counter
        self.file_handler = RagFileHandler(logger=self.logger, config=config)
        
        # Initialize component managers
        self.news_manager = NewsManager(logger, self.file_handler, cryptocompare_api, format_utils)
        self.market_data_manager = MarketDataManager(
            logger, self.file_handler, coingecko_api, cryptocompare_api, symbol_manager
        )
        self.index_manager = IndexManager(logger, format_utils)
        self.category_manager = CategoryManager(
            logger, self.file_handler, cryptocompare_api, symbol_manager
        )
        self.context_builder = ContextBuilder(logger, token_counter, format_utils)
        
        # API clients with dependency injection
        self.coingecko_api = coingecko_api
        self.cryptocompare_api = cryptocompare_api
        self.symbol_manager = symbol_manager

        # Update timestamps
        self.last_update: Optional[datetime] = None

        # Update intervals from config
        self.update_interval = timedelta(hours=config.RAG_UPDATE_INTERVAL_HOURS)

        # Task management
        self._periodic_update_task = None

        # Closure flag
        self._is_closed = False

    async def initialize(self) -> None:
        """Initialize RAG engine and load cached data"""
        try:
            # Initialize API clients if they weren't provided
            if self.coingecko_api is None:
                self.coingecko_api = CoinGeckoAPI(logger=self.logger)
                await self.coingecko_api.initialize()
                # Update market data manager with the API
                self.market_data_manager.coingecko_api = self.coingecko_api
                
            if self.cryptocompare_api is None:
                self.cryptocompare_api = CryptoCompareAPI(logger=self.logger)
                await self.cryptocompare_api.initialize()
                # Update managers with the API
                self.news_manager.cryptocompare_api = self.cryptocompare_api
                self.market_data_manager.cryptocompare_api = self.cryptocompare_api
                self.category_manager.cryptocompare_api = self.cryptocompare_api
                

            
            # Load known tickers
            await self.category_manager.load_known_tickers()

            # Ensure categories are up to date
            await self.category_manager.ensure_categories_updated()

            # Load news database
            await self.news_manager.load_cached_news()

            if self.news_manager.get_database_size() > 0:
                self.last_update = datetime.now()
                self._build_indices()
                self.logger.debug(f"Loaded {self.news_manager.get_database_size()} recent news articles")

            await self.update_known_tickers()

            if self.news_manager.get_database_size() < 10:
                await self.refresh_market_data()
                self.last_update = datetime.now()
        except Exception as e:
            self.logger.exception(f"Error initializing RAG engine: {e}")
            self.news_manager.clear_database()

    def _build_indices(self) -> None:
        """Build search indices from news database"""
        self.index_manager.build_indices(
            self.news_manager.news_database,
            self.category_manager.get_known_tickers(),
            self.category_manager.get_category_word_map()
        )

    async def update_if_needed(self) -> bool:
        """Update market data if needed based on time intervals"""
        if not self.last_update:
            self.logger.debug("No previous update, refreshing market knowledge base")
            try:
                await self.refresh_market_data()
                self.last_update = datetime.now()
                return True
            except Exception as e:
                self.logger.error(f"Failed to update market knowledge: {e}")
                return False

        time_since_update = datetime.now() - self.last_update
        if time_since_update > self.update_interval:
            self.logger.debug(f"Last update was {time_since_update.total_seconds()/60:.1f} minutes ago, refreshing market knowledge")
            try:
                await self.refresh_market_data()
                self.last_update = datetime.now()
                return True
            except Exception as e:
                self.logger.error(f"Failed to update market knowledge: {e}")
                return False

        try:
            categories_updated = await self.category_manager.ensure_categories_updated()
            if categories_updated:
                self._build_indices()
        except Exception as e:
            self.logger.error(f"Failed to update categories: {e}")

        return False

    async def refresh_market_data(self) -> None:
        """Refresh all market data from external sources"""
        await self.category_manager.ensure_categories_updated()

        self.logger.debug("Starting fetch of news data")
        
        # Always fetch news
        try:
            articles = await self.news_manager.fetch_fresh_news(
                self.category_manager.get_known_tickers()
            )
        except Exception as e:
            self.logger.error(f"Error fetching crypto news: {e}")
            articles = []
    
        # Update market overview if needed
        try:
            await self.market_data_manager.update_market_overview_if_needed(max_age_hours=24)
        except Exception as e:
            self.logger.error(f"Error updating market overview: {e}")
    
        # Process articles
        if articles:
            updated = self.news_manager.update_news_database(articles)
            if updated:
                self._build_indices()
                self.logger.debug("News database updated; rebuilt indices")
            else:
                self.logger.debug("No new articles to add or only duplicates found")

    async def retrieve_context(self, query: str, symbol: str, k: int = 3, max_tokens: int = 8096) -> str:
        """Retrieve relevant context for a query with token limiting
        
        Note: Market overview data is handled separately by PromptBuilder._build_market_overview_section()
        This method only returns news articles and market context to avoid redundancy.
        """
        if self.news_manager.get_database_size() == 0:
            self.logger.warning("News database is empty")
            return ""

        try:
            rebuild_indices = await self.category_manager.ensure_categories_updated()
            if rebuild_indices:
                self._build_indices()

            if not self.last_update or datetime.now() - self.last_update > timedelta(minutes=30):
                await self.update_if_needed()

            # Use context builder for keyword search
            scores = await self.context_builder.keyword_search(
                query, self.news_manager.news_database, symbol,
                self.index_manager.get_coin_indices(),
                self.category_manager.get_category_word_map(),
                self.category_manager.get_important_categories()
            )
            relevant_indices = [idx for idx, _ in scores[:k*2]]

            # Add coin-specific articles if needed
            if symbol and len(relevant_indices) < k:
                coin = self.category_manager.extract_base_coin(symbol)
                coin_indices = self.index_manager.search_by_coin(coin)
                for idx in coin_indices:
                    if idx not in relevant_indices:
                        relevant_indices.append(idx)
                        if len(relevant_indices) >= k*2:
                            break

            # Build context using context builder
            context_text, total_tokens = self.context_builder.add_articles_to_context(
                relevant_indices, self.news_manager.news_database, max_tokens, k
            )

            articles_added = len([idx for idx in relevant_indices 
                                if idx < self.news_manager.get_database_size()])

            self.logger.debug(f"Added {min(articles_added, k)} news articles to context (market overview handled separately)")
            self.logger.debug(f"Context tokens: {total_tokens}/{max_tokens}")

            return context_text
        except Exception as e:
            self.logger.error(f"Error retrieving context: {e}")
            return "Error retrieving market context."

    async def get_market_overview(self) -> Optional[Dict[str, Any]]:
        """Get current market overview data - now uses CoinGecko data directly"""
        try:
            # Try to get fresh CoinGecko global data directly
            if self.coingecko_api:
                coingecko_data = await self.coingecko_api.get_global_market_data()
                if coingecko_data:
                    # Format CoinGecko data as market overview
                    market_overview = {
                        "timestamp": self.coingecko_api.last_update.isoformat() if self.coingecko_api.last_update else "unknown",
                        "summary": "CRYPTO MARKET OVERVIEW",
                        "published_on": self.coingecko_api.last_update.timestamp() if self.coingecko_api.last_update else 0,
                        "data_sources": ["coingecko_global"],
                        "market_cap": coingecko_data.get("market_cap", {}),
                        "volume": coingecko_data.get("volume", {}),
                        "dominance": coingecko_data.get("dominance", {}),
                        "stats": coingecko_data.get("stats", {}),
                        "top_coins": coingecko_data.get("top_coins", []),
                        "defi": coingecko_data.get("defi", {})
                    }
                    return market_overview
            
            # Fallback to complex market data manager if CoinGecko fails
            current_overview = self.market_data_manager.get_current_overview()
            
            # Try to get market overview data directly from CoinGecko API
            if current_overview is None:
                self.logger.debug("No current market overview, fetching from CoinGecko")
                await self.market_data_manager.update_market_overview_if_needed(max_age_hours=1)
                current_overview = self.market_data_manager.get_current_overview()

            if current_overview is not None:
                if self.market_data_manager.is_overview_stale(max_age_hours=1):
                    self.logger.debug("Market overview data needs refresh")
                    await self.market_data_manager.update_market_overview_if_needed(max_age_hours=1)
                    current_overview = self.market_data_manager.get_current_overview()
            else:
                self.logger.debug("No market overview data, fetching fresh data")
                await self.market_data_manager.update_market_overview_if_needed(max_age_hours=0)
                current_overview = self.market_data_manager.get_current_overview()

            return current_overview
        except Exception as e:
            self.logger.error(f"Error getting market overview: {e}")
            return None

    def extract_base_coin(self, symbol: str) -> str:
        """Extract base coin from trading pair symbol"""
        return self.category_manager.extract_base_coin(symbol)

    async def get_coin_categories(self, symbol: str) -> List[str]:
        """Get categories associated with a coin symbol"""
        return self.category_manager.get_coin_categories(symbol, self.news_manager.news_database)

    async def update_known_tickers(self) -> None:
        """Update known cryptocurrency ticker symbols"""
        try:
            await self.category_manager.update_known_tickers(self.news_manager.news_database)
        except Exception as e:
            self.logger.error(f"Error updating known tickers: {e}")

    async def start_periodic_updates(self) -> None:
        """Start periodic data update task"""
        async def update_loop():
            while True:
                try:
                    await asyncio.sleep(self.update_interval.total_seconds())
                    await self.update_if_needed()
                except asyncio.CancelledError:
                    self.logger.debug("Periodic updates cancelled")
                    break
                except Exception as e:
                    self.logger.error(f"Error in periodic update: {e}")
                    await asyncio.sleep(60)

        if self._periodic_update_task is None:
            self._periodic_update_task = asyncio.create_task(update_loop())
            self.logger.debug(f"Started periodic news updates every {self.update_interval.total_seconds()/60:.1f} minutes")

    async def stop_periodic_updates(self) -> None:
        """Stop periodic data update task"""
        if self._periodic_update_task:
            self._periodic_update_task.cancel()
            try:
                await self._periodic_update_task
            except asyncio.CancelledError:
                pass
            self._periodic_update_task = None
            self.logger.debug("Stopped periodic news updates")

    async def close(self) -> None:
        """Close resources and mark as closed"""
        if self._is_closed:
            return
            
        self._is_closed = True
        
        # Cancel periodic update task if running
        if self._periodic_update_task and not self._periodic_update_task.done():
            self.logger.debug("Cancelling periodic update task")
            self._periodic_update_task.cancel()
            try:
                await self._periodic_update_task
            except asyncio.CancelledError:
                pass
            
        # Close API clients if they have close methods
        for client in [self.coingecko_api, self.cryptocompare_api]:
            if client and hasattr(client, 'close') and callable(client.close):
                try:
                    await client.close()
                except Exception as e:
                    self.logger.error(f"Error closing API client: {e}")
                    
        self.logger.info("RAG Engine resources released")

    def set_symbol_manager(self, symbol_manager) -> None:
        """Set the symbol manager reference"""
        self.symbol_manager = symbol_manager
        # Update managers with symbol manager
        self.market_data_manager.symbol_manager = symbol_manager
        self.category_manager.symbol_manager = symbol_manager
        self.logger.debug("SymbolManager set in RagEngine and component managers")