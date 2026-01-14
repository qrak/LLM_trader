import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from src.logger.logger import Logger
from src.utils.profiler import profile_performance
from src.utils.token_counter import TokenCounter


if TYPE_CHECKING:
    from src.config.protocol import ConfigProtocol
    from src.platforms.coingecko import CoinGeckoAPI


class RagEngine:
    def __init__(
        self,
        logger: Logger,
        token_counter: TokenCounter,
        config: "ConfigProtocol",
        coingecko_api: Optional["CoinGeckoAPI"] = None,
        exchange_manager=None,
        file_handler=None,
        news_manager=None,
        market_data_manager=None,
        index_manager=None,
        category_fetcher=None,
        category_processor=None,
        ticker_manager=None,
        news_category_analyzer=None,
        context_builder=None,
    ):
        """Initialize RagEngine with injected dependencies (DI pattern).
        
        Args:
            logger: Logger instance
            token_counter: TokenCounter instance
            config: ConfigProtocol instance for RAG update intervals
            coingecko_api: CoinGecko API client (optional)
            exchange_manager: Exchange manager (optional)
            file_handler: RagFileHandler instance (injected from app.py)
            news_manager: NewsManager instance (injected from app.py)
            market_data_manager: MarketDataManager instance (injected from app.py)
            index_manager: IndexManager instance (injected from app.py)
            category_fetcher: CategoryFetcher instance (injected from app.py)
            category_processor: CategoryProcessor instance (injected from app.py)
            ticker_manager: TickerManager instance (injected from app.py)
            news_category_analyzer: NewsCategoryAnalyzer instance (injected from app.py)
            context_builder: ContextBuilder instance (injected from app.py)
        """
        if file_handler is None:
            raise ValueError("file_handler is required - must be injected from app.py")
        if news_manager is None:
            raise ValueError("news_manager is required - must be injected from app.py")
        if market_data_manager is None:
            raise ValueError("market_data_manager is required - must be injected from app.py")
        if index_manager is None:
            raise ValueError("index_manager is required - must be injected from app.py")
        if category_fetcher is None:
            raise ValueError("category_fetcher is required - must be injected from app.py")
        if category_processor is None:
            raise ValueError("category_processor is required - must be injected from app.py")
        if ticker_manager is None:
            raise ValueError("ticker_manager is required - must be injected from app.py")
        if news_category_analyzer is None:
            raise ValueError("news_category_analyzer is required - must be injected from app.py")
        if context_builder is None:
            raise ValueError("context_builder is required - must be injected from app.py")
        
        self.logger = logger
        self.config = config
        self.token_counter = token_counter
        
        # Store injected components
        self.file_handler = file_handler
        self.news_manager = news_manager
        self.market_data_manager = market_data_manager
        self.index_manager = index_manager
        self.category_fetcher = category_fetcher
        self.category_processor = category_processor
        self.ticker_manager = ticker_manager
        self.news_category_analyzer = news_category_analyzer
        self.context_builder = context_builder

        self.coingecko_api = coingecko_api
        self.exchange_manager = exchange_manager

        # Update timestamps
        self.last_update: Optional[datetime] = None

        # Update intervals from config
        self.update_interval = timedelta(hours=config.RAG_UPDATE_INTERVAL_HOURS)

        # Task management
        self._periodic_update_task = None
        
        # Async lock to prevent concurrent updates
        self._update_lock = asyncio.Lock()

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
                
            
            # Load known tickers
            await self.ticker_manager.load_known_tickers()

            # Ensure categories are up to date
            await self._ensure_categories_updated()

            # Load news database
            await self.news_manager.load_cached_news()

            if self.news_manager.get_database_size() > 0:
                self.last_update = datetime.now()
                self._build_indices()
                self.logger.debug(f"Loaded {self.news_manager.get_database_size()} recent news articles")

            await self.ticker_manager.update_known_tickers(self.news_manager.news_database)

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
            self.ticker_manager.get_known_tickers(),
            self.category_processor.category_word_map
        )

    async def update_if_needed(self, force_update: bool = False) -> bool:
        """Update market data if needed based on time intervals or forced update"""
        async with self._update_lock:
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
            if force_update or time_since_update > self.update_interval:
                reason = "forced update" if force_update else f"{time_since_update.total_seconds()/60:.1f} minutes since last update"
                self.logger.debug(f"Refreshing market knowledge: {reason}")
                try:
                    await self.refresh_market_data()
                    self.last_update = datetime.now()
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to update market knowledge: {e}")
                    return False

            try:
                categories_updated = await self._ensure_categories_updated()
                if categories_updated:
                    self._build_indices()
            except Exception as e:
                self.logger.error(f"Failed to update categories: {e}")

            return False

    async def refresh_market_data(self) -> None:
        """Refresh all market data from external sources"""
        await self._ensure_categories_updated()
        
        # Fetch news
        try:
            articles = await self.news_manager.fetch_fresh_news(
                self.ticker_manager.get_known_tickers()
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

    @profile_performance
    async def retrieve_context(self, query: str, symbol: str, k: Optional[int] = None, max_tokens: int = 8096) -> str:
        """Retrieve relevant context for a query with token limiting.

        If `k` is None, the configured RAG news limit (`[rag] news_limit`) will be used.
        Note: Market overview data is handled separately by PromptBuilder._build_market_overview_section()
        This method only returns news articles and market context to avoid redundancy.
        """
        if k is None:
            try:
                k = int(self.config.RAG_NEWS_LIMIT)
            except Exception:
                k = 3

        if self.news_manager.get_database_size() == 0:
            self.logger.warning("News database is empty")
            return ""

        try:
            rebuild_indices = await self._ensure_categories_updated()
            if rebuild_indices:
                self._build_indices()

            if not self.last_update or datetime.now() - self.last_update > timedelta(minutes=30):
                await self.update_if_needed()

            # Extract keywords from query for smart sentence selection
            import re
            keywords = set(re.findall(r'\b\w{3,15}\b', query.lower()))

            # Use context builder for keyword search
            scores = await self.context_builder.keyword_search(
                query, self.news_manager.news_database, symbol,
                self.index_manager.get_coin_indices(),
                self.category_processor.category_word_map,
                self.category_processor.important_categories
            )
            relevant_indices = [idx for idx, _ in scores[:k*2]]
            scores_dict = {idx: score for idx, score in scores}

            # Add coin-specific articles if needed
            if symbol and len(relevant_indices) < k:
                coin = self.category_processor.extract_base_coin(symbol)
                coin_indices = self.index_manager.search_by_coin(coin)
                for idx in coin_indices:
                    if idx not in relevant_indices:
                        relevant_indices.append(idx)
                        if len(relevant_indices) >= k*2:
                            break

            # Build context using context builder (pass keywords and scores for smart selection)
            context_text, total_tokens = self.context_builder.add_articles_to_context(
                relevant_indices, self.news_manager.news_database, max_tokens, k, keywords, scores_dict
            )

            articles_added = len([idx for idx in relevant_indices 
                                if idx < self.news_manager.get_database_size()])

            # self.logger.debug(f"Added {min(articles_added, k)} news articles to context (market overview handled separately)")
            # self.logger.debug(f"Context tokens: {total_tokens}/{max_tokens}")

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

    async def update_known_tickers(self) -> None:
        """Update known cryptocurrency ticker symbols"""
        try:
            await self.ticker_manager.update_known_tickers(self.news_manager.news_database)
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
        if self.coingecko_api:
            try:
                await self.coingecko_api.close()
            except Exception as e:
                self.logger.error(f"Error closing CoinGecko API client: {e}")
                    
        self.logger.info("RAG Engine resources released")
    
    async def _ensure_categories_updated(self, force_refresh: bool = False) -> bool:
        """Ensure categories are loaded and up to date.
        
        Returns:
            True if categories were updated, False otherwise
        """
        try:
            categories = await self.category_fetcher.fetch_cryptocompare_categories(force_refresh)
            if categories:
                self.category_processor.process_api_categories(categories)
                return True
            return False
        except Exception as e:
            self.logger.exception(f"Error ensuring categories updated: {e}")
            return False
