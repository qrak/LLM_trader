import asyncio
from datetime import datetime, timedelta, timezone
import re
from typing import Dict, Any, Optional, TYPE_CHECKING

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
        file_handler=None,
        news_manager=None,
        market_data_manager=None,
        index_manager=None,
        category_fetcher=None,
        category_processor=None,
        ticker_manager=None,
        context_builder=None,
    ):
        """Initialize RagEngine with injected dependencies (DI pattern).

        Args:
            logger: Logger instance
            token_counter: TokenCounter instance
            config: ConfigProtocol instance for RAG update intervals
            coingecko_api: CoinGecko API client (optional)
            file_handler: RagFileHandler instance (injected from app.py)
            news_manager: NewsManager instance (injected from app.py)
            market_data_manager: MarketDataManager instance (injected from app.py)
            index_manager: IndexManager instance (injected from app.py)
            category_fetcher: LocalTaxonomyProvider instance (injected from start.py)
            category_processor: CategoryProcessor instance (injected from app.py)
            ticker_manager: TickerManager instance (injected from app.py)
            context_builder: ContextBuilder instance (injected from app.py)
        """


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
        self.context_builder = context_builder

        self.coingecko_api = coingecko_api

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

        # Last retrieval metadata snapshot for external consumers.
        self._latest_article_urls: Dict[str, str] = {}

    async def initialize(self) -> None:
        """Initialize RAG engine and load cached data"""
        try:
            # Load known tickers
            await self.ticker_manager.load_known_tickers()

            # Ensure categories are up to date
            await self._ensure_categories_updated()

            # Load news database
            await self.news_manager.load_cached_news()

            if self.news_manager.get_database_size() > 0:
                self.last_update = datetime.now(timezone.utc)
                self._build_indices()
                self.logger.debug("Loaded %s recent news articles", self.news_manager.get_database_size())

            await self.ticker_manager.update_known_tickers(self.news_manager.news_database)

            if self.news_manager.get_database_size() < 10:
                await self.refresh_market_data()
                self.last_update = datetime.now(timezone.utc)
        except Exception as e:
            self.logger.exception("Error initializing RAG engine: %s", e)
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
                    self.last_update = datetime.now(timezone.utc)
                    return True
                except Exception as e:
                    self.logger.error("Failed to update market knowledge: %s", e)
                    return False

            time_since_update = datetime.now(timezone.utc) - self.last_update
            if force_update or time_since_update > self.update_interval:
                reason = "forced update" if force_update else f"{time_since_update.total_seconds()/60:.1f} minutes since last update"
                self.logger.debug("Refreshing market knowledge: %s", reason)
                try:
                    await self.refresh_market_data()
                    self.last_update = datetime.now(timezone.utc)
                    return True
                except Exception as e:
                    self.logger.error("Failed to update market knowledge: %s", e)
                    return False

            try:
                categories_updated = await self._ensure_categories_updated()
                if categories_updated:
                    self._build_indices()
            except Exception as e:
                self.logger.error("Failed to update categories: %s", e)

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
            self.logger.error("Error fetching crypto news: %s", e)
            articles = []

        # Update market overview if needed
        try:
            await self.market_data_manager.update_market_overview_if_needed(max_age_hours=24)
        except Exception as e:
            self.logger.error("Error updating market overview: %s", e)

        # Process articles
        if articles:
            updated = self.news_manager.update_news_database(articles)
            if updated:
                self._build_indices()
                self.logger.debug("News database updated; rebuilt indices")

    def _resolve_retrieval_limits(self, k: Optional[int], max_tokens: Optional[int]) -> tuple[int, int]:
        """Resolve retrieval limits from explicit values or config with safe fallbacks."""
        resolved_k = k
        if resolved_k is None:
            try:
                resolved_k = int(self.config.RAG_NEWS_LIMIT)
            except Exception:
                resolved_k = 3

        resolved_max_tokens = max_tokens
        if resolved_max_tokens is None:
            try:
                article_max = int(self.config.RAG_ARTICLE_MAX_TOKENS)
                resolved_max_tokens = article_max * resolved_k
            except Exception:
                resolved_max_tokens = 250 * resolved_k

        return resolved_k, resolved_max_tokens

    @staticmethod
    def _extract_query_keywords(query: str) -> set[str]:
        """Extract query keywords used by context selection heuristics."""
        return set(re.findall(r'\b\w{3,15}\b', query.lower()))

    def _expand_candidate_indices_for_symbol(
        self,
        symbol: str,
        k: int,
        relevant_indices: list[int],
    ) -> list[int]:
        """Add symbol-linked candidates when ranked results are sparse."""
        if not symbol or len(relevant_indices) >= k:
            return relevant_indices

        coin = self.category_processor.extract_base_coin(symbol)
        coin_indices = self.index_manager.search_by_coin(coin)
        for idx in coin_indices:
            if idx not in relevant_indices:
                relevant_indices.append(idx)
                if len(relevant_indices) >= k * 10:
                    break

        return relevant_indices

    def _prioritize_full_body_candidates(self, relevant_indices: list[int]) -> list[int]:
        """Move full-body articles ahead of short summaries while preserving relative order."""
        min_body_chars = int(getattr(self.config, 'RAG_NEWS_ENRICH_MIN_CHARS', 400))
        full_body_indices = [
            idx for idx in relevant_indices
            if len(str(self.news_manager.news_database[idx].get('body', ''))) >= min_body_chars
        ]
        short_body_indices = [idx for idx in relevant_indices if idx not in full_body_indices]
        return full_body_indices + short_body_indices

    def build_context_query(self, symbol: str) -> str:
        """Build a default semantic query string for RAG retrieval based on the trading symbol."""
        base_coin = symbol
        if self.category_processor:
            base_coin = self.category_processor.extract_base_coin(symbol).upper()
        elif '/' in symbol:
            base_coin = symbol.split('/')[0].upper()
        else:
            base_coin = symbol.upper()

        coin_name = base_coin.lower()
        if self.context_builder:
            coin_name = self.context_builder.symbol_name_map.get(base_coin, coin_name)

        return f"{coin_name} price analysis market trends"

    @profile_performance
    async def retrieve_context(self, query: str, symbol: str, k: Optional[int] = None, max_tokens: Optional[int] = None) -> str:
        """Retrieve relevant context for a query with token limiting.

        If `k` is None, the configured RAG news limit (`[rag] news_limit`) will be used.
        Note: Market overview data is handled separately by PromptBuilder._build_market_overview_section()
        This method only returns news articles and market context to avoid redundancy.
        """
        k, max_tokens = self._resolve_retrieval_limits(k, max_tokens)

        if self.news_manager.get_database_size() == 0:
            self.logger.warning("News database is empty")
            return ""

        try:
            rebuild_indices = await self._ensure_categories_updated()
            if rebuild_indices:
                self._build_indices()

            if not self.last_update or datetime.now(timezone.utc) - self.last_update > timedelta(minutes=30):
                await self.update_if_needed()

            keywords = self._extract_query_keywords(query)

            # Use context builder for keyword search
            scores = await self.context_builder.keyword_search(
                query, self.news_manager.news_database, symbol,
                self.index_manager.get_coin_indices(),
                self.category_processor.category_word_map,
                self.category_processor.important_categories
            )
            relevant_indices = [idx for idx, _ in scores[:k*10]]
            scores_dict = {idx: score for idx, score in scores}

            relevant_indices = self._expand_candidate_indices_for_symbol(symbol, k, relevant_indices)
            relevant_indices = self._prioritize_full_body_candidates(relevant_indices)

            # Build context using context builder (pass keywords and scores for smart selection)
            context_text, total_tokens = self.context_builder.add_articles_to_context(
                relevant_indices, self.news_manager.news_database, max_tokens, k, keywords, scores_dict
            )
            self._latest_article_urls = self.context_builder.get_latest_article_urls()
            self.logger.debug("Retrieved context with %s tokens", total_tokens)

            return context_text
        except Exception as e:
            self.logger.error("Error retrieving context: %s", e)
            self._latest_article_urls = {}
            return "Error retrieving market context."

    def get_news_cache_snapshot(self, limit: Optional[int] = None) -> list[Dict[str, Any]]:
        """Return a copy of cached news articles for read-only external consumption."""
        if not self.news_manager:
            return []

        articles = self.news_manager.news_database
        if limit is not None and limit > 0:
            articles = articles[:limit]

        return [dict(article) for article in articles if isinstance(article, dict)]

    def get_latest_article_urls_snapshot(self) -> Dict[str, str]:
        """Return a copy of article URLs captured during the latest retrieve_context call."""
        return dict(self._latest_article_urls)

    async def get_market_overview(self) -> Optional[Dict[str, Any]]:
        """Get current market overview data using MarketDataManager (aggregates CoinGecko + DefiLlama)"""
        try:
            # Delegate to MarketDataManager which handles aggregation of all sources
            if self.market_data_manager:
                 # Check/Update if needed
                 await self.market_data_manager.update_market_overview_if_needed(max_age_hours=1)
                 return self.market_data_manager.get_current_overview()

            return None
        except Exception as e:
            self.logger.error("Error getting market overview: %s", e)
            return None

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
                self.logger.error("Error closing CoinGecko API client: %s", e)

        self.logger.info("RAG Engine resources released")

    async def _ensure_categories_updated(self, force_refresh: bool = False) -> bool:
        """Ensure categories are loaded and up to date.

        Returns:
            True if categories were updated, False otherwise
        """
        try:
            categories = await self.category_fetcher.fetch_categories(force_refresh)
            if categories:
                self.category_processor.process_api_categories(categories)
                return True
            return False
        except Exception as e:
            self.logger.exception("Error ensuring categories updated: %s", e)
            return False
