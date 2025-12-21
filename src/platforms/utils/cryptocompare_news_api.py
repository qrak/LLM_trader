"""
CryptoCompare News API - Refactored with specialized components
Orchestrates news fetching, caching, filtering, and processing operations.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, TYPE_CHECKING

import aiohttp

if TYPE_CHECKING:
    from src.contracts.config import ConfigProtocol
from src.logger.logger import Logger
from src.utils.decorators import retry_api_call
from .news_components import CryptoCompareNewsClient, NewsCache, NewsProcessor, NewsFilter


class CryptoCompareNewsAPI:
    """
    Orchestrates CryptoCompare news operations using specialized components.
    Handles fetching, caching, filtering, and processing of news articles.
    """
    
    def __init__(
        self,
        logger: Logger,
        config: "ConfigProtocol",
        cache_dir: str = 'data/news_cache',
        update_interval_hours: int = 1
    ) -> None:
        self.logger = logger
        self.config = config
        self.update_interval = timedelta(hours=update_interval_hours)
        
        # Initialize specialized components
        self.client = CryptoCompareNewsClient(logger, config)
        self.cache = NewsCache(cache_dir, logger)
        self.processor = NewsProcessor(logger)
        self.filter = NewsFilter(logger)
    
    async def initialize(self) -> None:
        """Initialize the news API and load cached data"""
        self.cache.initialize()
    
    @retry_api_call(max_retries=3)
    async def get_latest_news(
        self, 
        limit: int = 50, 
        max_age_hours: int = 24,
        session: Optional[aiohttp.ClientSession] = None,
        api_categories: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get latest cryptocurrency news articles
        
        Args:
            limit: Maximum number of articles to return
            max_age_hours: Maximum age of articles in hours
            session: Optional aiohttp session to use
            api_categories: Optional list of API categories for filtering
            
        Returns:
            List of news articles
        """
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=max_age_hours)
        
        # Check if we need to update
        if self.cache.should_fetch_fresh_news(self.update_interval):
            return await self._fetch_and_process_fresh_news(limit, cutoff_time, session, api_categories)
        else:
            # Use cached data if it's recent enough
            return self.cache.get_cached_news(limit, cutoff_time)
    
    @retry_api_call(max_retries=3)
    async def get_news_by_category(
        self, 
        category: str, 
        limit: int = 10,
        category_word_map: Optional[Dict[str, str]] = None,
        session: Optional[aiohttp.ClientSession] = None,
        api_categories: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get news articles filtered by category
        
        Args:
            category: Category name to filter by
            limit: Maximum number of articles to return
            category_word_map: Optional mapping of words to categories
            session: Optional aiohttp session to use
            api_categories: Optional list of API categories for filtering
            
        Returns:
            List of news articles matching the category
        """
        # Get all recent news
        all_news = await self.get_latest_news(limit=0, session=session, api_categories=api_categories)
        
        # Filter by category
        filtered_news = self.filter.filter_by_category(all_news, category, limit, category_word_map or {})
        
        # Apply limit and return
        return filtered_news[:limit]
    
    async def detect_coins_in_article(self, article: Dict[str, Any], known_tickers: Set[str]) -> Set[str]:
        """
        Detect cryptocurrency mentions in article content
        
        Args:
            article: Article data
            known_tickers: Set of known cryptocurrency tickers
            
        Returns:
            Set of detected coin tickers
        """
        return self.processor.detect_coins_in_article(article, known_tickers)
    
    async def _fetch_and_process_fresh_news(
        self,
        limit: int,
        cutoff_time: datetime,
        session: Optional[aiohttp.ClientSession],
        api_categories: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Fetch fresh news from API and process it"""
        self.logger.debug("Fetching fresh news data from CryptoCompare")
        articles = await self.client.fetch_news(session, api_categories)
        
        if articles:
            # Process fresh articles
            return self._process_fresh_articles(articles, limit, cutoff_time)
        else:
            # If API fetch failed, try to use cached data
            return self.cache.get_cached_news(limit, cutoff_time)
    
    def _process_fresh_articles(
        self,
        articles: List[Dict[str, Any]],
        limit: int,
        cutoff_time: datetime
    ) -> List[Dict[str, Any]]:
        """Process fresh articles by filtering, sorting, and caching"""
        # Process articles using the processor component
        result = self.processor.process_and_sort_articles(articles, cutoff_time, limit)
        
        # Cache the results
        self.cache.save_news_data(result)
        
        return result
