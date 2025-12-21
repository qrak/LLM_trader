import os
from typing import Dict, List, Any, Set, TYPE_CHECKING

import aiohttp

from src.logger.logger import Logger
from src.utils.decorators import retry_api_call
from .utils.cryptocompare_news_api import CryptoCompareNewsAPI
from .utils.cryptocompare_categories_api import CryptoCompareCategoriesAPI
from .utils.cryptocompare_market_api import CryptoCompareMarketAPI
from .utils.cryptocompare_data_processor import CryptoCompareDataProcessor

if TYPE_CHECKING:
    from src.contracts.config import ConfigProtocol


class CryptoCompareAPI:
    """
    API client for CryptoCompare services.
    Handles news fetching, price data, and categories with proper caching.
    
    This class acts as an orchestrator for specialized API components.
    """
    
    def __init__(
        self,
        logger: Logger,
        config: "ConfigProtocol",
        data_dir: str = 'data',
        cache_dir: str = 'data/news_cache',
        update_interval_hours: int = 1,
        categories_update_interval_hours: int = 24
    ) -> None:
        """Initialize CryptoCompareAPI with logger and config.
        
        Args:
            logger: Logger instance
            config: ConfigProtocol instance for API keys and URLs
            data_dir: Data directory path
            cache_dir: Cache directory path
            update_interval_hours: News update interval
            categories_update_interval_hours: Categories update interval
        """
        # Initialize specialized components
        self.logger = logger
        self.config = config
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        
        # Initialize specialized API components
        self.news_api = CryptoCompareNewsAPI(logger, config, cache_dir, update_interval_hours)
        self.categories_api = CryptoCompareCategoriesAPI(logger, config, data_dir, categories_update_interval_hours)
        self.market_api = CryptoCompareMarketAPI(logger, config)
        self.data_processor = CryptoCompareDataProcessor(logger)
        
        # Ensure directories exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        self.session = None
    
    async def initialize(self) -> None:
        """Initialize the API client and load cached data"""
        # Create a shared session
        self.session = aiohttp.ClientSession()
        
        # Initialize specialized components
        await self.news_api.initialize()
        await self.categories_api.initialize()
    
    async def close(self) -> None:
        """Close resources"""
        if hasattr(self, 'session') and self.session:
            try:
                self.logger.debug("Closing CryptoCompare API session")
                await self.session.close()
                self.session = None
            except Exception as e:
                self.logger.error(f"Error closing CryptoCompare API session: {e}")
    
    # Delegate news operations to news API component
    @retry_api_call(max_retries=3)
    async def get_latest_news(self, limit: int = 50, max_age_hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get latest cryptocurrency news articles
        
        Args:
            limit: Maximum number of articles to return
            max_age_hours: Maximum age of articles in hours
            
        Returns:
            List of news articles
        """
        return await self.news_api.get_latest_news(
            limit=limit,
            max_age_hours=max_age_hours,
            session=self.session,
            api_categories=self.categories_api.get_api_categories()
        )
    
    @retry_api_call(max_retries=3)
    async def get_news_by_category(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get news articles filtered by category
        
        Args:
            category: Category name to filter by
            limit: Maximum number of articles to return
            
        Returns:
            List of news articles matching the category
        """
        return await self.news_api.get_news_by_category(
            category=category,
            limit=limit,
            category_word_map=self.categories_api.get_category_word_map(),
            session=self.session,
            api_categories=self.categories_api.get_api_categories()
        )
    
    # Delegate categories operations to categories API component
    @retry_api_call(max_retries=3)
    async def get_categories(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get cryptocurrency categories data
        
        Args:
            force_refresh: Force refresh from API instead of using cache
            
        Returns:
            List of category objects
        """
        return await self.categories_api.get_categories(force_refresh=force_refresh)
    
    # Delegate market operations to market API component
    @retry_api_call(max_retries=3)
    async def get_multi_price_data(self, coins: List[str] = None, vs_currencies: List[str] = None) -> Dict[str, Any]:
        """
        Get price data for multiple coins
        
        Args:
            coins: List of coin symbols (default: BTC,ETH,XRP,LTC,BCH,BNB,ADA,DOT,LINK)
            vs_currencies: List of fiat currencies (default: USD)
            
        Returns:
            Dictionary with price data
        """
        return await self.market_api.get_multi_price_data(coins=coins, vs_currencies=vs_currencies)
    
    @retry_api_call(max_retries=3)
    async def get_coin_details(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed coin information including description, taxonomy, and Weiss ratings
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'LINK', 'BTC')
            
        Returns:
            Dictionary with coin details including description, algorithm, proof type,
            sponsored status, taxonomy classifications, and Weiss ratings
        """
        return await self.market_api.get_coin_details(symbol)
    
    # Delegate static methods to appropriate components
    async def detect_coins_in_article(self, article: Dict[str, Any], known_tickers: Set[str]) -> Set[str]:
        """
        Detect cryptocurrency mentions in article content
        
        Args:
            article: Article data
            known_tickers: Set of known cryptocurrency tickers
            
        Returns:
            Set of detected coin tickers
        """
        return await self.news_api.detect_coins_in_article(article, known_tickers)
