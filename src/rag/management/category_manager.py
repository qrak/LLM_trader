"""
Refactored Category Management Module for RAG Engine

Orchestrates category fetching, processing, ticker management, and news analysis
with reduced complexity through specialized components.
"""

from typing import List, Dict, Any, Optional, Set
from src.logger.logger import Logger
from ..data.file_handler import RagFileHandler
from .category_fetcher import CategoryFetcher
from .category_processor import CategoryProcessor
from .ticker_manager import TickerManager
from ..processing.news_category_analyzer import NewsCategoryAnalyzer


class CategoryManager:
    """
    Orchestrates cryptocurrency category and ticker management operations.
    Refactored to use specialized components for better maintainability.
    """
    
    def __init__(self, logger: Logger, file_handler: RagFileHandler, 
                 cryptocompare_api=None, exchange_manager=None):
        self.logger = logger
        self.file_handler = file_handler
        self.cryptocompare_api = cryptocompare_api
        self.exchange_manager = exchange_manager
        
        # Initialize specialized components
        self.category_fetcher = CategoryFetcher(logger, cryptocompare_api)
        self.category_processor = CategoryProcessor(logger)
        self.ticker_manager = TickerManager(logger, file_handler, exchange_manager)
        self.news_analyzer = NewsCategoryAnalyzer(logger, self.category_processor)
    
    # Core API methods - orchestrate component operations
    async def load_known_tickers(self) -> None:
        """Load known cryptocurrency tickers from disk."""
        await self.ticker_manager.load_known_tickers()
    
    async def fetch_cryptocompare_categories(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Fetch cryptocurrency categories from CryptoCompare API."""
        categories = await self.category_fetcher.fetch_cryptocompare_categories(force_refresh)
        
        # Process the fetched categories
        if categories:
            self.category_processor.process_api_categories(categories)
        
        return categories
    
    def process_api_categories(self, api_categories: List[Dict[str, Any]]) -> None:
        """Process API categories and update internal indices."""
        self.category_processor.process_api_categories(api_categories)
    
    async def ensure_categories_updated(self, force_refresh: bool = False) -> bool:
        """Ensure categories are loaded and up to date."""
        success = await self.category_fetcher.ensure_categories_updated(force_refresh)
        
        # If we fetched new categories, process them
        if success and force_refresh:
            categories = await self.fetch_cryptocompare_categories(force_refresh)
            return len(categories) > 0
        
        return success
    
    async def update_known_tickers(self, news_database: List[Dict[str, Any]]) -> None:
        """Update known tickers from news database and validation."""
        await self.ticker_manager.update_known_tickers(news_database)
    
    async def save_tickers(self) -> None:
        """Save known tickers to disk."""
        await self.ticker_manager.save_tickers()
    
    def get_coin_categories(self, symbol: str, news_database: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Get categories for a given coin symbol from multiple sources."""
        return self.news_analyzer.get_coin_categories(symbol, news_database)
    
    def extract_base_coin(self, symbol: str) -> str:
        """Extract base coin from trading pair symbol."""
        return self.category_processor.extract_base_coin(symbol)
    
    def get_known_tickers(self) -> Set[str]:
        """Get the set of known cryptocurrency tickers."""
        return self.ticker_manager.get_known_tickers()

    def get_category_word_map(self) -> Dict[str, str]:
        """Get the mapping of category words to category names."""
        return self.category_processor.category_word_map

    def get_important_categories(self) -> Set[str]:
        """Get the set of important categories used for scoring."""
        return self.category_processor.important_categories
    
