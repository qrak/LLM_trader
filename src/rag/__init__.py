"""
RAG System - Restructured with Clean Architecture

This package provides a well-organized RAG (Retrieval-Augmented Generation) system
with clear separation of concerns following the same principles as the analyzer restructure.
"""

from .rag_engine import RagEngine
from .context_builder import ContextBuilder
from .market_data_manager import MarketDataManager
from .news_manager import NewsManager
from .file_handler import RagFileHandler
from .index_manager import IndexManager
from .article_processor import ArticleProcessor
from .news_category_analyzer import NewsCategoryAnalyzer
from .ticker_manager import TickerManager
from .category_fetcher import CategoryFetcher
from .category_processor import CategoryProcessor

__all__ = [
    # Core RAG orchestration
    'RagEngine', 'ContextBuilder',
    
    # Data operations
    'MarketDataManager', 'NewsManager', 'RagFileHandler',
    
    # Search operations
    'IndexManager',
    
    # Content processing
    'ArticleProcessor', 'NewsCategoryAnalyzer',
    
    # Management operations
    'TickerManager', 'CategoryFetcher', 'CategoryProcessor'
]
