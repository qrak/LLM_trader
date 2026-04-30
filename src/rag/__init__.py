"""
RAG System - Restructured with Clean Architecture

This package provides a well-organized RAG (Retrieval-Augmented Generation) system
with clear separation of concerns following the same principles as the analyzer restructure.
"""

from .rag_engine import RagEngine
from .context_builder import ContextBuilder
from .market_data_manager import MarketDataManager
from .news_manager import NewsManager
from .news_repository import NewsRepository
from .file_handler import RagFileHandler
from .index_manager import IndexManager
from .scoring_policy import ArticleScoringPolicy
from .article_processor import ArticleProcessor
from .ticker_manager import TickerManager
from .category_processor import CategoryProcessor
from .local_taxonomy import LocalTaxonomyProvider
from .news_ingestion import RSSCrawl4AINewsProvider

__all__ = [
    # Core RAG orchestration
    'RagEngine', 'ContextBuilder',

    # Data operations
<<<<<<< HEAD
    'MarketDataManager', 'NewsManager', 'RagFileHandler',

    # Search operations
    'IndexManager',

    # Content processing
    'ArticleProcessor', 'NewsCategoryAnalyzer',
=======
    'MarketDataManager', 'NewsManager', 'NewsRepository', 'RagFileHandler',

    # Search operations
    'IndexManager', 'ArticleScoringPolicy',

    # Content processing
    'ArticleProcessor',
>>>>>>> main

    # Management operations
    'TickerManager', 'CategoryProcessor',

    # New ingestion providers
    'LocalTaxonomyProvider', 'RSSCrawl4AINewsProvider',
]
