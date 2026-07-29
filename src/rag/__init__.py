"""
RAG System - Restructured with Clean Architecture

This package provides a well-organized RAG (Retrieval-Augmented Generation) system
with clear separation of concerns following the same principles as the analyzer restructure.
"""

from .article_processor import ArticleProcessor
from .category_processor import CategoryProcessor
from .context_builder import ContextBuilder
from .file_handler import RagFileHandler
from .index_manager import IndexManager
from .local_taxonomy import LocalTaxonomyProvider
from .market_data_manager import MarketDataManager
from .news_ingestion import RSSCrawl4AINewsProvider
from .news_manager import NewsManager
from .news_repository import NewsRepository
from .rag_engine import RagEngine
from .scoring_policy import ArticleScoringPolicy
from .ticker_manager import TickerManager

__all__ = [
    # Content processing
    "ArticleProcessor",
    "ArticleScoringPolicy",
    "CategoryProcessor",
    "ContextBuilder",
    # Search operations
    "IndexManager",
    # New ingestion providers
    "LocalTaxonomyProvider",
    # Data operations
    "MarketDataManager",
    "NewsManager",
    "NewsRepository",
    "RSSCrawl4AINewsProvider",
    # Core RAG orchestration
    "RagEngine",
    "RagFileHandler",
    # Management operations
    "TickerManager",
]
