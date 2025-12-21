"""
RAG System - Restructured with Clean Architecture

This package provides a well-organized RAG (Retrieval-Augmented Generation) system
with clear separation of concerns following the same principles as the analyzer restructure.
"""

from .core import RagEngine, ContextBuilder
from .data import MarketDataManager, NewsManager, RagFileHandler
from .search import IndexManager
from .processing import ArticleProcessor, NewsCategoryAnalyzer
from .management import CategoryManager, TickerManager, CategoryFetcher, CategoryProcessor

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
    'CategoryManager', 'TickerManager', 'CategoryFetcher', 'CategoryProcessor'
]
