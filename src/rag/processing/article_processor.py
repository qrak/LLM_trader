"""
Shared article processing utilities for RAG components.
Eliminates code duplication between news_manager and context_builder.
"""
from typing import Dict, Any, Set
import logging

from src.parsing.unified_parser import UnifiedParser


class ArticleProcessor:
    """Utility class for common article processing operations."""
    
    def __init__(self, logger: logging.Logger = None, format_utils=None):
        self.logger = logger or logging.getLogger(__name__)
        self.parser = UnifiedParser(self.logger, format_utils)
        self.format_utils = format_utils
    
    def detect_coins_in_article(self, article: Dict[str, Any], known_crypto_tickers: Set[str]) -> Set[str]:
        """Detect cryptocurrency mentions in article content."""
        # Check categories first
        coins_mentioned = set()
        categories = article.get('categories', '').split('|')
        for category in categories:
            cat_upper = category.upper()
            if cat_upper in known_crypto_tickers:
                coins_mentioned.add(cat_upper)
        
        # Check title and body for coin mentions
        title = article.get('title', '')
        body = article.get('body', '')[:10000] if len(article.get('body', '')) >= 10000 else article.get('body', '')
        
        title_coins = self.parser.detect_coins_in_text(title, known_crypto_tickers)
        body_coins = self.parser.detect_coins_in_text(body, known_crypto_tickers)
        
        coins_mentioned.update(title_coins)
        coins_mentioned.update(body_coins)
        
        return coins_mentioned
    
    def get_article_timestamp(self, article: Dict[str, Any]) -> float:
        """Extract timestamp from article in a consistent format."""
        published_on = article.get('published_on', 0)
        return self.parser.parse_timestamp(published_on)
    
    def format_article_date(self, article: Dict[str, Any]) -> str:
        """Format article date in a consistent way."""
        timestamp = self.get_article_timestamp(article)
        if timestamp <= 0:
            return "Unknown Date"
        
        return self.format_utils.format_date_from_timestamp(timestamp)
    
    def extract_base_coin(self, symbol: str) -> str:
        """Extract base coin from trading pair symbol."""
        return self.parser.extract_base_coin(symbol)
