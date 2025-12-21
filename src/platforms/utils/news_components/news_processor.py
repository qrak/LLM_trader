"""
News Processing and Coin Detection
Handles article processing, sorting, and cryptocurrency detection.
"""
import re
from datetime import datetime
from typing import Dict, List, Any, Set

from src.logger.logger import Logger
from .timestamp_utils import get_article_timestamp


class NewsProcessor:
    """Handles news article processing, sorting, and coin detection."""
    
    def __init__(self, logger: Logger):
        self.logger = logger
    
    def process_and_sort_articles(
        self, 
        articles: List[Dict[str, Any]], 
        cutoff_time: datetime,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Process articles by filtering, sorting, and limiting"""
        # Filter by age
        filtered_articles = []
        for article in articles:
            pub_time = get_article_timestamp(article)
            if pub_time > cutoff_time.timestamp():
                filtered_articles.append(article)
        
        # Sort by publication date (newest first)
        filtered_articles.sort(key=lambda x: get_article_timestamp(x), reverse=True)
        
        # Trim to limit
        return filtered_articles[:limit] if limit > 0 else filtered_articles
    
    def detect_coins_in_article(self, article: Dict[str, Any], known_tickers: Set[str]) -> Set[str]:
        """
        Detect cryptocurrency mentions in article content
        
        Args:
            article: Article data
            known_tickers: Set of known cryptocurrency tickers
            
        Returns:
            Set of detected coin tickers
        """
        coins_mentioned = set()
        
        # Extract and prepare content
        title = article.get('title', '').upper()
        body = self._prepare_body_content(article)
        categories = article.get('categories', '').split('|')
        
        # Check different sources for coin mentions
        coins_mentioned.update(self._check_categories_for_coins(categories, known_tickers))
        coins_mentioned.update(self._check_content_for_ticker_patterns(title, body, known_tickers))
        coins_mentioned.update(self._check_special_coin_names(title, body))
        
        return coins_mentioned

    def _prepare_body_content(self, article: Dict[str, Any]) -> str:
        """Prepare and truncate body content for processing"""
        body = article.get('body', '')
        if len(body) > 10000:
            body = body[:10000]
        return body.upper()
    
    def _check_categories_for_coins(self, categories: List[str], known_tickers: Set[str]) -> Set[str]:
        """Check article categories for known cryptocurrency tickers"""
        coins_mentioned = set()
        for category in categories:
            cat_upper = category.upper()
            if cat_upper in known_tickers:
                coins_mentioned.add(cat_upper)
        return coins_mentioned
    
    def _check_content_for_ticker_patterns(self, title: str, body: str, known_tickers: Set[str]) -> Set[str]:
        """Check title and body content for ticker patterns"""
        coins_mentioned = set()
        ticker_regex = r'\b[A-Z]{2,6}\b'
        
        # Find potential tickers in both title and body
        potential_tickers_in_title = set(re.findall(ticker_regex, title))
        potential_tickers_in_body = set(re.findall(ticker_regex, body))
        
        # Add known tickers found in title
        for ticker in potential_tickers_in_title:
            if ticker in known_tickers:
                coins_mentioned.add(ticker)
        
        # Add known tickers found in body
        for ticker in potential_tickers_in_body:
            if ticker in known_tickers:
                coins_mentioned.add(ticker)
        
        return coins_mentioned
    
    def _check_special_coin_names(self, title: str, body: str) -> Set[str]:
        """Check for special cryptocurrency names that have common full names"""
        coins_mentioned = set()
        title_lower = title.lower()
        body_lower = body.lower()
        
        # Only handle the most common cases where full names are widely used
        if 'bitcoin' in title_lower or 'bitcoin' in body_lower:
            coins_mentioned.add('BTC')
        if 'ethereum' in title_lower or 'ethereum' in body_lower:
            coins_mentioned.add('ETH')
            
        return coins_mentioned
