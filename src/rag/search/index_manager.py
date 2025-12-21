"""
Index Management Module for RAG Engine

Handles building and maintaining search indices for news articles.
"""

import re
from collections import defaultdict
from typing import List, Dict, Any, Set
from src.logger.logger import Logger
from ..processing.article_processor import ArticleProcessor


class IndexManager:
    """Manages search indices for efficient article lookup."""
    
    def __init__(self, logger: Logger, format_utils=None):
        self.logger = logger
        self.article_processor = ArticleProcessor(logger, format_utils)
        
        # Search indices
        self.category_index: Dict[str, List[int]] = defaultdict(list)
        self.tag_index: Dict[str, List[int]] = defaultdict(list)
        self.coin_index: Dict[str, List[int]] = defaultdict(list)
        self.keyword_index: Dict[str, List[int]] = defaultdict(list)
    
    def build_indices(self, news_database: List[Dict[str, Any]], 
                     known_crypto_tickers: Set[str], 
                     category_word_map: Dict[str, str]) -> None:
        """Build search indices from news database."""
        self._clear_indices()
        
        for i, article in enumerate(news_database):
            self._index_article_categories(article, i, known_crypto_tickers)
            self._index_article_tags(article, i)
            self._index_article_coins(article, i, known_crypto_tickers)
            self._index_article_keywords(article, i, category_word_map)
    
    def _clear_indices(self) -> None:
        """Clear all search indices."""
        self.category_index.clear()
        self.tag_index.clear()
        self.coin_index.clear()
        self.keyword_index.clear()

    def _index_article_categories(self, article: Dict[str, Any], index: int, known_crypto_tickers: Set[str]) -> None:
        """Index article categories."""
        categories = article.get('categories', '').split('|')
        for category in categories:
            if not category:
                continue
                
            category_lower = category.lower()
            self.category_index[category_lower].append(index)

            # Use consistent case-insensitive comparison
            if category.strip().upper() in known_crypto_tickers:
                self.coin_index[category_lower].append(index)

    def _index_article_tags(self, article: Dict[str, Any], index: int) -> None:
        """Index article tags."""
        tags = article.get('tags', '').split('|')
        for tag in tags:
            if tag:
                self.tag_index[tag.lower()].append(index)

    def _index_article_coins(self, article: Dict[str, Any], index: int, known_crypto_tickers: Set[str]) -> None:
        """Detect and index coins mentioned in the article."""
        # Check if coins are already detected and stored as list
        if 'detected_coins' in article and isinstance(article['detected_coins'], list):
            coins_mentioned = set(article['detected_coins'])
        else:
            # Fall back to detection
            coins_mentioned = self._detect_coins_in_article(article, known_crypto_tickers)
            if coins_mentioned:
                # Store as list internally
                article['detected_coins'] = list(coins_mentioned)
                article['detected_coins_str'] = '|'.join(coins_mentioned)
            
        for coin in coins_mentioned:
            self.coin_index[coin.lower()].append(index)

    def _index_article_keywords(self, article: Dict[str, Any], index: int, category_word_map: Dict[str, str]) -> None:
        """Index keywords from article title and body with consistent case normalization."""
        title = article.get('title', '').lower()
        body = article.get('body', '').lower()

        # Index category-associated words
        self._index_category_words(title, body, index, category_word_map)
        
        # Index important title words
        self._index_title_words(title, index)

    def _index_category_words(self, title: str, body: str, index: int, category_word_map: Dict[str, str]) -> None:
        """Index words associated with categories with consistent lowercase normalization."""
        for word, category in category_word_map.items():
            # Ensure word is already lowercase (should be from the mapping)
            word_lower = word.lower()
            word_pattern = rf'\b{re.escape(word_lower)}\b'
            if re.search(word_pattern, title) or re.search(word_pattern, body):
                # Prevent duplicates
                if index not in self.keyword_index[word_lower]:
                    self.keyword_index[word_lower].append(index)
                # Note: This would need known_crypto_tickers to be passed in
                # For now, commenting out the coin index part
                # if category.upper() in known_crypto_tickers:
                #     self.coin_index[category.lower()].append(index)

    def _index_title_words(self, title: str, index: int) -> None:
        """Index important words from article title with consistent lowercase normalization."""
        title_words = set(re.findall(r'\b[a-z0-9]{3,15}\b', title))
        stop_words = {'the', 'and', 'for', 'with'}
        
        for word in title_words:
            if len(word) > 2 and word not in stop_words:
                # Ensure consistent lowercase normalization and prevent duplicates
                word_lower = word.lower()
                if index not in self.keyword_index[word_lower]:
                    self.keyword_index[word_lower].append(index)
    
    def _detect_coins_in_article(self, article: Dict[str, Any], known_crypto_tickers: Set[str]) -> Set[str]:
        """Detect cryptocurrency mentions in article content - delegates to ArticleProcessor."""
        return self.article_processor.detect_coins_in_article(article, known_crypto_tickers)
    
    def search_by_coin(self, coin: str) -> List[int]:
        """Search for articles mentioning a specific coin."""
        coin_lower = coin.lower()
        return self.coin_index.get(coin_lower, [])
    
    def get_coin_indices(self) -> Dict[str, List[int]]:
        """Get the coin index."""
        return dict(self.coin_index)
