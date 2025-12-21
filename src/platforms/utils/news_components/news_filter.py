"""
News Filtering System
Handles filtering news articles by categories and associated words.
"""
from typing import Dict, List, Any

from src.logger.logger import Logger


class NewsFilter:
    """Handles filtering news articles by categories and content matching."""
    
    def __init__(self, logger: Logger):
        self.logger = logger
    
    def filter_by_category(
        self, 
        articles: List[Dict[str, Any]], 
        category: str, 
        limit: int,
        category_word_map: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Filter news articles by category and associated words"""
        category_lower = category.lower()
        filtered_news = []
        
        for article in articles:
            if len(filtered_news) >= limit:
                break
                
            # Check direct category match
            if self._article_matches_category_directly(article, category_lower):
                filtered_news.append(article)
                continue
            
            # Check category-associated words in title and body
            if self._article_matches_category_words(article, category_lower, category_word_map):
                if article not in filtered_news:
                    filtered_news.append(article)
        
        return filtered_news
    
    def _article_matches_category_directly(self, article: Dict[str, Any], category_lower: str) -> bool:
        """Check if article categories directly match the target category"""
        categories = article.get('categories', '').lower().split('|')
        return category_lower in categories
    
    def _article_matches_category_words(
        self, 
        article: Dict[str, Any], 
        category_lower: str,
        category_word_map: Dict[str, str]
    ) -> bool:
        """Check if article content matches category-associated words"""
        title = article.get('title', '').lower()
        body = article.get('body', '').lower()
        
        for word, cat in category_word_map.items():
            if cat.lower() == category_lower and (word in title or word in body):
                return True
        
        return False
