"""
News Caching System
Handles loading, saving, and managing cached news data.
"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from src.logger.logger import Logger
from .timestamp_utils import get_article_timestamp


class NewsCache:
    """Handles news caching operations including load, save, and validation."""
    
    def __init__(self, cache_dir: str, logger: Logger):
        self.cache_dir = cache_dir
        self.logger = logger
        self.news_cache_file = os.path.join(cache_dir, "recent_news.json")
        self.last_news_update: Optional[datetime] = None
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
    
    def initialize(self) -> None:
        """Initialize cache and load existing data"""
        if os.path.exists(self.news_cache_file):
            try:
                with open(self.news_cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    if "last_updated" in cached_data:
                        self.last_news_update = datetime.fromisoformat(cached_data["last_updated"])
                        self.logger.debug(f"Found cached news data from {self.last_news_update.isoformat()}")
            except Exception as e:
                self.logger.error(f"Error loading news cache: {e}")
    
    def should_fetch_fresh_news(self, update_interval: timedelta) -> bool:
        """Determine if we should fetch fresh news from the API"""
        if not self.last_news_update:
            return True
        return datetime.now() - self.last_news_update > update_interval
    
    def get_cached_news(self, limit: int, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Get news from cache with filtering by age and limit"""
        try:
            if os.path.exists(self.news_cache_file):
                with open(self.news_cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    if "articles" in cached_data:
                        articles = cached_data["articles"]
                        # Filter by age using timestamp getter
                        filtered_articles = [
                            art for art in articles 
                            if get_article_timestamp(art) > cutoff_time.timestamp()
                        ]
                        
                        # Return with limit
                        return filtered_articles[:limit] if limit > 0 else filtered_articles
        except Exception as e:
            self.logger.error(f"Error reading cached news: {e}")
        
        return []
    
    def save_news_data(self, articles: List[Dict[str, Any]]) -> None:
        """Save news data to cache file"""
        if not articles:
            return
            
        try:
            cache_data = {
                "last_updated": datetime.now().isoformat(),
                "count": len(articles),
                "articles": articles
            }
            
            with open(self.news_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            self.last_news_update = datetime.now()
            self.logger.debug(f"Saved {len(articles)} news articles to cache")
        except Exception as e:
            self.logger.error(f"Error saving news cache: {e}")
