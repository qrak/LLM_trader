"""
News Management Module for RAG Engine

Fetches, deduplicates, and caches cryptocurrency news articles.
"""
from typing import List, Dict, Any, Set

from src.logger.logger import Logger
from .file_handler import RagFileHandler


class NewsManager:
<<<<<<< HEAD
    """Manages cryptocurrency news articles and related operations."""

    def __init__(self, logger: Logger, file_handler: RagFileHandler,
                 news_api=None, categories_api=None, session=None, article_processor=None):
=======
    """Fetches fresh news via CryptoCompareNewsClient and maintains the local news database."""
>>>>>>> new_features

    def __init__(
        self,
        logger: Logger,
        file_handler: RagFileHandler,
        news_client=None,
        categories_api=None,
        session=None,
        article_processor=None,
    ):
        self.logger = logger
        self.file_handler = file_handler
        self.news_client = news_client
        self.categories_api = categories_api
        self.session = session
        self.article_processor = article_processor

<<<<<<< HEAD
        # News database
        self.news_database: List[Dict[str, Any]] = []
        self.latest_article_urls: Dict[str, str] = {}
=======
        self.news_database: List[Dict[str, Any]] = []

    # ── Public API ────────────────────────────────────────────────────────────
>>>>>>> new_features

    async def load_cached_news(self) -> None:
        """Load cached news articles from disk."""
        try:
            self.news_database = self.file_handler.load_news_articles()
<<<<<<< HEAD
            if self.news_database:
                # Ensure all loaded articles are normalized
                for article in self.news_database:
                    if 'title_lower' not in article:
                        self._normalize_article(article)
                self.logger.debug("Loaded %s cached news articles", len(self.news_database))
=======
            for article in self.news_database:
                if "title_lower" not in article:
                    self._normalize(article)
            self.logger.debug("Loaded %s cached news articles", len(self.news_database))
>>>>>>> new_features
        except Exception as e:
            self.logger.exception("Error loading cached news: %s", e)
            self.news_database = []

    async def fetch_fresh_news(self, known_crypto_tickers: Set[str]) -> List[Dict[str, Any]]:
        """Fetch fresh articles from CryptoCompare; fall back to cache on failure."""
        if self.news_client is None:
            self.logger.error("News client not initialized")
            return []

        try:
<<<<<<< HEAD
            # Use the news API directly
            articles = await self.news_api.get_latest_news(
                limit=50,
                max_age_hours=24,
                session=self.session,
                api_categories=self.categories_api.get_api_categories() if self.categories_api else None
            )

            if articles:
                # Detect coins in articles using centralized method
                for article in articles:
                    coins_mentioned = self.article_processor.detect_coins_in_article(article, known_crypto_tickers)
                    if coins_mentioned:
                        # Store as list internally, convert to string for file storage
                        article['detected_coins'] = list(coins_mentioned)
                        article['detected_coins_str'] = '|'.join(coins_mentioned)

                self.logger.debug("Fetched %s recent news articles from CryptoCompare", len(articles))
                return articles
            else:
                self.logger.warning("No articles returned from CryptoCompare API")
                return self._get_fallback_articles()

        except Exception as e:
            self.logger.error("Error fetching CryptoCompare news: %s", e)
            return self._get_fallback_articles()

    def _get_fallback_articles(self) -> List[Dict[str, Any]]:
        """Get fallback articles when fresh fetch fails."""
        fallback_articles = self.file_handler.load_fallback_articles(max_age_hours=72)
        if fallback_articles:
            self.logger.info("Using %s cached articles as fallback", len(fallback_articles))
            return fallback_articles
        return []
=======
            api_categories = (
                self.categories_api.get_api_categories() if self.categories_api else None
            )
            raw = await self.news_client.fetch_news(
                session=self.session,
                api_categories=api_categories,
            )

            if not raw:
                self.logger.warning("No articles returned from CryptoCompare API")
                return self._fallback()

            articles = self.news_client.filter_by_age(raw, max_age_hours=24)

            for article in articles:
                coins = self.article_processor.detect_coins_in_article(article, known_crypto_tickers)
                if coins:
                    article["detected_coins"] = list(coins)
                    article["detected_coins_str"] = "|".join(coins)

            self.logger.debug("Fetched %s recent news articles from CryptoCompare", len(articles))
            return articles

        except Exception as e:
            self.logger.error("Error fetching CryptoCompare news: %s", e)
            return self._fallback()
>>>>>>> new_features

    def update_news_database(self, new_articles: List[Dict[str, Any]]) -> bool:
        """Merge new articles into the local database, deduplicate, and persist."""
        if not new_articles:
            self.logger.debug("No new articles to process")
            return False

<<<<<<< HEAD
        recent_articles = self.file_handler.filter_articles_by_age(new_articles, max_age_seconds=86400)

        existing_ids = {article.get('id') for article in self.news_database if article.get('id')}
        unique_articles = [art for art in recent_articles if art.get('id') and art.get('id') not in existing_ids]

        if unique_articles:
            # Pre-compute normalized fields for search optimization
            for article in unique_articles:
                self._normalize_article(article)

            self.logger.debug("Found %s new articles", len(unique_articles))
            combined_articles = self.news_database + unique_articles

            # Sort by timestamp, newest first
            combined_articles.sort(key=self.article_processor.get_article_timestamp, reverse=True)

            # Filter to keep only recent articles
            self.news_database = self.file_handler.filter_articles_by_age(combined_articles, max_age_seconds=86400)

            # Save updated database
            self.file_handler.save_news_articles(self.news_database)

            self.logger.debug("Updated news database with %s recent articles", len(self.news_database))
            return True
        else:
            self.logger.debug("No new articles to add or only duplicates found")
            return False


=======
        recent = self.file_handler.filter_articles_by_age(new_articles, max_age_seconds=86400)
        existing_ids = {a.get("id") for a in self.news_database if a.get("id")}
        unique = [a for a in recent if a.get("id") and a["id"] not in existing_ids]

        if not unique:
            self.logger.debug("No new articles to add or only duplicates found")
            return False

        for article in unique:
            self._normalize(article)

        combined = self.news_database + unique
        combined.sort(key=self.article_processor.get_article_timestamp, reverse=True)
        self.news_database = self.file_handler.filter_articles_by_age(combined, max_age_seconds=86400)
        self.file_handler.save_news_articles(self.news_database)

        self.logger.debug("Updated news database with %s recent articles", len(self.news_database))
        return True
>>>>>>> new_features

    def get_database_size(self) -> int:
        return len(self.news_database)

    def clear_database(self) -> None:
        self.news_database.clear()
<<<<<<< HEAD
        self.latest_article_urls.clear()

    def _normalize_article(self, article: Dict[str, Any]) -> None:
        """Pre-compute lowercased fields for search optimization."""
        article['title_lower'] = article.get('title', '').lower()
        article['body_lower'] = article.get('body', '').lower()
        article['categories_lower'] = article.get('categories', '').lower()
        article['tags_lower'] = article.get('tags', '').lower()
        article['detected_coins_str_lower'] = article.get('detected_coins_str', '').lower()
=======

    # ── Private ───────────────────────────────────────────────────────────────

    def _fallback(self) -> List[Dict[str, Any]]:
        articles = self.file_handler.load_fallback_articles(max_age_hours=72)
        if articles:
            self.logger.info("Using %s cached articles as fallback", len(articles))
        return articles

    def _normalize(self, article: Dict[str, Any]) -> None:
        """Pre-compute lowercased fields for fast keyword search."""
        article["title_lower"] = article.get("title", "").lower()
        article["body_lower"] = article.get("body", "").lower()
        article["categories_lower"] = article.get("categories", "").lower()
        article["tags_lower"] = article.get("tags", "").lower()
        article["detected_coins_str_lower"] = article.get("detected_coins_str", "").lower()
>>>>>>> new_features
