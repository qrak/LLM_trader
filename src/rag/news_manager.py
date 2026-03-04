"""
News Management Module for RAG Engine

Fetches, deduplicates, and caches cryptocurrency news articles.
"""
from typing import List, Dict, Any, Set

from src.logger.logger import Logger
from .file_handler import RagFileHandler


class NewsManager:
    """Fetches fresh news via CryptoCompareNewsClient and maintains the local news database."""

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

        self.news_database: List[Dict[str, Any]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    async def load_cached_news(self) -> None:
        """Load cached news articles from disk."""
        try:
            self.news_database = self.file_handler.load_news_articles()
            for article in self.news_database:
                if "title_lower" not in article:
                    self._normalize(article)
            self.logger.debug("Loaded %s cached news articles", len(self.news_database))
        except Exception as e:
            self.logger.exception("Error loading cached news: %s", e)
            self.news_database = []

    async def fetch_fresh_news(self, known_crypto_tickers: Set[str]) -> List[Dict[str, Any]]:
        """Fetch fresh articles from CryptoCompare; fall back to cache on failure."""
        if self.news_client is None:
            self.logger.error("News client not initialized")
            return []

        try:
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

    def update_news_database(self, new_articles: List[Dict[str, Any]]) -> bool:
        """Merge new articles into the local database, deduplicate, and persist."""
        if not new_articles:
            self.logger.debug("No new articles to process")
            return False

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

    def get_database_size(self) -> int:
        return len(self.news_database)

    def clear_database(self) -> None:
        self.news_database.clear()

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
