"""
News Management Module for RAG Engine

Fetches, deduplicates, and caches cryptocurrency news articles.
"""
from typing import Any, Set

from src.logger.logger import Logger
from .file_handler import RagFileHandler
from .news_repository import NewsRepository


class NewsManager:
    """Fetches fresh news via the configured NewsProvider and maintains the local news database."""

    def __init__(
        self,
        logger: Logger,
        file_handler: RagFileHandler,
        news_client=None,
        session=None,
        article_processor=None,
        news_repository: NewsRepository | None = None,
    ):
        self.logger = logger
        self.file_handler = file_handler
        self.news_client = news_client
        self.session = session
        self.article_processor = article_processor
        self.news_repository = news_repository or NewsRepository(logger=logger, file_handler=file_handler)

        self.news_database: list[dict[str, Any]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    async def load_cached_news(self) -> None:
        """Load cached news articles from disk."""
        try:
            self.news_database = self.news_repository.load_recent_articles(max_age_seconds=86400)
            for article in self.news_database:
                if "title_lower" not in article:
                    self._normalize(article)
            self.logger.debug("Loaded %s cached news articles", len(self.news_database))
        except Exception as e:
            self.logger.exception("Error loading cached news: %s", e)
            self.news_database = []

    async def fetch_fresh_news(self, known_crypto_tickers: Set[str]) -> list[dict[str, Any]]:
        """Fetch fresh articles from the news provider; fall back to cache on failure."""
        if self.news_client is None:
            self.logger.error("News client not initialized")
            return []

        try:
            raw = await self.news_client.fetch_news(
                session=self.session,
            )

            if not raw:
                self.logger.warning("No articles returned from news provider")
                return self._fallback()

            articles = self.news_client.filter_by_age(raw, max_age_hours=72)

            for article in articles:
                coins = self.article_processor.detect_coins_in_article(article, known_crypto_tickers)
                if coins:
                    article["detected_coins"] = list(coins)
                    article["detected_coins_str"] = "|".join(coins)

            self.logger.debug("Fetched %s recent news articles", len(articles))
            return articles

        except Exception as e:
            self.logger.error("Error fetching news: %s", e)
            return self._fallback()

    def update_news_database(self, new_articles: list[dict[str, Any]]) -> bool:
        """Merge new articles into the local database, deduplicate, and persist.

        Deduplication is URL-first: if a canonical URL already exists in the
        database the incoming article is skipped regardless of its ``id``.
        This is provider-agnostic and works for RSS/Crawl4AI sourced articles.
        A stable ``id`` is still required for
        downstream consumers; the new RSS provider generates a deterministic
        SHA-256-based id from the URL.
        """
        if not new_articles:
            self.logger.debug("No new articles to process")
            return False

        recent = self.news_repository.filter_recent_articles(new_articles, max_age_seconds=86400)
        # URL-first dedup with body-length-aware update.
        # If an existing article was cached with a short body (enrichment may have
        # failed previously) and the fresh fetch produced a longer body, replace it.
        _MIN_BODY = 400  # mirror news_min_body_chars default
        url_to_existing = {a["url"]: a for a in self.news_database if a.get("url")}
        existing_ids = {a.get("id") for a in self.news_database if a.get("id")}
        unique: list = []
        body_updates: list = []
        for a in recent:
            url = a.get("url")
            if url:
                if url not in url_to_existing:
                    unique.append(a)
                else:
                    existing = url_to_existing[url]
                    new_len = len(a.get("body", ""))
                    old_len = len(existing.get("body", ""))
                    if old_len < _MIN_BODY and new_len > old_len:
                        body_updates.append(a)
            elif a.get("id") and a["id"] not in existing_ids:
                unique.append(a)

        if not unique and not body_updates:
            self.logger.debug("No new articles to add or only duplicates found")
            return False

        # Apply in-place body updates first
        if body_updates:
            url_to_update = {a["url"]: a for a in body_updates}
            for i, existing in enumerate(self.news_database):
                url = existing.get("url")
                if url and url in url_to_update:
                    self.news_database[i] = url_to_update[url]
                    self._normalize(self.news_database[i])
            self.logger.debug("Re-enriched %d articles with longer bodies", len(body_updates))

        for article in unique:
            self._normalize(article)

        combined = self.news_database + unique
        combined.sort(key=self.article_processor.get_article_timestamp, reverse=True)
        self.news_database = self.news_repository.filter_recent_articles(combined, max_age_seconds=86400)
        self.news_repository.save_recent_articles(self.news_database, max_age_seconds=86400)

        self.logger.debug("Updated news database with %s recent articles", len(self.news_database))
        return True

    def get_database_size(self) -> int:
        return len(self.news_database)

    def clear_database(self) -> None:
        self.news_database.clear()

    # ── Private ───────────────────────────────────────────────────────────────

    def _fallback(self) -> list[dict[str, Any]]:
        articles = self.news_repository.load_fallback_articles(max_age_hours=72)
        if articles:
            self.logger.info("Using %s cached articles as fallback", len(articles))
        return articles

    def _normalize(self, article: dict[str, Any]) -> None:
        """Pre-compute lowercased fields for fast keyword search."""
        source_info = article.get("source_info")
        if source_info and source_info.get("name"):
            source_info["name"] = str(source_info["name"]).strip().lower()

        article["title_lower"] = article.get("title", "").lower()
        article["body_lower"] = article.get("body", "").lower()
        article["categories_lower"] = article.get("categories", "").lower()
        article["tags_lower"] = article.get("tags", "").lower()
        article["detected_coins_str_lower"] = article.get("detected_coins_str", "").lower()
