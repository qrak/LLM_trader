

from __future__ import annotations

from typing import Any

from src.logger.logger import Logger

from .file_handler import RagFileHandler


class NewsRepository:
    """Wraps news persistence and age filtering concerns for NewsManager."""

    def __init__(self, logger: Logger, file_handler: RagFileHandler):
        self.logger = logger
        self.file_handler = file_handler

    def load_recent_articles(self, max_age_seconds: int = 86400) -> list[dict[str, Any]]:

        if max_age_seconds == 86400:
            return self.file_handler.load_news_articles()

        articles = self.file_handler.load_news_articles()
        return self.file_handler.filter_articles_by_age(articles, max_age_seconds=max_age_seconds)

    def save_recent_articles(self, articles: list[dict[str, Any]], max_age_seconds: int = 86400) -> None:

        if max_age_seconds == 86400:
            self.file_handler.save_news_articles(articles)
            return

        recent_articles = self.file_handler.filter_articles_by_age(articles, max_age_seconds=max_age_seconds)
        self.file_handler.save_news_articles(recent_articles)

    def filter_recent_articles(self, articles: list[dict[str, Any]], max_age_seconds: int = 86400) -> list[dict[str, Any]]:

        return self.file_handler.filter_articles_by_age(articles, max_age_seconds=max_age_seconds)

    def load_fallback_articles(self, max_age_hours: int = 72) -> list[dict[str, Any]]:

        return self.file_handler.load_fallback_articles(max_age_hours=max_age_hours)
