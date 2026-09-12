"""
Index Management Module for RAG Engine

Handles building and maintaining search indices for news articles.
"""

from collections import defaultdict
from typing import Any

from src.logger.logger import Logger


class IndexManager:
    """Manages search indices for efficient article lookup."""

    def __init__(self, logger: Logger, article_processor=None):

        self.logger = logger
        self.article_processor = article_processor

        # Coin index: maps lowercased coin symbols to article positions.
        self.coin_index: dict[str, list[int]] = defaultdict(list)

    def build_indices(self, news_database: list[dict[str, Any]],
                     known_crypto_tickers: set[str]) -> None:
        """Build the coin search index from the news database."""
        self.coin_index.clear()

        for i, article in enumerate(news_database):
            self._index_ticker_categories(article, i, known_crypto_tickers)
            self._index_article_coins(article, i, known_crypto_tickers)

    def _index_ticker_categories(self, article: dict[str, Any], index: int, known_crypto_tickers: set[str]) -> None:
        """Index ticker-named categories so coin lookups find their articles."""
        for category in article.get("categories", "").split("|"):
            stripped = category.strip()
            if not stripped:
                continue
            if stripped.upper() in known_crypto_tickers:
                self.coin_index[stripped.lower()].append(index)

    def _index_article_coins(self, article: dict[str, Any], index: int, known_crypto_tickers: set[str]) -> None:
        """Detect and index coins mentioned in the article."""
        # Check if coins are already detected and stored as list
        if "detected_coins" in article:
            coins_mentioned = set(article["detected_coins"])
        else:
            # Fall back to detection
            coins_mentioned = self.article_processor.detect_coins_in_article(article, known_crypto_tickers)  # type: ignore
            if coins_mentioned:
                # Store as list internally
                article["detected_coins"] = list(coins_mentioned)
                article["detected_coins_str"] = "|".join(coins_mentioned)

        for coin in coins_mentioned:
            self.coin_index[coin.lower()].append(index)

    def search_by_coin(self, coin: str) -> list[int]:
        """Search for articles mentioning a specific coin."""
        coin_lower = coin.lower()
        return self.coin_index.get(coin_lower, [])

