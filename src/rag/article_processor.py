"""
Shared article processing utilities for RAG components.
Eliminates code duplication between news_manager and context_builder.
"""
from typing import Any, Set
import logging
import re


class ArticleProcessor:
    """Utility class for common article processing operations."""

    def __init__(
        self,
        logger: logging.Logger,
        format_utils=None,
        unified_parser=None,
        symbol_name_map: dict[str, str] | None = None,
    ):

        self.logger = logger
        self.parser = unified_parser
        self.format_utils = format_utils
        self.symbol_name_map = {
            str(symbol).upper(): str(name).lower().strip()
            for symbol, name in (symbol_name_map or {}).items()
            if symbol and name
        }

    @staticmethod
    def _normalize_coin_name(name: str) -> str:
        """Normalize coin names/ids so token names match article text better."""
        normalized = re.sub(r"[-_]+", " ", name.lower())
        return re.sub(r"\s+", " ", normalized).strip()

    def detect_coins_in_article(self, article: dict[str, Any], known_crypto_tickers: Set[str]) -> Set[str]:
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
        body = article.get('body', '')

        title_coins = self.parser.detect_coins_in_text(title, known_crypto_tickers)
        body_coins = self.parser.detect_coins_in_text(body, known_crypto_tickers)

        coins_mentioned.update(title_coins)
        coins_mentioned.update(body_coins)

        # Also resolve mentions through configured symbol_name_map entries.
        # This catches names that parser heuristics do not map yet.
        if self.symbol_name_map:
            combined_text = f"{title}\n{body}".lower()
            for symbol, raw_name in self.symbol_name_map.items():
                normalized_name = self._normalize_coin_name(raw_name)
                if not normalized_name:
                    continue
                name_pattern = r'\b' + r'[-\s]+'.join(re.escape(p) for p in normalized_name.split()) + r'\b'
                if re.search(name_pattern, combined_text):
                    if symbol in known_crypto_tickers:
                        coins_mentioned.add(symbol)

        return coins_mentioned

    def get_article_timestamp(self, article: dict[str, Any]) -> float:
        """Extract timestamp from article in a consistent format."""
        published_on = article.get('published_on', 0)
        return self.format_utils.parse_timestamp(published_on)

    def extract_base_coin(self, symbol: str) -> str:
        """Extract base coin from trading pair symbol."""

        return self.parser.extract_base_coin(symbol)
