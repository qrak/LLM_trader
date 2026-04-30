"""
Context Building Module for RAG Engine

Handles building analysis context from news articles and search results.
"""

import re
from datetime import datetime
from collections import namedtuple
from typing import List, Dict, Any, Optional, Tuple, Set
from src.logger.logger import Logger
from src.rag.article_processor import ArticleProcessor
from src.rag.news_ingestion.schema_mapper import normalize_article_whitespace
from src.rag.scoring_policy import ArticleScoringPolicy
from src.utils.profiler import profile_performance
from src.utils.token_counter import TokenCounter

ArticleContent = namedtuple('ArticleContent', ['title', 'body', 'categories', 'tags', 'detected_coins'])

class ContextBuilder:
    """Builds analysis context from various data sources."""

    def __init__(
        self,
        logger: Logger,
        token_counter: TokenCounter,
        config=None,
        article_processor=None,
        symbol_name_map: Optional[Dict[str, str]] = None,
    ):
        self.logger = logger
        self.config = config
        self.token_counter = token_counter
        self.article_processor = article_processor
        self.latest_article_urls: Dict[str, str] = {}
        self.scoring_policy = ArticleScoringPolicy(config=config)
        self.symbol_name_map = {
            str(symbol).upper(): ArticleProcessor._normalize_coin_name(str(name))
            for symbol, name in (symbol_name_map or {}).items()
            if symbol and name
        }

    @profile_performance
    async def keyword_search(self, query: str, news_database: List[Dict[str, Any]],
                           symbol: Optional[str] = None, coin_index: Optional[Dict[str, List[int]]] = None,
                           category_word_map: Optional[Dict[str, str]] = None,
                           important_categories: Optional[Set[str]] = None) -> List[Tuple[int, float]]:
        """Search for articles matching keywords with relevance scores."""
        # Provide default empty containers to avoid mutable defaults
        coin_index = coin_index or {}
        category_word_map = category_word_map or {}
        important_categories = important_categories or set()

        query = query.lower()
        keywords = set(re.findall(r'\b\w{3,15}\b', query))

        coin = None
        coin_patterns: Optional[Dict[str, Any]] = None
        if symbol and self.article_processor:
            coin = self.article_processor.extract_base_coin(symbol).upper()
            coin_lower = coin.lower()
            coin_patterns = {
                'coin_pattern': re.compile(rf'\b{re.escape(coin_lower)}\b'),
                'title_start_pattern': re.compile(rf'^\s*{re.escape(coin_lower)}\b'),
                'price_pattern': re.compile(rf'\b{re.escape(coin_lower)}\s+price\b')
            }
            coin_full_name = self.symbol_name_map.get(coin)
            coin_patterns['coin_name_pattern'] = (
                re.compile(rf"\b{re.escape(coin_full_name).replace('\\ ', r'[-\\s]+')}\b") if coin_full_name else None
            )

        # Pre-calculate relevant categories based on query
        relevant_categories = []
        if category_word_map:
            for word, category in category_word_map.items():
                if word in query:
                    relevant_categories.append(category.lower())

        scores: List[Tuple[int, float]] = []
        current_time = datetime.now().timestamp()

        for i, article in enumerate(news_database):
            score = self._calculate_article_relevance(
                article, keywords, coin, current_time,
                relevant_categories, important_categories, coin_patterns
            )
            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def _calculate_article_relevance(self, article: Dict[str, Any], keywords: Set[str],
                                   coin: Optional[str],
                                   current_time: float, relevant_categories: List[str],
                                   important_categories: Set[str],
                                   coin_patterns: Optional[Dict[str, Any]] = None) -> float:
        """Calculate article relevance score based on various factors."""
        content = self._extract_article_content(article)
        if self.article_processor:
            pub_time = self.article_processor.get_article_timestamp(article)
        else:
            pub_time = 0.0
        return self.scoring_policy.calculate_article_relevance(
            article=article,
            content=content,
            keywords=keywords,
            coin=coin,
            current_time=current_time,
            relevant_categories=relevant_categories,
            important_categories=important_categories,
            pub_time=pub_time,
            coin_patterns=coin_patterns,
        )

    def _extract_article_content(self, article: Dict[str, Any]):
        """Extract and normalize article content for scoring."""
        # Use pre-computed lowercase fields if available (from NewsManager), otherwise compute on the fly
        return ArticleContent(
            title=article.get('title_lower') or article.get('title', '').lower(),
            body=article.get('body_lower') or article.get('body', '').lower(),
            categories=article.get('categories_lower') or article.get('categories', '').lower(),
            tags=article.get('tags_lower') or article.get('tags', '').lower(),
            detected_coins=article.get('detected_coins_str_lower') or article.get('detected_coins_str', '').lower()
        )

    @profile_performance
    def build_context(self, news_items: List[Dict], max_tokens: int = 2000) -> str:
        """
        Build a context string from news items using simple lead paragraph extraction.

        Args:
            news_items: List of news dictionaries
            max_tokens: Maximum tokens for the entire context

        Returns:
            Formatted context string
        """
        if not news_items:
            return ""

        self.latest_article_urls.clear()
        context_parts = []
        current_tokens = 0

        # Use configured article_max_tokens to limit each article strictly
        # This ensures uniform article quality and control over content distribution
        try:
            article_max_tokens = int(self.config.RAG_ARTICLE_MAX_TOKENS) if self.config else 1500
        except Exception:
            article_max_tokens = 1500

        for item in news_items:
            # Stop if we're close to the total context limit
            if current_tokens >= max_tokens:
                break

            # Process article with strict per-article token limit
            processed_text = self._process_article_simple(item, article_max_tokens)

            if processed_text:
                token_count = self.token_counter.count_tokens(processed_text)

                # Skip this article if including it would exceed total context limit
                if current_tokens + token_count > max_tokens:
                    break

                context_parts.append(processed_text)
                current_tokens += token_count

                # Track article URL if available
                if 'url' in item:
                    title = item.get('title', 'Untitled')
                    self.latest_article_urls[title] = item['url']

        return "\n\n".join(context_parts)

    def _process_article_simple(self, item: Dict, max_tokens: int) -> str:
        """
        Process a single article: Title + article body (truncated only by budget).

        Args:
            item: News item dictionary
            max_tokens: Max tokens for this article chunk

        Returns:
            Formatted string: "Title\nSource (Date)\nArticle body..."
        """
        title = item.get('title', 'No Title').strip()
        source_raw = item.get('source_info') or item.get('source', 'Unknown Source')
        source = source_raw.get('name', 'Unknown Source') if isinstance(source_raw, dict) else str(source_raw)
        published_on = item.get('published_on', 0)
        published = datetime.fromtimestamp(published_on).strftime('%Y-%m-%d %H:%M UTC')

        body = item.get('body', '').strip()
        if not body:
            return ""

        body = normalize_article_whitespace(body)
        paragraphs = [paragraph.strip() for paragraph in body.split('\n\n') if paragraph.strip()]

        if not paragraphs:
            return ""

        article_body = "\n\n".join(paragraph.replace('\n', ' ') for paragraph in paragraphs)

        # Format header
        header = f"📰 {title}\nSrc: {source} ({published})"

        # Combine
        full_text = f"{header}\n{article_body}"

        # Check token count and truncate if necessary
        current_count = self.token_counter.count_tokens(full_text)

        if current_count <= max_tokens:
            return full_text

        # Truncation needed
        header_tokens = self.token_counter.count_tokens(header)
        body_token_budget = max(1, max_tokens - header_tokens)
        max_chars = body_token_budget * 4
        truncated = article_body[:max_chars].rstrip()

        # Backtrack to a paragraph or sentence boundary when possible.
        last_paragraph = truncated.rfind('\n\n')
        last_period = truncated.rfind('.')
        if last_period > len(truncated) * 0.5:
            truncated = truncated[:last_period + 1]
        elif last_paragraph > len(truncated) * 0.5:
            truncated = truncated[:last_paragraph]

        candidate = f"{header}\n{truncated}..."
        while self.token_counter.count_tokens(candidate) > max_tokens and len(truncated) > 200:
            truncated = truncated[: int(len(truncated) * 0.9)].rstrip()
            candidate = f"{header}\n{truncated}..."

        return candidate

    def add_articles_to_context(
        self,
        relevant_indices: List[int],
        news_database: List[Dict],
        max_tokens: int,
        k: int,
        _keywords: set,
        scores_dict: Dict[int, float]
    ) -> Tuple[str, int]:
        """
        Bridge method for rag_engine compatibility.
        Converts article indices to article dicts and builds context.

        Args:
            relevant_indices: List of article indices from news_database
            news_database: Full news database
            max_tokens: Maximum tokens for context
            k: Number of top articles to include
            keywords: Keywords for relevance (unused in simple approach)
            scores_dict: Relevance scores (used for sorting)

        Returns:
            Tuple of (context_text, total_tokens)
        """
        # Score-sort a wider candidate pool, then prefer full-body items first.
        pool_limit = max(k * 10, 50)
        candidate_sorted = sorted(
            relevant_indices[:pool_limit],
            key=lambda idx: scores_dict.get(idx, 0),
            reverse=True
        )

        min_body_chars = int(getattr(self.config, 'RAG_NEWS_ENRICH_MIN_CHARS', 400)) if self.config else 400
        full_body = [
            idx for idx in candidate_sorted
            if idx < len(news_database) and len(str(news_database[idx].get('body', ''))) >= min_body_chars
        ]
        short_body = [idx for idx in candidate_sorted if idx not in full_body]
        sorted_indices = (full_body + short_body)[:k]

        # Convert indices to article dicts
        articles = [news_database[idx] for idx in sorted_indices if idx < len(news_database)]

        # Build context using simple method
        context_text = self.build_context(articles, max_tokens)

        # Calculate actual token count
        total_tokens = self.token_counter.count_tokens(context_text)

        return context_text, total_tokens

    def get_latest_article_urls(self) -> Dict[str, str]:
        """Get the latest article URLs from the last context build."""
        return self.latest_article_urls.copy()
