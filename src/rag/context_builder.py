"""
Context Building Module for RAG Engine

Handles building analysis context from news articles and search results.
"""

import math
import re
from datetime import datetime
from collections import namedtuple
from typing import List, Dict, Any, Optional, Tuple, Set
from src.logger.logger import Logger
from src.utils.profiler import profile_performance
from src.utils.token_counter import TokenCounter

ArticleContent = namedtuple('ArticleContent', ['title', 'body', 'categories', 'tags', 'detected_coins'])
FIELD_WEIGHTS = {"title": 10, "body": 3, "categories": 5, "tags": 4}

class ContextBuilder:
    """Builds analysis context from various data sources."""

    def __init__(self, logger: Logger, token_counter: TokenCounter, config=None, article_processor=None):
        self.logger = logger
        self.config = config
        self.token_counter = token_counter
        self.article_processor = article_processor
        self.latest_article_urls: Dict[str, str] = {}

    @profile_performance
    async def keyword_search(self, query: str, news_database: List[Dict[str, Any]],
                           symbol: Optional[str] = None, coin_index: Dict[str, List[int]] = None,
                           category_word_map: Dict[str, str] = None,
                           important_categories: Set[str] = None) -> List[Tuple[int, float]]:
        """Search for articles matching keywords with relevance scores."""
        # Provide default empty containers to avoid mutable defaults
        coin_index = coin_index or {}
        category_word_map = category_word_map or {}
        important_categories = important_categories or set()

        query = query.lower()
        keywords = set(re.findall(r'\b\w{3,15}\b', query))

        coin = None
        coin_patterns = None
        if symbol:
            coin = self.article_processor.extract_base_coin(symbol).upper()
            coin_lower = coin.lower()
            coin_patterns = {
                'coin_pattern': re.compile(rf'\b{re.escape(coin_lower)}\b'),
                'title_start_pattern': re.compile(rf'^\s*{re.escape(coin_lower)}\b'),
                'price_pattern': re.compile(rf'\b{re.escape(coin_lower)}\s+price\b')
            }

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
        # Extract article content
        content = self._extract_article_content(article)

        # Calculate base scores
        keyword_score = self._calculate_keyword_score(keywords, content)
        category_score = self._calculate_category_score(relevant_categories, content.categories)
        coin_score = self._calculate_coin_score(coin, content, coin_patterns) if coin else 0.0
        importance_score = self._calculate_importance_score(content.categories, important_categories)

        # Apply recency weighting
        pub_time = self.article_processor.get_article_timestamp(article)
        recency = self._calculate_recency_factor(current_time, pub_time)

        base_score = keyword_score + category_score + coin_score + importance_score

        # Apply smart heuristic modifiers
        density_mult = self._calculate_density_modifier(article)
        cooc_mult = self._calculate_cooccurrence_modifier(keywords, content)

        final_score = base_score * density_mult * cooc_mult * (0.3 + 0.7 * recency)

        # Special boost for market overview
        if article.get('id') == 'market_overview':
            final_score += 10

        return final_score

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

    def _calculate_keyword_score(self, keywords: Set[str], content) -> float:
        """Calculate score based on keyword frequency with log-normal smoothing."""
        score = 0.0
        for keyword in keywords:
            content_dict = content._asdict()
            for field, weight in FIELD_WEIGHTS.items():
                count = content_dict.get(field, "").count(keyword)
                if count > 0:
                    score += weight * (1 + math.log(1 + count))
        return score

    def _calculate_category_score(self, relevant_categories: List[str], article_categories: str) -> float:
        """Calculate score based on pre-filtered category list."""
        score = 0.0
        for category in relevant_categories:
            if category in article_categories:
                score += 5
        return score

    def _calculate_coin_score(self, coin: str, content, coin_patterns: Optional[Dict[str, Any]] = None) -> float:
        """Calculate score based on coin-specific matches using word boundaries."""
        coin_lower = coin.lower()
        score = 0.0

        # Category matches
        if coin_lower in content.categories:
            score += 15

        if coin_patterns:
            if coin_patterns['coin_pattern'].search(content.title):
                score += 15
            if coin_patterns['coin_pattern'].search(content.body):
                score += 5
        else:
            # Title/body word-boundary regex matches
            coin_pattern = rf'\b{re.escape(coin_lower)}\b'
            if re.search(coin_pattern, content.title):
                score += 15
            if re.search(coin_pattern, content.body):
                score += 5

        # Detected coins
        if coin_lower in content.detected_coins:
            score += 8

        # Special title patterns with word boundaries
        if coin_patterns:
            if coin_patterns['title_start_pattern'].search(content.title) or \
               coin_patterns['price_pattern'].search(content.title):
                score += 20
        else:
            title_start_pattern = rf'^\s*{re.escape(coin_lower)}\b'
            price_pattern = rf'\b{re.escape(coin_lower)}\s+price\b'

            if re.search(title_start_pattern, content.title) or re.search(price_pattern, content.title):
                score += 20

        return score

    def _calculate_importance_score(self, categories: str, important_categories: Set[str]) -> float:
        """Calculate score based on important categories."""
        score = 0.0
        for category in important_categories:
            if category.lower() in categories:
                score += 3
        return score

    def _calculate_density_modifier(self, article: Dict[str, Any]) -> float:
        """Calculate score multiplier based on article body length (content quality)."""
        if not self.config:
            return 1.0
        body_len = len(article.get('body', ''))
        if body_len < self.config.RAG_DENSITY_PENALTY_THRESHOLD:
            return self.config.RAG_DENSITY_PENALTY_MULTIPLIER
        if body_len > self.config.RAG_DENSITY_BOOST_THRESHOLD:
            return self.config.RAG_DENSITY_BOOST_MULTIPLIER
        return 1.0

    def _calculate_cooccurrence_modifier(self, keywords: Set[str], content) -> float:
        """Calculate score multiplier when all query keywords appear in article."""
        if not self.config or len(keywords) < 2:
            return 1.0
        combined = content.title + " " + content.body
        if all(kw in combined for kw in keywords):
            return self.config.RAG_COOCCURRENCE_MULTIPLIER
        return 1.0

    def _calculate_recency_factor(self, current_time: float, pub_time: float) -> float:
        """Calculate recency weighting factor."""
        time_diff = current_time - pub_time
        return max(0.0, 1.0 - (time_diff / (24 * 3600)))

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

        # Calculate tokens per article to ensure fair distribution
        # Reserve some tokens for headers/separators
        tokens_per_article = max(200, (max_tokens - 100) // len(news_items))

        for item in news_items:
            # Stop if we're close to the total limit
            if current_tokens >= max_tokens:
                break

            processed_text = self._process_article_simple(item, tokens_per_article)

            if processed_text:
                token_count = self.token_counter.count_tokens(processed_text)

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
        Process a single article: Title + Lead Paragraph (truncated).

        Args:
            item: News item dictionary
            max_tokens: Max tokens for this article chunk

        Returns:
            Formatted string: "Title\nSource (Date)\nLead Paragraph..."
        """
        title = item.get('title', 'No Title').strip()
        source = item.get('source', 'Unknown Source')
        published_on = item.get('published_on', 0)
        published = datetime.fromtimestamp(published_on).strftime('%Y-%m-%d %H:%M UTC')

        # Clean up body: normalize whitespace but keep structure for a moment
        body = item.get('body', '')
        if not body:
            return ""

        # Extract Lead Paragraph (content before first double newline)
        paragraphs = [p.strip() for p in body.split('\n\n') if p.strip()]

        if not paragraphs:
            return ""

        # Start with just the first paragraph (Lead)
        lead_paragraph = paragraphs[0].replace('\n', ' ')

        # Format header
        header = f"📰 {title}\nSrc: {source} ({published})"

        # Combine
        full_text = f"{header}\n{lead_paragraph}"

        # Check token count and truncate if necessary
        current_count = self.token_counter.count_tokens(full_text)

        if current_count <= max_tokens:
            return full_text

        # Truncation needed
        # Approx chars (1 token ~= 4 chars)
        max_chars = max_tokens * 4
        truncated = lead_paragraph[:max_chars]

        # Backtrack to last period
        last_period = truncated.rfind('.')
        if last_period > len(truncated) * 0.5:
             truncated = truncated[:last_period+1]

        return f"{header}\n{truncated}..."

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
        # Take top k articles by score
        sorted_indices = sorted(
            relevant_indices[:k*2],
            key=lambda idx: scores_dict.get(idx, 0),
            reverse=True
        )[:k]

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
