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
from src.rag.article_processor import ArticleProcessor
from src.rag.news_ingestion.schema_mapper import normalize_article_whitespace
from src.rag.scoring_policy import ArticleScoringPolicy
from src.utils.profiler import profile_performance
from src.utils.token_counter import TokenCounter

ArticleContent = namedtuple('ArticleContent', ['title', 'body', 'categories', 'tags', 'detected_coins'])
<<<<<<< HEAD
FIELD_WEIGHTS = {"title": 10, "body": 3, "categories": 5, "tags": 4}
=======
>>>>>>> main

class ContextBuilder:
    """Builds analysis context from various data sources."""

<<<<<<< HEAD
    def __init__(self, logger: Logger, token_counter: TokenCounter, config=None, article_processor=None):
=======
    def __init__(
        self,
        logger: Logger,
        token_counter: TokenCounter,
        config=None,
        article_processor=None,
        symbol_name_map: Optional[Dict[str, str]] = None,
    ):
>>>>>>> main
        self.logger = logger
        self.config = config
        self.token_counter = token_counter
        self.article_processor = article_processor
        self.latest_article_urls: Dict[str, str] = {}
<<<<<<< HEAD

    @profile_performance
    async def keyword_search(self, query: str, news_database: List[Dict[str, Any]],
                           symbol: Optional[str] = None, coin_index: Dict[str, List[int]] = None,
                           category_word_map: Dict[str, str] = None,
                           important_categories: Set[str] = None) -> List[Tuple[int, float]]:
=======
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
>>>>>>> main
        """Search for articles matching keywords with relevance scores."""
        # Provide default empty containers to avoid mutable defaults
        coin_index = coin_index or {}
        category_word_map = category_word_map or {}
        important_categories = important_categories or set()

        query = query.lower()
        keywords = set(re.findall(r'\b\w{3,15}\b', query))

        coin = None
<<<<<<< HEAD
        coin_patterns = None
        if symbol:
=======
        coin_patterns: Optional[Dict[str, Any]] = None
        if symbol and self.article_processor:
>>>>>>> main
            coin = self.article_processor.extract_base_coin(symbol).upper()
            coin_lower = coin.lower()
            coin_patterns = {
                'coin_pattern': re.compile(rf'\b{re.escape(coin_lower)}\b'),
                'title_start_pattern': re.compile(rf'^\s*{re.escape(coin_lower)}\b'),
                'price_pattern': re.compile(rf'\b{re.escape(coin_lower)}\s+price\b')
            }
<<<<<<< HEAD
=======
            coin_full_name = self.symbol_name_map.get(coin)
            coin_patterns['coin_name_pattern'] = (
                re.compile(rf"\b{re.escape(coin_full_name).replace('\\ ', r'[-\\s]+')}\b") if coin_full_name else None
            )
>>>>>>> main

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
<<<<<<< HEAD

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

        # Demote articles with zero coin relevance when analyzing a specific symbol
        coin_relevance_mult = 1.0
        if coin and coin_score == 0 and article.get('id') != 'market_overview':
            coin_relevance_mult = 0.3

        final_score = base_score * density_mult * cooc_mult * coin_relevance_mult * (0.3 + 0.7 * recency)

        # Special boost for market overview
        if article.get('id') == 'market_overview':
            final_score += 10

        return final_score
=======
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
>>>>>>> main

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

<<<<<<< HEAD
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

=======
>>>>>>> main
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

<<<<<<< HEAD
        # Calculate tokens per article to ensure fair distribution
        # Reserve some tokens for headers/separators
        tokens_per_article = max(200, (max_tokens - 100) // len(news_items))

        for item in news_items:
            # Stop if we're close to the total limit
            if current_tokens >= max_tokens:
                break

            processed_text = self._process_article_simple(item, tokens_per_article)
=======
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
>>>>>>> main

            if processed_text:
                token_count = self.token_counter.count_tokens(processed_text)

<<<<<<< HEAD
=======
                # Skip this article if including it would exceed total context limit
>>>>>>> main
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
<<<<<<< HEAD
        Process a single article: Title + Lead Paragraph (truncated).
=======
        Process a single article: Title + article body (truncated only by budget).
>>>>>>> main

        Args:
            item: News item dictionary
            max_tokens: Max tokens for this article chunk

        Returns:
<<<<<<< HEAD
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
=======
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
>>>>>>> main

        if not paragraphs:
            return ""

<<<<<<< HEAD
        # Start with just the first paragraph (Lead)
        lead_paragraph = paragraphs[0].replace('\n', ' ')
=======
        article_body = "\n\n".join(paragraph.replace('\n', ' ') for paragraph in paragraphs)
>>>>>>> main

        # Format header
        header = f"📰 {title}\nSrc: {source} ({published})"

        # Combine
<<<<<<< HEAD
        full_text = f"{header}\n{lead_paragraph}"
=======
        full_text = f"{header}\n{article_body}"
>>>>>>> main

        # Check token count and truncate if necessary
        current_count = self.token_counter.count_tokens(full_text)

        if current_count <= max_tokens:
            return full_text

        # Truncation needed
<<<<<<< HEAD
        # Approx chars (1 token ~= 4 chars)
        max_chars = max_tokens * 4
        truncated = lead_paragraph[:max_chars]

        # Backtrack to last period
        last_period = truncated.rfind('.')
        if last_period > len(truncated) * 0.5:
             truncated = truncated[:last_period+1]

        return f"{header}\n{truncated}..."
=======
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
>>>>>>> main

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
<<<<<<< HEAD
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

=======
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

>>>>>>> main
    def get_latest_article_urls(self) -> Dict[str, str]:
        """Get the latest article URLs from the last context build."""
        return self.latest_article_urls.copy()
