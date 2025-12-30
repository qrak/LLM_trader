"""
Context Building Module for RAG Engine

Handles building analysis context from news articles and search results.
"""

import re
from datetime import datetime
from collections import namedtuple
from typing import List, Dict, Any, Optional, Tuple, Set
from src.logger.logger import Logger
from src.utils.profiler import profile_performance
from src.utils.token_counter import TokenCounter
from .text_splitting import SentenceSplitter


class ContextBuilder:
    """Builds analysis context from various data sources."""
    
    def __init__(self, logger: Logger, token_counter: TokenCounter, article_processor=None, sentence_splitter: Optional[SentenceSplitter] = None):
        if article_processor is None:
            raise ValueError("article_processor is required - must be injected from app.py")
        self.logger = logger
        self.config = None  # Will be set via setter or initialization update
        self.token_counter = token_counter
        self.article_processor = article_processor
        self.latest_article_urls: Dict[str, str] = {}
        
        # Initialize shared sentence splitter
        self.sentence_splitter = sentence_splitter
    
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
        if symbol:
            coin = self._extract_base_coin(symbol).upper()

        scores: List[Tuple[int, float]] = []
        current_time = datetime.now().timestamp()

        for i, article in enumerate(news_database):
            score = self._calculate_article_relevance(
                article, keywords, query, coin, current_time,
                category_word_map, important_categories
            )
            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def _calculate_article_relevance(self, article: Dict[str, Any], keywords: Set[str],
                                   query: str, coin: Optional[str],
                                   current_time: float, category_word_map: Dict[str, str],
                                   important_categories: Set[str]) -> float:
        """Calculate article relevance score based on various factors."""
        # Extract article content
        content = self._extract_article_content(article)
        
        # Calculate base scores
        keyword_score = self._calculate_keyword_score(keywords, content)
        category_score = self._calculate_category_score(query, content.categories, category_word_map)
        coin_score = self._calculate_coin_score(coin, content) if coin else 0.0
        importance_score = self._calculate_importance_score(content.categories, important_categories)
        
        # Apply recency weighting
        pub_time = self._get_article_timestamp(article)
        recency = self._calculate_recency_factor(current_time, pub_time)
        
        base_score = keyword_score + category_score + coin_score + importance_score
        final_score = base_score * (0.3 + 0.7 * recency)
        
        # Special boost for market overview
        if article.get('id') == 'market_overview':
            final_score += 10
        
        return final_score
    
    def _extract_article_content(self, article: Dict[str, Any]):
        """Extract and normalize article content for scoring."""
        ArticleContent = namedtuple('ArticleContent', ['title', 'body', 'categories', 'tags', 'detected_coins'])
        
        return ArticleContent(
            title=article.get('title', '').lower(),
            body=article.get('body', '').lower(),
            categories=article.get('categories', '').lower(),
            tags=article.get('tags', '').lower(),
            detected_coins=article.get('detected_coins_str', '').lower()
        )
    
    def _calculate_keyword_score(self, keywords: Set[str], content) -> float:
        """Calculate score based on keyword matches."""
        score = 0.0
        for keyword in keywords:
            if keyword in content.title:
                score += 10
            if keyword in content.body:
                score += 3
            if keyword in content.categories:
                score += 5
            if keyword in content.tags:
                score += 4
        return score
    
    def _calculate_category_score(self, query: str, categories: str, category_word_map: Dict[str, str]) -> float:
        """Calculate score based on category word mapping."""
        score = 0.0
        for word, category in category_word_map.items():
            if word in query and category.lower() in categories:
                score += 5
        return score
    
    def _calculate_coin_score(self, coin: str, content) -> float:
        """Calculate score based on coin-specific matches using word boundaries."""
        coin_lower = coin.lower()
        score = 0.0
        
        # Category matches
        if coin_lower in content.categories:
            score += 15
        
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
    
    def _calculate_recency_factor(self, current_time: float, pub_time: float) -> float:
        """Calculate recency weighting factor."""
        time_diff = current_time - pub_time
        return max(0.0, 1.0 - (time_diff / (24 * 3600)))

    def _get_article_timestamp(self, article: Dict[str, Any]) -> float:
        """Extract timestamp from article in a consistent format."""
        return self.article_processor.get_article_timestamp(article)
    
    def _extract_base_coin(self, symbol: str) -> str:
        """Extract base coin from trading pair symbol."""
        return self.article_processor.extract_base_coin(symbol)
    
    @profile_performance
    def add_articles_to_context(self, indices: List[int], news_database: List[Dict[str, Any]],
                               max_tokens: int, k: int, keywords: Optional[Set[str]] = None) -> Tuple[str, int]:
        """Add articles to context with token limiting.
        
        Args:
            indices: List of article indices to consider
            news_database: Full news database
            max_tokens: Maximum tokens for the entire context
            k: Maximum number of articles to include
            keywords: Optional set of keywords for smart sentence selection
        """
        context_parts = []
        total_tokens = 0
        articles_added = 0
        self.latest_article_urls = {}
        
        for idx in indices:
            if idx >= len(news_database) or articles_added >= k:
                break

            article = news_database[idx]
            article_text, article_tokens = self._format_single_article(article, keywords)
            
            if total_tokens + article_tokens <= max_tokens:
                context_parts.append(article_text)
                total_tokens += article_tokens
                articles_added += 1
                
                # Store URL for reference
                if 'url' in article:
                    title = article.get('title', 'No Title')
                    self.latest_article_urls[title] = article['url']
            else:
                break

        return "".join(context_parts), total_tokens
    
    def _format_single_article(self, article: Dict[str, Any], keywords: Optional[Set[str]] = None) -> Tuple[str, int]:
        """Format a single article for context inclusion (compressed format for token efficiency).
        
        Args:
            article: Article dictionary
            keywords: Optional set of keywords for smart sentence selection
        """
        published_date = self._format_article_date(article)
        title = article.get('title', 'No Title')
        source = article.get('source', 'Unknown Source')

        # Compressed format: headline (source, date) - key facts
        article_text = f"## {title}\n**Source:** {source} | **Date:** {published_date}\n\n"

        body = article.get('body', '')
        if body:
            # Clean text and split into sentences
            cleaned_body = body.replace('\n\n', ' ').replace('\n', ' ').strip()
            sentence_list = self.sentence_splitter.split_text(cleaned_body)
            
            # Apply configured limits
            max_sentences = 3
            max_article_tokens = 200
            
            if self.config:
                try:
                    max_sentences = int(self.config.RAG_ARTICLE_MAX_SENTENCES)
                    max_article_tokens = int(self.config.RAG_ARTICLE_MAX_TOKENS)
                except (AttributeError, ValueError):
                    pass  # Use defaults
            
            # Smart sentence selection: score by keyword matches if keywords provided
            if keywords and sentence_list:
                selected_sentences = self._select_relevant_sentences(
                    sentence_list, keywords, max_sentences
                )
            else:
                # Fallback: take first N sentences
                selected_sentences = sentence_list[:max_sentences]
            
            key_facts = ""
            current_tokens = self.token_counter.count_tokens(article_text)
            
            for sent in selected_sentences:
                sent_tokens = self.token_counter.count_tokens(sent + " ")
                
                if current_tokens + sent_tokens > max_article_tokens:
                    break
                    
                key_facts += sent + " "
                current_tokens += sent_tokens
                
            if key_facts:
                article_text += f"{key_facts}\n\n"

        categories = article.get('categories', '')
        tags = article.get('tags', '')
        if categories or tags:
            article_text += f"**Topics:** {categories} | {tags}\n\n"

        article_tokens = self.token_counter.count_tokens(article_text)
        return article_text, article_tokens
    
    def _select_relevant_sentences(self, sentences: List[str], keywords: Set[str], max_count: int) -> List[str]:
        """Select the most relevant sentences based on keyword matches.
        
        Returns sentences sorted by their original order to preserve narrative flow.
        
        Args:
            sentences: List of sentences from the article
            keywords: Set of keywords to match against
            max_count: Maximum number of sentences to return
        """
        if not sentences or not keywords:
            return sentences[:max_count]
        
        # Score each sentence: (original_index, score, sentence)
        scored = []
        for i, sent in enumerate(sentences):
            sent_lower = sent.lower()
            kw_count = sum(1 for kw in keywords if kw in sent_lower)
            score = float(kw_count)
            
            # Boost for numbers (likely data points)
            # Count distinct number groups (e.g. 49.34, 50, 2025)
            number_groups = len(re.findall(r'\d+(?:\.\d+)?', sent))
            if number_groups > 0:
                score += (number_groups * 0.5)
            
            # Boost for percentage signs (key metrics)
            percent_count = sent.count('%')
            if percent_count > 0:
                score += (percent_count * 2.0)
            
            # Boost for currency symbols (prices)
            if '$' in sent:
                score += 2.0

            # Boost for "Key: Value" patterns (often used in data reporting)
            if ':' in sent:
                score += 1.5

            # SYNERGY BOOST: If sentence has both a keyword match AND a number/currency, it is highly relevant
            # This filters out random financial noise (like "Source: ...") that doesn't match the query
            if kw_count > 0 and ('$' in sent or percent_count > 0):
                score += 2.0
                
            scored.append((i, score, sent))
        
        # Sort by score descending, take top N
        scored.sort(key=lambda x: x[1], reverse=True)
        top_sentences = scored[:max_count]
        
        # Re-sort by original index to preserve narrative order
        top_sentences.sort(key=lambda x: x[0])
        
        return [sent for _, _, sent in top_sentences]

    def _format_article_date(self, article: Dict[str, Any]) -> str:
        """Format article date in a consistent way."""
        return self.article_processor.format_article_date(article)
    
    def get_latest_article_urls(self) -> Dict[str, str]:
        """Get the latest article URLs from the last context build."""
        return self.latest_article_urls.copy()
