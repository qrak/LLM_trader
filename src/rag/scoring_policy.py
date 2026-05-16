"""
Scoring policy for RAG article relevance.

This module isolates ranking heuristics from ContextBuilder orchestration.
"""
from __future__ import annotations
import math
import re
from typing import Any, Set


FIELD_WEIGHTS = {"title": 10, "body": 3, "categories": 5, "tags": 4}


class ArticleScoringPolicy:
    """Encapsulates article relevance scoring heuristics."""

    def __init__(self, config: Any | None = None):
        self.config = config

    def calculate_article_relevance(
        self,
        article: dict[str, Any],
        content: Any,
        keywords: Set[str],
        coin: str | None,
        current_time: float,
        relevant_categories: list[str],
        important_categories: Set[str],
        pub_time: float,
        coin_patterns: dict[str, Any] | None = None,
    ) -> float:
        """Calculate article relevance score based on current scoring policy."""
        keyword_score = self.calculate_keyword_score(keywords, content)
        category_score = self.calculate_category_score(relevant_categories, content.categories)
        coin_score = self.calculate_coin_score(coin, content, coin_patterns) if coin else 0.0
        importance_score = self.calculate_importance_score(content.categories, important_categories)

        recency = self.calculate_recency_factor(current_time, pub_time)
        base_score = keyword_score + category_score + coin_score + importance_score

        density_mult = self.calculate_density_modifier(article)
        cooc_mult = self.calculate_cooccurrence_modifier(keywords, content)
        coin_relevance_mult = self._calculate_coin_relevance_multiplier(article, coin, coin_score, content, coin_patterns)

        final_score = base_score * density_mult * cooc_mult * coin_relevance_mult * (0.3 + 0.7 * recency)

        if article.get("id") == "market_overview":
            final_score += 10

        return final_score

    def calculate_keyword_score(self, keywords: Set[str], content: Any) -> float:
        """Calculate score based on keyword frequency with log-normal smoothing."""
        score = 0.0
        content_dict = content._asdict()
        for keyword in keywords:
            for field, weight in FIELD_WEIGHTS.items():
                count = content_dict.get(field, "").count(keyword)
                if count > 0:
                    score += weight * (1 + math.log(1 + count))
        return score

    @staticmethod
    def calculate_category_score(relevant_categories: list[str], article_categories: str) -> float:
        """Calculate score based on pre-filtered category list."""
        score = 0.0
        for category in relevant_categories:
            if category in article_categories:
                score += 5
        return score

    @staticmethod
    def calculate_coin_score(coin: str, content: Any, coin_patterns: dict[str, Any] | None = None) -> float:
        """Calculate score based on coin-specific matches using word boundaries."""
        coin_lower = coin.lower()
        score = 0.0

        if coin_lower in content.categories:
            score += 15

        if coin_patterns:
            if coin_patterns["coin_pattern"].search(content.title):
                score += 15
            if coin_patterns["coin_pattern"].search(content.body):
                score += 5
            name_pat = coin_patterns.get("coin_name_pattern")
            if name_pat:
                if name_pat.search(content.title):
                    score += 15
                if name_pat.search(content.body):
                    score += 5
        else:
            coin_pattern = rf"\b{re.escape(coin_lower)}\b"
            if re.search(coin_pattern, content.title):
                score += 15
            if re.search(coin_pattern, content.body):
                score += 5

        if coin_lower in content.detected_coins:
            score += 8

        if coin_patterns:
            if coin_patterns["title_start_pattern"].search(content.title) or coin_patterns["price_pattern"].search(content.title):
                score += 20
        else:
            title_start_pattern = rf"^\s*{re.escape(coin_lower)}\b"
            price_pattern = rf"\b{re.escape(coin_lower)}\s+price\b"
            if re.search(title_start_pattern, content.title) or re.search(price_pattern, content.title):
                score += 20

        return score

    @staticmethod
    def calculate_importance_score(categories: str, important_categories: Set[str]) -> float:
        """Calculate score based on important categories."""
        score = 0.0
        for category in important_categories:
            if category.lower() in categories:
                score += 3
        return score

    def calculate_density_modifier(self, article: dict[str, Any]) -> float:
        """Calculate score multiplier based on article body length (content quality)."""
        if not self.config:
            return 1.0
        body_len = len(article.get("body", ""))
        if body_len < self.config.RAG_DENSITY_PENALTY_THRESHOLD:
            return self.config.RAG_DENSITY_PENALTY_MULTIPLIER
        if body_len > self.config.RAG_DENSITY_BOOST_THRESHOLD:
            return self.config.RAG_DENSITY_BOOST_MULTIPLIER
        return 1.0

    def calculate_cooccurrence_modifier(self, keywords: Set[str], content: Any) -> float:
        """Calculate score multiplier when all query keywords appear in article."""
        if not self.config or len(keywords) < 2:
            return 1.0
        combined = content.title + " " + content.body
        if all(kw in combined for kw in keywords):
            return self.config.RAG_COOCCURRENCE_MULTIPLIER
        return 1.0

    @staticmethod
    def calculate_recency_factor(current_time: float, pub_time: float) -> float:
        """Calculate recency weighting factor."""
        time_diff = current_time - pub_time
        return max(0.0, 1.0 - (time_diff / (24 * 3600)))

    @staticmethod
    def _calculate_coin_relevance_multiplier(
        article: dict[str, Any],
        coin: str | None,
        coin_score: float,
        content: Any,
        coin_patterns: dict[str, Any] | None,
    ) -> float:
        """Demote low-coin-relevance items for symbol-specific analysis."""
        if not coin or article.get("id") == "market_overview":
            return 1.0
        if coin_score == 0:
            return 0.1

        name_pat = coin_patterns.get("coin_name_pattern") if coin_patterns else None
        coin_in_title = bool(
            coin_patterns
            and (
                coin_patterns["coin_pattern"].search(content.title)
                or coin_patterns["title_start_pattern"].search(content.title)
                or (name_pat and name_pat.search(content.title))
            )
        )
        if not coin_in_title:
            return 0.35
        return 1.0
