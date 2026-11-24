
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.config.loader import Config

FIELD_WEIGHTS = {"title": 10, "body": 3, "categories": 5, "tags": 4}


class ArticleScoringPolicy:
    def __init__(self, config: Config):
        self.config = config

    def calculate_article_relevance(
        self,
        article: dict[str, Any],
        content: Any,
        keywords: set[str],
        coin: str | None,
        current_time: float,
        relevant_categories: list[str],
        important_categories: set[str],
        pub_time: float,
        coin_patterns: dict[str, Any] | None = None,
    ) -> float:

        keyword_score = self.calculate_keyword_score(keywords, content)
        category_score = self.calculate_category_score(relevant_categories, content.categories)
        importance_score = self.calculate_importance_score(content.categories, important_categories)

        if coin and coin_patterns:
            coin_score = self.calculate_coin_score(coin, content, coin_patterns)
            coin_relevance_mult = self._calculate_coin_relevance_multiplier(coin_score, content, coin_patterns)
        else:
            coin_score = 0.0
            coin_relevance_mult = 1.0

        recency = self.calculate_recency_factor(current_time, pub_time)
        base_score = keyword_score + category_score + coin_score + importance_score

        density_mult = self.calculate_density_modifier(article)
        source_mult = self._source_weight(article)

        final_score = base_score * density_mult * coin_relevance_mult * recency * source_mult

        return final_score

    def _source_weight(self, article: dict[str, Any]) -> float:

        name = str((article.get("source_info") or {}).get("name", "")).lower()
        return self.config.RAG_NEWS_SOURCE_WEIGHTS.get(name, 1.0)

    def calculate_keyword_score(self, keywords: set[str], content: Any) -> float:

        score = 0.0
        content_dict = content._asdict()
        for keyword in keywords:
            for field, weight in FIELD_WEIGHTS.items():
                count = content_dict.get(field, "").count(keyword)
                if count > 0:
                    score += weight * (1 + math.log(1 + count))
        return score * self._length_scale(content.body)

    @staticmethod
    def _length_scale(body: str) -> float:

        return min(1.0, math.sqrt(1000 / max(len(body), 1)))

    @staticmethod
    def calculate_category_score(relevant_categories: list[str], article_categories: str) -> float:

        score = 0.0
        for category in relevant_categories:
            if category in article_categories:
                score += 5
        return score

    @staticmethod
    def calculate_coin_score(coin: str, content: Any, coin_patterns: dict[str, Any]) -> float:

        coin_lower = coin.lower()
        score = 0.0

        if coin_lower in content.categories:
            score += 15

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

        if coin_lower in content.detected_coins:
            score += 8

        if coin_patterns["title_start_pattern"].search(content.title) or coin_patterns["price_pattern"].search(content.title):
            score += 20

        return score

    @staticmethod
    def calculate_importance_score(categories: str, important_categories: set[str]) -> float:

        score = 0.0
        for category in important_categories:
            if category.lower() in categories:
                score += 3
        return score

    def calculate_density_modifier(self, article: dict[str, Any]) -> float:

        body_len = len(article.get("body", ""))
        if body_len < self.config.RAG_DENSITY_PENALTY_THRESHOLD:
            return self.config.RAG_DENSITY_PENALTY_MULTIPLIER
        if body_len > self.config.RAG_DENSITY_BOOST_THRESHOLD:
            return self.config.RAG_DENSITY_BOOST_MULTIPLIER
        return 1.0

    @staticmethod
    def calculate_recency_factor(current_time: float, pub_time: float) -> float:

        time_diff = current_time - pub_time
        return min(1.0, max(0.0, 1.0 - (time_diff / (24 * 3600))))

    @staticmethod
    def _calculate_coin_relevance_multiplier(
        coin_score: float,
        content: Any,
        coin_patterns: dict[str, Any],
    ) -> float:

        if coin_score == 0:
            return 0.1

        name_pat = coin_patterns.get("coin_name_pattern")
        coin_in_title = bool(
            coin_patterns["coin_pattern"].search(content.title)
            or coin_patterns["title_start_pattern"].search(content.title)
            or (name_pat and name_pat.search(content.title))
        )
        if not coin_in_title:
            return 0.35
        return 1.0
