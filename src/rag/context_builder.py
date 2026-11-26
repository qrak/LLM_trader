
from __future__ import annotations

import logging
import re
from collections import namedtuple
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from src.logger.logger import Logger
from src.rag.article_processor import ArticleProcessor
from src.rag.news_ingestion.schema_mapper import strip_news_boilerplate
from src.rag.scoring_policy import ArticleScore, ArticleScoringPolicy
from src.utils.profiler import profile_performance
from src.utils.token_counter import TokenCounter

if TYPE_CHECKING:
    from src.config.loader import Config

ArticleContent = namedtuple("ArticleContent", ["title", "body", "categories", "tags", "detected_coins"])

class ContextBuilder:
    def __init__(
        self,
        logger: Logger,
        token_counter: TokenCounter,
        config: Config,
        scoring_policy: ArticleScoringPolicy,
        article_processor: ArticleProcessor,
        symbol_name_map: dict[str, str] | None = None,
    ):
        self.logger = logger
        self.config = config
        self.token_counter = token_counter
        self.scoring_policy = scoring_policy
        self.article_processor = article_processor
        self.latest_article_urls: dict[str, str] = {}
        self._rendered_positions: list[int] = []
        self._last_selection: tuple[list[int], list[int]] = ([], [])
        self.symbol_name_map = {
            str(symbol).upper(): ArticleProcessor._normalize_coin_name(str(name))
            for symbol, name in (symbol_name_map or {}).items()
            if symbol and name
        }

    @profile_performance
    async def keyword_search(self, query: str, news_database: list[dict[str, Any]],
                           symbol: str | None = None,
                           category_word_map: dict[str, str] | None = None,
                           important_categories: set[str] | None = None
                           ) -> tuple[list[tuple[int, float]], dict[int, ArticleScore]]:

        category_word_map = category_word_map or {}
        important_categories = important_categories or set()

        query = query.lower()
        keywords = set(re.findall(r"\b\w{3,15}\b", query))

        coin = None
        coin_patterns: dict[str, Any] | None = None
        if symbol:
            coin = self.article_processor.extract_base_coin(symbol).upper()
            coin_lower = coin.lower()
            coin_patterns = {
                "coin_pattern": re.compile(rf"\b{re.escape(coin_lower)}\b"),
                "title_start_pattern": re.compile(rf"^\s*{re.escape(coin_lower)}\b"),
                "price_pattern": re.compile(rf"\b{re.escape(coin_lower)}\s+price\b")
            }
            coin_full_name = self.symbol_name_map.get(coin)
            coin_patterns["coin_name_pattern"] = (
                re.compile(r"\b" + r"[-\s]+".join(re.escape(p) for p in coin_full_name.split()) + r"\b") if coin_full_name else None
            )

        relevant_categories = []
        if category_word_map:
            for word, category in category_word_map.items():
                if word in query:
                    relevant_categories.append(category.lower())

        scores: list[tuple[int, float]] = []
        breakdowns: dict[int, ArticleScore] = {}
        trace = self.logger.isEnabledFor(logging.DEBUG)
        current_time = datetime.now(timezone.utc).timestamp()

        for i, article in enumerate(news_database):
            content = self._extract_article_content(article)
            score = self.scoring_policy.calculate_article_relevance(
                article=article,
                content=content,
                keywords=keywords,
                coin=coin,
                current_time=current_time,
                relevant_categories=relevant_categories,
                important_categories=important_categories,
                pub_time=self.article_processor.get_article_timestamp(article),
                coin_patterns=coin_patterns,
            )
            if trace:
                breakdowns[i] = score
            if score.score > 0:
                scores.append((i, score.score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores, breakdowns

    def _extract_article_content(self, article: dict[str, Any]):

        return ArticleContent(
            title=article.get("title", "").lower(),
            body=article.get("body", "").lower(),
            categories=article.get("categories", "").lower(),
            tags=article.get("tags", "").lower(),
            detected_coins=article.get("detected_coins_str", "").lower()
        )

    @profile_performance
    def build_context(self, news_items: list[dict], max_tokens: int | None = None) -> str:

        self._rendered_positions = []
        if not news_items:
            return ""

        self.latest_article_urls.clear()
        context_parts = []
        current_tokens = 0

        article_max_tokens = self.config.RAG_ARTICLE_MAX_TOKENS
        resolved_max_tokens = max_tokens
        if resolved_max_tokens is None:
            resolved_max_tokens = article_max_tokens * self.config.RAG_NEWS_LIMIT

        for position, item in enumerate(news_items):
            if current_tokens >= resolved_max_tokens:
                break

            processed_text = self._process_article_simple(item, article_max_tokens)

            if processed_text:
                token_count = self.token_counter.count_tokens(processed_text)

                if current_tokens + token_count > resolved_max_tokens:
                    break

                context_parts.append(processed_text)
                self._rendered_positions.append(position)
                current_tokens += token_count

                if "url" in item:
                    title = item.get("title", "Untitled")
                    self.latest_article_urls[title] = item["url"]

        return "\n\n".join(context_parts)

    @staticmethod
    def _last_sentence_boundary(text: str) -> int | None:

        boundary = None
        for match in re.finditer(r"[.!?](?=\s|$)", text):
            if match.start() > len(text) * 0.5:
                boundary = match.end()
        return boundary

    def _process_article_simple(self, item: dict, max_tokens: int) -> str:

        title = item.get("title", "No Title")
        title = str(title).strip() if title is not None else "No Title"
        source = item.get("source_info", {"name": "Unknown Source"}).get("name", "Unknown Source")
        published_on = item.get("published_on", 0)
        published = datetime.fromtimestamp(published_on, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        body = item.get("body", "").strip()
        if not body:
            return ""

        body = strip_news_boilerplate(body)
        if not body:
            return ""

        paragraphs = [paragraph.strip() for paragraph in body.split("\n\n") if paragraph.strip()]

        if not paragraphs:
            return ""

        article_body = "\n\n".join(paragraph.replace("\n", " ") for paragraph in paragraphs)

        header = f"📰 {title}\nSrc: {source} ({published})"

        full_text = f"{header}\n{article_body}"

        current_count = self.token_counter.count_tokens(full_text)

        if current_count <= max_tokens:
            return full_text

        header_tokens = self.token_counter.count_tokens(header)
        body_token_budget = max(1, max_tokens - header_tokens)
        max_chars = body_token_budget * 4
        truncated = article_body[:max_chars].rstrip()

        last_paragraph = truncated.rfind("\n\n")
        sentence_end = self._last_sentence_boundary(truncated)
        if sentence_end is not None:
            truncated = truncated[:sentence_end]
        elif last_paragraph > len(truncated) * 0.5:
            truncated = truncated[:last_paragraph]
        elif " " in truncated:
            truncated = truncated.rsplit(" ", 1)[0].rstrip(" ,;:")

        candidate = f"{header}\n{truncated}..."
        while self.token_counter.count_tokens(candidate) > max_tokens and len(truncated) > 200:
            truncated = truncated[: int(len(truncated) * 0.9)].rstrip()
            candidate = f"{header}\n{truncated}..."

        return candidate

    def add_articles_to_context(
        self,
        relevant_indices: list[int],
        news_database: list[dict],
        max_tokens: int,
        k: int,
        scores_dict: dict[int, float]
    ) -> tuple[str, int]:

        pool_limit = max(k * 10, 50)
        candidate_sorted = sorted(
            relevant_indices[:pool_limit],
            key=lambda idx: scores_dict.get(idx, 0),
            reverse=True
        )

        min_body_chars = self.config.RAG_NEWS_ENRICH_MIN_CHARS
        full_body = [
            idx for idx in candidate_sorted
            if idx < len(news_database) and len(str(news_database[idx].get("body", ""))) >= min_body_chars
        ]
        short_body = [idx for idx in candidate_sorted if idx not in full_body]
        sorted_indices = (full_body + short_body)[:k]

        present = [idx for idx in sorted_indices if idx < len(news_database)]
        articles = [news_database[idx] for idx in present]

        context_text = self.build_context(articles, max_tokens)
        self._last_selection = (
            list(sorted_indices),
            [present[position] for position in self._rendered_positions],
        )

        total_tokens = self.token_counter.count_tokens(context_text)

        return context_text, total_tokens

    def get_last_selection(self) -> tuple[list[int], list[int]]:

        shortlist, rendered = self._last_selection
        return list(shortlist), list(rendered)

    def get_latest_article_urls(self) -> dict[str, str]:

        return self.latest_article_urls.copy()

