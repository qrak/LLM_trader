from collections import namedtuple
import re
from types import SimpleNamespace

from src.rag.scoring_policy import ArticleScoringPolicy


ArticleContent = namedtuple("ArticleContent", ["title", "body", "categories", "tags", "detected_coins"])


def _policy() -> ArticleScoringPolicy:
    config = SimpleNamespace(
        RAG_DENSITY_PENALTY_THRESHOLD=20,
        RAG_DENSITY_PENALTY_MULTIPLIER=0.7,
        RAG_DENSITY_BOOST_THRESHOLD=80,
        RAG_DENSITY_BOOST_MULTIPLIER=1.2,
        RAG_COOCCURRENCE_MULTIPLIER=1.3,
    )
    return ArticleScoringPolicy(config=config)


def test_market_overview_receives_bonus_boost():
    policy = _policy()
    current_time = 1000.0
    pub_time = 1000.0

    content = ArticleContent(
        title="market update",
        body="macro outlook",
        categories="macro",
        tags="overview",
        detected_coins="",
    )

    base_score = policy.calculate_article_relevance(
        article={"id": "article_1", "body": "macro outlook"},
        content=content,
        keywords={"market"},
        coin=None,
        current_time=current_time,
        relevant_categories=[],
        important_categories=set(),
        pub_time=pub_time,
        coin_patterns=None,
    )
    boosted_score = policy.calculate_article_relevance(
        article={"id": "market_overview", "body": "macro outlook"},
        content=content,
        keywords={"market"},
        coin=None,
        current_time=current_time,
        relevant_categories=[],
        important_categories=set(),
        pub_time=pub_time,
        coin_patterns=None,
    )

    assert boosted_score > base_score


def test_symbol_relevance_demotes_non_coin_articles():
    policy = _policy()
    current_time = 1000.0
    pub_time = 1000.0

    coin_patterns = {
        "coin_pattern": re.compile(r"\bbtc\b"),
        "title_start_pattern": re.compile(r"^\s*btc\b"),
        "price_pattern": re.compile(r"\bbtc\s+price\b"),
        "coin_name_pattern": re.compile(r"\bbitcoin\b"),
    }

    non_coin_content = ArticleContent(
        title="Altcoin round-up",
        body="general market coverage",
        categories="markets",
        tags="summary",
        detected_coins="eth|sol",
    )
    coin_content = ArticleContent(
        title="BTC price jumps",
        body="bitcoin momentum continues",
        categories="btc|markets",
        tags="btc",
        detected_coins="btc",
    )

    non_coin_score = policy.calculate_article_relevance(
        article={"id": "article_non_coin", "body": "general market coverage"},
        content=non_coin_content,
        keywords={"price"},
        coin="BTC",
        current_time=current_time,
        relevant_categories=[],
        important_categories=set(),
        pub_time=pub_time,
        coin_patterns=coin_patterns,
    )
    coin_score = policy.calculate_article_relevance(
        article={"id": "article_coin", "body": "bitcoin momentum continues"},
        content=coin_content,
        keywords={"price"},
        coin="BTC",
        current_time=current_time,
        relevant_categories=[],
        important_categories=set(),
        pub_time=pub_time,
        coin_patterns=coin_patterns,
    )

    assert coin_score > non_coin_score


def test_cooccurrence_modifier_applies_when_all_keywords_present():
    policy = _policy()
    content = ArticleContent(
        title="bitcoin regulation",
        body="bitcoin regulation framework",
        categories="regulation",
        tags="policy",
        detected_coins="btc",
    )

    modifier = policy.calculate_cooccurrence_modifier({"bitcoin", "regulation"}, content)

    assert modifier == 1.3
