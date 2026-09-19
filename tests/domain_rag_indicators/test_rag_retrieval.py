"""Dense domain tests for the RAG retrieval path: RagEngine retrieval and update
gates, ContextBuilder budgets and candidate ordering, ArticleScoringPolicy
weights, TickerManager category validation and RedditSentimentAnalyst parsing.
Every collaborator is a mock or an in-memory double, so nothing here touches the
network, an embedding model or a browser."""

import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.analyzer.sentiment_analyst import RedditSentimentAnalyst
from src.rag.context_builder import ArticleContent, ContextBuilder
from src.rag.rag_engine import RagEngine
from src.rag.scoring_policy import ArticleScoringPolicy
from src.rag.ticker_manager import TickerManager
from src.utils.token_counter import TokenCounter
from tests.conftest import make_config, null_logger

RAG_CONFIG = {
    "RAG_UPDATE_INTERVAL_HOURS": 1,
    "RAG_NEWS_LIMIT": 2,
    "RAG_ARTICLE_MAX_TOKENS": 200,
    "RAG_NEWS_ENRICH_MIN_CHARS": 20,
}

COIN_PATTERNS = {
    "coin_pattern": re.compile(r"\bbtc\b"),
    "title_start_pattern": re.compile(r"^\s*btc\b"),
    "price_pattern": re.compile(r"\bbtc\s+price\b"),
    "coin_name_pattern": re.compile(r"\bbitcoin\b"),
}

ATOM_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:media="http://search.yahoo.com/mrss/">
  <category term="Bitcoin" label="r/Bitcoin"/>
  <updated>2026-08-13T12:10:17+00:00</updated>
  <id>/r/Bitcoin/.rss?limit=10</id>
  <link rel="self"
        href="https://www.reddit.com/r/Bitcoin/.rss?limit=10" type="application/atom+xml"/>
  <title>Bitcoin</title>
  <entry>
    <author><name>/u/satoshi</name><uri>https://www.reddit.com/user/satoshi</uri></author>
    <category term="Bitcoin" label="r/Bitcoin"/>
    <content type="html">&lt;p&gt;Post body text&lt;/p&gt;</content>
    <id>t3_abc123</id>
    <link href="https://www.reddit.com/r/Bitcoin/comments/abc123/bitcoin_breaks_70k/"/>
    <title>Bitcoin breaks $70k resistance</title>
    <updated>2026-08-13T12:09:00+00:00</updated>
  </entry>
  <entry>
    <author><name>/u/bear_bot</name><uri>https://www.reddit.com/user/bear_bot</uri></author>
    <category term="Bitcoin" label="r/Bitcoin"/>
    <content type="html">&lt;p&gt;Post body text&lt;/p&gt;</content>
    <id>t3_def456</id>
    <link href="https://www.reddit.com/r/Bitcoin/comments/def456/etf_outflows_crash/"/>
    <title>ETF outflows spark crash fears</title>
    <updated>2026-08-13T12:08:00+00:00</updated>
  </entry>
  <entry>
    <author><name>/u/neutral_guy</name><uri>https://www.reddit.com/user/neutral_guy</uri></author>
    <category term="Bitcoin" label="r/Bitcoin"/>
    <content type="html">&lt;p&gt;Post body text&lt;/p&gt;</content>
    <id>t3_ghi789</id>
    <link href="https://www.reddit.com/r/Bitcoin/comments/ghi789/monday_thread/"/>
    <title>Daily discussion thread</title>
    <updated>2026-08-13T12:07:00+00:00</updated>
  </entry>
</feed>
"""

BARE_ENTRY_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><title>Bare entry title</title></entry>
  <entry><title>Second bare entry</title></entry>
</feed>
"""

EMPTY_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>Nothing</title></feed>
"""

RSS_INSTEAD_OF_ATOM = """<?xml version="1.0"?>
<rss version="2.0"><channel><item><title>RSS item</title></item></channel></rss>"""


def make_engine(logger: MagicMock | None = None) -> SimpleNamespace:
    """RagEngine on mocked collaborators, returned with the doubles it was given."""
    news_manager = MagicMock()
    news_manager.news_database = [
        {"id": "a1", "title": "Long body first", "body": "L" * 60, "published_on": 1758000000.0},
        {
            "id": "a2",
            "title": "Short body high score",
            "body": "tiny",
            "published_on": 1758000000.0,
        },
        {"id": "a3", "title": "Long body second", "body": "B" * 40, "published_on": 1758000000.0},
    ]
    news_manager.get_database_size.return_value = len(news_manager.news_database)
    news_manager.fetch_fresh_news = AsyncMock(return_value=[])

    category_fetcher = MagicMock()
    category_fetcher.fetch_categories = AsyncMock(return_value=None)

    category_processor = MagicMock()
    category_processor.category_word_map = {}
    category_processor.important_categories = set()
    category_processor.extract_base_coin.return_value = "BTC"

    index_manager = MagicMock()
    index_manager.search_by_coin.return_value = []

    context_builder = MagicMock()
    context_builder.keyword_search = AsyncMock(return_value=[(1, 10.0), (0, 9.0), (2, 8.0)])
    context_builder.add_articles_to_context.return_value = ("context text", 123)
    context_builder.get_latest_article_urls.return_value = {
        "Long body first": "https://example.com/a1"
    }

    market_data_manager = MagicMock()
    market_data_manager.update_market_overview_if_needed = AsyncMock(return_value=False)

    ticker_manager = MagicMock()
    ticker_manager.get_known_tickers.return_value = {"BTC"}

    engine = RagEngine(
        logger=logger or null_logger(),
        config=make_config(**RAG_CONFIG),
        news_manager=news_manager,
        market_data_manager=market_data_manager,
        index_manager=index_manager,
        category_fetcher=category_fetcher,
        category_processor=category_processor,
        ticker_manager=ticker_manager,
        context_builder=context_builder,
    )
    return SimpleNamespace(
        engine=engine,
        news_manager=news_manager,
        context_builder=context_builder,
        index_manager=index_manager,
        category_processor=category_processor,
        category_fetcher=category_fetcher,
        market_data_manager=market_data_manager,
    )


class WordCounter:
    """Token counter double counting one token per whitespace-separated word."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def make_builder(token_counter: Any = None, **overrides: Any) -> ContextBuilder:
    """ContextBuilder on the word-counting double and a mocked article processor."""
    article_processor = MagicMock()
    article_processor.extract_base_coin.return_value = "BTC"
    article_processor.get_article_timestamp.return_value = 0.0
    config = {"RAG_ARTICLE_MAX_TOKENS": 300, "RAG_NEWS_LIMIT": 2, "RAG_NEWS_ENRICH_MIN_CHARS": 20}
    config.update(overrides)
    return ContextBuilder(
        logger=null_logger(),
        token_counter=token_counter or WordCounter(),
        config=make_config(**config),
        scoring_policy=MagicMock(),
        article_processor=article_processor,
    )


def make_policy(**overrides: Any) -> ArticleScoringPolicy:
    """Scoring policy wired to the density and co-occurrence knobs it reads from config."""
    return ArticleScoringPolicy(
        config=make_config(
            RAG_DENSITY_PENALTY_THRESHOLD=20,
            RAG_DENSITY_PENALTY_MULTIPLIER=0.7,
            RAG_DENSITY_BOOST_THRESHOLD=80,
            RAG_DENSITY_BOOST_MULTIPLIER=1.2,
            RAG_COOCCURRENCE_MULTIPLIER=1.3,
            **overrides,
        )
    )


def make_content(
    title: str, body: str, categories: str = "", tags: str = "", detected_coins: str = ""
) -> ArticleContent:
    """ArticleContent double standing in for an article's normalised scoring fields."""
    return ArticleContent(title, body, categories, tags, detected_coins)


class ExchangeDouble:
    """ExchangeManager double exposing stored symbols and a lazy preload."""

    def __init__(self, symbols: set[str], symbols_after_load: set[str] | None = None) -> None:
        self._symbols = set(symbols)
        self._symbols_after_load = symbols_after_load
        self.ensure_calls = 0

    def get_all_symbols(self) -> set[str]:
        return set(self._symbols)

    async def ensure_symbols_loaded(self) -> None:
        self.ensure_calls += 1
        if self._symbols_after_load is not None:
            self._symbols = set(self._symbols_after_load)


def make_ticker_manager(
    symbols: set[str] | None = None, logger: MagicMock | None = None
) -> tuple[TickerManager, MagicMock, ExchangeDouble]:
    """TickerManager with its file-handler double and an exchange-symbol double."""
    file_handler = MagicMock()
    exchange = ExchangeDouble(symbols or set())
    manager = TickerManager(
        logger=logger or null_logger(), file_handler=file_handler, exchange_manager=exchange
    )
    return manager, file_handler, exchange


def long_body_articles(count: int) -> list[dict]:
    """Canonical news rows whose bodies clear the enrich minimum."""
    return [
        {"title": f"A{index}", "body": "body " * 20, "published_on": 1710000000}
        for index in range(count)
    ]


async def test_retrieve_context_reranks_candidates_and_records_article_urls():
    doubles = make_engine()

    context = await doubles.engine.retrieve_context(
        query="bitcoin regulation headlines", symbol="BTCUSDT", k=2, max_tokens=300
    )

    assert context == "context text"
    assert doubles.context_builder.add_articles_to_context.call_args.args == (
        [0, 2, 1],
        doubles.news_manager.news_database,
        300,
        2,
        {1: 10.0, 0: 9.0, 2: 8.0},
    )
    assert doubles.engine.get_latest_article_urls_snapshot() == {
        "Long body first": "https://example.com/a1"
    }


async def test_retrieve_context_expands_sparse_results_and_caps_the_pool():
    sparse = make_engine()
    sparse.context_builder.keyword_search = AsyncMock(return_value=[(1, 12.0)])
    sparse.index_manager.search_by_coin.return_value = [2]

    await sparse.engine.retrieve_context(
        query="btc liquidity", symbol="BTCUSDT", k=3, max_tokens=300
    )

    assert sparse.index_manager.search_by_coin.call_args.args == ("BTC",)
    assert sparse.context_builder.add_articles_to_context.call_args.args[0] == [2, 1]

    crowded = make_engine()
    crowded.news_manager.news_database = long_body_articles(30)
    crowded.news_manager.get_database_size.return_value = 30
    crowded.context_builder.keyword_search = AsyncMock(
        return_value=[(index, 100.0 - index) for index in range(30)]
    )

    await crowded.engine.retrieve_context(query="btc", symbol="BTCUSDT", k=2, max_tokens=300)

    assert crowded.context_builder.add_articles_to_context.call_args.args[0] == list(range(20))


async def test_retrieve_context_short_circuits_without_news_failing_scores_or_within_backoff():
    empty = make_engine()
    empty.news_manager.news_database = []
    empty.news_manager.get_database_size.return_value = 0

    assert await empty.engine.retrieve_context(query="anything", symbol="BTCUSDT") == ""
    assert empty.context_builder.keyword_search.await_count == 0

    failing = make_engine()
    failing.context_builder.keyword_search = AsyncMock(side_effect=RuntimeError("boom"))
    failing.engine._latest_article_urls = {"stale headline": "https://example.com/stale"}

    assert await failing.engine.retrieve_context(query="eth staking", symbol="ETHUSDT", k=2) == ""
    assert failing.engine.get_latest_article_urls_snapshot() == {}

    backed_off = make_engine()
    backed_off.engine.update_if_needed = AsyncMock(return_value=True)
    backed_off.engine._last_update_attempt = datetime.now(timezone.utc)

    assert await backed_off.engine.retrieve_context(
        query="btc", symbol="BTCUSDT", k=2, max_tokens=300
    ) == "context text"
    assert backed_off.engine.update_if_needed.await_count == 0

    expired_window = make_engine()
    expired_window.engine.update_if_needed = AsyncMock(return_value=True)

    await expired_window.engine.retrieve_context(query="btc", symbol="BTCUSDT", k=2, max_tokens=300)

    assert expired_window.engine.update_if_needed.await_count == 1


async def test_update_if_needed_refreshes_stale_forced_or_missing_timestamps(logger_double):
    stale = make_engine()
    stale.engine.refresh_market_data = AsyncMock(return_value=None)
    stale.engine.last_update = datetime.now(timezone.utc) - timedelta(hours=2)

    assert await stale.engine.update_if_needed(force_update=False) is True
    assert stale.engine.refresh_market_data.await_count == 1
    assert stale.engine.last_update > datetime.now(timezone.utc) - timedelta(seconds=5)

    unknown = make_engine()
    unknown.engine.refresh_market_data = AsyncMock(return_value=None)

    assert await unknown.engine.update_if_needed() is True
    assert unknown.engine.refresh_market_data.await_count == 1

    fresh = make_engine()
    fresh.engine.refresh_market_data = AsyncMock(return_value=None)
    fresh.engine.last_update = datetime.now(timezone.utc)

    assert await fresh.engine.update_if_needed() is False
    assert fresh.engine.refresh_market_data.await_count == 0

    forced = make_engine()
    forced.engine.refresh_market_data = AsyncMock(return_value=None)
    forced.engine.last_update = datetime.now(timezone.utc)

    assert await forced.engine.update_if_needed(force_update=True) is True

    live_categories = make_engine()
    live_categories.category_fetcher.fetch_categories = AsyncMock(
        return_value=[{"categoryName": "BTC"}]
    )
    live_categories.engine.last_update = datetime.now(timezone.utc)

    assert await live_categories.engine.update_if_needed() is False
    assert live_categories.category_processor.process_api_categories.call_args.args == (
        [{"categoryName": "BTC"}],
    )
    assert live_categories.index_manager.build_indices.call_args.args == (
        live_categories.news_manager.news_database,
        {"BTC"},
    )

    categories_down = make_engine(logger=logger_double)
    categories_down.category_fetcher.fetch_categories = AsyncMock(
        side_effect=RuntimeError("taxonomy api down")
    )
    categories_down.engine.last_update = datetime.now(timezone.utc)

    assert await categories_down.engine.update_if_needed() is False
    assert logger_double.exception.call_count == 1

    failing = make_engine(logger=logger_double)
    failing.engine.refresh_market_data = AsyncMock(side_effect=RuntimeError("exchange down"))
    failing.engine.last_update = datetime.now(timezone.utc) - timedelta(hours=3)

    assert await failing.engine.update_if_needed() is False
    assert logger_double.error.call_args.args[0] == "Failed to update market knowledge: %s"


async def test_engine_readers_expose_detached_snapshots_and_an_optional_overview():
    doubles = make_engine()
    snapshot = doubles.engine.get_news_cache_snapshot()

    assert snapshot == doubles.news_manager.news_database

    snapshot[0]["title"] = "mutated"

    assert doubles.news_manager.news_database[0]["title"] == "Long body first"
    assert doubles.engine.get_news_cache_snapshot(limit=2) == doubles.news_manager.news_database[:2]
    assert len(doubles.engine.get_news_cache_snapshot(limit=0)) == 3
    assert doubles.engine.get_latest_article_urls_snapshot() == {}

    doubles.market_data_manager.get_current_overview.return_value = {"total_market_cap_usd": 1}

    assert await doubles.engine.get_market_overview() == {"total_market_cap_usd": 1}
    assert doubles.market_data_manager.update_market_overview_if_needed.call_args.kwargs == {
        "max_age_hours": 1
    }

    doubles.market_data_manager.get_current_overview.side_effect = RuntimeError("coingecko down")

    assert await doubles.engine.get_market_overview() is None

    bare = RagEngine(logger=null_logger(), config=make_config())

    assert bare.get_news_cache_snapshot() == []
    assert bare.get_latest_article_urls_snapshot() == {}
    assert await bare.get_market_overview() is None


def test_retrieval_limits_and_context_query_derive_from_config():
    doubles = make_engine()
    limits = [(None, None), (5, None), (None, 77), (3, 77)]

    assert [doubles.engine._resolve_retrieval_limits(k, tokens) for k, tokens in limits] == [
        (2, 400),
        (5, 1000),
        (2, 77),
        (3, 77),
    ]

    doubles.context_builder.symbol_name_map = {"BTC": "bitcoin"}
    bare = RagEngine(logger=null_logger(), config=make_config())

    assert doubles.engine.build_context_query("BTCUSDT") == "bitcoin price analysis market trends"
    assert bare.build_context_query("ETH/USDT") == "eth price analysis market trends"
    assert bare.build_context_query("SOLUSDT") == "solusdt price analysis market trends"
    assert bare._resolve_retrieval_limits(None, None) == (5, 5000)


def test_builder_reads_the_config_budget_and_normalizes_the_symbol_name_map():
    builder = make_builder()
    builder.config.RAG_ARTICLE_MAX_TOKENS = 10
    builder.config.RAG_NEWS_LIMIT = 1
    called_limits: list[int] = []

    def process(_item: dict, max_tokens: int) -> str:
        called_limits.append(max_tokens)
        return "word " * 8

    builder._process_article_simple = process

    context_text = builder.build_context(
        [
            {"title": "A0", "url": "https://example.com/0"},
            {"title": "A1", "url": "https://example.com/1"},
        ]
    )

    assert context_text == "word " * 8
    assert called_limits == [10, 10]

    mapped = ContextBuilder(
        logger=null_logger(),
        token_counter=WordCounter(),
        config=make_config(**RAG_CONFIG),
        scoring_policy=MagicMock(),
        article_processor=MagicMock(),
        symbol_name_map={"btc": "Bitcoin", "eth": ""},
    )

    assert mapped.symbol_name_map == {"BTC": "bitcoin"}


def test_add_articles_to_context_prefers_full_body_over_score():
    builder = make_builder()
    news_database = [
        {
            "title": "Long body lower score",
            "body": "x " * 40,
            "source": "unit",
            "published_on": 1710000000,
        },
        {
            "title": "Short body higher score",
            "body": "tiny",
            "source": "unit",
            "published_on": 1710000000,
        },
    ]

    context_text, total_tokens = builder.add_articles_to_context(
        relevant_indices=[1, 0],
        news_database=news_database,
        max_tokens=1000,
        k=2,
        scores_dict={1: 10.0, 0: 5.0},
    )

    assert context_text.index("Long body lower score") < context_text.index(
        "Short body higher score"
    )
    header = "📰 Long body lower score\nSrc: Unknown Source (2024-03-09 16:00 UTC)"
    assert context_text.startswith(header)
    assert total_tokens == 63
    assert builder.get_latest_article_urls() == {}


def test_add_articles_to_context_limits_the_article_count_to_k():
    builder = make_builder()

    context_text, total_tokens = builder.add_articles_to_context(
        relevant_indices=[0, 1, 2],
        news_database=long_body_articles(3),
        max_tokens=1000,
        k=2,
        scores_dict={0: 5.0, 1: 4.0, 2: 3.0},
    )

    assert [f"A{index}" in context_text for index in range(3)] == [True, True, False]
    assert total_tokens == 56


def test_add_articles_to_context_truncates_the_pool_and_drops_missing_indices():
    builder = make_builder()
    news_database = long_body_articles(60)
    scores = {index: float(index) for index in range(60)}
    scores[55] = 999.0

    context_text, _ = builder.add_articles_to_context(
        list(range(60)), news_database, 1000, 2, scores
    )

    assert context_text.startswith("📰 A49")
    assert ["A55" in context_text, "A59" in context_text] == [False, False]

    missing_text, missing_tokens = builder.add_articles_to_context(
        [99, 0], news_database, 1000, 2, {99: 50.0, 0: 10.0}
    )

    assert missing_text.startswith("📰 A0")
    assert missing_tokens == 28
    assert builder.add_articles_to_context([], news_database, 1000, 2, {}) == ("", 0)
    assert builder.add_articles_to_context([0, 1], news_database, 1000, 0, {}) == ("", 0)


def test_process_article_respects_the_token_budget():
    counter = TokenCounter()
    builder = make_builder(token_counter=counter, RAG_ARTICLE_MAX_TOKENS=80)
    item = {
        "title": "Token Budget Check",
        "body": " ".join(["bitcoin liquidity rotation volatility"] * 220),
        "source_info": {"name": "coindesk"},
        "published_on": 1710000000,
    }

    processed = builder._process_article_simple(item, 80)

    assert processed.startswith("📰 Token Budget Check\nSrc: coindesk (2024-03-09 16:00 UTC)")
    assert processed.endswith("...")
    assert counter.count_tokens(processed) == 50


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        ({"title": "X", "body": ""}, ""),
        ({"title": "X", "body": "   \n\n  "}, ""),
        (
            {"title": None, "body": "Some body text here."},
            "📰 No Title\nSrc: Unknown Source (1970-01-01 00:00 UTC)\nSome body text here.",
        ),
    ],
    ids=["empty-body", "whitespace-body", "missing-title"],
)
def test_process_article_degenerate_payloads(item, expected):
    assert make_builder()._process_article_simple(item, 80) == expected


def test_process_article_overflows_a_budget_below_its_header():
    counter = TokenCounter()
    builder = make_builder(token_counter=counter)
    item = {
        "title": "Token Budget Check",
        "body": " ".join(["bitcoin liquidity rotation volatility"] * 220),
        "source_info": {"name": "coindesk"},
        "published_on": 1710000000,
    }

    processed = builder._process_article_simple(item, 20)

    assert counter.count_tokens(processed) == 28
    assert counter.count_tokens(processed) > 20


def test_build_context_stops_at_the_budget_and_resets_article_urls():
    builder = make_builder()
    articles = [
        {"title": f"T{index}", "body": "word " * 50, "published_on": 0} for index in range(5)
    ]

    capped = builder.build_context(articles, max_tokens=60)

    assert ["T0" in capped, "T1" in capped] == [True, False]

    builder.build_context(
        [
            {
                "title": "T",
                "url": "https://example.com/t",
                "body": "body text",
                "published_on": 1710000000,
            }
        ]
    )

    assert builder.get_latest_article_urls() == {"T": "https://example.com/t"}

    builder.build_context(
        [
            {
                "title": "U",
                "url": "https://example.com/u",
                "body": "body text",
                "published_on": 1710000000,
            }
        ]
    )

    assert builder.build_context([]) == ""
    assert builder.get_latest_article_urls() == {"U": "https://example.com/u"}


def test_symbol_relevance_demotes_articles_without_a_coin_match():
    policy = make_policy()
    non_coin = make_content(
        "altcoin round-up", "general market coverage", "markets", "summary", "eth|sol"
    )
    coin = make_content(
        "btc price jumps", "bitcoin momentum continues", "btc|markets", "btc", "btc"
    )

    def relevance(content: ArticleContent, article_body: str) -> float:
        return policy.calculate_article_relevance(
            article={"id": "article", "body": article_body},
            content=content,
            keywords={"price"},
            coin="BTC",
            current_time=1000.0,
            relevant_categories=[],
            important_categories=set(),
            pub_time=1000.0,
            coin_patterns=COIN_PATTERNS,
        )

    assert relevance(non_coin, "general market coverage") == 0.0
    assert relevance(coin, "bitcoin momentum continues") == pytest.approx(79.93147180559946)
    assert relevance(coin, "tiny") == pytest.approx(79.93147180559946 * 0.7)


def test_cooccurrence_modifier_requires_every_keyword():
    policy = make_policy()
    content = make_content(
        "bitcoin regulation", "bitcoin regulation framework", "regulation", "policy", "btc"
    )

    assert policy.calculate_cooccurrence_modifier({"bitcoin", "regulation"}, content) == 1.3
    assert policy.calculate_cooccurrence_modifier({"bitcoin"}, content) == 1.0
    assert policy.calculate_cooccurrence_modifier({"bitcoin", "etf"}, content) == 1.0
    assert policy.calculate_cooccurrence_modifier(set(), content) == 1.0

    stateless = ArticleScoringPolicy()

    assert stateless.calculate_cooccurrence_modifier({"bitcoin", "etf"}, content) == 1.0


@pytest.mark.parametrize(
    ("body_length", "expected"),
    [(19, 0.7), (20, 1.0), (80, 1.0), (81, 1.2)],
    ids=[
        "below-penalty-threshold",
        "at-penalty-threshold",
        "at-boost-threshold",
        "above-boost-threshold",
    ],
)
def test_density_modifier_uses_exclusive_thresholds(body_length, expected):
    assert make_policy().calculate_density_modifier({"body": "x" * body_length}) == expected


@pytest.mark.parametrize(
    ("published_offset", "expected"),
    [
        (0.0, 1.0),
        (3600.0, 0.9583333333333334),
        (90000.0, 0.0),
        (-3600.0, 1.0416666666666667),
    ],
    ids=["same-instant", "one-hour-old", "beyond-one-day", "future-timestamp"],
)
def test_recency_factor_decays_and_clamps_at_zero(published_offset, expected):
    factor = ArticleScoringPolicy.calculate_recency_factor(1000.0, 1000.0 - published_offset)

    assert factor == expected


def test_score_component_tables_cover_coin_keyword_category_and_importance():
    policy = make_policy()
    content = make_content(
        "btc price jumps", "bitcoin momentum continues", "btc|markets", "btc", "btc"
    )

    assert policy.calculate_coin_score("BTC", content, COIN_PATTERNS) == 63.0
    assert policy.calculate_coin_score("BTC", content, None) == 58.0
    assert policy.calculate_keyword_score({"price"}, content) == pytest.approx(16.931471805599452)
    assert policy.calculate_category_score(["btc", "eth"], "btc|eth") == 10.0
    assert policy.calculate_category_score(["btc", "eth"], "BTC|ETH") == 0.0
    assert ArticleScoringPolicy.calculate_importance_score("btc|markets", {"BTC", "MARKETS"}) == 6.0
    assert ArticleScoringPolicy().calculate_density_modifier({"body": "x"}) == 1.0
    assert policy.calculate_density_modifier({}) == 0.7


def test_coin_relevance_multiplier_demotes_body_only_and_unmatched_articles():
    policy = make_policy()
    body_only = make_content("new listing", "bitcoin momentum continues", "btc", "btc", "btc")
    unmatched = make_content(
        "altcoin round-up", "general market coverage", "markets", "summary", "eth|sol"
    )

    assert policy._calculate_coin_relevance_multiplier(None, 0.0, body_only, COIN_PATTERNS) == 1.0
    assert policy._calculate_coin_relevance_multiplier("BTC", 0.0, body_only, COIN_PATTERNS) == 0.1
    assert policy._calculate_coin_relevance_multiplier("BTC", 5.0, body_only, COIN_PATTERNS) == 0.35
    assert policy._calculate_coin_relevance_multiplier("BTC", 5.0, unmatched, COIN_PATTERNS) == 0.35
    assert policy._calculate_coin_relevance_multiplier("BTC", 63.0, body_only, None) == 0.35


def test_category_and_ticker_candidate_tables():
    manager, _, _ = make_ticker_manager({"BTC/USDT", "ETH/USDT"})
    category_coins = {
        "BTC|ETH": {"BTC", "ETH"},
        "Markets|BTC|DeFi": {"BTC"},
        "Technology": {"TECHNOLOGY"},
        "": set(),
    }
    widened = {
        "BTC-USD": "BTC",
        "ETH-USDT": "ETH",
        "SOL-BTC": "SOL",
        "X-ETH": None,
        "ABCDEFGHIJ": "ABCDEFGHIJ",
        "ABCDEFGHIJK": None,
        "A": None,
        "BTC/USDT": None,
    }

    assert {
        categories: manager._extract_category_coins([{"categories": categories}])
        for categories in category_coins
    } == category_coins
    assert manager._extract_category_coins([{}]) == set()
    assert {
        category: manager._extract_ticker_from_category(category) for category in widened
    } == widened


def test_candidate_validation_policy_tables():
    manager, _, _ = make_ticker_manager({"BTC/USDT"})
    potentials = {
        "BTC": True,
        "AB": True,
        "ABCDEFGHIJ": True,
        "ABCDEFGHIJK": False,
        "A": False,
        "": False,
        "USD": False,
        "usd": False,
        "NEWS": False,
    }
    sections = {
        "Technology": True,
        "Markets": False,
        "News": False,
        "DeFi": False,
        "BTC": True,
        "btc": True,
        "A": False,
        "": False,
    }

    assert {coin: manager._is_potential_valid_coin(coin) for coin in potentials} == potentials
    assert {value: manager._is_valid_ticker_category(value) for value in sections} == sections
    assert manager._should_add_coin("BTC", {"BTC/USDT", "ETH/USDT"}) is True
    assert manager._should_add_coin("BTC", {"BTCUSDT"}) is False
    assert manager._should_add_coin("ETH", {"BTC/USDT"}) is False
    assert manager._should_add_coin("BTC", set()) is False

    manager.known_tickers = {"BTC"}

    assert manager._should_add_coin("BTC", {"BTC/USDT"}) is False


async def test_update_known_tickers_adds_only_exchange_validated_base_assets():
    manager, file_handler, _ = make_ticker_manager({"BTC/USDT", "ETH/USDT"})

    articles = [{"categories": "Technology|BTC", "detected_coins": ["btc"]}]

    await manager.update_known_tickers(articles)

    assert manager.known_tickers == {"BTC"}
    assert file_handler.save_known_tickers.call_args.args == (["BTC"],)


async def test_update_known_tickers_requires_exchange_symbols_and_preloads_once():
    logger = null_logger()
    no_symbols, _, _ = make_ticker_manager(set(), logger=logger)

    await no_symbols.update_known_tickers([{"categories": "Technology|BTC"}])

    assert no_symbols.known_tickers == set()
    assert logger.warning.call_args.args[:2] == (
        "Exchange symbol data unavailable; skipping %d unvalidated candidate tickers",
        2,
    )

    missing_exchange, _, _ = make_ticker_manager({"BTC/USDT"}, logger=logger)
    missing_exchange.exchange_manager = None

    await missing_exchange.update_known_tickers([{"categories": "BTC"}])

    assert missing_exchange.known_tickers == set()
    assert logger.warning.call_args.args[1] == 1

    exchange = ExchangeDouble(set(), symbols_after_load={"SOL/USDT", "BTC/USDT"})
    lazy = TickerManager(logger=null_logger(), file_handler=MagicMock(), exchange_manager=exchange)

    await lazy.update_known_tickers([{"categories": "SOL|TECHNOLOGY"}])

    assert exchange.ensure_calls == 1
    assert lazy.known_tickers == {"SOL"}

    preloaded, _, loaded = make_ticker_manager({"BTC/USDT"})

    await preloaded.update_known_tickers([{"categories": "BTC"}])

    assert loaded.ensure_calls == 0
    assert preloaded.known_tickers == {"BTC"}


async def test_load_known_tickers_handles_data_falsy_and_unreadable_states():
    manager, file_handler, _ = make_ticker_manager()
    file_handler.load_known_tickers.return_value = ["BTC", "ETH"]

    await manager.load_known_tickers()

    assert manager.known_tickers == {"BTC", "ETH"}

    manager.known_tickers = {"OLD"}
    file_handler.load_known_tickers.return_value = None

    await manager.load_known_tickers()

    assert manager.known_tickers == {"OLD"}

    file_handler.load_known_tickers.side_effect = RuntimeError("disk unreadable")

    await manager.load_known_tickers()

    assert manager.known_tickers == set()
    assert manager.logger.exception.call_count == 1


async def test_save_tickers_persists_a_sorted_detached_list():
    manager, file_handler, _ = make_ticker_manager()
    manager.known_tickers = {"ETH", "BTC"}

    await manager.save_tickers()

    assert file_handler.save_known_tickers.call_args.args == (["BTC", "ETH"],)
    assert manager.get_known_tickers() == {"BTC", "ETH"}
    assert manager.get_known_tickers() is not manager.known_tickers

    file_handler.save_known_tickers.side_effect = RuntimeError("read-only volume")

    await manager.save_tickers()

    assert manager.known_tickers == {"BTC", "ETH"}
    assert manager.logger.exception.call_count == 1


def test_parse_atom_feed_maps_the_post_contract_and_respects_the_limit():
    analyst = RedditSentimentAnalyst(logger=null_logger())
    posts = analyst._parse_atom_feed(ATOM_SAMPLE, "Bitcoin", limit=10)

    assert len(posts) == 3
    assert posts[0] == {
        "subreddit": "Bitcoin",
        "title": "Bitcoin breaks $70k resistance",
        "author": "/u/satoshi",
        "url": "https://www.reddit.com/r/Bitcoin/comments/abc123/bitcoin_breaks_70k/",
        "score": 0,
        "num_comments": 0,
        "upvote_ratio": 0.0,
        "created_utc": 1786622940,
    }
    assert [post["title"] for post in posts] == [
        "Bitcoin breaks $70k resistance",
        "ETF outflows spark crash fears",
        "Daily discussion thread",
    ]
    assert [post["created_utc"] for post in posts] == [1786622940, 1786622880, 1786622820]
    lengths = [
        len(analyst._parse_atom_feed(ATOM_SAMPLE, "Bitcoin", limit=limit)) for limit in (10, 2, 0)
    ]

    assert lengths == [3, 2, 0]


def test_parse_atom_feed_rejects_unusable_or_emptied_payloads():
    analyst = RedditSentimentAnalyst(logger=null_logger())
    emptied_title = "<title>Bitcoin breaks $70k resistance</title>"
    emptied = ATOM_SAMPLE.replace(emptied_title, "<title></title>")

    assert analyst._parse_atom_feed("<not xml", "Bitcoin", limit=10) == []
    assert analyst._parse_atom_feed("", "Bitcoin", limit=10) == []
    assert analyst._parse_atom_feed(EMPTY_FEED, "Bitcoin", limit=10) == []
    assert analyst._parse_atom_feed(RSS_INSTEAD_OF_ATOM, "Bitcoin", limit=10) == []
    filled = analyst._parse_atom_feed(emptied, "Bitcoin", limit=10)

    assert [post["title"] for post in filled] == [
        "ETF outflows spark crash fears",
        "Daily discussion thread",
    ]
    bare = analyst._parse_atom_feed(BARE_ENTRY_FEED, "Bitcoin", limit=10)

    assert [
        (post["title"], post["author"], post["url"], post["created_utc"]) for post in bare
    ] == [("Bare entry title", "", "", 0), ("Second bare entry", "", "", 0)]


def test_parse_iso8601_normalizes_offsets_and_falls_back_to_zero():
    assert RedditSentimentAnalyst._parse_iso8601("2026-08-13T12:09:00+00:00") == 1786622940
    assert RedditSentimentAnalyst._parse_iso8601("2026-08-13T12:09:00Z") == 1786622940
    assert RedditSentimentAnalyst._parse_iso8601("2026-08-13T12:09:00") == 1786622940
    assert RedditSentimentAnalyst._parse_iso8601("not-a-date") == 0
    assert RedditSentimentAnalyst._parse_iso8601("") == 0
    assert RedditSentimentAnalyst._parse_iso8601("2026-13-45T99:00:00+00:00") == 0


@pytest.mark.parametrize(
    ("titles", "expected"),
    [
        (
            [
                "Bitcoin breaks $70k resistance, new all-time high",
                "ETF inflows hit record, institutional adoption surges",
            ],
            "BULLISH",
        ),
        (
            [
                "Bitcoin crashes below support, panic selling",
                "ETF outflows spark sell-off, bear market fears",
            ],
            "BEARISH",
        ),
        (["Daily discussion thread", "Weekly market update"], "NEUTRAL"),
        ([], "NO_DATA"),
        (["rally"], "SLIGHTLY_BULLISH"),
        (["crash"], "SLIGHTLY_BEARISH"),
    ],
    ids=[
        "bullish-pair",
        "bearish-pair",
        "no-lexicon-hits",
        "no-posts",
        "single-bull",
        "single-bear",
    ],
)
def test_compute_overall_sentiment_uses_the_title_lexicon_ratio(titles, expected):
    posts = [{"title": title} for title in titles]

    assert RedditSentimentAnalyst._compute_overall_sentiment(posts) == expected


def test_extract_top_topics_ranks_words_and_drops_stopwords():
    analyst = RedditSentimentAnalyst(logger=null_logger())
    posts = [
        {"title": "Bitcoin ETF approval expected soon halving"},
        {"title": "ETF inflows break records again Bitcoin"},
    ]

    assert analyst._extract_top_topics(posts) == [
        "approval",
        "expected",
        "soon",
        "halving",
        "inflows",
    ]
    assert analyst._extract_top_topics(posts, top_n=2) == ["approval", "expected"]
    assert analyst._extract_top_topics([]) == []


def test_format_sentiment_section_renders_sorted_capped_posts():
    analyst = RedditSentimentAnalyst(logger=null_logger())
    posts = [
        {
            "subreddit": "CryptoCurrency",
            "title": "Bitcoin breaks $70k resistance",
            "author": "/u/bull",
            "created_utc": 1755080000,
        },
        {
            "subreddit": "Bitcoin",
            "title": "ETF inflows hit $1B weekly",
            "author": "/u/hodler",
            "created_utc": 1755081000,
        },
    ]

    section = analyst.format_sentiment_section(
        {
            "posts": posts,
            "overall_sentiment": "BULLISH",
            "top_topics": ["etf", "halving", "breakout"],
        }
    )

    subreddits = "r/CryptoCurrency, r/Bitcoin, r/ethereum, r/CryptoMarkets"
    header = f"## Social Sentiment (Reddit — {subreddits})"

    assert header in section
    assert "Overall: **BULLISH**" in section
    assert "Trending topics: etf, halving, breakout" in section
    assert "1. [Bitcoin] **ETF inflows hit $1B weekly** (by /u/hodler)" in section
    assert "2. [CryptoCurrency] **Bitcoin breaks $70k resistance** (by /u/bull)" in section
    assert "⚠" not in section

    ranked = [
        {"subreddit": f"s{index}", "title": f"T{index}", "author": "", "created_utc": index}
        for index in reversed(range(7))
    ]
    topics = [f"t{index}" for index in range(7)]
    capped = analyst.format_sentiment_section(
        {"posts": ranked, "overall_sentiment": "NEUTRAL", "top_topics": topics}
    )
    numbers = enumerate(range(6, 1, -1), 1)

    assert capped.startswith("\n## Social Sentiment")
    assert "Overall: **NEUTRAL**" in capped
    assert "Trending topics: t0, t1, t2, t3, t4\n" in capped
    assert [f"{rank}. [s{index}] **T{index}**" in capped for rank, index in numbers] == [True] * 5
    assert "T1" not in capped

    truncated = analyst.format_sentiment_section(
        {"posts": [{"subreddit": "s", "title": "L" * 200, "author": "", "created_utc": 5}]}
    )

    assert "Overall: **N/A**" in truncated
    assert f"1. [s] **{'L' * 120}**" in truncated
    assert "L" * 121 not in truncated
    assert analyst.format_sentiment_section({"posts": [], "overall_sentiment": "NO_DATA"}) == ""
    assert analyst.format_sentiment_section({}) == ""

    errored = analyst.format_sentiment_section(
        {
            "posts": [
                {
                    "subreddit": "CryptoCurrency",
                    "title": "Test post",
                    "author": "",
                    "created_utc": 0,
                }
            ],
            "overall_sentiment": "NEUTRAL",
            "top_topics": [],
            "error": "CryptoMarkets: timeout",
        }
    )

    assert "⚠️ Fetch errors: CryptoMarkets: timeout" in errored
