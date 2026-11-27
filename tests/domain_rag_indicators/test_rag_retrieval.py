

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.analyzer.sentiment_analyst import SocialSentimentAnalyst
from src.rag.context_builder import ArticleContent, ContextBuilder
from src.rag.file_handler import RagFileHandler
from src.rag.rag_engine import RagEngine
from src.rag.scoring_policy import ArticleScore, ArticleScoringPolicy
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

STREAM_SAMPLE = json.dumps({
    "messages": [
        {"body": "Bitcoin breaks $70k resistance &#39;finally&#39;",
         "created_at": "2026-08-13T12:09:00Z"},
        {"body": "ETF outflows spark crash fears",
         "created_at": "2026-08-13T12:08:00Z"},
        {"body": "Daily discussion thread about the $BTC.X chart",
         "created_at": "2026-08-13T12:07:00Z"},
    ],
})


def make_engine(logger: MagicMock | None = None) -> SimpleNamespace:

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
    context_builder.keyword_search = AsyncMock(return_value=([(1, 10.0), (0, 9.0), (2, 8.0)], {}))
    context_builder.add_articles_to_context.return_value = ("context text", 123)
    context_builder.get_last_selection.return_value = ([], [])
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
    def count_tokens(self, text: str) -> int:
        return len(text.split())


def make_builder(token_counter: Any = None, **overrides: Any) -> ContextBuilder:

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

    return ArticleScoringPolicy(
        config=make_config(
            RAG_DENSITY_PENALTY_THRESHOLD=20,
            RAG_DENSITY_PENALTY_MULTIPLIER=0.7,
            RAG_DENSITY_BOOST_THRESHOLD=80,
            RAG_DENSITY_BOOST_MULTIPLIER=1.2,
            **overrides,
        )
    )


def make_content(
    title: str, body: str, categories: str = "", tags: str = "", detected_coins: str = ""
) -> ArticleContent:

    return ArticleContent(title, body, categories, tags, detected_coins)


class ExchangeDouble:
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

    file_handler = MagicMock()
    exchange = ExchangeDouble(symbols or set())
    manager = TickerManager(
        logger=logger or null_logger(), file_handler=file_handler, exchange_manager=exchange
    )
    return manager, file_handler, exchange


def long_body_articles(count: int) -> list[dict]:

    return [
        {"title": f"A{index}", "body": "body " * 20, "published_on": 1710000000}
        for index in range(count)
    ]


async def test_retrieve_context_forwards_ranked_candidates_and_records_article_urls():
    doubles = make_engine()

    context = await doubles.engine.retrieve_context(
        query="bitcoin regulation headlines", symbol="BTCUSDT", k=2, max_tokens=300
    )

    assert context == "context text"
    assert doubles.context_builder.add_articles_to_context.call_args.args == (
        [1, 0, 2],
        doubles.news_manager.news_database,
        300,
        2,
        {1: 10.0, 0: 9.0, 2: 8.0},
    )
    assert doubles.engine.get_latest_article_urls_snapshot() == {
        "Long body first": "https://example.com/a1"
    }


async def test_retrieve_context_reports_one_ranking_trace():
    doubles = make_engine()
    breakdown = ArticleScore(
        score=10.0,
        keyword_score=6.0,
        category_score=0.0,
        coin_score=4.0,
        importance_score=0.0,
        density_mult=1.0,
        coin_relevance_mult=1.0,
        recency=1.0,
        source_mult=1.0,
    )
    doubles.context_builder.keyword_search = AsyncMock(
        return_value=([(1, 10.0), (0, 9.0)], {0: breakdown, 1: breakdown})
    )
    doubles.context_builder.get_last_selection.return_value = ([1, 0], [0])

    await doubles.engine.retrieve_context(
        query="bitcoin regulation headlines", symbol="BTCUSDT", k=2, max_tokens=300
    )

    traces = [
        call for call in doubles.engine.logger.debug.call_args_list
        if call.args and str(call.args[0]).startswith("RAG ranking")
    ]
    assert len(traces) == 1
    payload = json.loads(traces[0].args[1])
    assert payload["symbol"] == "BTCUSDT"
    assert payload["query"] == "bitcoin regulation headlines"
    assert payload["stages"] == {"deduped": 2, "pool": 2, "shortlist": 2, "rendered": 1}
    assert {entry["index"]: entry["status"] for entry in payload["articles"]} == {
        0: "rendered",
        1: "shortlisted_not_rendered",
    }
    assert traces[0].kwargs["extra"] == {"skip_live_stream": True}


async def test_retrieve_context_skips_the_trace_without_scored_breakdowns():
    doubles = make_engine()

    await doubles.engine.retrieve_context(
        query="bitcoin regulation headlines", symbol="BTCUSDT", k=2, max_tokens=300
    )

    assert not [
        call for call in doubles.engine.logger.debug.call_args_list
        if call.args and str(call.args[0]).startswith("RAG ranking")
    ]


async def test_retrieve_context_expands_sparse_results_and_caps_the_pool():
    sparse = make_engine()
    sparse.context_builder.keyword_search = AsyncMock(return_value=([(1, 12.0)], {}))
    sparse.index_manager.search_by_coin.return_value = [2]
    sparse.news_manager.news_database[2]["published_on"] = datetime.now(timezone.utc).timestamp()

    await sparse.engine.retrieve_context(
        query="btc liquidity", symbol="BTCUSDT", k=3, max_tokens=300
    )

    assert sparse.index_manager.search_by_coin.call_args.args == ("BTC",)
    assert sparse.context_builder.add_articles_to_context.call_args.args[0] == [1, 2]

    crowded = make_engine()
    crowded.news_manager.news_database = long_body_articles(30)
    crowded.news_manager.get_database_size.return_value = 30
    crowded.context_builder.keyword_search = AsyncMock(
        return_value=([(index, 100.0 - index) for index in range(30)], {})
    )

    await crowded.engine.retrieve_context(query="btc", symbol="BTCUSDT", k=2, max_tokens=300)

    assert crowded.context_builder.add_articles_to_context.call_args.args[0] == list(range(20))


async def test_backfill_skips_scored_and_expired_articles(monkeypatch):
    expired = make_engine()
    expired.context_builder.keyword_search = AsyncMock(return_value=([(1, 12.0), (0, 3.0)], {}))
    monkeypatch.setattr(expired.engine, "_dedupe_by_story", lambda ranked, database: [1])
    expired.news_manager.news_database[2]["published_on"] = (
        datetime.now(timezone.utc).timestamp() - 90000.0
    )
    expired.index_manager.search_by_coin.return_value = [0, 2]

    await expired.engine.retrieve_context(query="btc", symbol="BTCUSDT", k=3, max_tokens=300)

    assert expired.context_builder.add_articles_to_context.call_args.args[0] == [1]

    fresh = make_engine()
    fresh.context_builder.keyword_search = AsyncMock(return_value=([(1, 12.0), (0, 3.0)], {}))
    monkeypatch.setattr(fresh.engine, "_dedupe_by_story", lambda ranked, database: [1])
    fresh.news_manager.news_database[2]["published_on"] = datetime.now(timezone.utc).timestamp()
    fresh.index_manager.search_by_coin.return_value = [0, 2]

    await fresh.engine.retrieve_context(query="btc", symbol="BTCUSDT", k=3, max_tokens=300)

    assert fresh.context_builder.add_articles_to_context.call_args.args[0] == [1, 2]


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

    assert doubles.engine.build_context_query("BTCUSDT") == "bitcoin price range consolidation"
    assert bare.build_context_query("ETH/USDT") == "eth price range consolidation"
    assert bare.build_context_query("SOLUSDT") == "solusdt price range consolidation"
    assert bare._resolve_retrieval_limits(None, None) == (5, 5000)


@pytest.mark.parametrize(
    ("percentage", "phrase"),
    [
        (-5.0, "selloff liquidation crash"),
        (-3.0, "selloff liquidation crash"),
        (-2.9, "decline pullback"),
        (-1.0, "decline pullback"),
        (-0.9, "range consolidation"),
        (0.0, "range consolidation"),
        (0.9, "range consolidation"),
        (1.0, "rebound gains"),
        (2.9, "rebound gains"),
        (3.0, "rally breakout"),
        (9.0, "rally breakout"),
    ],
)
def test_the_context_query_follows_the_market_state(percentage, phrase):

    engine = RagEngine(logger=null_logger(), config=make_config())

    assert engine.build_context_query("BTC/USDT", {"percentage": percentage}) == f"btc price {phrase}"


@pytest.mark.parametrize(
    "ticker",
    [None, {}, {"percentage": None}],
    ids=["no-ticker", "empty-ticker", "no-percentage"],
)
def test_a_missing_ticker_falls_back_to_the_neutral_phrase(ticker):

    engine = RagEngine(logger=null_logger(), config=make_config())

    assert engine.build_context_query("BTC/USDT", ticker) == "btc price range consolidation"


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
    assert counter.count_tokens(processed) <= 50
    assert re.fullmatch(r"[a-z]+\.\.\.", processed.rsplit(" ", 1)[-1])


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
        ).score

    assert relevance(non_coin, "general market coverage") == 0.0
    assert relevance(coin, "bitcoin momentum continues") == pytest.approx(79.93147180559946)
    assert relevance(coin, "tiny") == pytest.approx(79.93147180559946 * 0.7)



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
        (-3600.0, 1.0),
    ],
    ids=["same-instant", "one-hour-old", "beyond-one-day", "future-timestamp"],
)
def test_recency_factor_decays_and_clamps_at_zero(published_offset, expected):
    factor = ArticleScoringPolicy.calculate_recency_factor(1000.0, 1000.0 - published_offset)

    assert factor == expected


def test_an_article_older_than_a_day_scores_zero():
    policy = make_policy()
    content = make_content(
        "btc price jumps", "bitcoin momentum continues", "btc|markets", "btc", "btc"
    )

    def relevance(pub_time: float) -> float:
        return policy.calculate_article_relevance(
            article={"id": "article", "body": "bitcoin momentum continues"},
            content=content,
            keywords={"price"},
            coin="BTC",
            current_time=1000.0,
            relevant_categories=[],
            important_categories=set(),
            pub_time=pub_time,
            coin_patterns=COIN_PATTERNS,
        ).score

    assert relevance(1000.0) > 0.0
    assert relevance(1000.0 - 86399.0) > 0.0
    assert relevance(1000.0 - 86400.0) == 0.0
    assert relevance(1000.0 - 200000.0) == 0.0


def test_a_long_body_does_not_multiply_the_keyword_score():
    policy = make_policy()
    short = make_content("btc price jumps", "price " * 10, "btc", "btc", "btc")
    long = make_content("btc price jumps", "price " * 1000, "btc", "btc", "btc")

    short_score = policy.calculate_keyword_score({"price"}, short)
    long_score = policy.calculate_keyword_score({"price"}, long)

    assert long.body.count("price") > short.body.count("price")
    assert long_score < short_score


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("coindesk", 1.0),
        ("cointelegraph", 1.0),
        ("decrypt", 0.9),
        ("cryptoslate", 0.8),
        ("unknown-feed", 1.0),
        (None, 1.0),
    ],
    ids=["coindesk", "cointelegraph", "decrypt", "cryptoslate", "unknown", "missing"],
)
def test_the_source_weight_scales_the_article_score(source, expected):
    policy = make_policy()
    article = {"id": "article", "body": "bitcoin momentum continues"}
    if source is not None:
        article["source_info"] = {"name": source}

    assert policy._source_weight(article) == expected


def test_score_component_tables_cover_coin_keyword_category_and_importance():
    policy = make_policy()
    content = make_content(
        "btc price jumps", "bitcoin momentum continues", "btc|markets", "btc", "btc"
    )

    assert policy.calculate_coin_score("BTC", content, COIN_PATTERNS) == 63.0
    assert policy.calculate_keyword_score({"price"}, content) == pytest.approx(16.931471805599452)
    assert policy.calculate_category_score(["btc", "eth"], "btc|eth") == 10.0
    assert policy.calculate_category_score(["btc", "eth"], "BTC|ETH") == 0.0
    assert ArticleScoringPolicy.calculate_importance_score("btc|markets", {"BTC", "MARKETS"}) == 6.0
    assert policy.calculate_density_modifier({}) == 0.7


def test_article_without_a_symbol_scores_keywords_only():
    policy = make_policy()
    content = make_content(
        "btc price jumps", "bitcoin momentum continues", "btc|markets", "btc", "btc"
    )
    article = {"id": "article", "body": "bitcoin momentum continues"}

    score = policy.calculate_article_relevance(
        article=article,
        content=content,
        keywords={"price"},
        coin=None,
        current_time=1000.0,
        relevant_categories=[],
        important_categories=set(),
        pub_time=1000.0,
        coin_patterns=None,
    ).score

    assert score == pytest.approx(
        policy.calculate_keyword_score({"price"}, content)
        * policy.calculate_density_modifier(article)
    )


def debug_builder(enabled: bool) -> ContextBuilder:

    logger = null_logger()
    logger.isEnabledFor.return_value = enabled
    article_processor = MagicMock()
    article_processor.extract_base_coin.return_value = "BTC"
    article_processor.get_article_timestamp.side_effect = lambda article: float(
        article.get("published_on") or 0.0
    )
    return ContextBuilder(
        logger=logger,
        token_counter=WordCounter(),
        config=make_config(**RAG_CONFIG),
        scoring_policy=make_policy(),
        article_processor=article_processor,
    )


async def test_keyword_search_keeps_rankings_identical_when_the_trace_is_off():
    published = datetime.now(timezone.utc).timestamp()
    database = [
        {
            "id": "a1",
            "title": "Bitcoin price rallies",
            "body": "bitcoin momentum builds",
            "categories": "btc|markets",
            "published_on": published,
        },
        {
            "id": "a2",
            "title": "Unrelated weather",
            "body": "nothing here",
            "categories": "misc",
            "published_on": published,
        },
    ]

    scores_on, breakdowns_on = await debug_builder(True).keyword_search(
        "bitcoin price", database, "BTCUSDT", {}, set()
    )
    scores_off, breakdowns_off = await debug_builder(False).keyword_search(
        "bitcoin price", database, "BTCUSDT", {}, set()
    )

    assert [index for index, _score in scores_on] == [index for index, _score in scores_off]
    assert scores_on[0][1] == pytest.approx(scores_off[0][1], rel=1e-6)
    assert breakdowns_off == {}
    assert set(breakdowns_on) == {0, 1}
    assert breakdowns_on[0].score == pytest.approx(scores_on[0][1])
    assert breakdowns_on[0].coin_score > 0
    assert breakdowns_on[1].score == 0.0


def make_file_handler(tmp_path, coin_names=None):
    handler = RagFileHandler(logger=null_logger(), config=make_config(**RAG_CONFIG), coin_names=coin_names)
    handler.base_dir = str(tmp_path)
    handler.tickers_file = str(tmp_path / "known_tickers.json")
    return handler


def test_save_known_tickers_falls_back_to_the_lowercase_ticker_without_a_catalog(tmp_path):
    handler = make_file_handler(tmp_path)

    handler.save_known_tickers(["BTC", "XYZ"])

    data = json.loads((tmp_path / "known_tickers.json").read_text(encoding="utf-8"))
    assert data["symbol_name_map"] == {"BTC": "btc", "XYZ": "xyz"}


def test_save_known_tickers_replaces_a_ticker_placeholder_with_the_catalog_name(tmp_path):
    handler = make_file_handler(
        tmp_path, coin_names=lambda symbol: "Bitcoin" if symbol == "BTC" else None
    )
    (tmp_path / "known_tickers.json").write_text(
        json.dumps({"symbol_name_map": {"BTC": "btc"}}), encoding="utf-8"
    )

    handler.save_known_tickers(["BTC", "XYZ"])

    data = json.loads((tmp_path / "known_tickers.json").read_text(encoding="utf-8"))
    assert data["symbol_name_map"] == {"BTC": "bitcoin", "XYZ": "xyz"}


def test_save_known_tickers_keeps_a_hand_written_name_over_the_catalog(tmp_path):
    handler = make_file_handler(tmp_path, coin_names=lambda symbol: "Bitcoin")
    (tmp_path / "known_tickers.json").write_text(
        json.dumps({"symbol_name_map": {"BTC": "Moja Nazwa"}}), encoding="utf-8"
    )

    handler.save_known_tickers(["BTC"])

    data = json.loads((tmp_path / "known_tickers.json").read_text(encoding="utf-8"))
    assert data["symbol_name_map"] == {"BTC": "moja nazwa"}


def test_a_bitcoin_title_without_the_ticker_still_matches_the_coin():
    builder = ContextBuilder(
        logger=null_logger(),
        token_counter=WordCounter(),
        config=make_config(**RAG_CONFIG),
        scoring_policy=make_policy(),
        article_processor=MagicMock(
            extract_base_coin=MagicMock(return_value="BTC"),
            get_article_timestamp=MagicMock(return_value=1000.0),
        ),
        symbol_name_map={"BTC": "bitcoin"},
    )
    database = [
        {
            "title": "Bitcoin price rallies",
            "body": "Bitcoin price rallies " * 10,
            "published_on": 1000.0,
        }
    ]

    _scores, breakdowns = asyncio.run(
        builder.keyword_search("bitcoin price rallies", database, "BTC/USDC", {}, set())
    )

    assert breakdowns[0].coin_relevance_mult == 1.0


def test_coin_relevance_multiplier_demotes_body_only_and_unmatched_articles():
    policy = make_policy()
    body_only = make_content("new listing", "bitcoin momentum continues", "btc", "btc", "btc")
    unmatched = make_content(
        "altcoin round-up", "general market coverage", "markets", "summary", "eth|sol"
    )

    assert policy._calculate_coin_relevance_multiplier(0.0, body_only, COIN_PATTERNS) == 0.1
    assert policy._calculate_coin_relevance_multiplier(5.0, body_only, COIN_PATTERNS) == 0.35
    assert policy._calculate_coin_relevance_multiplier(5.0, unmatched, COIN_PATTERNS) == 0.35


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


def test_parse_stream_maps_the_message_contract_and_respects_the_limit():
    analyst = SocialSentimentAnalyst(logger=null_logger(), token_counter=TokenCounter())
    posts = analyst._parse_stream(STREAM_SAMPLE, limit=10)

    assert len(posts) == 3
    assert posts[0] == {
        "title": "Bitcoin breaks $70k resistance 'finally'",
        "published_utc": 1786622940,
    }
    assert [post["published_utc"] for post in posts] == [1786622940, 1786622880, 1786622820]
    lengths = [
        len(analyst._parse_stream(STREAM_SAMPLE, limit=limit)) for limit in (10, 2, 0)
    ]

    assert lengths == [3, 2, 0]


def test_parse_stream_rejects_unusable_payloads_and_short_bodies():
    analyst = SocialSentimentAnalyst(logger=null_logger(), token_counter=TokenCounter())

    assert analyst._parse_stream("<not json", limit=10) == []
    assert analyst._parse_stream("", limit=10) == []
    assert analyst._parse_stream({"messages": []}, limit=10) == []
    assert analyst._parse_stream({"unexpected": True}, limit=10) == []
    assert analyst._parse_stream(["not", "a", "dict"], limit=10) == []

    payload = json.loads(STREAM_SAMPLE)
    payload["messages"][0]["body"] = "$BTC.X 🚀"
    payload["messages"][1]["body"] = "   "

    assert [post["title"] for post in analyst._parse_stream(payload, limit=10)] == [
        "Daily discussion thread about the $BTC.X chart",
    ]
    undated = analyst._parse_stream({"messages": [{"body": "Bitcoin price outlook for next week"}]}, limit=10)
    assert undated[0]["published_utc"] == 0


def test_parse_timestamp_normalizes_offsets_and_falls_back_to_zero():
    assert SocialSentimentAnalyst._parse_timestamp("2026-08-13T12:09:00+00:00") == 1786622940
    assert SocialSentimentAnalyst._parse_timestamp("2026-08-13T12:09:00Z") == 1786622940
    assert SocialSentimentAnalyst._parse_timestamp("2026-08-13T12:09:00") == 1786622940
    assert SocialSentimentAnalyst._parse_timestamp("not-a-date") == 0
    assert SocialSentimentAnalyst._parse_timestamp(None) == 0
    assert SocialSentimentAnalyst._parse_timestamp("2026-13-45T99:00:00+00:00") == 0


def test_the_same_story_from_two_feeds_is_kept_once():

    engine = RagEngine(logger=null_logger(), config=make_config())
    database = [
        {
            "title": "Live updates: Bitcoin climbs above $63,000 as Nasdaq soars 6% on Huang return",
            "published_on": 1795438880.0,
        },
        {
            "title": "Live updates: Bitcoin climbs back above $61,000 as stocks rebound from AI rout",
            "published_on": 1795433349.0,
        },
        {"title": "Ethereum upgrade ships next week", "published_on": 1795438880.0},
    ]

    kept = engine._dedupe_by_story([(0, 3.0), (1, 2.0), (2, 1.0)], database)

    assert kept == [0, 2]


def test_the_same_headline_outside_the_two_hour_window_is_kept():

    engine = RagEngine(logger=null_logger(), config=make_config())
    title = "Live updates: Bitcoin climbs above $63,000 as Nasdaq soars 6% on Huang return"
    database = [
        {"title": title, "published_on": 1795438880.0},
        {"title": title, "published_on": 1795438880.0 + 7201.0},
    ]

    assert engine._dedupe_by_story([(0, 2.0), (1, 1.0)], database) == [0, 1]