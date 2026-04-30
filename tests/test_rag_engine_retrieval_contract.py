from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.rag.rag_engine import RagEngine


class _Config:
    RAG_UPDATE_INTERVAL_HOURS = 1
    RAG_NEWS_LIMIT = 2
    RAG_ARTICLE_MAX_TOKENS = 200
    RAG_NEWS_ENRICH_MIN_CHARS = 20


@pytest.fixture
def logger() -> MagicMock:
    return MagicMock()


@pytest.fixture
def base_components(logger: MagicMock):
    news_manager = MagicMock()
    news_manager.news_database = [
        {
            "id": "a1",
            "title": "Long body first",
            "body": "L" * 60,
            "published_on": datetime.now(timezone.utc).timestamp(),
        },
        {
            "id": "a2",
            "title": "Short body high score",
            "body": "tiny",
            "published_on": datetime.now(timezone.utc).timestamp(),
        },
        {
            "id": "a3",
            "title": "Long body second",
            "body": "B" * 40,
            "published_on": datetime.now(timezone.utc).timestamp(),
        },
    ]
    news_manager.get_database_size.return_value = len(news_manager.news_database)

    category_fetcher = MagicMock()
    category_fetcher.fetch_categories = AsyncMock(return_value=None)

    category_processor = MagicMock()
    category_processor.category_word_map = {}
    category_processor.important_categories = set()
    category_processor.extract_base_coin.return_value = "BTC"
    category_processor.process_api_categories.return_value = None

    index_manager = MagicMock()
    index_manager.get_coin_indices.return_value = {}
    index_manager.search_by_coin.return_value = []

    context_builder = MagicMock()
    context_builder.keyword_search = AsyncMock(return_value=[(1, 10.0), (0, 9.0), (2, 8.0)])
    context_builder.add_articles_to_context.return_value = ("context text", 123)
    context_builder.get_latest_article_urls.return_value = {"Long body first": "https://example.com/a1"}

    market_data_manager = MagicMock()
    market_data_manager.update_market_overview_if_needed = AsyncMock(return_value=False)

    ticker_manager = MagicMock()

    engine = RagEngine(
        logger=logger,
        token_counter=MagicMock(),
        config=cast(Any, _Config()),
        file_handler=MagicMock(),
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
    )


@pytest.mark.asyncio
async def test_retrieve_context_prioritizes_full_body_candidates(base_components):
    ctx = await base_components.engine.retrieve_context(
        query="bitcoin regulation headlines",
        symbol="BTCUSDT",
        k=2,
        max_tokens=300,
    )

    assert ctx == "context text"

    add_call = base_components.context_builder.add_articles_to_context.call_args
    relevant_indices = add_call.args[0]
    assert relevant_indices[:3] == [0, 2, 1]


@pytest.mark.asyncio
async def test_retrieve_context_uses_coin_fallback_when_scores_are_sparse(base_components):
    base_components.context_builder.keyword_search = AsyncMock(return_value=[(1, 12.0)])
    base_components.index_manager.search_by_coin.return_value = [2]

    await base_components.engine.retrieve_context(
        query="btc liquidity",
        symbol="BTCUSDT",
        k=3,
        max_tokens=300,
    )

    add_call = base_components.context_builder.add_articles_to_context.call_args
    relevant_indices = add_call.args[0]
    assert 2 in relevant_indices
    base_components.category_processor.extract_base_coin.assert_called_once_with("BTCUSDT")


@pytest.mark.asyncio
async def test_retrieve_context_returns_empty_for_empty_news_db(logger: MagicMock):
    news_manager = MagicMock()
    news_manager.news_database = []
    news_manager.get_database_size.return_value = 0

    engine = RagEngine(
        logger=logger,
        token_counter=MagicMock(),
        config=cast(Any, _Config()),
        file_handler=MagicMock(),
        news_manager=news_manager,
        market_data_manager=MagicMock(),
        index_manager=MagicMock(),
        category_fetcher=MagicMock(fetch_categories=AsyncMock(return_value=None)),
        category_processor=MagicMock(
            category_word_map={},
            important_categories=set(),
            process_api_categories=MagicMock(),
        ),
        ticker_manager=MagicMock(),
        context_builder=MagicMock(),
    )

    result = await engine.retrieve_context(query="anything", symbol="BTCUSDT")

    assert result == ""


@pytest.mark.asyncio
async def test_retrieve_context_returns_error_message_on_exception(base_components):
    base_components.context_builder.keyword_search = AsyncMock(side_effect=RuntimeError("boom"))

    result = await base_components.engine.retrieve_context(
        query="eth staking",
        symbol="ETHUSDT",
        k=2,
    )

    assert result == "Error retrieving market context."


@pytest.mark.asyncio
async def test_update_if_needed_triggers_refresh_when_stale(base_components):
    base_components.engine.refresh_market_data = AsyncMock(return_value=None)
    base_components.engine.last_update = datetime.now(timezone.utc) - timedelta(hours=2)

    updated = await base_components.engine.update_if_needed(force_update=False)

    assert updated is True
    base_components.engine.refresh_market_data.assert_awaited_once()


@pytest.mark.asyncio
async def test_retrieve_context_updates_latest_article_urls_snapshot(base_components):
    await base_components.engine.retrieve_context(
        query="btc headlines",
        symbol="BTCUSDT",
        k=2,
        max_tokens=300,
    )

    urls = base_components.engine.get_latest_article_urls_snapshot()
    assert urls == {"Long body first": "https://example.com/a1"}


def test_get_news_cache_snapshot_returns_copy(base_components):
    articles_snapshot = base_components.engine.get_news_cache_snapshot()

    assert len(articles_snapshot) == len(base_components.news_manager.news_database)

    # Ensure caller mutations do not alter manager state.
    articles_snapshot[0]["title"] = "mutated"
    assert base_components.news_manager.news_database[0]["title"] == "Long body first"


def test_build_context_query_uses_symbol_name_map_when_available(base_components):
    base_components.engine.context_builder.symbol_name_map = {"BTC": "bitcoin"}
    base_components.category_processor.extract_base_coin.return_value = "BTC"

    query = base_components.engine.build_context_query("BTCUSDT")

    assert query == "bitcoin price analysis market trends"
