"""
Contract tests for RSSCrawl4AINewsProvider.

Locks filter_by_age and the orchestration of fetch_news (via mocked stages)
before extracting private helper methods.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.news_ingestion.rss_provider import RSSCrawl4AINewsProvider


# ---------------------------------------------------------------------------
# Minimal config stub
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> SimpleNamespace:
    defaults = dict(
        RAG_NEWS_SOURCES=None,
        RAG_NEWS_SOURCE_URLS=None,
        RAG_NEWS_MAX_ITEMS_PER_SOURCE=50,
        RAG_NEWS_FETCH_TIMEOUT=30,
        RAG_NEWS_FETCH_TOTAL_TIMEOUT=45,
        RAG_NEWS_PAGE_ENRICHMENT=False,
        RAG_NEWS_ENRICH_TIMEOUT=120,
        RAG_NEWS_CRAWL4AI_ENABLED=False,
        RAG_NEWS_CRAWL_CONCURRENCY=3,
        RAG_NEWS_CRAWL_TIMEOUT=30,
        RAG_NEWS_ENRICH_MIN_CHARS=400,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_provider(config=None) -> RSSCrawl4AINewsProvider:
    return RSSCrawl4AINewsProvider(
        logger=MagicMock(),
        config=config or _make_config(),
        enricher=MagicMock(),
    )


def _article(age_hours: float, **kwargs) -> dict[str, Any]:
    ts = (datetime.now(timezone.utc) - timedelta(hours=age_hours)).timestamp()
    return {"published_on": ts, "url": "https://example.com/a", "title": "t", **kwargs}


# ---------------------------------------------------------------------------
# filter_by_age
# ---------------------------------------------------------------------------

class TestFilterByAge:
    def test_recent_articles_kept(self):
        provider = _make_provider()
        articles = [_article(1), _article(5)]
        result = provider.filter_by_age(articles, max_age_hours=24)
        assert len(result) == 2

    def test_old_articles_removed(self):
        provider = _make_provider()
        articles = [_article(1), _article(48)]
        result = provider.filter_by_age(articles, max_age_hours=24)
        assert len(result) == 1

    def test_all_old_articles_returns_empty(self):
        provider = _make_provider()
        articles = [_article(48), _article(72)]
        result = provider.filter_by_age(articles, max_age_hours=24)
        assert result == []

    def test_returns_sorted_newest_first(self):
        provider = _make_provider()
        articles = [_article(10), _article(2), _article(6)]
        result = provider.filter_by_age(articles, max_age_hours=24)
        timestamps = [a["published_on"] for a in result]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_missing_published_on_excluded(self):
        provider = _make_provider()
        articles = [{"url": "x", "title": "t"}]  # no published_on
        result = provider.filter_by_age(articles, max_age_hours=24)
        assert result == []

    def test_empty_input_returns_empty(self):
        provider = _make_provider()
        assert provider.filter_by_age([], 24) == []


# ---------------------------------------------------------------------------
# _enabled_source_names
# ---------------------------------------------------------------------------

class TestEnabledSourceNames:
    def test_none_config_returns_none(self):
        provider = _make_provider(_make_config(RAG_NEWS_SOURCES=None))
        assert provider._enabled_source_names() is None

    def test_empty_string_returns_none(self):
        provider = _make_provider(_make_config(RAG_NEWS_SOURCES=""))
        assert provider._enabled_source_names() is None

    def test_comma_separated_string_parsed(self):
        provider = _make_provider(_make_config(RAG_NEWS_SOURCES="coindesk, decrypt"))
        assert provider._enabled_source_names() == ["coindesk", "decrypt"]

    def test_list_passthrough(self):
        provider = _make_provider(_make_config(RAG_NEWS_SOURCES=["coindesk", "decrypt"]))
        assert provider._enabled_source_names() == ["coindesk", "decrypt"]

    def test_list_strips_whitespace(self):
        provider = _make_provider(_make_config(RAG_NEWS_SOURCES=["  coindesk  "]))
        assert provider._enabled_source_names() == ["coindesk"]


# ---------------------------------------------------------------------------
# fetch_news – integration contract via mocked primitives
# ---------------------------------------------------------------------------

class TestFetchNewsOrchestration:
    """Test fetch_news orchestration by mocking the I/O boundaries."""

    @pytest.fixture
    def provider(self):
        return _make_provider(_make_config(RAG_NEWS_PAGE_ENRICHMENT=False))

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_sources(self, provider):
        with patch("src.rag.news_ingestion.rss_provider.get_sources", return_value=[]):
            result = await provider.fetch_news()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_all_sources_fail(self, provider):
        from src.rag.news_ingestion.rss_primitives import FetchResult
        failing = FetchResult(source_name="coindesk", source_type="rss", url="x", success=False, status_code=None, error="timeout", normalized_items=[])
        with patch("src.rag.news_ingestion.rss_provider.get_sources", return_value=[{"name": "coindesk", "url": "x"}]), \
             patch("src.rag.news_ingestion.rss_provider.fetch_source", new=AsyncMock(return_value=failing)):
            result = await provider.fetch_news()
        assert result == []

    @pytest.mark.asyncio
    async def test_articles_mapped_to_schema(self, provider):
        from src.rag.news_ingestion.rss_primitives import FetchResult
        raw_item = {
            "url": "https://example.com/btc-news",
            "title": "BTC News",
            "body_text": "Some body text",
            "summary": "",
            "categories": ["BTC"],
            "published_at_epoch": 1_700_000_000.0,
            "source_name": "coindesk",
        }
        ok_result = FetchResult(source_name="coindesk", source_type="rss", url="x", success=True, status_code=200, error=None, normalized_items=[raw_item])
        with patch("src.rag.news_ingestion.rss_provider.get_sources", return_value=[{"name": "coindesk", "url": "x"}]), \
             patch("src.rag.news_ingestion.rss_provider.fetch_source", new=AsyncMock(return_value=ok_result)), \
             patch("src.rag.news_ingestion.rss_provider.dedupe_by_url", side_effect=lambda x: x), \
             patch("src.rag.news_ingestion.rss_provider.sort_by_date", side_effect=lambda x: x):
            result = await provider.fetch_news()

        assert len(result) == 1
        assert result[0]["url"] == "https://example.com/btc-news"
        assert result[0]["title"] == "BTC News"
        assert "id" in result[0]

    @pytest.mark.asyncio
    async def test_enrichment_called_when_enabled(self):
        config = _make_config(RAG_NEWS_PAGE_ENRICHMENT=True)
        provider = _make_provider(config)
        from src.rag.news_ingestion.rss_primitives import FetchResult
        raw_item = {
            "url": "https://example.com/eth-news",
            "title": "ETH News",
            "body_text": "Short",
            "summary": "",
            "categories": [],
            "published_at_epoch": 1_700_000_000.0,
            "source_name": "coindesk",
        }
        ok_result = FetchResult(source_name="coindesk", source_type="rss", url="x", success=True, status_code=200, error=None, normalized_items=[raw_item])
        mock_enrich = AsyncMock(return_value=1)
        provider._enricher.enrich_items = mock_enrich

        with patch("src.rag.news_ingestion.rss_provider.get_sources", return_value=[{"name": "coindesk", "url": "x"}]), \
             patch("src.rag.news_ingestion.rss_provider.fetch_source", new=AsyncMock(return_value=ok_result)), \
             patch("src.rag.news_ingestion.rss_provider.dedupe_by_url", side_effect=lambda x: x), \
             patch("src.rag.news_ingestion.rss_provider.sort_by_date", side_effect=lambda x: x):
            await provider.fetch_news()

        mock_enrich.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_enrichment_skipped_when_disabled(self):
        provider = _make_provider(_make_config(RAG_NEWS_PAGE_ENRICHMENT=False))
        from src.rag.news_ingestion.rss_primitives import FetchResult
        raw_item = {
            "url": "https://example.com/sol-news",
            "title": "SOL News",
            "body_text": "Short",
            "summary": "",
            "categories": [],
            "published_at_epoch": 1_700_000_000.0,
            "source_name": "decrypt",
        }
        ok_result = FetchResult(source_name="decrypt", source_type="rss", url="x", success=True, status_code=200, error=None, normalized_items=[raw_item])
        mock_enrich = AsyncMock(return_value=0)
        provider._enricher.enrich_items = mock_enrich

        with patch("src.rag.news_ingestion.rss_provider.get_sources", return_value=[{"name": "decrypt", "url": "x"}]), \
             patch("src.rag.news_ingestion.rss_provider.fetch_source", new=AsyncMock(return_value=ok_result)), \
             patch("src.rag.news_ingestion.rss_provider.dedupe_by_url", side_effect=lambda x: x), \
             patch("src.rag.news_ingestion.rss_provider.sort_by_date", side_effect=lambda x: x):
            await provider.fetch_news()

        mock_enrich.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fetch_stage_timeout_returns_empty(self, provider):
        with patch("src.rag.news_ingestion.rss_provider.get_sources", return_value=[{"name": "coindesk", "url": "x"}]), \
             patch("src.rag.news_ingestion.rss_provider.asyncio.wait_for", new=AsyncMock(side_effect=asyncio.TimeoutError())):
            result = await provider.fetch_news()

        assert result == []

    @pytest.mark.asyncio
    async def test_enrichment_timeout_continues_with_mapped_articles(self):
        config = _make_config(RAG_NEWS_PAGE_ENRICHMENT=True)
        provider = _make_provider(config)
        provider._enricher.enrich_items = AsyncMock(side_effect=asyncio.TimeoutError())
        merged = [{
            "url": "https://example.com/ada-news",
            "title": "ADA News",
            "body_text": "Short body",
            "summary": "",
            "categories": [],
            "published_at_epoch": 1_700_000_000.0,
            "source_name": "coindesk",
        }]

        with patch("src.rag.news_ingestion.rss_provider.dedupe_by_url", side_effect=lambda x: x), \
             patch("src.rag.news_ingestion.rss_provider.sort_by_date", side_effect=lambda x: x):
            result = await provider._postprocess_items(merged, session=MagicMock())

        assert len(result) == 1
        assert result[0]["url"] == "https://example.com/ada-news"
