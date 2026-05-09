"""Tests for the new RSS/Crawl4AI news ingestion pipeline.

Covers:
- URL normalisation
- RSS item parsing
- Publish-date → epoch conversion
- URL-first deduplication
- Deterministic article ID generation
- Canonical schema mapping
- NewsManager.update_news_database() URL-first dedup compatibility
- LocalTaxonomyProvider loading from file
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from src.rag.news_ingestion.rss_primitives import (
    normalize_url,
    parse_pub_date_to_epoch,
    strip_html,
    extract_html_body_text,
    parse_rss_items,
    dedupe_by_url,
    sort_by_date,
)
from src.rag.news_ingestion.schema_mapper import make_article_id, to_article_schema
from src.rag.local_taxonomy import LocalTaxonomyProvider


# ---------------------------------------------------------------------------
# URL normalisation
# ---------------------------------------------------------------------------

class TestNormalizeUrl:
    def test_strips_utm_params(self):
        url = "https://example.com/article?utm_source=twitter&utm_medium=social"
        assert normalize_url(url) == "https://example.com/article"

    def test_preserves_meaningful_params(self):
        url = "https://example.com/article?page=2"
        assert normalize_url(url) == "https://example.com/article?page=2"

    def test_strips_trailing_slash(self):
        url = "https://example.com/article/"
        assert normalize_url(url) == "https://example.com/article"

    def test_preserves_root_slash(self):
        url = "https://example.com/"
        assert normalize_url(url) == "https://example.com/"

    def test_strips_fragment(self):
        url = "https://example.com/article#comments"
        assert normalize_url(url) == "https://example.com/article"

    def test_empty_string(self):
        assert normalize_url("") == ""

    def test_strips_fbclid(self):
        url = "https://example.com/post?fbclid=IwAR123"
        assert normalize_url(url) == "https://example.com/post"


# ---------------------------------------------------------------------------
# Publish-date parsing
# ---------------------------------------------------------------------------

class TestParsePubDateToEpoch:
    def test_rfc2822(self):
        # Wed, 02 Apr 2025 12:00:00 GMT
        raw = "Wed, 02 Apr 2025 12:00:00 GMT"
        epoch = parse_pub_date_to_epoch(raw)
        assert isinstance(epoch, float)
        assert epoch > 0
        dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
        assert dt.year == 2025
        assert dt.month == 4
        assert dt.day == 2

    def test_iso8601(self):
        raw = "2025-04-02T12:00:00+00:00"
        epoch = parse_pub_date_to_epoch(raw)
        assert isinstance(epoch, float)
        dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
        assert dt.year == 2025

    def test_none_returns_zero(self):
        assert parse_pub_date_to_epoch(None) == 0.0

    def test_invalid_returns_zero(self):
        assert parse_pub_date_to_epoch("not-a-date") == 0.0


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------

class TestStripHtml:
    def test_removes_tags(self):
        assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_collapses_whitespace(self):
        assert strip_html("  hello   world  ") == "hello world"


class TestExtractHtmlBodyText:
    def test_extracts_paragraphs(self):
        html = "<html><body><p>First para.</p><p>Second para.</p></body></html>"
        result = extract_html_body_text(html)
        assert "First para." in result
        assert "Second para." in result

    def test_prefers_article_tag(self):
        html = (
            "<html><body><div>sidebar</div>"
            "<article><p>Main content.</p></article></body></html>"
        )
        result = extract_html_body_text(html)
        assert "Main content." in result
        assert "sidebar" not in result

    def test_strips_scripts(self):
        html = "<html><script>alert('x')</script><p>Content</p></html>"
        result = extract_html_body_text(html)
        assert "alert" not in result
        assert "Content" in result


# ---------------------------------------------------------------------------
# RSS parsing
# ---------------------------------------------------------------------------

_SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Crypto News</title>
    <item>
      <title>Bitcoin Hits New High</title>
      <link>https://example.com/btc-high?utm_source=rss</link>
      <guid>https://example.com/btc-high</guid>
      <pubDate>Wed, 02 Apr 2025 10:00:00 GMT</pubDate>
      <description>&lt;p&gt;Bitcoin reached $100k today.&lt;/p&gt;</description>
      <category>BTC</category>
      <category>Bitcoin</category>
    </item>
    <item>
      <title>Ethereum Update</title>
      <link>https://example.com/eth-update</link>
      <pubDate>Wed, 02 Apr 2025 09:00:00 GMT</pubDate>
      <description>Ethereum has a new upgrade.</description>
    </item>
  </channel>
</rss>"""


class TestParseRssItems:
    def test_parses_two_items(self):
        items = parse_rss_items(_SAMPLE_RSS, "TestSource")
        assert len(items) == 2

    def test_strips_utm_from_url(self):
        items = parse_rss_items(_SAMPLE_RSS, "TestSource")
        assert "utm_source" not in items[0]["url"]
        assert items[0]["url"] == "https://example.com/btc-high"

    def test_parses_categories(self):
        items = parse_rss_items(_SAMPLE_RSS, "TestSource")
        assert "BTC" in items[0]["categories"]
        assert "Bitcoin" in items[0]["categories"]

    def test_epoch_timestamp(self):
        items = parse_rss_items(_SAMPLE_RSS, "TestSource")
        assert items[0]["published_at_epoch"] > 0

    def test_source_name(self):
        items = parse_rss_items(_SAMPLE_RSS, "CoinDesk")
        assert items[0]["source_name"] == "CoinDesk"

    def test_body_text_extracted_from_html_description(self):
        items = parse_rss_items(_SAMPLE_RSS, "TestSource")
        assert "Bitcoin reached" in items[0]["body_text"]

    def test_max_items_respected(self):
        items = parse_rss_items(_SAMPLE_RSS, "TestSource", max_items=1)
        assert len(items) == 1

    def test_skips_items_without_title_or_url(self):
        rss = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item><title></title><link>https://example.com/a</link></item>
  <item><title>Good title</title><link></link></item>
  <item><title>Valid</title><link>https://example.com/valid</link></item>
</channel></rss>"""
        items = parse_rss_items(rss, "src")
        assert len(items) == 1
        assert items[0]["title"] == "Valid"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDedupeByUrl:
    def _make_item(self, url: str, epoch: float, title: str = "T") -> dict[str, Any]:
        return {"url": url, "published_at_epoch": epoch, "title": title}

    def test_removes_duplicate_urls(self):
        items = [
            self._make_item("https://a.com/1", 1000.0),
            self._make_item("https://a.com/1", 2000.0, "newer"),
            self._make_item("https://a.com/2", 1500.0),
        ]
        result = dedupe_by_url(items)
        assert len(result) == 2

    def test_keeps_newer_version(self):
        items = [
            self._make_item("https://a.com/1", 1000.0, "old"),
            self._make_item("https://a.com/1", 2000.0, "new"),
        ]
        result = dedupe_by_url(items)
        assert result[0]["title"] == "new"

    def test_skips_empty_url(self):
        items = [
            {"url": "", "published_at_epoch": 1000.0},
            {"url": None, "published_at_epoch": 1000.0},
            {"url": "https://a.com/1", "published_at_epoch": 1000.0},
        ]
        result = dedupe_by_url(items)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Schema mapping and deterministic ID
# ---------------------------------------------------------------------------

class TestMakeArticleId:
    def test_deterministic(self):
        url = "https://coindesk.com/article/btc-100k"
        assert make_article_id(url) == make_article_id(url)

    def test_different_urls_different_ids(self):
        id1 = make_article_id("https://coindesk.com/a")
        id2 = make_article_id("https://coindesk.com/b")
        assert id1 != id2

    def test_length_16_hex(self):
        article_id = make_article_id("https://example.com/x")
        assert len(article_id) == 16
        assert all(c in "0123456789abcdef" for c in article_id)


class TestToArticleSchema:
    def _make_raw(self, **overrides) -> dict[str, Any]:
        base = {
            "url": "https://coindesk.com/article/btc",
            "title": "BTC Hits 100k",
            "body_text": "Full article body here.",
            "summary": "Short summary.",
            "published_at_epoch": 1743590400.0,
            "source_name": "coindesk",
            "categories": ["BTC", "Bitcoin"],
        }
        base.update(overrides)
        return base

    def test_required_fields_present(self):
        article = to_article_schema(self._make_raw())
        for field in ("id", "title", "body", "categories", "tags", "published_on", "source_info", "url"):
            assert field in article

    def test_id_is_deterministic(self):
        raw = self._make_raw()
        a1 = to_article_schema(raw)
        a2 = to_article_schema(raw)
        assert a1["id"] == a2["id"]

    def test_body_prefers_body_text(self):
        raw = self._make_raw(body_text="Full body", summary="Short")
        article = to_article_schema(raw)
        assert article["body"] == "Full body"

    def test_body_falls_back_to_summary(self):
        raw = self._make_raw(body_text="", summary="Short summary")
        article = to_article_schema(raw)
        assert article["body"] == "Short summary"

    def test_categories_pipe_separated(self):
        raw = self._make_raw(categories=["BTC", "DeFi"])
        article = to_article_schema(raw)
        assert article["categories"] == "BTC|DeFi"

    def test_published_on_is_float(self):
        raw = self._make_raw(published_at_epoch=1743590400.0)
        article = to_article_schema(raw)
        assert isinstance(article["published_on"], float)
        assert article["published_on"] == 1743590400.0

    def test_source_info_has_name(self):
        raw = self._make_raw(source_name="cointelegraph")
        article = to_article_schema(raw)
        assert article["source_info"] == {"name": "cointelegraph"}

    def test_coindesk_tail_navigation_clutter_is_trimmed(self):
        content_prefix = " ".join(["Bitcoin market structure remains constructive."] * 40)
        clutter_tail = (
            " * * * * * * About  * About Us  * Masthead  * Careers  * Blog  * Investor Relations "
            "Contact  * Contact Us  * Accessibility  * Advertise  * Media Kit  * Sitemap  "
            "Newsletters  * CoinDesk Headlines  * Crypto Daybook Americas"
        )
        raw = self._make_raw(
            source_name="coindesk",
            body_text=f"{content_prefix}\n\n{clutter_tail}",
        )

        article = to_article_schema(raw)

        assert "Bitcoin market structure remains constructive." in article["body"]
        assert "CoinDesk Headlines" not in article["body"]
        assert "Contact Us" not in article["body"]

    def test_early_boilerplate_marker_is_trimmed(self):
        body = (
            "Bitcoin market structure remains constructive. "
            "More For You * About Us * Contact Us * Newsletters * Latest Crypto News"
        )
        raw = self._make_raw(source_name="coindesk", body_text=body)

        article = to_article_schema(raw)

        assert article["body"] == "Bitcoin market structure remains constructive."

    def test_decrypt_price_ticker_prefix_is_trimmed(self):
        price_rows = "\n".join(
            f"${index + 1}.00\n{index + 0.25:.2f}%"
            for index in range(12)
        )
        title = "Banking Industry Says Clarity Act Stablecoin Proposal Would Enable Evasion"
        body = (
            "* * *\n"
            "NewsPredictAILearnGaming\n"
            f"{price_rows}\n"
            "Price data by\n"
            "* * *\n"
            "DecryptNewsLaw and Order\n"
            "* * *\n"
            f"{title}. "
            "The article continues with policy context."
        )
        raw = self._make_raw(title=title, source_name="decrypt", body_text=body)

        article = to_article_schema(raw)

        assert article["body"].startswith("The article continues")
        assert "NewsPredictAILearnGaming" not in article["body"]
        assert "Price data by" not in article["body"]
        assert "$" not in article["body"][:100]

    def test_price_prose_is_preserved(self):
        body = (
            "Bitcoin ETF demand rose 12% while market cap held near $1.5 trillion. "
            "Analysts said spot liquidity remained orderly."
        )
        raw = self._make_raw(source_name="decrypt", body_text=body)

        article = to_article_schema(raw)

        assert article["body"] == body

    def test_early_marker_like_phrase_is_not_trimmed(self):
        body = (
            "Top Stories in derivatives can still be misleading without volume context. "
            + " ".join(["This paragraph is analysis, not site navigation."] * 35)
        )
        raw = self._make_raw(source_name="coindesk", body_text=body)

        article = to_article_schema(raw)

        assert "Top Stories in derivatives" in article["body"]


# ---------------------------------------------------------------------------
# NewsManager.update_news_database() - URL-first dedup
# ---------------------------------------------------------------------------

class TestNewsManagerUrlDedup:
    """Verify that update_news_database uses URL-first deduplication."""

    def _make_manager(self):
        from src.rag.news_manager import NewsManager
        logger = MagicMock()
        file_handler = MagicMock()
        article_processor = MagicMock()
        article_processor.get_article_timestamp.side_effect = lambda a: a.get("published_on", 0)

        def fake_filter_by_age(articles, max_age_seconds):
            now = time.time()
            return [a for a in articles if a.get("published_on", 0) > now - max_age_seconds]

        file_handler.filter_articles_by_age.side_effect = fake_filter_by_age

        manager = NewsManager(
            logger=logger,
            file_handler=file_handler,
            article_processor=article_processor,
        )
        return manager

    def test_new_url_is_added(self):
        mgr = self._make_manager()
        now = time.time()
        article = {
            "id": "abc123",
            "url": "https://coindesk.com/btc",
            "published_on": now - 3600,
            "title": "BTC article",
        }
        result = mgr.update_news_database([article])
        assert result is True
        assert len(mgr.news_database) == 1

    def test_duplicate_url_is_rejected(self):
        mgr = self._make_manager()
        now = time.time()
        article = {
            "id": "abc123",
            "url": "https://coindesk.com/btc",
            "published_on": now - 3600,
        }
        mgr.update_news_database([article])

        # Same URL, different ID (as would happen if ID scheme changed)
        duplicate = {
            "id": "xyz999",
            "url": "https://coindesk.com/btc",
            "published_on": now - 1800,
        }
        result = mgr.update_news_database([duplicate])
        assert result is False


# ---------------------------------------------------------------------------
# LocalTaxonomyProvider
# ---------------------------------------------------------------------------

_SAMPLE_CATEGORIES = [
    {
        "categoryName": "BTC",
        "wordsAssociatedWithCategory": ["BTC", "Bitcoin", "bitcoin"],
        "includedPhrases": ["BITCOIN NETWORK"],
    },
    {
        "categoryName": "ETH",
        "wordsAssociatedWithCategory": ["ETH", "Ethereum"],
    },
]

_SAMPLE_CATEGORIES_JSON = {
    "timestamp": "2026-01-01T00:00:00+00:00",
    "categories": _SAMPLE_CATEGORIES,
}


class TestLocalTaxonomyProvider:
    @pytest.mark.asyncio
    async def test_loads_from_file(self):
        logger = MagicMock()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as fh:
            json.dump(_SAMPLE_CATEGORIES_JSON, fh)
            tmp_path = fh.name

        try:
            provider = LocalTaxonomyProvider(logger, categories_file=tmp_path)
            categories = await provider.fetch_categories()
            assert len(categories) == 2
            names = [c["categoryName"] for c in categories]
            assert "BTC" in names
            assert "ETH" in names
        finally:
            os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_loads_plain_list(self):
        logger = MagicMock()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as fh:
            json.dump(_SAMPLE_CATEGORIES, fh)
            tmp_path = fh.name

        try:
            provider = LocalTaxonomyProvider(logger, categories_file=tmp_path)
            categories = await provider.fetch_categories()
            assert len(categories) == 2
        finally:
            os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_caches_in_memory(self):
        logger = MagicMock()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as fh:
            json.dump(_SAMPLE_CATEGORIES_JSON, fh)
            tmp_path = fh.name

        try:
            provider = LocalTaxonomyProvider(logger, categories_file=tmp_path)
            await provider.fetch_categories()
            # Remove file; second call should still succeed from cache
            os.unlink(tmp_path)
            categories = await provider.fetch_categories()
            assert len(categories) == 2
        except FileNotFoundError:
            os.unlink(tmp_path) if os.path.exists(tmp_path) else None

    @pytest.mark.asyncio
    async def test_force_refresh_reloads(self):
        logger = MagicMock()
        initial = [{"categoryName": "BTC", "wordsAssociatedWithCategory": ["BTC"]}]
        updated = [
            {"categoryName": "BTC", "wordsAssociatedWithCategory": ["BTC"]},
            {"categoryName": "SOL", "wordsAssociatedWithCategory": ["SOL", "Solana"]},
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as fh:
            json.dump(initial, fh)
            tmp_path = fh.name

        try:
            provider = LocalTaxonomyProvider(logger, categories_file=tmp_path)
            cats1 = await provider.fetch_categories()
            assert len(cats1) == 1

            # Overwrite file
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(updated, fh)

            cats2 = await provider.fetch_categories(force_refresh=True)
            assert len(cats2) == 2
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_missing_file_returns_empty(self):
        logger = MagicMock()
        provider = LocalTaxonomyProvider(
            logger, categories_file="/nonexistent/path/categories.json"
        )
        categories = await provider.fetch_categories()
        assert categories == []
