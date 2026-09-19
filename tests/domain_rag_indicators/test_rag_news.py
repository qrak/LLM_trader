"""Dense domain tests for the RSS/Crawl4AI news ingestion pipeline.

Covers the shared ingestion primitives (URL canonicalisation, RSS parsing,
publish-date conversion, HTML body extraction), the deterministic canonical
article schema, the provider's fetch/enrich orchestration contract, the
Crawl4AI enricher, the news repository persistence boundary and the two
consumers that close the loop: NewsManager's URL-first merge and
LocalTaxonomyProvider.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import ExitStack, nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from src.rag.article_processor import ArticleProcessor
from src.rag.local_taxonomy import LocalTaxonomyProvider
from src.rag.news_ingestion.crawl4ai_enricher import Crawl4AIEnricher
from src.rag.news_ingestion.rss_primitives import (
    FetchResult,
    dedupe_by_normalized_title,
    dedupe_by_url,
    extract_html_body_text,
    normalize_url,
    parse_pub_date_to_epoch,
    parse_rss_items,
    sort_by_date,
    strip_html,
)
from src.rag.news_ingestion.rss_provider import RSSCrawl4AINewsProvider
from src.rag.news_ingestion.schema_mapper import make_article_id, to_article_schema
from src.rag.news_manager import NewsManager
from src.rag.news_repository import NewsRepository
from tests.conftest import make_config, null_logger

GET_SOURCES = "src.rag.news_ingestion.rss_provider.get_sources"
FETCH_SOURCE = "src.rag.news_ingestion.rss_provider.fetch_source"
STAGE_WAIT_FOR = "src.rag.news_ingestion.rss_provider.asyncio.wait_for"

SOURCE_ENTRY = {"name": "coindesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/"}
FAILED_FETCH = FetchResult(
    source_name="coindesk",
    source_type="rss",
    url="x",
    success=False,
    status_code=None,
    error="timeout",
)

SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
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

UNDATED_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Invalid date</title>
      <link>https://example.com/bad-date</link>
      <pubDate>not-a-date</pubDate>
      <description>Body text.</description>
    </item>
    <item>
      <title>No date</title>
      <link>https://example.com/no-date</link>
      <description>Body text.</description>
    </item>
  </channel>
</rss>"""

COINDESK_BODY_PREFIX = " ".join(["Bitcoin market structure remains constructive."] * 40)
COINDESK_NAV_TAIL = (
    " * * * * * * About  * About Us  * Masthead  * Careers  * Blog  * Investor Relations "
    "Contact  * Contact Us  * Accessibility  * Advertise  * Media Kit  * Sitemap  "
    "Newsletters  * CoinDesk Headlines  * Crypto Daybook Americas"
)
DECRYPT_TITLE = "Banking Industry Says Clarity Act Stablecoin Proposal Would Enable Evasion"
DECRYPT_PRICE_ROWS = "\n".join(f"${index}.00\n{index + 0.25:.2f}%" for index in range(12))
DECRYPT_BODY = (
    "* * *\nNewsPredictAILearnGaming\n"
    f"{DECRYPT_PRICE_ROWS}\n"
    "Price data by\n"
    "* * *\n"
    "DecryptNewsLaw and Order\n"
    "* * *\n"
    f"{DECRYPT_TITLE}. The article continues with policy context."
)
NAVIGATION_LIKE_PROSE = (
    "Top Stories in derivatives can still be misleading without volume context. "
    + " ".join(["This paragraph is analysis, not site navigation."] * 35)
)
PRICE_PROSE = (
    "Bitcoin ETF demand rose 12% while market cap held near $1.5 trillion. "
    "Analysts said spot liquidity remained orderly."
)

TAXONOMY_CATEGORIES = [
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
TAXONOMY_DOCUMENT = {
    "timestamp": "2026-01-01T00:00:00+00:00",
    "categories": TAXONOMY_CATEGORIES,
}
TAXONOMY_INITIAL = [{"categoryName": "BTC", "wordsAssociatedWithCategory": ["BTC"]}]
TAXONOMY_UPDATED = TAXONOMY_INITIAL + [
    {"categoryName": "SOL", "wordsAssociatedWithCategory": ["SOL", "Solana"]}
]


def provider_config(**overrides: Any) -> SimpleNamespace:
    """Config double for the ingestion provider.

    conftest production defaults plus the RAG_* keys the provider and the
    enricher actually read; only changed values need stating.
    """
    values: dict[str, Any] = {
        "RAG_NEWS_SOURCE_URLS": None,
        "RAG_NEWS_MAX_ITEMS_PER_SOURCE": 50,
        "RAG_NEWS_FETCH_TOTAL_TIMEOUT": 45,
        "RAG_NEWS_PAGE_ENRICHMENT": False,
        "RAG_NEWS_CRAWL_TIMEOUT": 30,
        "RAG_NEWS_CRAWL_CONCURRENCY": 3,
        "RAG_NEWS_ENRICH_MIN_CHARS": 400,
    }
    values.update(overrides)
    return make_config(**values)


def make_provider(**overrides: Any) -> tuple[RSSCrawl4AINewsProvider, MagicMock]:
    """Provider wired to a mocked enricher, returned together with that double."""
    enricher = MagicMock()
    provider = RSSCrawl4AINewsProvider(
        logger=null_logger(),
        config=provider_config(**overrides),
        enricher=enricher,
    )
    return provider, enricher


def raw_item(**overrides: Any) -> dict[str, Any]:
    """Raw ingested item, i.e. the pre-canonical shape produced by parse_rss_items."""
    values: dict[str, Any] = {
        "url": "https://coindesk.com/article/btc",
        "title": "BTC Hits 100k",
        "body_text": "Full article body here.",
        "summary": "Short summary.",
        "published_at_epoch": 1743590400.0,
        "source_name": "coindesk",
        "categories": ["BTC", "Bitcoin"],
    }
    values.update(overrides)
    return values


def canonical_article(age_hours: float, **overrides: Any) -> dict[str, Any]:
    """Canonical-schema article published *age_hours* ago."""
    published_on = (datetime.now(timezone.utc) - timedelta(hours=age_hours)).timestamp()
    return {
        "url": "https://example.com/a",
        "title": "t",
        "published_on": published_on,
        **overrides,
    }


def make_news_manager() -> tuple[NewsManager, MagicMock]:
    """NewsManager over mocked persistence with a real wall-clock age filter."""
    file_handler = MagicMock()
    article_processor = MagicMock()
    article_processor.get_article_timestamp.side_effect = lambda item: item.get("published_on", 0)
    file_handler.filter_articles_by_age.side_effect = lambda items, max_age_seconds: [
        item for item in items if item.get("published_on", 0) > time.time() - max_age_seconds
    ]
    manager = NewsManager(
        logger=null_logger(),
        file_handler=file_handler,
        article_processor=article_processor,
    )
    return manager, file_handler


def make_news_repository() -> tuple[NewsRepository, MagicMock]:
    """NewsRepository over a mocked file handler."""
    file_handler = MagicMock()
    return NewsRepository(logger=null_logger(), file_handler=file_handler), file_handler


def write_taxonomy(path: Path, payload: Any) -> str:
    """Write taxonomy JSON from sync context and return the path as a string."""
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize(
    ("raw_url", "expected"),
    [
        (
            "https://example.com/article?utm_source=twitter&utm_medium=social",
            "https://example.com/article",
        ),
        ("https://example.com/article?page=2", "https://example.com/article?page=2"),
        ("https://example.com/article/", "https://example.com/article"),
        ("https://example.com/", "https://example.com/"),
        ("https://example.com/article#comments", "https://example.com/article"),
        ("", ""),
        ("https://example.com/post?fbclid=IwAR123", "https://example.com/post"),
    ],
    ids=[
        "tracking-params",
        "meaningful-query",
        "trailing-slash",
        "root-slash",
        "fragment",
        "empty",
        "fbclid",
    ],
)
def test_normalize_url_canonicalizes_tracking_slashes_and_fragments(raw_url, expected):
    assert normalize_url(raw_url) == expected


@pytest.mark.parametrize(
    ("raw_date", "expected_date"),
    [
        ("Wed, 02 Apr 2025 12:00:00 GMT", (2025, 4, 2)),
        ("2025-04-02T12:00:00+00:00", (2025, 4, 2)),
    ],
    ids=["rfc2822", "iso8601"],
)
def test_parse_pub_date_to_epoch_returns_utc_epoch_floats(raw_date, expected_date):
    epoch = parse_pub_date_to_epoch(raw_date)

    assert type(epoch) is float
    assert epoch > 0
    assert datetime.fromtimestamp(epoch, tz=timezone.utc).timetuple()[:3] == expected_date


@pytest.mark.parametrize("raw_date", [None, "not-a-date"], ids=["missing", "invalid"])
def test_parse_pub_date_to_epoch_falls_back_to_zero(raw_date):
    assert parse_pub_date_to_epoch(raw_date) == 0.0


def test_strip_html_removes_tags_and_collapses_whitespace():
    assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"
    assert strip_html("  hello   world  ") == "hello world"


def test_extract_html_body_text_reads_paragraphs_and_ignores_chrome():
    paragraphs = extract_html_body_text(
        "<html><body><p>First para.</p><p>Second para.</p></body></html>"
    )
    article = extract_html_body_text(
        "<html><body><div>sidebar</div><article><p>Main content.</p></article></body></html>"
    )
    scripted = extract_html_body_text(
        "<html><script>alert('x')</script><p>Content</p></html>"
    )

    assert "First para." in paragraphs
    assert "Second para." in paragraphs
    assert "Main content." in article
    assert "sidebar" not in article
    assert "alert" not in scripted
    assert "Content" in scripted


def test_parse_rss_items_maps_the_whole_raw_item_contract():
    items = parse_rss_items(SAMPLE_RSS, "CoinDesk")

    assert len(items) == 2
    first, second = items
    assert first["url"] == "https://example.com/btc-high"
    assert first["title"] == "Bitcoin Hits New High"
    assert first["source_name"] == "CoinDesk"
    assert first["source_type"] == "rss"
    assert first["categories"] == ["BTC", "Bitcoin"]
    assert first["published_at_epoch"] > 0
    assert "Bitcoin reached" in first["body_text"]
    assert first["summary"] == "Bitcoin reached $100k today."
    assert first["raw_source_id"] == "https://example.com/btc-high"
    assert second["categories"] == []
    assert second["raw_source_id"] is None
    assert second["body_text"] == "Ethereum has a new upgrade."


@pytest.mark.parametrize(
    ("max_items", "expected_count"),
    [(50, 2), (1, 1), (0, 0)],
    ids=["all", "single", "zero"],
)
def test_parse_rss_items_respects_the_item_cap(max_items, expected_count):
    assert len(parse_rss_items(SAMPLE_RSS, "src", max_items=max_items)) == expected_count


def test_parse_rss_items_skips_items_missing_a_title_or_link():
    rss = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item><title></title><link>https://example.com/a</link></item>
  <item><title>Good title</title><link></link></item>
  <item><title>Valid</title><link>https://example.com/valid</link></item>
</channel></rss>"""

    items = parse_rss_items(rss, "src")

    assert len(items) == 1
    assert items[0]["title"] == "Valid"


@pytest.mark.parametrize(
    "payload",
    [
        '<?xml version="1.0"?><rss version="2.0"><channel><item><title>Broken</title></channel></rss>',
        (
            '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">'
            '<entry><title>Atom News</title><link href="https://example.com/a"/></entry></feed>'
        ),
        '<?xml version="1.0"?><rss version="2.0"><channel><title>Empty</title></channel></rss>',
        "403 Forbidden: rate limited",
    ],
    ids=["unclosed-tag", "atom-feed", "empty-feed", "non-xml-body"],
)
def test_parse_rss_items_returns_nothing_for_malformed_or_non_rss_payloads(payload):
    assert parse_rss_items(payload, "src") == []


def test_dedupe_by_url_keeps_the_newer_version_and_skips_empty_urls():
    items = [
        {"url": "https://a.com/1", "published_at_epoch": 1000.0, "title": "old"},
        {"url": "https://a.com/1", "published_at_epoch": 2000.0, "title": "new"},
        {"url": "https://a.com/2", "published_at_epoch": 1500.0, "title": "other"},
        {"url": "", "published_at_epoch": 1000.0},
        {"url": None, "published_at_epoch": 1000.0},
    ]

    result = dedupe_by_url(items)

    assert len(result) == 2
    assert {item["url"] for item in result} == {"https://a.com/1", "https://a.com/2"}
    assert next(item for item in result if item["url"] == "https://a.com/1")["title"] == "new"


def test_dedupe_by_url_never_replaces_the_current_version_with_an_older_duplicate():
    items = [
        {"url": "https://a.com/1", "published_at_epoch": 2000.0, "title": "newer"},
        {"url": "https://a.com/1", "published_at_epoch": 1000.0, "title": "older"},
        {"url": "https://a.com/1", "published_at_epoch": 2000.0, "title": "same-age"},
    ]

    result = dedupe_by_url(items)

    assert [item["title"] for item in result] == ["newer"]


def test_dedupe_by_url_then_sort_by_date_yields_one_newest_first_item_per_url():
    items = [
        {"url": "https://a.com/1", "published_at_epoch": 1000.0, "title": "old"},
        {"url": "https://a.com/2", "published_at_epoch": 3000.0, "title": "newest"},
        {"url": "https://a.com/1", "published_at_epoch": 2000.0, "title": "updated"},
    ]

    result = sort_by_date(dedupe_by_url(items))

    assert [(item["title"], item["published_at_epoch"]) for item in result] == [
        ("newest", 3000.0),
        ("updated", 2000.0),
    ]


def test_dedupe_by_normalized_title_keeps_the_longest_raw_body_text():
    items = [
        {
            "title": "BTC Hits New High!",
            "url": "https://a.com/1",
            "body_text": "short",
            "published_at_epoch": 1.0,
        },
        {
            "title": "btc hits new high",
            "url": "https://b.com/9",
            "body_text": "much longer body text than the first feed carried",
            "published_at_epoch": 2.0,
        },
        {"title": "", "url": "https://c.com/3", "body_text": "no title at all"},
    ]

    kept = dedupe_by_normalized_title(items)

    assert [(item["url"], item["body_text"]) for item in kept] == [
        ("https://b.com/9", "much longer body text than the first feed carried")
    ]


def test_dedupe_by_normalized_title_never_reads_the_canonical_body_field():
    items = [
        {"title": "Same Story", "url": "https://a.com/1", "body": "x" * 500},
        {"title": "same story", "url": "https://b.com/2", "body": "y"},
    ]

    kept = dedupe_by_normalized_title(items)

    assert [item["url"] for item in kept] == ["https://a.com/1"]


def test_make_article_id_is_deterministic_sixteen_hex_and_url_unique():
    url = "https://coindesk.com/article/btc-100k"
    article_id = make_article_id(url)

    assert article_id == make_article_id(url)
    assert article_id != make_article_id("https://coindesk.com/other-article")
    assert len(article_id) == 16
    assert set(article_id) <= set("0123456789abcdef")


def test_to_article_schema_emits_the_full_canonical_contract():
    item = raw_item(source_name="  CoinTelegraph  ", categories=["BTC", "DeFi"])

    article = to_article_schema(item)

    assert set(article) == {
        "id",
        "title",
        "body",
        "categories",
        "tags",
        "published_on",
        "source_info",
        "url",
    }
    assert article["id"] == make_article_id(item["url"])
    assert article["id"] == to_article_schema(raw_item())["id"]
    assert article["title"] == "BTC Hits 100k"
    assert article["body"] == "Full article body here."
    assert article["categories"] == "BTC|DeFi"
    assert article["tags"] == ""
    assert type(article["published_on"]) is float
    assert article["published_on"] == 1743590400.0
    assert article["source_info"] == {"name": "cointelegraph"}
    assert article["url"] == item["url"]


@pytest.mark.parametrize(
    ("overrides", "expected_body"),
    [
        ({"body_text": "Full body", "summary": "Short"}, "Full body"),
        ({"body_text": "", "summary": "Short summary"}, "Short summary"),
    ],
    ids=["body-text-preferred", "summary-fallback"],
)
def test_to_article_schema_selects_the_body_source(overrides, expected_body):
    assert to_article_schema(raw_item(**overrides))["body"] == expected_body


@pytest.mark.parametrize(
    ("title", "source_name", "body_text", "expected_body"),
    [
        ("BTC Hits 100k", "coindesk", f"{COINDESK_BODY_PREFIX}\n\n{COINDESK_NAV_TAIL}", COINDESK_BODY_PREFIX),
        (
            "BTC Hits 100k",
            "coindesk",
            (
                "Bitcoin market structure remains constructive. "
                "More For You * About Us * Contact Us * Newsletters * Latest Crypto News"
            ),
            "Bitcoin market structure remains constructive.",
        ),
        (DECRYPT_TITLE, "decrypt", DECRYPT_BODY, "The article continues with policy context."),
        ("BTC Hits 100k", "decrypt", PRICE_PROSE, PRICE_PROSE),
        ("BTC Hits 100k", "coindesk", NAVIGATION_LIKE_PROSE, NAVIGATION_LIKE_PROSE),
    ],
    ids=[
        "coindesk-tail-navigation",
        "early-boilerplate-marker",
        "decrypt-price-ticker-prefix",
        "price-prose-kept",
        "navigation-like-prose-kept",
    ],
)
def test_to_article_schema_strips_navigation_clutter_only(title, source_name, body_text, expected_body):
    article = to_article_schema(
        raw_item(title=title, source_name=source_name, body_text=body_text)
    )

    assert article["body"] == expected_body


@pytest.mark.parametrize(
    ("ages", "expected_count"),
    [((1, 5), 2), ((1, 48), 1), ((48, 72), 0), ((10, 2, 6), 3), ((), 0)],
    ids=["all-recent", "mixed", "all-stale", "unsorted", "empty"],
)
def test_filter_by_age_keeps_the_window_newest_first(ages, expected_count):
    provider, _ = make_provider()

    recent = provider.filter_by_age([canonical_article(age) for age in ages], max_age_hours=24)

    assert len(recent) == expected_count
    timestamps = [article["published_on"] for article in recent]
    assert timestamps == sorted(timestamps, reverse=True)


def test_filter_by_age_drops_articles_without_a_published_timestamp():
    provider, _ = make_provider()

    assert provider.filter_by_age([{"url": "x", "title": "t"}], max_age_hours=24) == []


@pytest.mark.parametrize(
    ("sources", "expected"),
    [
        (None, None),
        ("", None),
        ([], None),
        ("coindesk, decrypt", ["coindesk", "decrypt"]),
        (["coindesk", "decrypt"], ["coindesk", "decrypt"]),
        (["  coindesk  "], ["coindesk"]),
    ],
    ids=["none", "empty-string", "empty-list", "csv-string", "list", "whitespace"],
)
def test_enabled_source_names_normalizes_config_values(sources, expected):
    provider, _ = make_provider(RAG_NEWS_SOURCES=sources)

    assert provider._enabled_source_names() == expected


@pytest.mark.parametrize(
    ("sources", "registry"),
    [(None, []), (["ghost-source"], None)],
    ids=["empty-registry", "no-matching-name"],
)
async def test_fetch_news_returns_empty_when_no_source_is_enabled(sources, registry):
    provider, _ = make_provider(RAG_NEWS_SOURCES=sources)
    patcher = (
        patch(GET_SOURCES, return_value=registry) if registry is not None else nullcontext()
    )

    with patcher:
        assert await provider.fetch_news() == []


@pytest.mark.parametrize(
    "extra_patch",
    [
        {FETCH_SOURCE: AsyncMock(return_value=FAILED_FETCH)},
        {STAGE_WAIT_FOR: AsyncMock(side_effect=asyncio.TimeoutError)},
    ],
    ids=["all-sources-failed", "fetch-stage-timeout"],
)
async def test_fetch_news_yields_nothing_when_the_fetch_stage_fails(extra_patch):
    provider, _ = make_provider()

    with ExitStack() as stack:
        stack.enter_context(patch(GET_SOURCES, return_value=[SOURCE_ENTRY]))
        for target, replacement in extra_patch.items():
            stack.enter_context(patch(target, new=replacement))
        result = await provider.fetch_news()

    assert result == []


async def test_fetch_news_maps_raw_items_to_the_canonical_schema():
    provider, _ = make_provider(RAG_NEWS_PAGE_ENRICHMENT=False)
    fetched = FetchResult(
        source_name="coindesk",
        source_type="rss",
        url="x",
        success=True,
        status_code=200,
        error=None,
        normalized_items=[
            raw_item(
                url="https://example.com/btc-news",
                title="BTC News",
                body_text="Some body text",
                summary="",
                categories=["BTC"],
            )
        ],
    )

    with ExitStack() as stack:
        stack.enter_context(patch(GET_SOURCES, return_value=[SOURCE_ENTRY]))
        stack.enter_context(patch(FETCH_SOURCE, new=AsyncMock(return_value=fetched)))
        result = await provider.fetch_news()

    assert len(result) == 1
    assert result[0]["url"] == "https://example.com/btc-news"
    assert result[0]["title"] == "BTC News"
    assert result[0]["id"] == make_article_id("https://example.com/btc-news")
    assert result[0]["body"] == "Some body text"
    assert result[0]["source_info"] == {"name": "coindesk"}


@pytest.mark.parametrize(
    ("enrichment", "expected_calls"),
    [(True, 1), (False, 0)],
    ids=["enabled", "disabled"],
)
async def test_fetch_news_enriches_only_when_page_enrichment_is_enabled(enrichment, expected_calls):
    provider, enricher = make_provider(RAG_NEWS_PAGE_ENRICHMENT=enrichment)
    enricher.enrich_items = AsyncMock(return_value=1)
    fetched = FetchResult(
        source_name="coindesk",
        source_type="rss",
        url="x",
        success=True,
        status_code=200,
        error=None,
        normalized_items=[
            raw_item(
                url="https://example.com/eth-news",
                title="ETH News",
                body_text="Short",
                categories=[],
            )
        ],
    )

    with ExitStack() as stack:
        stack.enter_context(patch(GET_SOURCES, return_value=[SOURCE_ENTRY]))
        stack.enter_context(patch(FETCH_SOURCE, new=AsyncMock(return_value=fetched)))
        articles = await provider.fetch_news()

    assert enricher.enrich_items.await_count == expected_calls
    assert [article["url"] for article in articles] == ["https://example.com/eth-news"]


@pytest.mark.parametrize(
    "error",
    [asyncio.TimeoutError(), RuntimeError("browser crashed mid-batch")],
    ids=["enrich-timeout", "enrich-raises"],
)
async def test_postprocess_items_keeps_mapped_articles_when_enrichment_fails(error):
    provider, enricher = make_provider(RAG_NEWS_PAGE_ENRICHMENT=True)
    enricher.enrich_items = AsyncMock(side_effect=error)
    merged = [
        raw_item(
            url="https://example.com/ada-news",
            title="ADA News",
            body_text="Short body",
            categories=[],
        )
    ]

    result = await provider._postprocess_items(merged, session=MagicMock())

    assert len(result) == 1
    assert result[0]["url"] == "https://example.com/ada-news"
    assert result[0]["id"] == make_article_id("https://example.com/ada-news")


def test_articles_with_a_missing_or_invalid_pub_date_never_count_as_fresh():
    items = parse_rss_items(UNDATED_RSS, "src")

    assert [item["published_at_epoch"] for item in items] == [0.0, 0.0]

    articles = [to_article_schema(item) for item in items]
    provider, _ = make_provider()

    assert [article["published_on"] for article in articles] == [0.0, 0.0]
    assert all(type(article["published_on"]) is float for article in articles)
    assert provider.filter_by_age(articles, max_age_hours=24) == []


def test_crawl4ai_enricher_resolves_explicit_di_overrides():
    logger = null_logger()

    enricher = Crawl4AIEnricher(
        logger=logger, concurrency=5, timeout=45.0, min_chars=500, use_crawl4ai=False
    )

    assert enricher.logger is logger
    assert enricher.concurrency == 5
    assert enricher.timeout == 45.0
    assert enricher.min_chars == 500
    assert enricher._use_crawl4ai is False


def test_crawl4ai_enricher_defaults_without_config_or_kwargs():
    enricher = Crawl4AIEnricher(logger=null_logger())

    assert enricher.concurrency == 3
    assert enricher.timeout == 30.0
    assert enricher.min_chars == 400
    assert type(enricher._use_crawl4ai) is bool


def test_crawl4ai_enricher_reads_config_when_kwargs_are_absent():
    config = provider_config(
        RAG_NEWS_CRAWL_CONCURRENCY=4,
        RAG_NEWS_CRAWL_TIMEOUT="25.0",
        RAG_NEWS_ENRICH_MIN_CHARS=350,
        RAG_NEWS_CRAWL4AI_ENABLED=False,
    )

    enricher = Crawl4AIEnricher(logger=null_logger(), config=config)

    assert enricher.concurrency == 4
    assert enricher.timeout == 25.0
    assert enricher.min_chars == 350
    assert enricher._use_crawl4ai is False


def test_enricher_cleans_markdown_into_plain_article_text():
    raw_markdown = (
        "# Heading\n\n[Link](https://example.com)\n"
        "![Image](https://example.com/img.png)\n\nParagraph text."
    )

    cleaned = Crawl4AIEnricher._clean_markdown_text(raw_markdown)

    assert cleaned == "Heading\n\nLink\n\nParagraph text."


@pytest.mark.parametrize(
    ("body_text", "expected"),
    [
        ("404 Not Found", True),
        ("Oops! Something went wrong", True),
        ("ARTICLE NOT FOUND", True),
        ("We're sorry for the inconvenience. Please try again", True),
        ("This is a valid news article about crypto trading.", False),
        ("", False),
    ],
    ids=["404", "oops", "case-insensitive", "generic-error", "valid-article", "empty"],
)
def test_enricher_flags_unusable_bodies(body_text, expected):
    assert Crawl4AIEnricher._is_unusable_body(body_text) is expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://coindesk.com/article", True),
        ("http://example.com/article", True),
        ("https://example.com:8443/article", True),
        ("", False),
        ("ftp://example.com/article", False),
        ("file:///etc/passwd", False),
        ("https:///no-host", False),
        ("http://localhost:8000/article", False),
        ("http://127.0.0.1/article", False),
        ("http://[::1]/article", False),
        ("http://10.1.2.3/article", False),
        ("http://192.168.0.10/article", False),
        ("http://169.254.1.1/article", False),
    ],
    ids=[
        "https",
        "http",
        "https-port",
        "empty",
        "ftp",
        "file",
        "no-host",
        "localhost",
        "loopback-ip",
        "ipv6-loopback",
        "private-10",
        "private-192",
        "link-local",
    ],
)
def test_enricher_ssrf_guard_rejects_internal_and_non_http_targets(url, expected):
    assert Crawl4AIEnricher._is_safe_external_url(url) is expected


@pytest.mark.parametrize(
    "items",
    [
        [],
        [{"url": "https://example.com/1", "body_text": "a" * 150}],
        [
            {"url": "http://localhost:9199/decision", "body_text": "short"},
            {"url": "file:///etc/passwd", "body_text": ""},
        ],
    ],
    ids=["empty-batch", "sufficient-body", "unsafe-urls"],
)
async def test_enrich_items_returns_zero_when_no_item_needs_a_safe_crawl(items):
    enricher = Crawl4AIEnricher(logger=null_logger(), min_chars=100, use_crawl4ai=False)

    assert await enricher.enrich_items(items) == 0


def test_news_repository_loads_uncut_for_the_default_24h_window_and_filters_otherwise():
    repo, file_handler = make_news_repository()
    articles = [{"id": "a"}, {"id": "b"}]
    file_handler.load_news_articles.return_value = articles

    assert repo.load_recent_articles(max_age_seconds=86400) == articles
    file_handler.load_news_articles.assert_called_once()
    file_handler.filter_articles_by_age.assert_not_called()

    file_handler.filter_articles_by_age.return_value = [{"id": "b"}]

    assert repo.load_recent_articles(max_age_seconds=3600) == [{"id": "b"}]
    file_handler.filter_articles_by_age.assert_called_once_with(
        articles, max_age_seconds=3600
    )


def test_news_repository_saves_uncut_for_the_default_24h_window_and_filters_otherwise():
    repo, file_handler = make_news_repository()
    articles = [{"id": "a"}, {"id": "b"}]

    repo.save_recent_articles(articles, max_age_seconds=86400)
    file_handler.filter_articles_by_age.assert_not_called()

    filtered = [{"id": "b"}]
    file_handler.filter_articles_by_age.return_value = filtered

    repo.save_recent_articles(articles, max_age_seconds=3600)

    assert file_handler.filter_articles_by_age.call_args_list[-1] == call(
        articles, max_age_seconds=3600
    )
    assert file_handler.save_news_articles.call_args_list == [call(articles), call(filtered)]


def test_news_repository_filter_recent_articles_delegates_every_window():
    repo, file_handler = make_news_repository()
    articles = [{"id": "a"}]
    file_handler.filter_articles_by_age.return_value = [{"id": "b"}]

    assert repo.filter_recent_articles(articles) == [{"id": "b"}]
    file_handler.filter_articles_by_age.assert_called_once_with(
        articles, max_age_seconds=86400
    )

    assert repo.filter_recent_articles(articles, max_age_seconds=0) == [{"id": "b"}]
    assert file_handler.filter_articles_by_age.call_args_list[-1] == call(
        articles, max_age_seconds=0
    )


def test_news_repository_load_fallback_articles_forwards_the_hours_window():
    repo, file_handler = make_news_repository()
    file_handler.load_fallback_articles.return_value = [{"id": "cached"}]

    result = repo.load_fallback_articles(max_age_hours=48)

    assert result == [{"id": "cached"}]
    file_handler.load_fallback_articles.assert_called_once_with(max_age_hours=48)


def test_news_manager_update_news_database_is_url_first():
    manager, file_handler = make_news_manager()
    now = time.time()
    first = {
        "id": "abc123",
        "url": "https://coindesk.com/btc",
        "published_on": now - 3600,
        "title": "BTC article",
    }

    assert manager.update_news_database([first]) is True
    assert len(manager.news_database) == 1

    duplicate = {
        "id": "xyz999",
        "url": "https://coindesk.com/btc",
        "published_on": now - 1800,
        "title": "same story, new id",
    }

    assert manager.update_news_database([duplicate]) is False
    assert len(manager.news_database) == 1
    file_handler.save_news_articles.assert_called_once_with(manager.news_database)


def test_news_manager_update_news_database_refreshes_a_truncated_body_for_a_known_url():
    manager, _ = make_news_manager()
    now = time.time()
    url = "https://coindesk.com/btc"

    assert (
        manager.update_news_database(
            [{"id": "abc123", "url": url, "published_on": now - 3600, "body": "teaser"}]
        )
        is True
    )

    full_body = "x" * 800

    assert (
        manager.update_news_database(
            [
                {
                    "id": "abc123",
                    "url": url,
                    "published_on": now - 3600,
                    "title": "BTC article",
                    "body": full_body,
                }
            ]
        )
        is True
    )
    assert len(manager.news_database) == 1
    assert manager.news_database[0]["body"] == full_body
    assert manager.news_database[0]["title"] == "BTC article"
    assert manager.news_database[0]["title_lower"] == "btc article"


@pytest.mark.parametrize(
    "payload",
    [TAXONOMY_DOCUMENT, TAXONOMY_CATEGORIES],
    ids=["wrapped-document", "plain-list"],
)
async def test_local_taxonomy_provider_loads_both_json_shapes(tmp_path, payload):
    provider = LocalTaxonomyProvider(
        null_logger(), categories_file=write_taxonomy(tmp_path / "categories.json", payload)
    )

    categories = await provider.fetch_categories()

    assert [category["categoryName"] for category in categories] == ["BTC", "ETH"]


async def test_local_taxonomy_provider_caches_until_force_refresh(tmp_path):
    categories_file = tmp_path / "categories.json"
    write_taxonomy(categories_file, TAXONOMY_INITIAL)
    provider = LocalTaxonomyProvider(null_logger(), categories_file=str(categories_file))

    assert len(await provider.fetch_categories()) == 1
    categories_file.unlink()

    assert len(await provider.fetch_categories()) == 1

    write_taxonomy(categories_file, TAXONOMY_UPDATED)

    assert len(await provider.fetch_categories(force_refresh=True)) == 2


@pytest.mark.parametrize(
    "content",
    [
        None,
        "{not json",
        json.dumps({"timestamp": "2026-01-01T00:00:00+00:00", "unexpected": []}),
    ],
    ids=["missing-file", "malformed-json", "unexpected-shape"],
)
async def test_local_taxonomy_provider_returns_empty_for_unusable_files(tmp_path, content):
    categories_file = tmp_path / "categories.json"
    if content is not None:
        categories_file.write_text(content, encoding="utf-8")
    provider = LocalTaxonomyProvider(null_logger(), categories_file=str(categories_file))

    assert await provider.fetch_categories() == []


class ParserStub:
    """Mirrors UnifiedParser's coin-scan surface without the real parser backend."""

    @staticmethod
    def detect_coins_in_text(text: str, known_crypto_tickers: set[str]) -> set[str]:
        lowered = text.lower()
        return {ticker for ticker in known_crypto_tickers if ticker.lower() in lowered}

    @staticmethod
    def extract_base_coin(symbol: str) -> str:
        return symbol.split("/")[0]


def test_article_processor_detects_coins_across_categories_title_and_whole_body():
    processor = ArticleProcessor(
        logger=null_logger(),
        format_utils=MagicMock(),
        unified_parser=ParserStub(),
    )
    long_prefix = "x" * 12050
    late_mention = {
        "title": "No coin in title",
        "categories": "",
        "body": f"{long_prefix} btc appears late in body",
    }
    labelled = {"title": "ETH merger", "categories": "BTC|SOL", "body": "unrelated"}

    assert processor.detect_coins_in_article(late_mention, {"BTC"}) == {"BTC"}
    assert processor.detect_coins_in_article(labelled, {"BTC", "SOL", "ETH"}) == {
        "BTC",
        "SOL",
        "ETH",
    }
