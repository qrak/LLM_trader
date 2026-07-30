"""News ingestion package: RSS-seeded, Crawl4AI-enriched news provider."""
from .crawl4ai_enricher import Crawl4AIEnricher
from .rss_primitives import (
    RSS_SOURCES,
    FetchResult,
    dedupe_by_url,
    extract_html_body_text,
    fetch_source,
    normalize_url,
    parse_pub_date_to_epoch,
    parse_rss_items,
    strip_html,
)
from .rss_provider import RSSCrawl4AINewsProvider
from .schema_mapper import make_article_id, to_article_schema

__all__ = [
    "RSS_SOURCES",
    "Crawl4AIEnricher",
    "FetchResult",
    "RSSCrawl4AINewsProvider",
    "dedupe_by_url",
    "extract_html_body_text",
    "fetch_source",
    "make_article_id",
    "normalize_url",
    "parse_pub_date_to_epoch",
    "parse_rss_items",
    "strip_html",
    "to_article_schema",
]
