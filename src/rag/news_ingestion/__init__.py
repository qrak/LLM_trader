"""News ingestion package: RSS-seeded, Crawl4AI-enriched news provider."""
from .rss_primitives import (
    RSS_SOURCES,
    FetchResult,
    normalize_url,
    parse_pub_date_to_epoch,
    strip_html,
    extract_html_body_text,
    parse_rss_items,
    fetch_source,
    dedupe_by_url,
)
from .schema_mapper import make_article_id, to_article_schema
from .crawl4ai_enricher import Crawl4AIEnricher
from .rss_provider import RSSCrawl4AINewsProvider

__all__ = [
    "RSS_SOURCES",
    "FetchResult",
    "normalize_url",
    "parse_pub_date_to_epoch",
    "strip_html",
    "extract_html_body_text",
    "parse_rss_items",
    "fetch_source",
    "dedupe_by_url",
    "make_article_id",
    "to_article_schema",
    "Crawl4AIEnricher",
    "RSSCrawl4AINewsProvider",
]
