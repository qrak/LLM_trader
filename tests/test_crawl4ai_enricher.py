"""Unit tests for Crawl4AIEnricher."""

from unittest.mock import MagicMock, AsyncMock, patch
import pytest
from src.rag.news_ingestion.crawl4ai_enricher import Crawl4AIEnricher


def test_crawl4ai_enricher_di_defaults():
    logger = MagicMock()
    enricher = Crawl4AIEnricher(logger=logger, concurrency=5, timeout=45.0, min_chars=500, use_crawl4ai=False)

    assert enricher.logger is logger
    assert enricher.concurrency == 5
    assert enricher.timeout == 45.0
    assert enricher.min_chars == 500
    assert enricher._use_crawl4ai is False


def test_crawl4ai_enricher_config_fallbacks():
    logger = MagicMock()
    config = MagicMock()
    config.RAG_NEWS_CRAWL_CONCURRENCY = 4
    config.RAG_NEWS_CRAWL_TIMEOUT = "25.0"
    config.RAG_NEWS_ENRICH_MIN_CHARS = 350
    config.RAG_NEWS_CRAWL4AI_ENABLED = False

    enricher = Crawl4AIEnricher(logger=logger, config=config)

    assert enricher.concurrency == 4
    assert enricher.timeout == 25.0
    assert enricher.min_chars == 350
    assert enricher._use_crawl4ai is False


def test_clean_markdown_text():
    raw_md = "# Heading\n\n[Link](https://example.com)\n![Image](https://example.com/img.png)\n\nParagraph text."
    cleaned = Crawl4AIEnricher._clean_markdown_text(raw_md)
    assert "Heading" in cleaned
    assert "Link" in cleaned
    assert "Image" not in cleaned
    assert "Paragraph text." in cleaned


def test_is_unusable_body():
    assert Crawl4AIEnricher._is_unusable_body("404 Not Found") is True
    assert Crawl4AIEnricher._is_unusable_body("Oops! Something went wrong") is True
    assert Crawl4AIEnricher._is_unusable_body("This is a valid news article about crypto trading.") is False


@pytest.mark.asyncio
async def test_enrich_items_skips_sufficient_body():
    logger = MagicMock()
    enricher = Crawl4AIEnricher(logger=logger, min_chars=100, use_crawl4ai=False)
    items = [{"url": "https://example.com/1", "body_text": "a" * 150}]

    count = await enricher.enrich_items(items)
    assert count == 0
