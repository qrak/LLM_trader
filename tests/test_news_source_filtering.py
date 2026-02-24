"""
Test News Source Filtering Implementation
Verifies that only Tier 1 sources are fetched from CryptoCompare API
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.loader import config
from src.logger.logger import Logger
from src.platforms.cryptocompare.news_components.news_client import (
    CryptoCompareNewsClient,
    RELIABLE_NEWS_FEEDS
)


async def test_news_filtering():
    """Test that news filtering returns only Tier 1 sources"""
    logger = Logger("NewsFilterTest", "NewsFilterTest", log_dir=config.LOG_DIR, logger_debug=config.LOGGER_DEBUG)
    client = CryptoCompareNewsClient(logger, config)

    logger.info("=" * 60)
    logger.info("Testing News Source Filtering")
    logger.info("=" * 60)
    logger.info("Filter enabled: %s", config.RAG_NEWS_FILTER_SOURCES)
    logger.info("Allowed feeds: %s", config.RAG_NEWS_ALLOWED_FEEDS or RELIABLE_NEWS_FEEDS)
    logger.info("")

    # Fetch news
    articles = await client.fetch_news()

    if not articles:
        logger.error("Failed to fetch news or empty response")
        return False

    logger.info("Total articles fetched: %s", len(articles))

    # Analyze sources
    sources = {}
    for article in articles:
        source = article.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1

    logger.info("\nArticle sources breakdown:")
    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        tier = "✓ TIER 1" if source in RELIABLE_NEWS_FEEDS else "✗ UNEXPECTED"
        logger.info("  %s: %s articles [%s]", source, count, tier)

    # Verify all sources are from whitelist
    unexpected_sources = [s for s in sources.keys() if s not in RELIABLE_NEWS_FEEDS]

    logger.info("\n" + "=" * 60)
    if unexpected_sources:
        logger.error("FAILED: Found unexpected sources: %s", unexpected_sources)
        return False
    else:
        logger.info("SUCCESS: All articles are from Tier 1 sources only!")
        logger.info("No bitcoinworld or other unreliable sources detected ✓")
        return True


if __name__ == "__main__":
    result = asyncio.run(test_news_filtering())
    sys.exit(0 if result else 1)
