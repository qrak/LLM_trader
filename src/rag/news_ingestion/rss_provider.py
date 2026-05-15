"""RSS + Crawl4AI news provider.

Implements the `fetch_news` + `filter_by_age` surface used by `NewsManager`.
``fetch_news()`` / ``filter_by_age()`` surface so NewsManager works without
modification.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from time import perf_counter
from typing import Any, TYPE_CHECKING

import aiohttp

from src.logger.logger import Logger
from .crawl4ai_enricher import Crawl4AIEnricher
from .rss_primitives import (
    FetchResult,
    dedupe_by_url,
    fetch_source,
    get_sources,
    sort_by_date,
)
from .schema_mapper import to_article_schema

if TYPE_CHECKING:
    from src.config.protocol import ConfigProtocol


class RSSCrawl4AINewsProvider:
    """Fetch crypto news from free RSS feeds, optionally enriching with Crawl4AI.

    This class satisfies the implicit ``news_client`` protocol expected by
    :class:`src.rag.news_manager.NewsManager`:

    * ``async fetch_news(session, api_categories) -> list[dict]``
    * ``filter_by_age(articles, max_age_hours) -> list[dict]``
    """

    def __init__(self, logger: Logger, config: "ConfigProtocol", enricher: Crawl4AIEnricher) -> None:
        self.logger = logger
        self.config = config
        self._enricher = enricher

    # ------------------------------------------------------------------
    # Public API expected by NewsManager
    # ------------------------------------------------------------------

    async def fetch_news(
        self,
        session: aiohttp.ClientSession | None = None,
        api_categories: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch, deduplicate, and enrich articles from configured RSS sources.

        *api_categories* is accepted for interface compatibility but is not used;
        category detection is handled by ArticleProcessor after ingestion.
        """
        enabled_names = self._enabled_source_names()
        sources = get_sources(enabled_names, self.config.RAG_NEWS_SOURCE_URLS)

        if not sources:
            self.logger.error("No RSS sources configured or none enabled")
            return []

        own_session = session is None
        if own_session:
            timeout = aiohttp.ClientTimeout(total=float(self.config.RAG_NEWS_FETCH_TIMEOUT))
            session = aiohttp.ClientSession(timeout=timeout)

        try:
            merged = await self._fetch_raw_items(sources, session)
            if not merged:
                self.logger.warning("All RSS sources returned empty; no articles ingested")
                return []

            articles = await self._postprocess_items(merged, session)
            self.logger.debug(
                "RSS ingestion complete: sources=%d raw=%d articles=%d",
                len(sources),
                len(merged),
                len(articles),
            )
            return articles

        finally:
            if own_session:
                await session.close()

    # ------------------------------------------------------------------
    # Stage helpers (split from fetch_news for readability)
    # ------------------------------------------------------------------

    async def _fetch_raw_items(
        self,
        sources: list[dict[str, Any]],
        session: aiohttp.ClientSession,
    ) -> list[dict[str, Any]]:
        """Concurrently fetch all RSS sources; return merged raw item list."""
        fetch_timeout = max(1, int(self.config.RAG_NEWS_FETCH_TIMEOUT))
        fetch_total_timeout = max(
            fetch_timeout,
            int(self.config.RAG_NEWS_FETCH_TOTAL_TIMEOUT),
        )
        self.logger.info(
            "Fetching crypto news from %d RSS sources "
            "(per-source timeout=%ss, stage timeout=%ss).",
            len(sources),
            fetch_timeout,
            fetch_total_timeout,
        )
        fetch_start = perf_counter()
        try:
            results: list[FetchResult] = await asyncio.wait_for(
                asyncio.gather(*(
                    fetch_source(
                        session,
                        s["name"],
                        s["url"],
                        max_items=self.config.RAG_NEWS_MAX_ITEMS_PER_SOURCE,
                        timeout=float(fetch_timeout),
                    )
                    for s in sources
                )),
                timeout=float(fetch_total_timeout),
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                "RSS fetch stage timed out after %.1fs (limit=%ss); "
                "continuing with empty fresh-news set",
                perf_counter() - fetch_start,
                fetch_total_timeout,
            )
            return []
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "RSS fetch stage failed after %.1fs: %s; continuing with empty fresh-news set",
                perf_counter() - fetch_start,
                exc,
            )
            return []

        for r in results:
            if not r.success:
                self.logger.warning("RSS source '%s' failed: %s", r.source_name, r.error)

        merged: list[dict[str, Any]] = []
        for r in results:
            merged.extend(r.normalized_items)
        self.logger.debug(
            "RSS fetch stage completed in %.1fs with %d raw items",
            perf_counter() - fetch_start,
            len(merged),
        )
        return merged

    async def _postprocess_items(
        self,
        merged: list[dict[str, Any]],
        session: aiohttp.ClientSession,
    ) -> list[dict[str, Any]]:
        """Enrich bodies, deduplicate, sort, and map to canonical article schema."""
        if self.config.RAG_NEWS_PAGE_ENRICHMENT:
            enrich_timeout = max(1, int(self.config.RAG_NEWS_ENRICH_TIMEOUT))
            self.logger.info(
                "Enriching %d news article bodies for better analysis "
                "(stage timeout=%ss, crawl per-page timeout=%ss).",
                len(merged),
                enrich_timeout,
                self.config.RAG_NEWS_CRAWL_TIMEOUT,
            )
            enrich_start = perf_counter()
            try:
                enriched_count = await asyncio.wait_for(
                    self._enricher.enrich_items(merged, session),
                    timeout=float(enrich_timeout),
                )
                self.logger.debug(
                    "Body enrichment: %d/%d items enriched in %.1fs",
                    enriched_count,
                    len(merged),
                    perf_counter() - enrich_start,
                )
            except asyncio.TimeoutError:
                self.logger.warning(
                    "News enrichment timed out after %.1fs (limit=%ss); "
                    "skipping remaining enrichment and continuing",
                    perf_counter() - enrich_start,
                    enrich_timeout,
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(
                    "News enrichment failed after %.1fs: %s; continuing with available bodies",
                    perf_counter() - enrich_start,
                    exc,
                )

        deduped = dedupe_by_url(merged)
        sorted_items = sort_by_date(deduped)
        return [to_article_schema(item) for item in sorted_items]

    def filter_by_age(
        self,
        articles: list[dict[str, Any]],
        max_age_hours: int = 24,
    ) -> list[dict[str, Any]]:
        """Filter articles older than *max_age_hours* and sort newest-first."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        ).timestamp()
        recent = [a for a in articles if a.get("published_on", 0) > cutoff]
        recent.sort(key=lambda a: a.get("published_on", 0), reverse=True)
        return recent

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _enabled_source_names(self) -> list[str] | None:
        """Return enabled source-name filter, or None to use all configured URLs."""
        raw = self.config.RAG_NEWS_SOURCES
        if not raw:
            return None
        if isinstance(raw, str):
            return [source.strip() for source in raw.split(",") if source.strip()]
        return [str(source).strip() for source in raw if str(source).strip()]
