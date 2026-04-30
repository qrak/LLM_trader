"""Crawl4AI-based article body enricher.

Enriches items whose body text is below *min_chars* by crawling the article
URL with Crawl4AI's AsyncWebCrawler.  Falls back to a plain aiohttp HTML
extraction if crawl4ai is not installed or the browser setup is unavailable.
"""
from __future__ import annotations

import asyncio
import logging
import re
import sys
from typing import Any

import aiohttp

from .rss_primitives import extract_html_body_text, normalize_url

logger = logging.getLogger(__name__)

_CRAWL4AI_AVAILABLE: bool | None = None  # lazy-checked on first use

_UNUSABLE_BODY_MARKERS: tuple[str, ...] = (
    "article not found",
    "oops! something went wrong",
    "we're sorry for the inconvenience. please try again",
    "page not found",
    "404 not found",
)


def _clean_markdown_text(markdown_text: str) -> str:
    """Normalize Crawl4AI markdown into prompt-friendly article text."""
    text = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?m)^\s*\[[^\]]+\]:\s+\S+.*$", "", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _markdown_value(markdown_obj: Any, attr_name: str) -> str:
    value = getattr(markdown_obj, attr_name, "")
    return value if isinstance(value, str) else ""


def _is_unusable_body(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return any(marker in normalized for marker in _UNUSABLE_BODY_MARKERS)


def _extract_crawl4ai_body(result: Any, min_chars: int) -> str | None:
    """Return the best full-article text from a Crawl4AI result."""
    markdown_obj = getattr(result, "markdown", None)
    candidates: list[str] = []

    if isinstance(markdown_obj, str):
        candidates.append(markdown_obj)
    elif markdown_obj is not None:
        fit_markdown = _markdown_value(markdown_obj, "fit_markdown")
        raw_markdown = _markdown_value(markdown_obj, "raw_markdown")
        cited_markdown = _markdown_value(markdown_obj, "markdown_with_citations")

        if fit_markdown:
            candidates.append(fit_markdown)
        if raw_markdown and len(raw_markdown) > len(fit_markdown):
            candidates.append(raw_markdown)
        if cited_markdown and cited_markdown not in candidates:
            candidates.append(cited_markdown)

    for candidate in candidates:
        text = _clean_markdown_text(candidate)
        if len(text) >= min_chars and not _is_unusable_body(text):
            return text

    html = getattr(result, "cleaned_html", None) or getattr(result, "html", None) or ""
    if html:
        text = extract_html_body_text(html).strip()
        if len(text) >= min_chars and not _is_unusable_body(text):
            return text

    return None


def _check_crawl4ai() -> bool:
    global _CRAWL4AI_AVAILABLE  # noqa: PLW0603
    if _CRAWL4AI_AVAILABLE is None:
        try:
            import crawl4ai  # noqa: F401
            _CRAWL4AI_AVAILABLE = True
        except ImportError:
            _CRAWL4AI_AVAILABLE = False
    return _CRAWL4AI_AVAILABLE


def _requires_dedicated_crawl_loop() -> bool:
    """Return True when the current Windows loop cannot spawn subprocesses.

    Playwright launches a browser subprocess. The app currently runs on
    Windows' selector loop, which does not implement subprocess transports and
    raises NotImplementedError. In that case we run Crawl4AI on a dedicated
    Proactor loop in a worker thread.
    """
    if sys.platform != "win32":
        return False

    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        return False

    return isinstance(current_loop, asyncio.SelectorEventLoop)


# ---------------------------------------------------------------------------
# aiohttp fallback enricher
# ---------------------------------------------------------------------------

async def _fetch_body_aiohttp(
    session: aiohttp.ClientSession,
    url: str,
    timeout: float,
    min_chars: int,
) -> str | None:
    """Fetch article body via plain aiohttp + HTML extraction."""
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                return None
            html = await resp.text(encoding="utf-8", errors="replace")
            body = extract_html_body_text(html)
            return body if len(body) >= min_chars and not _is_unusable_body(body) else None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Crawl4AI enricher
# ---------------------------------------------------------------------------

class Crawl4AIEnricher:
    """Enriches article items with full body text using Crawl4AI.

    If Crawl4AI is not available the enricher degrades gracefully to an
    aiohttp-based plain-HTML extractor so the pipeline never hard-fails.

    Parameters
    ----------
    concurrency:
        Max simultaneous Crawl4AI crawl sessions.
    timeout:
        Per-page timeout in seconds.
    min_chars:
        Skip items whose existing body already meets this length.
    use_crawl4ai:
        Force-enable or force-disable the Crawl4AI path regardless of
        whether the package is installed.  ``None`` = auto-detect.
    """

    def __init__(
        self,
        concurrency: int = 3,
        timeout: float = 30.0,
        min_chars: int = 400,
        use_crawl4ai: bool | None = None,
    ) -> None:
        self.concurrency = concurrency
        self.timeout = timeout
        self.min_chars = min_chars
        self._use_crawl4ai: bool = (
            _check_crawl4ai() if use_crawl4ai is None else use_crawl4ai
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def enrich_items(
        self,
        items: list[dict[str, Any]],
        session: aiohttp.ClientSession | None = None,
    ) -> int:
        """Enrich *items* in-place; returns the count of items enriched.

        Items whose ``body_text`` already meets ``min_chars`` are skipped.
        """
        targets = [
            item
            for item in items
            if str(item.get("url") or "").startswith("http")
            and len(str(item.get("body_text") or "")) < self.min_chars
        ]
        if not targets:
            return 0

        if self._use_crawl4ai:
            return await self._enrich_crawl4ai(targets)
        else:
            return await self._enrich_aiohttp(targets, session)

    # ------------------------------------------------------------------
    # Crawl4AI path
    # ------------------------------------------------------------------

    async def _enrich_crawl4ai(self, targets: list[dict[str, Any]]) -> int:
        if _requires_dedicated_crawl_loop():
            logger.debug(
                "Running Crawl4AI in a dedicated Proactor loop because the "
                "current Windows event loop does not support subprocesses"
            )
            return await asyncio.to_thread(
                self._run_crawl4ai_in_dedicated_loop,
                targets,
            )

        return await self._enrich_crawl4ai_batch(targets)

    async def _enrich_crawl4ai_batch(self, targets: list[dict[str, Any]]) -> int:
        """Use AsyncWebCrawler to extract article bodies.

        A single browser instance is shared across all targets to avoid the
        per-item Playwright startup/teardown overhead and the CancelledError
        that Playwright raises when the event loop tears down mid-close.
        """
        try:
            from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig  # type: ignore[import]
            from crawl4ai.content_filter_strategy import PruningContentFilter  # type: ignore[import]
            from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator  # type: ignore[import]
        except ImportError:
            logger.warning(
                "crawl4ai import failed at enrichment time; falling back to aiohttp"
            )
            return await self._enrich_aiohttp(targets, None)

        enriched = 0

        browser_cfg = BrowserConfig(headless=True, verbose=False)
        run_cfg = CrawlerRunConfig(
            page_timeout=int(self.timeout * 1000),
            word_count_threshold=5,
            remove_overlay_elements=True,
            remove_consent_popups=True,
            semaphore_count=max(1, min(self.concurrency, 2)),
            stream=False,
            verbose=False,
            magic=True,
            simulate_user=True,
            override_navigator=True,
            wait_until="load",
            delay_before_return_html=1.0,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(
                    threshold=0.48,
                    threshold_type="dynamic",
                    min_word_threshold=5,
                ),
                options={
                    "ignore_links": True,
                    "ignore_images": True,
                    "skip_internal_links": True,
                    "body_width": 0,
                },
            ),
        )

        urls = [str(item.get("url")) for item in targets]
        url_to_item = {
            normalize_url(str(item.get("url"))): item
            for item in targets
            if item.get("url")
        }

        try:
            async with AsyncWebCrawler(config=browser_cfg) as crawler:
                results = list(await crawler.arun_many(urls=urls, config=run_cfg))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Crawl4AI browser session error, falling back to aiohttp: %s", exc)
            return await self._enrich_aiohttp(targets, None)

        fallback_targets: list[dict[str, Any]] = []

        for result in results:
            result_url = normalize_url(
                str(
                    getattr(result, "url", None)
                    or getattr(result, "redirected_url", None)
                    or ""
                )
            )
            item = url_to_item.pop(result_url, None)
            if item is None:
                continue

            if not (result and result.success):
                fallback_targets.append(item)
                continue

            text = _extract_crawl4ai_body(result, self.min_chars)
            if not text:
                fallback_targets.append(item)
                continue

            item["body_text"] = text
            item["body_source"] = "crawl4ai"
            enriched += 1

        fallback_targets.extend(url_to_item.values())

        if fallback_targets:
            enriched += await self._enrich_aiohttp(fallback_targets, None)

        return enriched

    def _run_crawl4ai_in_dedicated_loop(self, targets: list[dict[str, Any]]) -> int:
        """Run Crawl4AI on a subprocess-capable loop from a worker thread."""
        if sys.platform != "win32":
            return asyncio.run(self._enrich_crawl4ai_batch(targets))

        loop = asyncio.ProactorEventLoop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._enrich_crawl4ai_batch(targets))
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:  # noqa: BLE001
                pass
            asyncio.set_event_loop(None)
            loop.close()

    # ------------------------------------------------------------------
    # aiohttp fallback path
    # ------------------------------------------------------------------

    async def _enrich_aiohttp(
        self,
        targets: list[dict[str, Any]],
        session: aiohttp.ClientSession | None,
    ) -> int:
        """Fallback: plain HTTP fetch + HTML extraction."""
        own_session = session is None
        if own_session:
            session = aiohttp.ClientSession()

        semaphore = asyncio.Semaphore(self.concurrency)
        enriched = 0

        async def worker(item: dict[str, Any]) -> None:
            nonlocal enriched
            url = str(item.get("url"))
            async with semaphore:
                body = await _fetch_body_aiohttp(
                    session, url, self.timeout, self.min_chars  # type: ignore[arg-type]
                )
                if body:
                    item["body_text"] = body
                    item["body_source"] = "article_page"
                    enriched += 1

        try:
            await asyncio.gather(*(worker(item) for item in targets))
        finally:
            if own_session:
                await session.close()  # type: ignore[union-attr]

        return enriched
