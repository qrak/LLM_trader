from __future__ import annotations

import asyncio
import ipaddress
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import aiohttp
from trafilatura import extract

from .rss_primitives import extract_html_body_text
from .schema_mapper import has_news_boilerplate, strip_news_boilerplate

if TYPE_CHECKING:
    from src.config.loader import Config
    from src.logger.logger import Logger


class ArticleBodyEnricher:
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
    )

    UNUSABLE_BODY_MARKERS: tuple[str, ...] = (
        "article not found",
        "oops! something went wrong",
        "we're sorry for the inconvenience. please try again",
        "page not found",
        "404 not found",
    )

    def __init__(
        self,
        logger: Logger,
        config: Config | None = None,
        concurrency: int | None = None,
        timeout: float | None = None,
        min_chars: int | None = None,
    ) -> None:
        self.logger = logger
        self.config = config

        self.concurrency: int = (
            concurrency
            if concurrency is not None
            else (config.RAG_NEWS_PAGE_CONCURRENCY if config else 3)
        )
        self.timeout: float = (
            float(timeout)
            if timeout is not None
            else (float(config.RAG_NEWS_FETCH_TIMEOUT) if config else 30.0)
        )
        self.min_chars: int = (
            min_chars
            if min_chars is not None
            else (config.RAG_NEWS_ENRICH_MIN_CHARS if config else 400)
        )

    @staticmethod
    def _is_safe_external_url(url: str) -> bool:

        if not url:
            return False
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        if hostname.lower() in ("localhost", "127.0.0.1", "::1"):
            return False
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return False
        except ValueError:
            pass
        return True

    @classmethod
    def _is_unusable_body(cls, text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text).strip().lower()
        return any(marker in normalized for marker in cls.UNUSABLE_BODY_MARKERS)

    def _extract_body(self, html: str) -> str | None:

        try:
            extracted = extract(
                html,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            ) or ""
        except (TypeError, ValueError):
            extracted = ""

        for candidate in (extracted, extract_html_body_text(html)):
            text = strip_news_boilerplate(candidate)
            if len(text) >= self.min_chars and not self._is_unusable_body(text):
                return text
        return None

    async def _fetch_body_aiohttp(
        self,
        session: aiohttp.ClientSession,
        url: str,
    ) -> str | None:

        try:
            async with session.get(
                url,
                headers={"User-Agent": self.USER_AGENT},
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                if resp.status != 200:
                    return None
                return self._extract_body(await resp.text(errors="replace"))
        except Exception:  # noqa: BLE001
            return None

    async def enrich_items(
        self,
        items: list[dict[str, Any]],
        session: aiohttp.ClientSession | None = None,
    ) -> int:

        targets = [
            item
            for item in items
            if self._is_safe_external_url(str(item.get("url") or ""))
            and (
                len(str(item.get("body_text") or "")) < self.min_chars
                or has_news_boilerplate(str(item.get("body_text") or ""))
            )
        ]
        if not targets:
            return 0

        own_session = session is None
        if own_session:
            session = aiohttp.ClientSession()

        semaphore = asyncio.Semaphore(self.concurrency)
        enriched = 0

        async def worker(item: dict[str, Any]) -> None:
            nonlocal enriched
            async with semaphore:
                body = await self._fetch_body_aiohttp(session, str(item.get("url")))  # type: ignore[arg-type]
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
