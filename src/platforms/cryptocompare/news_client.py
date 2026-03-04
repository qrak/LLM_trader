"""
CryptoCompare News Client
Handles direct API interactions with the CryptoCompare news service.
"""
import time
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, TYPE_CHECKING

import aiohttp

from src.logger.logger import Logger

if TYPE_CHECKING:
    from src.config.protocol import ConfigProtocol

RELIABLE_NEWS_FEEDS = [
    'cointelegraph',
    'coindesk',
    'theblock',
    'decrypt',
    'bitcoinmagazine',
    'cryptoslate',
]

_IMPORTANT_CATEGORIES = ["BTC", "ETH", "DeFi", "NFT", "Layer 2", "Stablecoin", "Altcoin"]


class CryptoCompareNewsClient:
    """Fetches raw news articles from CryptoCompare with resilient retry logic."""

    def __init__(self, logger: Logger, config: "ConfigProtocol"):
        self.logger = logger
        self.config = config

    async def fetch_news(
        self,
        session: Optional[aiohttp.ClientSession] = None,
        api_categories: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch crypto news with automatic retry and CDN-cache-busting fallback."""
        session_to_use = session or aiohttp.ClientSession()
        use_temp_session = session is None

        try:
            # Attempt 1: Full URL with categories + feeds
            url = self._build_url(api_categories)
            articles = await self._fetch_once(session_to_use, url)
            if articles:
                return articles

            # Attempt 2: Cache-buster — bypasses CDN cached error responses (the 0.1s failure trap)
            self.logger.warning("Empty response from CryptoCompare. Retrying in 5s with cache-buster...")
            await asyncio.sleep(5)
            articles = await self._fetch_once(session_to_use, url + f"&_t={int(time.time())}")
            if articles:
                return articles

            # Attempt 3: Drop categories to simplify the DB query and avoid upstream timeouts
            self.logger.warning("Empty response again. Retrying in 5s without category filters...")
            await asyncio.sleep(5)
            simplified = self._build_url(None) + f"&_t={int(time.time())}"
            return await self._fetch_once(session_to_use, simplified)

        finally:
            if use_temp_session:
                await session_to_use.close()

    def filter_by_age(
        self,
        articles: List[Dict[str, Any]],
        max_age_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Filter articles older than max_age_hours and sort newest-first."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).timestamp()
        recent = [a for a in articles if a.get("published_on", 0) > cutoff]
        recent.sort(key=lambda a: a.get("published_on", 0), reverse=True)
        return recent

    # ── Private ───────────────────────────────────────────────────────────────

    async def _fetch_once(self, session: aiohttp.ClientSession, url: str) -> List[Dict[str, Any]]:
        """Execute one HTTP request, returning the raw article list."""
        start = time.monotonic()
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=90)) as resp:
            elapsed = time.monotonic() - start
            return self._parse_response(resp.status, await resp.json(), elapsed)

    def _parse_response(self, status: int, data: Any, elapsed: float) -> List[Dict[str, Any]]:
        if status != 200:
            self.logger.error("News API request failed: status=%s, elapsed=%.1fs", status, elapsed)
            return []

        if not data or "Data" not in data:
            self.logger.warning(
                "News API unexpected body (no 'Data' key): keys=%s, elapsed=%.1fs",
                list(data.keys()) if data else "None", elapsed
            )
            return []

        articles = data["Data"]
        self.logger.debug("Fetched %s news articles from CryptoCompare (%.1fs)", len(articles), elapsed)

        if not articles:
            self.logger.warning(
                "CryptoCompare returned 0 articles — Message=%s, Type=%s, "
                "RateLimit=%s, HasWarning=%s, elapsed=%.1fs",
                data.get("Message"), data.get("Type"),
                data.get("RateLimit", {}), data.get("HasWarning"), elapsed
            )
        return articles

    def _build_url(self, api_categories: Optional[List[Dict[str, Any]]]) -> str:
        url = self.config.RAG_NEWS_API_URL
        url += self._categories_param(api_categories)
        url += self._feeds_param()
        return self._with_api_key(url)

    def _categories_param(self, api_categories: Optional[List[Dict[str, Any]]]) -> str:
        if not api_categories:
            return ""
        cats = [c["categoryName"] for c in api_categories
                if c.get("categoryName") in _IMPORTANT_CATEGORIES]
        return f"&categories={','.join(cats[:5])}" if cats else ""

    def _feeds_param(self) -> str:
        if not getattr(self.config, "RAG_NEWS_FILTER_SOURCES", True):
            return ""
        feeds = getattr(self.config, "RAG_NEWS_ALLOWED_FEEDS", None) or RELIABLE_NEWS_FEEDS
        return f"&feeds={','.join(feeds)}" if feeds else ""

    def _with_api_key(self, url: str) -> str:
        key = self.config.CRYPTOCOMPARE_API_KEY
        if key and "api_key=" not in url:
            sep = "&" if "?" in url else "?"
            return f"{url}{sep}api_key={key}"
        return url
