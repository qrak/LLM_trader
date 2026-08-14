"""Social media sentiment analyst using Reddit public Atom feeds.

Fetches top posts from crypto subreddits to gauge community sentiment.
No API key required.

NOTE (2026-08): old.reddit.com HTML scraping is dead — Reddit serves an
IP-level login-wall redirect (``/login/?reason=lor2``) to datacenter
addresses, so the classic ``<div class="thing">`` parse returns nothing.
The official www.reddit.com ``.rss`` Atom endpoint remains reachable and
is the source used here. Atom feeds carry no score/comment/ratio data,
so overall sentiment is computed from a title keyword lexicon instead of
score thresholds.

Based on TradingAgents (Xiao et al., 2024) sentiment analyst role.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, ClassVar
from xml.etree import ElementTree as ET

import aiohttp

if TYPE_CHECKING:
    from src.logger.logger import Logger


# Title keyword lexicons for sentiment scoring (Atom feeds have no scores).
_BULLISH_TERMS = (
    "bullish", "breakout", "surge", "rally", "pump", "ath", "all-time high",
    "gains", "adoption", "inflow", "etf", "halving", "institutional",
    "recovery", "rebound", "accumulate", "buy", "bull", "moon", "record",
    "upgrade", "launch", "partnership",
)
_BEARISH_TERMS = (
    "bearish", "crash", "dump", "plunge", "sell-off", "selloff", "correction",
    "capitulation", "hack", "exploit", "scam", "fraud", "lawsuit", "ban",
    "fud", "fear", "panic", "liquidation", "recession", "decline", "drop",
    "bear", "rug", "outflow", "underwater",
)
_BULLISH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in _BULLISH_TERMS
]
_BEARISH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in _BEARISH_TERMS
]

_ATOM_NS = "{http://www.w3.org/2005/Atom}"


class RedditSentimentAnalyst:
    """Fetches and analyzes sentiment from crypto subreddits via Atom RSS."""

    # Subreddits to query — ordered by relevance to crypto trading
    SUBREDDITS: ClassVar[list[str]] = [
        "CryptoCurrency",
        "Bitcoin",
        "ethereum",
        "CryptoMarkets",
    ]

    # Max posts per subreddit
    POST_LIMIT = 10

    # Official Atom feed endpoint (no auth, no API key)
    BASE_URL = "https://www.reddit.com/r/{subreddit}/.rss"

    # User-Agent to present as a normal browser
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    # Politeness: Reddit rate-limits bursts from flagged IPs (HTTP 429).
    # Observed: ~1 request / 20-30s succeeds, bursts get 429 + Retry-After.
    REQUEST_DELAY_SECONDS = 20.0
    RETRY_AFTER_SECONDS = 30.0
    MAX_RETRIES = 3

    def __init__(self, logger: Logger, session: Any = None) -> None:
        """Initialize the sentiment analyst.

        Args:
            logger: Logger instance.
            session: Optional aiohttp.ClientSession for HTTP requests.
                If None, a new session is created per fetch.
        """
        self.logger = logger
        self._session = session

    async def fetch_sentiment(self, limit: int = POST_LIMIT) -> dict[str, Any]:
        """Fetch hot posts from all configured subreddits.

        Args:
            limit: Max posts per subreddit.

        Returns:
            dict with keys: 'posts' (list of post dicts), 'overall_sentiment',
            'top_topics', 'error' (if any).
        """
        headers = {"User-Agent": self.USER_AGENT}
        all_posts: list[dict[str, Any]] = []
        errors: list[str] = []

        own_session = self._session is None
        session = self._session
        if own_session:
            session = aiohttp.ClientSession(headers=headers)

        try:
            for i, subreddit in enumerate(self.SUBREDDITS):
                if i > 0:
                    await asyncio.sleep(self.REQUEST_DELAY_SECONDS)
                try:
                    posts, error = await self._fetch_subreddit(
                        session, subreddit, limit  # type: ignore[arg-type]
                    )
                    if error:
                        errors.append(error)
                    all_posts.extend(posts)
                except asyncio.TimeoutError:
                    errors.append(f"{subreddit}: timeout")
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{subreddit}: {e}")
        finally:
            if own_session and session is not None:
                await session.close()  # type: ignore[func-returns-value]

        result: dict[str, Any] = {
            "posts": all_posts,
            "overall_sentiment": self._compute_overall_sentiment(all_posts),
            "top_topics": self._extract_top_topics(all_posts),
        }
        if errors:
            result["error"] = "; ".join(errors)
            self.logger.warning("Reddit fetch had errors: %s", result["error"])
        if not all_posts:
            self.logger.warning(
                "Reddit returned 0 posts across %d subreddits", len(self.SUBREDDITS)
            )

        return result

    async def _fetch_subreddit(
        self,
        session: aiohttp.ClientSession,
        subreddit: str,
        limit: int,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Fetch one subreddit's Atom feed with 429 backoff.

        Returns:
            (posts, error_or_None).
        """
        url = self.BASE_URL.format(subreddit=subreddit)
        timeout = aiohttp.ClientTimeout(total=15)
        for attempt in range(self.MAX_RETRIES):
            async with session.get(
                url, params={"limit": limit}, timeout=timeout
            ) as resp:
                if resp.status == 429:
                    retry_after = resp.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else self.RETRY_AFTER_SECONDS
                    await asyncio.sleep(delay)
                    continue
                if resp.status != 200:
                    return [], f"{subreddit}: HTTP {resp.status}"
                xml_text = await resp.text()
                posts = self._parse_atom_feed(xml_text, subreddit, limit)
                if not posts:
                    return [], f"{subreddit}: no posts parsed"
                return posts, None
        return [], f"{subreddit}: HTTP 429 (rate limited)"

    @staticmethod
    def _parse_atom_feed(xml_text: str, subreddit: str, limit: int) -> list[dict[str, Any]]:
        """Parse a www.reddit.com Atom feed into post dicts.

        Atom entries expose title/author/link/updated only — no scores,
        comment counts, or upvote ratios.
        """
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []

        posts: list[dict[str, Any]] = []
        for entry in root.findall(f"{_ATOM_NS}entry"):
            if len(posts) >= limit:
                break

            title_el = entry.find(f"{_ATOM_NS}title")
            title = (title_el.text or "").strip() if title_el is not None else ""
            if not title:
                continue

            author = ""
            author_el = entry.find(f"{_ATOM_NS}author/{_ATOM_NS}name")
            if author_el is not None and author_el.text:
                author = author_el.text.strip()

            url = ""
            link_el = entry.find(f"{_ATOM_NS}link")
            if link_el is not None:
                url = link_el.get("href") or ""

            created_utc = 0
            updated_el = entry.find(f"{_ATOM_NS}updated")
            if updated_el is not None and updated_el.text:
                created_utc = RedditSentimentAnalyst._parse_iso8601(updated_el.text)

            posts.append({
                "subreddit": subreddit,
                "title": title,
                "author": author,
                "url": url,
                "score": 0,
                "num_comments": 0,
                "upvote_ratio": 0.0,
                "created_utc": created_utc,
            })

        return posts

    @staticmethod
    def _parse_iso8601(value: str) -> int:
        """Parse an Atom ``updated`` timestamp to epoch seconds (UTC)."""
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _compute_overall_sentiment(posts: list[dict[str, Any]]) -> str:
        """Compute overall sentiment from bullish/bearish title keywords.

        Args:
            posts: List of post dicts (from fetch_sentiment).

        Returns:
            One of NO_DATA / BULLISH / SLIGHTLY_BULLISH / NEUTRAL /
            SLIGHTLY_BEARISH / BEARISH.
        """
        if not posts:
            return "NO_DATA"

        bull = bear = 0
        for post in posts:
            title = post.get("title", "")
            bull += sum(1 for pat in _BULLISH_PATTERNS if pat.search(title))
            bear += sum(1 for pat in _BEARISH_PATTERNS if pat.search(title))

        total = bull + bear
        if total == 0:
            return "NEUTRAL"

        ratio = bull / total
        if ratio >= 0.75 and bull >= 2:
            return "BULLISH"
        if ratio >= 0.6:
            return "SLIGHTLY_BULLISH"
        if ratio <= 0.25 and bear >= 2:
            return "BEARISH"
        if ratio <= 0.4:
            return "SLIGHTLY_BEARISH"
        return "NEUTRAL"

    @staticmethod
    def _extract_top_topics(posts: list[dict[str, Any]], top_n: int = 5) -> list[str]:
        """Extract most frequent keywords from post titles."""
        all_words: list[str] = []
        for post in posts:
            title = post.get("title", "").lower()
            words = re.findall(r"[a-z]{4,}", title)
            all_words.extend(words)

        stopwords = {
            "this", "that", "with", "from", "have", "been", "will",
            "just", "what", "when", "your", "about", "which", "their",
            "there", "would", "could", "should", "more", "some", "than",
            "like", "price", "market", "crypto", "bitcoin", "ethereum",
        }
        filtered = [w for w in all_words if w not in stopwords]

        counter = Counter(filtered)
        return [word for word, _ in counter.most_common(top_n)]

    def format_sentiment_section(self, sentiment_data: dict[str, Any]) -> str:
        """Build a formatted sentiment section for the LLM prompt.

        Args:
            sentiment_data: Output from fetch_sentiment().

        Returns:
            Formatted markdown string, or empty string if no data.
        """
        posts = sentiment_data.get("posts", [])
        if not posts:
            return ""

        subreddits = ", ".join(f"r/{s}" for s in self.SUBREDDITS)
        lines = [
            "",
            f"## Social Sentiment (Reddit — {subreddits})",
            f"Overall: **{sentiment_data.get('overall_sentiment', 'N/A')}**",
        ]

        topics = sentiment_data.get("top_topics", [])
        if topics:
            lines.append(f"Trending topics: {', '.join(topics[:5])}")
        lines.append("")

        # Newest first — Atom feeds carry no scores to rank by.
        top_posts = sorted(
            posts, key=lambda p: p.get("created_utc", 0), reverse=True
        )[:5]
        for i, post in enumerate(top_posts, 1):
            author = post.get("author", "")
            by = f" (by {author})" if author else ""
            lines.append(
                f"{i}. [{post['subreddit']}] **{post['title'][:120]}**{by}"
            )

        error = sentiment_data.get("error")
        if error:
            lines.append(f"\n⚠️ Fetch errors: {error}")
        lines.append("")

        return "\n".join(lines)
