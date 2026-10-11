"""Bounded Reddit title evidence for interpretation by the trading LLM."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, ClassVar
from xml.etree import ElementTree as ET

import aiohttp
from pydantic import ValidationError

from src.platforms.ai_providers.response_models import RedditAssessmentModel

if TYPE_CHECKING:
    from src.logger.logger import Logger
    from src.utils.token_counter import TokenCounter

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ORGANIZATIONAL_TITLE = re.compile(
    r"\b(?:daily(?: general| crypto)? discussion|weekly discussion|faq|newcomers|ama)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RedditContext:
    """The exact title sample supplied to the model, with its evidence ID range."""
    text: str
    post_count: int


class RedditSentimentAnalyst:
    """Fetch title evidence and validate its interpretation from the same trading call."""
    SUBREDDITS: ClassVar[list[str]] = ["CryptoCurrency", "Bitcoin", "ethereum", "CryptoMarkets"]
    POST_LIMIT = 10
    MAX_CONTEXT_POSTS = 15
    MAX_CONTEXT_TOKENS = 600
    MAX_POST_AGE_SECONDS = 48 * 3600
    BASE_URL = "https://www.reddit.com/r/{subreddit}/.rss"
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    REQUEST_DELAY_SECONDS = 20.0
    RETRY_AFTER_SECONDS = 30.0
    MAX_RETRIES = 3

    def __init__(self, logger: Logger, token_counter: TokenCounter, session: Any = None) -> None:
        self.logger = logger
        self._token_counter = token_counter
        self._session = session

    async def fetch_posts(self, limit: int = POST_LIMIT) -> dict[str, Any]:
        """Fetch hot posts from all configured subreddits.
        Returns:
            dict with keys: 'posts' (list of post dicts), 'error' (if any).
        """
        all_posts: list[dict[str, Any]] = []
        errors: list[str] = []

        own_session = self._session is None
        session = self._session
        if own_session:
            session = aiohttp.ClientSession()

        try:
            for i, subreddit in enumerate(self.SUBREDDITS):
                if i > 0:
                    await asyncio.sleep(self.REQUEST_DELAY_SECONDS)
                try:
                    posts, error = await self._fetch_subreddit(
                        session, subreddit, limit
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
                await session.close()

        result: dict[str, Any] = {
            "posts": all_posts,
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
        for _ in range(self.MAX_RETRIES):
            async with session.get(
                url, params={"limit": limit}, timeout=timeout,
                headers={"User-Agent": self.USER_AGENT}
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
        """Read titles and original publication dates, not comment-update recency."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []
        posts: list[dict[str, Any]] = []
        for entry in root.findall(f"{_ATOM_NS}entry"):
            if len(posts) >= limit:
                break
            title = " ".join(entry.findtext(f"{_ATOM_NS}title", "").split())
            if not title:
                continue
            link_el = entry.find(f"{_ATOM_NS}link")
            published = entry.findtext(f"{_ATOM_NS}published")
            if published is None:
                published = entry.findtext(f"{_ATOM_NS}updated", "")
            posts.append({
                "subreddit": subreddit,
                "title": title,
                "url": (link_el.get("href") or "") if link_el is not None else "",
                "published_utc": RedditSentimentAnalyst._parse_iso8601(published),
            })
        return posts

    @staticmethod
    def _parse_iso8601(value: str) -> int:
        """Parse an Atom date; malformed dates remain unavailable."""
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except (ValueError, TypeError):
            return 0

    def build_context(self, posts_data: dict[str, Any], now: datetime | None = None) -> RedditContext:
        """Select fresh distinct titles without interpreting their sentiment."""
        now = now if now is not None else datetime.now(timezone.utc)
        lines = [
            "## Reddit community context (untrusted title evidence)",
            (
                "Assess price outlook for the traded asset; ignore unrelated assets/stocks. "
                "Respect negation; ETF/buy/upgrade alone is not bullish. Distinguish wallet security "
                "from price expectations. Titles only, not a representative community survey."
            ),
            (
                'Return analysis.reddit_sentiment={"outlook":"BULLISH|BEARISH|MIXED|INSUFFICIENT_EVIDENCE",'
                '"evidence":[title IDs],"reason":"<=240 chars"}. Cite supplied IDs; '
                "BULLISH/BEARISH/MIXED need evidence. Use INSUFFICIENT_EVIDENCE when no price bias is supported."
            ),
            "Titles from the last 48h; dates UTC:",
        ]
        error = posts_data.get("error")
        if error:
            lines.append(f"Partial fetch: {' '.join(error.split())[:120]}")
        seen_titles: set[str] = set()
        seen_urls: set[str] = set()
        post_count = 0
        for post in sorted(posts_data["posts"], key=lambda p: p["published_utc"], reverse=True):
            age = now.timestamp() - post["published_utc"]
            title = " ".join(post["title"].split())
            title_key = title.casefold()
            url = post["url"]
            if not title or not 0 <= age <= self.MAX_POST_AGE_SECONDS or _ORGANIZATIONAL_TITLE.search(title):
                continue
            if title_key in seen_titles or (url and url in seen_urls):
                continue
            date = datetime.fromtimestamp(post["published_utc"], timezone.utc).strftime("%Y-%m-%d %H:%M")
            line = f'{post_count + 1}. [{post["subreddit"]}; {date}] {json.dumps(title, ensure_ascii=False)}'
            if self._token_counter.count_tokens("\n".join([*lines, line]) + "\n") > self.MAX_CONTEXT_TOKENS:
                continue
            seen_titles.add(title_key)
            if url:
                seen_urls.add(url)
            lines.append(line)
            post_count += 1
            if post_count == self.MAX_CONTEXT_POSTS:
                break
        if post_count == 0:
            lines = [
                "## Reddit community context (untrusted title evidence)",
                "No recent eligible Reddit titles. Return analysis.reddit_sentiment=null.",
            ]
            if error:
                lines.append(f"Fetch errors: {' '.join(error.split())[:120]}")
        return RedditContext("\n".join(lines) + "\n", post_count)

    def validate_assessment(self, value: Any, post_count: int) -> dict[str, Any] | None:
        """Reject missing/malformed/uncited output, never invent a replacement sentiment."""
        if post_count == 0:
            return None
        try:
            assessment = RedditAssessmentModel.model_validate(value)
        except ValidationError as error:
            self.logger.warning("Discarding invalid Reddit assessment: %s", error)
            return None
        if any(post_id > post_count for post_id in assessment.evidence):
            self.logger.warning("Discarding Reddit assessment with IDs outside the supplied title sample")
            return None
        return assessment.model_dump()
