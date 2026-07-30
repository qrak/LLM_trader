"""X/Twitter sentiment via Nitter (no API key, no login).

Nitter is a free, open-source Twitter frontend that exposes public RSS/JSON
endpoints without authentication. This module scrapes crypto-related tweets
through Nitter instances and builds a sentiment summary for the LLM prompt.

Nitter instances:
- nitter.net (primary)
- nitter.poast.org (fallback)
- nitter.privacydev.net (fallback)

On failure, falls back gracefully with empty result.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from typing import TYPE_CHECKING, Any, ClassVar

import aiohttp

if TYPE_CHECKING:
    from src.logger.logger import Logger


class NitterSentimentAnalyst:
    """Fetches recent crypto tweets via Nitter RSS feeds."""

    # Nitter instances to try, in order
    NITTER_INSTANCES: ClassVar[list[str]] = [
        "https://nitter.net",
        "https://nitter.poast.org",
        "https://nitter.privacydev.net",
    ]

    # Max items to parse per query
    MAX_ITEMS = 15

    # Timeout per request (seconds)
    TIMEOUT = 15

    def __init__(self, logger: Logger, symbols: list[str] | None = None) -> None:
        self.logger = logger
        self.symbols = symbols or []

    async def fetch_sentiment(self) -> dict[str, Any]:
        """Fetch recent tweets from Nitter RSS feeds.

        Returns:
            dict with keys: 'tweets' (list), 'overall_sentiment', 'error' (if any).
        """

        all_tweets: list[dict[str, str]] = []
        errors: list[str] = []
        instance = self.NITTER_INSTANCES[0]  # try primary first, fall back

        for nitter_url in self.NITTER_INSTANCES:
            try:
                tweets = await self._fetch_from_instance(nitter_url)
                if tweets:
                    all_tweets = tweets
                    instance = nitter_url
                    break
            except Exception as e:  # noqa: BLE001
                errors.append(f"{nitter_url}: {e}")

        result: dict[str, Any] = {
            "tweets": all_tweets,
            "instance": instance,
            "overall_sentiment": self._compute_sentiment(all_tweets),
            "top_terms": self._extract_terms(all_tweets),
        }
        if errors:
            result["error"] = "; ".join(errors)

        return result

    async def _fetch_from_instance(self, base_url: str) -> list[dict[str, str]]:
        """Fetch tweets from a Nitter instance using RSS search."""
        all_items: list[dict[str, str]] = []
        seen_titles: set[str] = set()

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.TIMEOUT)
        ) as session:
            for query in self.symbols:
                url = f"{base_url}/search?f=tweets&q={query}"
                try:
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            continue
                        html = await resp.text()

                        # Parse tweet-like content from HTML
                        # Nitter renders tweets as: <div class="tweet-content">text</div>
                        # or in <meta> / <title> tags
                        items = self._parse_nitter_html(html)
                        for item in items:
                            title = item.get("title", "")
                            if title and title not in seen_titles:
                                seen_titles.add(title)
                                all_items.append(item)

                        if len(all_items) >= self.MAX_ITEMS:
                            break

                except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                    self.logger.debug("Nitter fetch failed for %s: %s", url, e)
                    continue

        return all_items[: self.MAX_ITEMS]

    @staticmethod
    def _parse_nitter_html(html: str) -> list[dict[str, str]]:
        """Extract tweet text and metadata from Nitter HTML.

        Nitter uses specific CSS classes and DOM structure.
        This is a best-effort parser — we look for common patterns.
        """
        items: list[dict[str, str]] = []

        # Pattern 1: tweet-content divs
        content_pattern = re.compile(
            r'<div[^>]*class="[^"]*tweet-content[^"]*"[^>]*>(.*?)</div>',
            re.DOTALL,
        )
        for match in content_pattern.finditer(html):
            text = re.sub(r"<[^>]+>", "", match.group(1)).strip()
            text = re.sub(r"&amp;", "&", text)
            text = re.sub(r"&lt;", "<", text)
            text = re.sub(r"&gt;", ">", text)
            text = re.sub(r"&quot;", '"', text)
            if text and len(text) > 10:
                items.append({"title": text, "source": "tweet"})

        # Pattern 2: meta tags (og:title, twitter:title)
        meta_pattern = re.compile(
            r'<meta[^>]+(?:name|property)="(?:og:title|twitter:title)"[^>]+content="([^"]+)"',
            re.IGNORECASE,
        )
        for match in meta_pattern.finditer(html):
            title = match.group(1)
            if title and len(title) > 10 and not any(
                title == item["title"] for item in items
            ):
                items.append({"title": title, "source": "meta"})

        # Pattern 3: tweet-body in newer Nitter layouts
        body_pattern = re.compile(
            r'<div[^>]*class="[^"]*tweet-body[^"]*"[^>]*>(.*?)</div>',
            re.DOTALL,
        )
        for match in body_pattern.finditer(html):
            text = re.sub(r"<[^>]+>", "", match.group(1)).strip()
            text = re.sub(r"&amp;", "&", text)
            if text and len(text) > 10:
                items.append({"title": text, "source": "tweet-body"})

        return items

    @staticmethod
    def _compute_sentiment(tweets: list[dict[str, str]]) -> str:
        """Simple keyword-based sentiment from tweet text."""
        if not tweets:
            return "NO_DATA"

        bullish_words = [
            "bullish", "breakout", "pump", "moon", "long", "buy", "accumulation",
            "rally", "surge", "ath", "all-time high", "green", "up only",
        ]
        bearish_words = [
            "bearish", "dump", "crash", "short", "sell", "distribution",
            "correction", "dip", "red", "rekt", "liquidation", "dead",
        ]

        bullish_score = 0
        bearish_score = 0
        total_tweets = len(tweets)

        for tweet in tweets:
            text = tweet.get("title", "").lower()
            for word in bullish_words:
                if word in text:
                    bullish_score += 1
            for word in bearish_words:
                if word in text:
                    bearish_score += 1

        if total_tweets == 0:
            return "NEUTRAL"

        if bullish_score == 0 and bearish_score == 0:
            return "NEUTRAL"

        ratio = bullish_score / max(bearish_score, 1)
        if ratio > 2.0:
            return "BULLISH"
        if ratio > 1.3:
            return "SLIGHTLY_BULLISH"
        if ratio < 0.5:
            return "BEARISH"
        if ratio < 0.75:
            return "SLIGHTLY_BEARISH"
        return "NEUTRAL"

    @staticmethod
    def _extract_terms(tweets: list[dict[str, str]], top_n: int = 8) -> list[str]:
        """Extract most common significant terms from tweet text."""
        all_words: list[str] = []
        for tweet in tweets:
            text = tweet.get("title", "").lower()
            # Extract $TICKER and #hashtag and 4+ letter words
            tickers = re.findall(r"\$[a-z]{2,8}", text)
            hashtags = re.findall(r"#[a-z]{3,20}", text)
            words = re.findall(r"[a-z]{4,}", text)
            all_words.extend(tickers + hashtags + words)

        stopwords = {
            "this", "that", "with", "from", "have", "been", "will", "just",
            "what", "when", "your", "about", "which", "their", "there",
            "would", "could", "should", "more", "some", "than", "like",
        }
        filtered = [w for w in all_words if w not in stopwords]
        counter = Counter(filtered)
        return [word for word, _ in counter.most_common(top_n)]

    def format_sentiment_section(self, data: dict[str, Any]) -> str:
        """Build a formatted sentiment section for the LLM prompt.

        Args:
            data: Output from fetch_sentiment().

        Returns:
            Formatted markdown string, or empty string if no tweets.
        """
        tweets = data.get("tweets", [])
        if not tweets:
            return ""

        lines = [
            "",
            "## X/Twitter Sentiment (via Nitter — near real-time)",
            f"Overall: **{data.get('overall_sentiment', 'N/A')}** ({len(tweets)} tweets)",
        ]

        terms = data.get("top_terms", [])
        if terms:
            lines.append(f"Trending: {' • '.join(terms[:6])}")
        lines.append("")

        # Show top tweets (first 100 chars each)
        for i, tweet in enumerate(tweets[:6], 1):
            text = tweet.get("title", "")[:100]
            lines.append(f"{i}. {text}")

        error = data.get("error")
        if error:
            lines.append(f"\n⚠️ Fetch errors: {error}")

        lines.append(
            "\n*Note: X/Twitter sentiment via Nitter. Faster than Reddit but less "
            "filtered. Weigh against technical analysis — social media amplifies noise.*"
        )
        return "\n".join(lines)
