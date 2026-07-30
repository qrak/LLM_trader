"""Social media sentiment analyst using Reddit's public JSON API.

Fetches top posts from crypto subreddits to gauge community sentiment.
No API key required — uses Reddit's free public JSON endpoint.

Based on TradingAgents (Xiao et al., 2024) sentiment analyst role.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from src.logger.logger import Logger


class RedditSentimentAnalyst:
    """Fetches and analyzes sentiment from crypto subreddits."""

    # Subreddits to query — ordered by relevance to crypto trading
    SUBREDDITS: ClassVar[list[str]] = [
        "CryptoCurrency",
        "Bitcoin",
        "ethereum",
        "CryptoMarkets",
    ]

    # Max posts per subreddit
    POST_LIMIT = 10

    # Base URL for Reddit's public JSON API (no auth needed)
    BASE_URL = "https://www.reddit.com/r/{subreddit}/hot.json"

    # User-Agent required by Reddit's API policy
    USER_AGENT = "LLM_trader/1.0 (sentiment analysis bot; contact via GitHub)"

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
        import aiohttp

        headers = {"User-Agent": self.USER_AGENT}
        all_posts: list[dict[str, Any]] = []
        errors: list[str] = []

        own_session = self._session is None
        session = self._session
        if own_session:
            session = aiohttp.ClientSession(headers=headers)

        try:
            for subreddit in self.SUBREDDITS:
                url = self.BASE_URL.format(subreddit=subreddit)
                try:
                    async with session.get(  # type: ignore[union-attr]
                        url, params={"limit": limit}, timeout=aiohttp.ClientTimeout(total=15)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            posts = data.get("data", {}).get("children", [])
                            for post in posts:
                                post_data = post.get("data", {})
                                all_posts.append({
                                    "subreddit": subreddit,
                                    "title": post_data.get("title", ""),
                                    "score": post_data.get("score", 0),
                                    "num_comments": post_data.get("num_comments", 0),
                                    "upvote_ratio": post_data.get("upvote_ratio", 0.0),
                                    "created_utc": post_data.get("created_utc", 0),
                                })
                        else:
                            errors.append(f"{subreddit}: HTTP {resp.status}")
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

        return result

    @staticmethod
    def _compute_overall_sentiment(posts: list[dict[str, Any]]) -> str:
        """Compute overall sentiment from post scores and ratios."""
        if not posts:
            return "NO_DATA"

        total_score = sum(p.get("score", 0) for p in posts)
        avg_ratio = (
            sum(p.get("upvote_ratio", 0.0) for p in posts) / len(posts)
            if posts else 0.0
        )

        if total_score > 5000 and avg_ratio > 0.85:
            return "BULLISH"
        if total_score > 2000 and avg_ratio > 0.75:
            return "SLIGHTLY_BULLISH"
        if total_score < 500 and avg_ratio < 0.65:
            return "BEARISH"
        if avg_ratio < 0.70:
            return "SLIGHTLY_BEARISH"
        return "NEUTRAL"

    @staticmethod
    def _extract_top_topics(posts: list[dict[str, Any]], top_n: int = 5) -> list[str]:
        """Extract most frequent keywords from post titles."""
        import re
        from collections import Counter

        all_words: list[str] = []
        for post in posts:
            title = post.get("title", "").lower()
            # Extract words, filter out short/common ones
            words = re.findall(r"[a-z]{4,}", title)
            all_words.extend(words)

        # Filter out common stopwords
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

        lines = [
            "",
            "## Social Sentiment (Reddit — r/CryptoCurrency, r/Bitcoin, r/ethereum)",
            f"Overall: **{sentiment_data.get('overall_sentiment', 'N/A')}**",
        ]

        topics = sentiment_data.get("top_topics", [])
        if topics:
            lines.append(f"Trending topics: {', '.join(topics[:5])}")
        lines.append("")

        # Top 5 posts by score
        top_posts = sorted(posts, key=lambda p: p.get("score", 0), reverse=True)[:5]
        for i, post in enumerate(top_posts, 1):
            score = post.get("score", 0)
            comments = post.get("num_comments", 0)
            ratio = post.get("upvote_ratio", 0.0)
            lines.append(
                f"{i}. [{post['subreddit']}] **{score}**↑ ({ratio:.0%} ratio, {comments} comments): "
                f"{post['title'][:120]}"
            )

        error = sentiment_data.get("error")
        if error:
            lines.append(f"\n⚠️ Fetch errors: {error}")

        lines.append(
            "\n*Note: Social sentiment is supplementary. Weigh against "
            "technical indicators — crowd sentiment often peaks at reversals.*"
        )
        return "\n".join(lines)
