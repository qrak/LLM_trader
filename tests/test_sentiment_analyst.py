"""Tests for RedditSentimentAnalyst — parsing and formatting methods (no network)."""

import pytest

from src.analyzer.sentiment_analyst import RedditSentimentAnalyst

ATOM_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:media="http://search.yahoo.com/mrss/">
  <category term="Bitcoin" label="r/Bitcoin"/>
  <updated>2026-08-13T12:10:17+00:00</updated>
  <id>/r/Bitcoin/.rss?limit=10</id>
  <link rel="self" href="https://www.reddit.com/r/Bitcoin/.rss?limit=10" type="application/atom+xml"/>
  <title>Bitcoin</title>
  <entry>
    <author><name>/u/satoshi</name><uri>https://www.reddit.com/user/satoshi</uri></author>
    <category term="Bitcoin" label="r/Bitcoin"/>
    <content type="html">&lt;p&gt;Post body text&lt;/p&gt;</content>
    <id>t3_abc123</id>
    <link href="https://www.reddit.com/r/Bitcoin/comments/abc123/bitcoin_breaks_70k/"/>
    <title>Bitcoin breaks $70k resistance</title>
    <updated>2026-08-13T12:09:00+00:00</updated>
  </entry>
  <entry>
    <author><name>/u/bear_bot</name><uri>https://www.reddit.com/user/bear_bot</uri></author>
    <category term="Bitcoin" label="r/Bitcoin"/>
    <content type="html">&lt;p&gt;Post body text&lt;/p&gt;</content>
    <id>t3_def456</id>
    <link href="https://www.reddit.com/r/Bitcoin/comments/def456/etf_outflows_crash/"/>
    <title>ETF outflows spark crash fears</title>
    <updated>2026-08-13T12:08:00+00:00</updated>
  </entry>
  <entry>
    <author><name>/u/neutral_guy</name><uri>https://www.reddit.com/user/neutral_guy</uri></author>
    <category term="Bitcoin" label="r/Bitcoin"/>
    <content type="html">&lt;p&gt;Post body text&lt;/p&gt;</content>
    <id>t3_ghi789</id>
    <link href="https://www.reddit.com/r/Bitcoin/comments/ghi789/monday_thread/"/>
    <title>Daily discussion thread</title>
    <updated>2026-08-13T12:07:00+00:00</updated>
  </entry>
</feed>
"""


class TestRedditSentimentFormatting:
    """Unit tests for sentiment formatting and static methods."""

    @pytest.fixture
    def analyst(self):
        from unittest.mock import MagicMock

        logger = MagicMock()
        return RedditSentimentAnalyst(logger=logger)

    def test_format_sentiment_section_with_posts(self, analyst):
        data = {
            "posts": [
                {
                    "subreddit": "CryptoCurrency",
                    "title": "Bitcoin breaks $70k resistance",
                    "author": "/u/bull",
                    "created_utc": 1755080000,
                },
                {
                    "subreddit": "Bitcoin",
                    "title": "ETF inflows hit $1B weekly",
                    "author": "/u/hodler",
                    "created_utc": 1755081000,
                },
            ],
            "overall_sentiment": "BULLISH",
            "top_topics": ["etf", "halving", "breakout"],
        }
        section = analyst.format_sentiment_section(data)
        assert "Social Sentiment" in section
        assert "BULLISH" in section
        assert "Bitcoin breaks" in section
        assert "ETF inflows" in section
        assert "/u/bull" in section
        assert "⚠" not in section  # no errors

    def test_format_sentiment_section_empty(self, analyst):
        section = analyst.format_sentiment_section({"posts": [], "overall_sentiment": "NO_DATA"})
        assert section == ""

    def test_format_sentiment_section_with_errors(self, analyst):
        data = {
            "posts": [
                {
                    "subreddit": "CryptoCurrency",
                    "title": "Test post",
                    "author": "",
                    "created_utc": 0,
                },
            ],
            "overall_sentiment": "NEUTRAL",
            "top_topics": [],
            "error": "CryptoMarkets: timeout",
        }
        section = analyst.format_sentiment_section(data)
        assert "⚠" in section
        assert "timeout" in section

    def test_compute_overall_sentiment_bullish(self, analyst):
        posts = [
            {"title": "Bitcoin breaks $70k resistance, new all-time high"},
            {"title": "ETF inflows hit record, institutional adoption surges"},
        ]
        assert analyst._compute_overall_sentiment(posts) == "BULLISH"

    def test_compute_overall_sentiment_bearish(self, analyst):
        posts = [
            {"title": "Bitcoin crashes below support, panic selling"},
            {"title": "ETF outflows spark sell-off, bear market fears"},
        ]
        assert analyst._compute_overall_sentiment(posts) == "BEARISH"

    def test_compute_overall_sentiment_neutral(self, analyst):
        posts = [
            {"title": "Daily discussion thread"},
            {"title": "Weekly market update"},
        ]
        assert analyst._compute_overall_sentiment(posts) == "NEUTRAL"

    def test_compute_sentiment_no_data(self, analyst):
        assert analyst._compute_overall_sentiment([]) == "NO_DATA"

    def test_extract_top_topics(self, analyst):
        posts = [
            {"title": "Bitcoin ETF approval expected soon halving"},
            {"title": "ETF inflows break records again Bitcoin"},
        ]
        topics = analyst._extract_top_topics(posts)
        assert "approval" in topics or "expected" in topics or "halving" in topics
        assert "bitcoin" not in topics  # filtered stopword

    def test_extract_top_topics_empty(self, analyst):
        assert analyst._extract_top_topics([]) == []

    def test_parse_atom_feed_extracts_posts(self, analyst):
        posts = analyst._parse_atom_feed(ATOM_SAMPLE, "Bitcoin", limit=10)
        assert len(posts) == 3
        assert posts[0]["title"] == "Bitcoin breaks $70k resistance"
        assert posts[0]["author"] == "/u/satoshi"
        assert posts[0]["subreddit"] == "Bitcoin"
        assert posts[0]["url"].startswith("https://www.reddit.com/")
        assert posts[0]["created_utc"] > 0  # parsed from <updated>
        assert posts[1]["title"] == "ETF outflows spark crash fears"

    def test_parse_atom_feed_respects_limit(self, analyst):
        posts = analyst._parse_atom_feed(ATOM_SAMPLE, "Bitcoin", limit=2)
        assert len(posts) == 2

    def test_parse_atom_feed_invalid_xml(self, analyst):
        assert analyst._parse_atom_feed("<not xml", "Bitcoin", limit=10) == []

    def test_parse_atom_feed_skips_empty_titles(self, analyst):
        sample = ATOM_SAMPLE.replace(
            "<title>Bitcoin breaks $70k resistance</title>", "<title></title>"
        )
        posts = analyst._parse_atom_feed(sample, "Bitcoin", limit=10)
        assert len(posts) == 2
        assert all(p["title"] for p in posts)

    def test_parse_iso8601_utc(self, analyst):
        assert analyst._parse_iso8601("2026-08-13T12:09:00+00:00") == 1786622940

    def test_parse_iso8601_invalid(self, analyst):
        assert analyst._parse_iso8601("not-a-date") == 0
