"""Tests for RedditSentimentAnalyst — parsing and formatting methods (no network)."""

import pytest

from src.analyzer.sentiment_analyst import RedditSentimentAnalyst


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
                    "score": 2431,
                    "num_comments": 312,
                    "upvote_ratio": 0.94,
                },
                {
                    "subreddit": "Bitcoin",
                    "title": "ETF inflows hit $1B weekly",
                    "score": 1856,
                    "num_comments": 245,
                    "upvote_ratio": 0.91,
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
        assert "⚠" not in section  # no errors

    def test_format_sentiment_section_empty(self, analyst):
        section = analyst.format_sentiment_section({"posts": [], "overall_sentiment": "NO_DATA"})
        assert section == ""

    def test_format_sentiment_section_with_errors(self, analyst):
        data = {
            "posts": [
                {"subreddit": "CryptoCurrency", "title": "Test", "score": 10, "num_comments": 5, "upvote_ratio": 0.5},
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
            {"score": 3000, "upvote_ratio": 0.90},
            {"score": 4000, "upvote_ratio": 0.88},
        ]
        assert analyst._compute_overall_sentiment(posts) == "BULLISH"

    def test_compute_overall_sentiment_bearish(self, analyst):
        posts = [
            {"score": 100, "upvote_ratio": 0.50},
            {"score": 200, "upvote_ratio": 0.60},
        ]
        assert analyst._compute_overall_sentiment(posts) == "BEARISH"

    def test_compute_overall_sentiment_neutral(self, analyst):
        posts = [
            {"score": 1500, "upvote_ratio": 0.72},
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
