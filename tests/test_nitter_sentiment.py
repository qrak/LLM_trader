"""Tests for NitterSentimentAnalyst — parsing and formatting (no network)."""

import pytest

from src.analyzer.nitter_sentiment import NitterSentimentAnalyst


class TestNitterHTMLParsing:
    """Unit tests for Nitter HTML parsing static methods."""

    def test_parse_tweet_content_divs(self):
        html = """
        <html>
        <div class="tweet-content">Bitcoin just broke $70k resistance!</div>
        <div class="tweet-content">ETH looking strong above $4k</div>
        </html>
        """
        items = NitterSentimentAnalyst._parse_nitter_html(html)
        assert len(items) >= 2
        texts = {item["title"] for item in items}
        assert "Bitcoin just broke $70k resistance!" in texts
        assert "ETH looking strong above $4k" in texts

    def test_parse_meta_tags(self):
        html = """
        <html>
        <meta property="og:title" content="BTC reaches new ATH — Crypto Twitter">
        </html>
        """
        items = NitterSentimentAnalyst._parse_nitter_html(html)
        assert any("BTC reaches new ATH" in item["title"] for item in items)

    def test_parse_tweet_body_divs(self):
        html = """
        <html>
        <div class="tweet-body">SOL ecosystem growing fast with new DeFi protocols</div>
        </html>
        """
        items = NitterSentimentAnalyst._parse_nitter_html(html)
        assert any("SOL ecosystem" in item["title"] for item in items)

    def test_parse_empty_html(self):
        items = NitterSentimentAnalyst._parse_nitter_html("")
        assert items == []

    def test_parse_short_content_filtered(self):
        html = '<div class="tweet-content">OK</div>'
        items = NitterSentimentAnalyst._parse_nitter_html(html)
        # "OK" is too short (< 10 chars)
        assert items == []

    def test_html_entities_decoded(self):
        html = '<div class="tweet-content">Bitcoin &amp; Ethereum rally &lt; 5%</div>'
        items = NitterSentimentAnalyst._parse_nitter_html(html)
        assert any("Bitcoin & Ethereum" in item["title"] for item in items)


class TestNitterSentimentComputation:
    """Unit tests for sentiment and term extraction."""

    def test_bullish_sentiment(self):
        tweets = [
            {"title": "BTC bullish breakout imminent, long position ready"},
            {"title": "Accumulation phase confirmed, buy the dip"},
            {"title": "Bull market rally continues, moon soon"},
        ]
        sentiment = NitterSentimentAnalyst._compute_sentiment(tweets)
        assert sentiment == "BULLISH"

    def test_bearish_sentiment(self):
        tweets = [
            {"title": "Market crash incoming, sell everything"},
            {"title": "Bearish divergence on daily — short setup"},
            {"title": "Dump it, correction to $40k likely"},
        ]
        sentiment = NitterSentimentAnalyst._compute_sentiment(tweets)
        assert sentiment == "BEARISH"

    def test_neutral_sentiment(self):
        tweets = [
            {"title": "BTC trading sideways at $60k support"},
            {"title": "Interesting setup developing on daily"},
        ]
        sentiment = NitterSentimentAnalyst._compute_sentiment(tweets)
        assert sentiment == "NEUTRAL"

    def test_no_data_sentiment(self):
        assert NitterSentimentAnalyst._compute_sentiment([]) == "NO_DATA"

    def test_extract_tickers_and_hashtags(self):
        tweets = [
            {"title": "Breaking: $BTC and $ETH leading the rally. #bullish #crypto"},
        ]
        terms = NitterSentimentAnalyst._extract_terms(tweets)
        assert "$btc" in terms
        assert "$eth" in terms
        assert "#bullish" in terms


class TestNitterSentimentFormatting:
    """Unit tests for format_sentiment_section."""

    @pytest.fixture
    def analyst(self):
        from unittest.mock import MagicMock
        logger = MagicMock()
        return NitterSentimentAnalyst(logger=logger, symbols=["BTC", "ETH"])

    def test_format_section_with_tweets(self, analyst):
        data = {
            "tweets": [
                {"title": "BTC breaking out above resistance", "source": "tweet"},
                {"title": "ETH following BTC lead", "source": "tweet"},
            ],
            "instance": "https://nitter.net",
            "overall_sentiment": "BULLISH",
            "top_terms": ["btc", "eth", "breakout"],
        }
        section = analyst.format_sentiment_section(data)
        assert "X/Twitter Sentiment" in section
        assert "Nitter" in section
        assert "BULLISH" in section
        assert "BTC breaking" in section
        assert "ETH following" in section

    def test_format_section_empty(self, analyst):
        data = {"tweets": [], "overall_sentiment": "NO_DATA"}
        assert analyst.format_sentiment_section(data) == ""

    def test_format_section_with_errors(self, analyst):
        data = {
            "tweets": [{"title": "Test tweet", "source": "tweet"}],
            "instance": "https://nitter.net",
            "overall_sentiment": "NEUTRAL",
            "top_terms": [],
            "error": "https://nitter.poast.org: timeout",
        }
        section = analyst.format_sentiment_section(data)
        assert "⚠" in section
        assert "timeout" in section

    def test_symbols_passed_to_analyst(self):
        from unittest.mock import MagicMock
        logger = MagicMock()
        analyst = NitterSentimentAnalyst(logger=logger, symbols=["DOGE", "SHIB"])
        assert analyst.symbols == ["DOGE", "SHIB"]

    def test_default_symbols(self):
        from unittest.mock import MagicMock
        logger = MagicMock()
        analyst = NitterSentimentAnalyst(logger=logger)
        assert analyst.symbols == []  # No hardcoded fallback — symbols come from config
