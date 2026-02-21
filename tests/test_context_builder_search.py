
import pytest
import re
from unittest.mock import MagicMock
from src.rag.context_builder import ContextBuilder
from src.utils.token_counter import TokenCounter

@pytest.mark.asyncio
async def test_keyword_search_score_calculation():
    # Mock dependencies
    logger = MagicMock()
    token_counter = TokenCounter()
    article_processor = MagicMock()
    article_processor.extract_base_coin.return_value = "Bitcoin"
    article_processor.get_article_timestamp.return_value = 0 # old article

    builder = ContextBuilder(logger, token_counter, article_processor=article_processor)

    # Test case 1: Coin in title (word boundary)
    article1 = {
        'id': '1',
        'title': 'Bitcoin price soars',
        'body': 'Some body text',
        'categories': 'market',
        'tags': '',
        'detected_coins_str': '',
        'published_on': 0
    }

    # Test case 2: Coin in body
    article2 = {
        'id': '2',
        'title': 'Crypto news',
        'body': 'Bitcoin is going up',
        'categories': 'market',
        'tags': '',
        'detected_coins_str': '',
        'published_on': 0
    }

    # Test case 3: Coin not present
    article3 = {
        'id': '3',
        'title': 'Ethereum news',
        'body': 'ETH is good',
        'categories': 'market',
        'tags': '',
        'detected_coins_str': '',
        'published_on': 0
    }

    # Test case 4: Coin in title but part of another word (should NOT match with \b)
    article4 = {
        'id': '4',
        'title': 'Bitcoinds is not a real thing',
        'body': 'Body',
        'categories': '',
        'tags': '',
        'detected_coins_str': '',
        'published_on': 0
    }

    news_database = [article1, article2, article3, article4]

    # Run search
    scores = await builder.keyword_search(
        query="market",
        news_database=news_database,
        symbol="BTC/USD"
    )

    # Scores are tuples (index, score)
    # Filter only relevant articles (score > 0)
    score_map = {idx: score for idx, score in scores}

    # Article 1: "Bitcoin price soars". Matches "bitcoin" (title) -> +15. +5 (keyword). Total base=20.
    # Article 2: "Bitcoin is going up". Matches "bitcoin" (body) -> +5. +5 (keyword). Total base=10.
    # Article 3: "ETH is good". No coin match. +5 (keyword). Total base=5.

    # Check that article 1 > article 2
    assert score_map.get(0, 0) > score_map.get(1, 0)

    # Check that article 2 > article 3
    assert score_map.get(1, 0) > score_map.get(2, 0)

    # Article 4 check: "Bitcoinds" should not match "Bitcoin" pattern \bbitcoin\b
    # Keyword "market" is not in article 4.
    # So score should be 0 (and filtered out)
    assert 3 not in score_map

@pytest.mark.asyncio
async def test_keyword_search_special_patterns():
    # Mock dependencies
    logger = MagicMock()
    token_counter = TokenCounter()
    article_processor = MagicMock()
    article_processor.extract_base_coin.return_value = "BTC"
    article_processor.get_article_timestamp.return_value = 0

    builder = ContextBuilder(logger, token_counter, article_processor=article_processor)

    # Case: "BTC Price" in title -> Special boost +20
    article_special = {
        'id': 's',
        'title': 'BTC Price Analysis',
        'body': '...',
        'categories': '',
        'tags': '',
        'detected_coins_str': '',
        'published_on': 0
    }

    scores = await builder.keyword_search(
        query="analysis",
        news_database=[article_special],
        symbol="BTC/USD"
    )

    assert len(scores) > 0
    score = scores[0][1]
    # Should be at least 20 (special) + 15 (title match) = 35 * recency factor
    assert score > 5


# --- Smart Heuristic Scoring Tests ---

def _make_mock_config():
    """Create a mock config with smart scoring defaults."""
    config = MagicMock()
    config.RAG_DENSITY_PENALTY_THRESHOLD = 300
    config.RAG_DENSITY_BOOST_THRESHOLD = 1000
    config.RAG_DENSITY_PENALTY_MULTIPLIER = 0.5
    config.RAG_DENSITY_BOOST_MULTIPLIER = 1.2
    config.RAG_COOCCURRENCE_MULTIPLIER = 1.5
    return config


@pytest.mark.asyncio
async def test_frequency_scoring_prefers_more_mentions():
    """Article with many keyword mentions should score higher than one with few."""
    logger = MagicMock()
    token_counter = TokenCounter()
    article_processor = MagicMock()
    article_processor.extract_base_coin.return_value = "BTC"
    article_processor.get_article_timestamp.return_value = 0

    config = _make_mock_config()
    builder = ContextBuilder(logger, token_counter, config=config, article_processor=article_processor)

    # Article A: "bitcoin" mentioned 10 times in body
    article_many = {
        'id': 'a',
        'title': 'crypto news',
        'body': ' '.join(['bitcoin'] * 10 + ['analysis'] * 5),
        'categories': '',
        'tags': '',
        'detected_coins_str': '',
        'published_on': 0
    }

    # Article B: "bitcoin" mentioned once in body
    article_few = {
        'id': 'b',
        'title': 'crypto news',
        'body': 'bitcoin analysis today',
        'categories': '',
        'tags': '',
        'detected_coins_str': '',
        'published_on': 0
    }

    scores = await builder.keyword_search(
        query="bitcoin",
        news_database=[article_many, article_few]
    )
    score_map = {idx: score for idx, score in scores}

    # Article with 10 mentions should score higher than article with 1
    assert score_map.get(0, 0) > score_map.get(1, 0)


@pytest.mark.asyncio
async def test_density_penalty_demotes_short_articles():
    """Short stub articles should score lower than substantial articles."""
    logger = MagicMock()
    token_counter = TokenCounter()
    article_processor = MagicMock()
    article_processor.extract_base_coin.return_value = "BTC"
    article_processor.get_article_timestamp.return_value = 0

    config = _make_mock_config()
    builder = ContextBuilder(logger, token_counter, config=config, article_processor=article_processor)

    # Short stub (< 300 chars) -> should get 0.5x penalty
    article_short = {
        'id': 'short',
        'title': 'bitcoin alert',
        'body': 'bitcoin price moved.',
        'categories': '',
        'tags': '',
        'detected_coins_str': '',
        'published_on': 0
    }

    # Long article (> 1000 chars) -> should get 1.2x boost
    article_long = {
        'id': 'long',
        'title': 'bitcoin alert',
        'body': 'bitcoin price moved. ' + ('Analysis of market conditions. ' * 50),
        'categories': '',
        'tags': '',
        'detected_coins_str': '',
        'published_on': 0
    }

    scores = await builder.keyword_search(
        query="bitcoin",
        news_database=[article_short, article_long]
    )
    score_map = {idx: score for idx, score in scores}

    # Long article should score higher despite same keyword in title
    assert score_map.get(1, 0) > score_map.get(0, 0)


@pytest.mark.asyncio
async def test_cooccurrence_boost_for_multi_keyword_queries():
    """Articles containing ALL query keywords should score higher."""
    logger = MagicMock()
    token_counter = TokenCounter()
    article_processor = MagicMock()
    article_processor.extract_base_coin.return_value = "ETH"
    article_processor.get_article_timestamp.return_value = 0

    config = _make_mock_config()
    builder = ContextBuilder(logger, token_counter, config=config, article_processor=article_processor)

    # Article A: contains both "ethereum" and "merge"
    article_both = {
        'id': 'both',
        'title': 'ethereum merge update',
        'body': 'The ethereum merge is proceeding as planned with validators ready.',
        'categories': '',
        'tags': '',
        'detected_coins_str': '',
        'published_on': 0
    }

    # Article B: contains only "ethereum", no "merge"
    article_partial = {
        'id': 'partial',
        'title': 'ethereum price update',
        'body': 'The ethereum network saw increased activity today with high volume.',
        'categories': '',
        'tags': '',
        'detected_coins_str': '',
        'published_on': 0
    }

    scores = await builder.keyword_search(
        query="ethereum merge",
        news_database=[article_both, article_partial]
    )
    score_map = {idx: score for idx, score in scores}

    # Article with both keywords should score higher (1.5x co-occurrence boost)
    assert score_map.get(0, 0) > score_map.get(1, 0)


@pytest.mark.asyncio
async def test_single_keyword_no_cooccurrence_bonus():
    """Single-keyword queries should not get co-occurrence multiplier."""
    logger = MagicMock()
    token_counter = TokenCounter()
    article_processor = MagicMock()
    article_processor.extract_base_coin.return_value = "BTC"
    article_processor.get_article_timestamp.return_value = 0

    config = _make_mock_config()
    builder = ContextBuilder(logger, token_counter, config=config, article_processor=article_processor)

    article = {
        'id': '1',
        'title': 'bitcoin news',
        'body': 'bitcoin price is stable today.',
        'categories': '',
        'tags': '',
        'detected_coins_str': '',
        'published_on': 0
    }

    # Single keyword - co-occurrence modifier should return 1.0
    from collections import namedtuple
    content = builder._extract_article_content(article)
    modifier = builder._calculate_cooccurrence_modifier({'bitcoin'}, content)
    assert modifier == 1.0
