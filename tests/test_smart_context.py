"""
Test Smart Sentence Selection for RAG Context Building.

Verifies that ContextBuilder._select_relevant_sentences selects sentences
containing query keywords rather than just taking the first N sentences.
"""
import pytest
from unittest.mock import MagicMock, patch


# Sample article body from the user's example
SAMPLE_ARTICLE_BODY = """
For any serious Bitcoin trader, understanding market sentiment is not just helpful—it's essential.
One of the most powerful, real-time gauges of this sentiment is the BTC perpetual futures long/short ratio.
This metric reveals whether traders are leaning bullish or bearish on the world's leading cryptocurrency at any given moment.
Let's decode the latest data from the world's top exchanges and uncover what it truly means for your next move.

What Does the BTC Perpetual Futures Long/Short Ratio Tell Us?
The BTC perpetual futures long/short ratio is a simple yet profound indicator.
It shows the percentage of traders holding long positions (betting the price will rise) versus those holding short positions (betting the price will fall) on perpetual futures contracts.
Therefore, a ratio above 50% long suggests bullish sentiment, while below 50% indicates bearish leanings.
However, interpreting this data requires looking beyond a single number.

Breaking Down the Latest Market Sentiment
Current data paints a fascinating picture of a market in near-perfect equilibrium, but with subtle nuances.
The overall 24-hour BTC perpetual futures long/short ratio across major platforms shows a market almost perfectly balanced between fear and greed.

Overall: 49.34% long, 50.66% short
Binance: 48.99% long, 51.01% short
OKX: 49.95% long, 50.05% short
Bybit: 48.69% long, 51.31% short

This collective data reveals a market that is cautiously bearish in the very short term.
The slight edge towards short positions suggests traders are hedging or anticipating potential downward pressure.
"""


class TestSmartSentenceSelection:
    """Test suite for smart sentence selection."""

    def test_select_relevant_sentences_finds_numbers(self):
        """Verify that sentences with numbers/percentages are prioritized."""
        # Import the real class
        from src.rag.context_builder import ContextBuilder
        from src.rag.text_splitting import SentenceSplitter
        
        # Create mocks for dependencies
        mock_logger = MagicMock()
        mock_token_counter = MagicMock()
        mock_token_counter.count_tokens.return_value = 10
        mock_article_processor = MagicMock()
        
        # Create real sentence splitter
        sentence_splitter = SentenceSplitter(mock_logger)
        
        # Create ContextBuilder
        builder = ContextBuilder(
            logger=mock_logger,
            token_counter=mock_token_counter,
            article_processor=mock_article_processor,
            sentence_splitter=sentence_splitter
        )
        
        # Split the sample text
        sentences = sentence_splitter.split_text(SAMPLE_ARTICLE_BODY.strip())
        
        # Keywords from a query like "What is the long short ratio?"
        keywords = {'long', 'short', 'ratio'}
        
        # Select top 5 sentences
        selected = builder._select_relevant_sentences(sentences, keywords, max_count=5)
        
        # Join selected sentences for assertion
        selected_text = ' '.join(selected)
        
        # Verify that the key data points are included
        assert '49.34%' in selected_text, "Should contain Overall ratio"
        assert 'long' in selected_text.lower(), "Should contain 'long'"
        assert 'short' in selected_text.lower(), "Should contain 'short'"
    
    def test_select_relevant_sentences_preserves_order(self):
        """Verify that selected sentences maintain their original order."""
        from src.rag.context_builder import ContextBuilder
        from src.rag.text_splitting import SentenceSplitter
        
        mock_logger = MagicMock()
        mock_token_counter = MagicMock()
        mock_token_counter.count_tokens.return_value = 10
        mock_article_processor = MagicMock()
        sentence_splitter = SentenceSplitter(mock_logger)
        
        builder = ContextBuilder(
            logger=mock_logger,
            token_counter=mock_token_counter,
            article_processor=mock_article_processor,
            sentence_splitter=sentence_splitter
        )
        
        sentences = ["First sentence about ratio.", "Second sentence.", "Third sentence about long positions.", "Fourth about ratio."]
        keywords = {'ratio', 'long'}
        
        selected = builder._select_relevant_sentences(sentences, keywords, max_count=3)
        
        # Should have kept order: first, third, fourth (all have keywords)
        assert selected[0] == "First sentence about ratio."
        assert "long" in selected[1].lower() or "ratio" in selected[1].lower()
    
    def test_fallback_without_keywords(self):
        """Verify that without keywords, first N sentences are returned."""
        from src.rag.context_builder import ContextBuilder
        from src.rag.text_splitting import SentenceSplitter
        
        mock_logger = MagicMock()
        mock_token_counter = MagicMock()
        mock_token_counter.count_tokens.return_value = 10
        mock_article_processor = MagicMock()
        sentence_splitter = SentenceSplitter(mock_logger)
        
        builder = ContextBuilder(
            logger=mock_logger,
            token_counter=mock_token_counter,
            article_processor=mock_article_processor,
            sentence_splitter=sentence_splitter
        )
        
        sentences = ["First.", "Second.", "Third.", "Fourth.", "Fifth."]
        
        # Empty keywords should fallback to first N
        selected = builder._select_relevant_sentences(sentences, set(), max_count=3)
        
        assert selected == ["First.", "Second.", "Third."]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
