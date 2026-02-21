import pytest
from unittest.mock import MagicMock
from src.rag.context_builder import ContextBuilder
from src.utils.token_counter import TokenCounter

class TestSimpleContextBuilder:
    @pytest.fixture
    def builder(self):
        logger = MagicMock()
        token_counter = TokenCounter()
        article_processor = MagicMock()
        article_processor.format_article_date.return_value = "Today"
        return ContextBuilder(logger, token_counter, article_processor=article_processor)

    def test_process_article_simple_extracts_lead(self, builder):
        item = {
            'title': 'Bitcoin Soars',
            'source': 'TestNews',
            'published_on': 1704067200,
            'body': 'This is the lead paragraph.\n\nThis is the second paragraph.'
        }
        # Max tokens ample enough
        res = builder._process_article_simple(item, max_tokens=100)
        assert "This is the lead paragraph" in res
        # Should stop after first paragraph (double newline)
        assert "This is the second paragraph" not in res
        assert "Bitcoin Soars" in res

    def test_process_article_truncates_long_lead(self, builder):
        # Create a very long lead paragraph
        long_lead = "Word " * 200
        item = {
            'title': 'Long Article',
            'source': 'TestNews',
            'published_on': 1704067200,
            'body': long_lead + '\n\nSecond para'
        }
        # Max token 50 -> approx 200 chars. 
        res = builder._process_article_simple(item, max_tokens=50)
        
        # Should be truncated
        assert len(res) < len(long_lead) + 50 # +50 for header slack
        assert "..." in res

    def test_build_context_respects_total_limit(self, builder):
        items = []
        for i in range(10):
            items.append({
                'title': f'Article {i}',
                'source': 'Source',
                'published_on': 1704067200,
                'body': f'Content for article {i}\n\nMore content'
            })
            
        # Very tight limit
        # This will test the loop in build_context
        context = builder.build_context(items, max_tokens=100)
        
        # Should likely only contain the first 1 or 2 articles
        assert "Article 0" in context
        # Given 100 tokens, it's unlikely to fit 10 articles
        assert "Article 9" not in context

    def test_article_urls_are_tracked(self, builder):
        items = [
            {
                'title': 'Article 1',
                'source': 'Source',
                'published_on': 1704067200,
                'body': 'Body 1',
                'url': 'https://example.com/1'
            },
            {
                'title': 'Article 2',
                'source': 'Source',
                'published_on': 1704067200,
                'body': 'Body 2',
                'url': 'https://example.com/2'
            }
        ]

        context = builder.build_context(items, max_tokens=500)

        # Verify URLs are tracked
        urls = builder.get_latest_article_urls()
        assert 'Article 1' in urls
        assert urls['Article 1'] == 'https://example.com/1'
        assert 'Article 2' in urls
        assert urls['Article 2'] == 'https://example.com/2'
