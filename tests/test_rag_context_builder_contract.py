from types import SimpleNamespace
from unittest.mock import MagicMock

from src.rag.context_builder import ContextBuilder
from src.utils.token_counter import TokenCounter


class _TokenCounter:
    def count_tokens(self, text: str) -> int:
        return len(text.split())


def _builder() -> ContextBuilder:
    config = SimpleNamespace(
        RAG_ARTICLE_MAX_TOKENS=300,
        RAG_NEWS_LIMIT=2,
        RAG_NEWS_ENRICH_MIN_CHARS=20,
    )
    article_processor = MagicMock()
    article_processor.extract_base_coin.return_value = "BTC"
    article_processor.get_article_timestamp.return_value = 0.0

    return ContextBuilder(
        logger=MagicMock(),
        token_counter=_TokenCounter(),
        config=config,
        scoring_policy=MagicMock(),
        article_processor=article_processor,
    )


def test_build_context_uses_config_derived_default_budget():
    builder = _builder()
    builder.config.RAG_ARTICLE_MAX_TOKENS = 10
    builder.config.RAG_NEWS_LIMIT = 1
    called_limits: list[int] = []

    def _stub_process(_item: dict, max_tokens: int) -> str:
        called_limits.append(max_tokens)
        return "word " * 8

    builder._process_article_simple = _stub_process

    context_text = builder.build_context([
        {"title": "A0", "url": "https://example.com/0"},
        {"title": "A1", "url": "https://example.com/1"},
    ])

    assert context_text.strip() == ("word " * 8).strip()
    assert called_limits == [10, 10]


def test_add_articles_to_context_prefers_full_body_even_if_score_is_lower():
    builder = _builder()
    news_database = [
        {
            "title": "Long body lower score",
            "body": "x " * 40,
            "source": "unit",
            "published_on": 1710000000,
        },
        {
            "title": "Short body higher score",
            "body": "tiny",
            "source": "unit",
            "published_on": 1710000000,
        },
    ]

    context_text, total_tokens = builder.add_articles_to_context(
        relevant_indices=[1, 0],
        news_database=news_database,
        max_tokens=1000,
        k=2,
        _keywords={"btc"},
        scores_dict={1: 10.0, 0: 5.0},
    )

    assert "Long body lower score" in context_text
    assert "Short body higher score" in context_text
    assert context_text.index("Long body lower score") < context_text.index("Short body higher score")
    assert total_tokens > 0


def test_add_articles_to_context_limits_article_count_to_k():
    builder = _builder()
    news_database = [
        {
            "title": "A0",
            "body": "body " * 20,
            "source": "unit",
            "published_on": 1710000000,
        },
        {
            "title": "A1",
            "body": "body " * 20,
            "source": "unit",
            "published_on": 1710000000,
        },
        {
            "title": "A2",
            "body": "body " * 20,
            "source": "unit",
            "published_on": 1710000000,
        },
    ]

    context_text, _ = builder.add_articles_to_context(
        relevant_indices=[0, 1, 2],
        news_database=news_database,
        max_tokens=1000,
        k=2,
        _keywords={"btc"},
        scores_dict={0: 5.0, 1: 4.0, 2: 3.0},
    )

    assert "A0" in context_text
    assert "A1" in context_text
    assert "A2" not in context_text


def test_process_article_respects_token_budget_with_tiktoken_counter():
    max_article_tokens = 80
    token_counter = TokenCounter()
    config = SimpleNamespace(
        RAG_ARTICLE_MAX_TOKENS=max_article_tokens,
        RAG_NEWS_ENRICH_MIN_CHARS=20,
    )
    article_processor = MagicMock()
    article_processor.extract_base_coin.return_value = "BTC"
    article_processor.get_article_timestamp.return_value = 0.0

    builder = ContextBuilder(
        logger=MagicMock(),
        token_counter=token_counter,
        config=config,
        scoring_policy=MagicMock(),
        article_processor=article_processor,
    )

    item = {
        "title": "Token Budget Check",
        "body": " ".join(["bitcoin liquidity rotation volatility"] * 220),
        "source_info": {"name": "coindesk"},
        "published_on": 1710000000,
    }

    processed = builder._process_article_simple(item, max_article_tokens)
    assert processed
    assert token_counter.count_tokens(processed) <= max_article_tokens
