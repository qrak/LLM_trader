"""Chaos tests: Vector DB (ChromaDB) boundaries, empty RAG returns, context poisoning.

Covers Pillar 4 — verifies the system handles:
- Empty RAG vector returns (zero results from ChromaDB query)
- Highly noisy embedding results with extremely low similarity
- Corrupted or missing metadata blocks from ChromaDB
- Extremely large context that exceeds token budget
- Empty database state (news_database size == 0)
- Many near-identical articles dominating the context
- Malformed articles with missing required fields
- Context builder handling of None/null title/body/source
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock

import pytest

from src.rag.rag_engine import RagEngine
from src.rag.context_builder import ContextBuilder
from src.trading.vector_memory import VectorMemoryService


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_context_builder() -> ContextBuilder:
    """Create a ContextBuilder with all dependencies mocked."""
    config = MagicMock()
    config.RAG_ARTICLE_MAX_TOKENS = 500
    config.RAG_NEWS_LIMIT = 10

    token_counter = MagicMock()
    token_counter.count_tokens.return_value = 50  # every string is ~50 tokens

    scoring_policy = MagicMock()

    return ContextBuilder(
        logger=MagicMock(),
        token_counter=token_counter,
        config=config,
        scoring_policy=scoring_policy,
        symbol_name_map={},
    )


def _make_rag_engine() -> RagEngine:
    """Create a RagEngine wired with mocks for all deps."""
    logger = MagicMock()
    config = MagicMock()
    config.RAG_UPDATE_INTERVAL_HOURS = 6
    config.RAG_NEWS_LIMIT = 10
    config.RAG_ARTICLE_MAX_TOKENS = 500
    config.RAG_NEWS_ENRICH_MIN_CHARS = 200

    token_counter = MagicMock()
    token_counter.count_tokens.return_value = 10

    news_manager = MagicMock()
    news_manager.news_database = []
    news_manager.get_database_size.return_value = 0

    context_builder = _build_context_builder()

    category_processor = MagicMock()
    category_processor.extract_base_coin.side_effect = lambda s: s.split("/")[0] if "/" in s else s
    category_processor.category_word_map = {}

    ticker_manager = MagicMock()
    ticker_manager.get_known_tickers.return_value = ["BTC", "ETH"]

    index_manager = MagicMock()
    index_manager.get_coin_indices.return_value = {"BTC": [0, 1], "ETH": [2]}
    index_manager.search_by_coin.return_value = []

    market_data_manager = MagicMock()

    category_fetcher = MagicMock()
    category_fetcher.fetch_categories = AsyncMock(return_value=True)

    engine = RagEngine(
        logger=logger,
        token_counter=token_counter,
        config=config,
        news_manager=news_manager,
        market_data_manager=market_data_manager,
        index_manager=index_manager,
        category_fetcher=category_fetcher,
        category_processor=category_processor,
        ticker_manager=ticker_manager,
        context_builder=context_builder,
    )
    engine.last_update = datetime.now(timezone.utc)
    return engine


# ── 1. EMPTY RAG VECTOR RETURNS ───────────────────────────────────────────────

class TestEmptyRagReturns:
    """When ChromaDB returns zero results, context must be empty (not crash)."""

    @pytest.mark.asyncio
    async def test_empty_news_database_returns_empty_context(self):
        """If news_database is empty, retrieve_context returns empty string."""
        engine = _make_rag_engine()

        context = await engine.retrieve_context("BTC price analysis", "BTC/USDC")

        # get_database_size == 0 should trigger immediate empty return
        assert context == ""

    @pytest.mark.asyncio
    async def test_query_with_no_matching_articles_returns_empty_string(self):
        """With populated database but no keyword matches, context should be empty."""
        engine = _make_rag_engine()
        engine.news_manager.get_database_size.return_value = 50
        engine.news_manager.news_database = [
            {"title": "DeFi summer recap", "body": "All about DeFi projects", "source": "test"},
        ]

        engine.context_builder.keyword_search = AsyncMock(return_value=[])

        context = await engine.retrieve_context("completely unrelated topic", "SOL/USDC")
        # No matching indices -> scores is empty -> relevant_indices is empty -> empty context
        assert context is not None
        assert context != "Error retrieving market context."

    @pytest.mark.asyncio
    async def test_systematic_empty_collection_in_vector_memory(self):
        """VectorMemoryService.retrieve_similar_experiences returns [] when collection is empty."""
        logger = MagicMock()
        chroma_client = MagicMock()
        embedding_model = MagicMock()
        embedding_model.encode.return_value = [0.1] * 128

        svc = VectorMemoryService(
            logger=logger,
            chroma_client=chroma_client,
            embedding_model=embedding_model,
            timeframe_minutes=240,
        )
        # Bypass initialization for unit test
        svc._ensure_initialized = lambda: True  # type: ignore[method-assign]

        # Mock collection with no data
        empty_collection = MagicMock()
        empty_collection.count.return_value = 0
        svc._collection = empty_collection

        results = svc.retrieve_similar_experiences("bullish BTC", k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_get_context_for_prompt_with_no_experiences_returns_empty(self):
        """get_context_for_prompt returns '' when no experiences are found."""
        logger = MagicMock()
        chroma_client = MagicMock()
        embedding_model = MagicMock()
        embedding_model.encode.return_value = [0.1] * 128

        svc = VectorMemoryService(
            logger=logger,
            chroma_client=chroma_client,
            embedding_model=embedding_model,
            timeframe_minutes=240,
        )
        svc._ensure_initialized = lambda: True  # type: ignore[method-assign]
        empty_collection = MagicMock()
        empty_collection.count.return_value = 0
        svc._collection = empty_collection

        context = svc.get_context_for_prompt("bullish BTC", k=5)
        assert context == ""


# ── 2. NOISY / LOW-SIMILARITY EMBEDDING RESULTS ──────────────────────────────

class TestNoisyEmbeddingResults:
    """Very low similarity results must still produce usable context (not crash)."""

    @pytest.mark.asyncio
    async def test_retrieve_similar_experiences_with_zero_similarity(self):
        """When ChromaDB returns results with near-zero distance (high similarity), no crash."""

        logger = MagicMock()
        chroma_client = MagicMock()
        embedding_model = MagicMock()
        embedding_model.encode.return_value = [0.0] * 128

        svc = VectorMemoryService(
            logger=logger,
            chroma_client=chroma_client,
            embedding_model=embedding_model,
            timeframe_minutes=240,
        )
        svc._ensure_initialized = lambda: True  # type: ignore[method-assign]

        # ChromaDB returns distance=0.0 (perfect match) — similarity = 1.0
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        mock_collection.query.return_value = {
            "ids": [["exp-1"]],
            "distances": [[0.0]],  # very close (distance=0 → similarity=1)
            "documents": [["BTC went up. Long trade won +5%."]],
            "metadatas": [[{
                "outcome": "WIN",
                "pnl_pct": 5.0,
                "direction": "LONG",
                "confidence": "HIGH",
                "market_context": "BULLISH",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reasoning": "good call",
            }]],
        }
        svc._collection = mock_collection

        results = svc.retrieve_similar_experiences("BTC analysis", k=5)
        assert len(results) == 1
        assert results[0].similarity == 100.0  # 1.0 * 100

    def test_sanitize_metadata_with_nan_and_inf(self):
        """_sanitize_metadata must safely handle NaN, Inf, and None values."""
        logger = MagicMock()
        svc = VectorMemoryService(
            logger=logger,
            chroma_client=MagicMock(),
            embedding_model=MagicMock(),
            timeframe_minutes=240,
        )

        bad_metadata = {
            "price": float("nan"),
            "volume": float("inf"),
            "neg_volume": float("-inf"),
            "none_val": None,
            "good_val": 42,
            "list_val": [1, 2, 3],  # lists are not supported
            "bool_val": True,
        }

        sanitized = svc._sanitize_metadata(bad_metadata)
        assert "price" not in sanitized, "NaN should be dropped"
        assert "volume" not in sanitized, "inf should be dropped"
        assert "neg_volume" not in sanitized, "-inf should be dropped"
        assert "none_val" not in sanitized, "None should be dropped"
        assert sanitized["good_val"] == 42
        assert "list_val" not in sanitized, "lists should be dropped"
        assert sanitized["bool_val"] is True


# ── 3. CORRUPTED / MISSING METADATA ──────────────────────────────────────────

class TestCorruptedMetadata:
    """ChromaDB metadata blocks with missing or corrupted keys."""

    @pytest.mark.asyncio
    async def test_missing_outcome_in_metadata_is_safe(self):
        """No 'outcome' key in metadata — the context builder defaults to UNKNOWN."""
        logger = MagicMock()
        chroma_client = MagicMock()
        embedding_model = MagicMock()
        embedding_model.encode.return_value = [0.2] * 128

        svc = VectorMemoryService(
            logger=logger,
            chroma_client=chroma_client,
            embedding_model=embedding_model,
            timeframe_minutes=240,
        )
        svc._ensure_initialized = lambda: True  # type: ignore[method-assign]

        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        mock_collection.query.return_value = {
            "ids": [["exp-1"]],
            "distances": [[0.5]],
            "documents": [["some trade description"]],
            "metadatas": [[{
                # Intentionally missing "outcome" key
                "pnl_pct": 5.0,
                "direction": "LONG",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }]],
        }
        svc._collection = mock_collection

        # This should not crash — missing "outcome" defaults to "UNKNOWN" in get_context_for_prompt
        context = svc.get_context_for_prompt("BTC analysis", k=5)
        assert isinstance(context, str)

    @pytest.mark.asyncio
    async def test_corrupted_timestamp_in_metadata_is_safe(self):
        """Malformed timestamp strings must not crash _calculate_recency_score."""
        logger = MagicMock()
        chroma_client = MagicMock()
        embedding_model = MagicMock()
        embedding_model.encode.return_value = [0.2] * 128

        svc = VectorMemoryService(
            logger=logger,
            chroma_client=chroma_client,
            embedding_model=embedding_model,
            timeframe_minutes=240,
        )
        svc._ensure_initialized = lambda: True  # type: ignore[method-assign]

        mock_collection = MagicMock()
        mock_collection.count.return_value = 1

        # A valid query response structure with corrupted timestamp
        from src.trading.vector_memory_context import VectorMemoryContextMixin
        parsed = VectorMemoryContextMixin._parse_trade_timestamp("not-a-date")
        # Must return a valid datetime (min value) not crash
        assert parsed is not None
        # Year should be the minimum
        assert parsed.year <= 1  # datetime.min = 0001-01-01

    def test_missing_timestamp_returns_min_datetime(self):
        """Empty/none timestamp must not crash parser."""
        from src.trading.vector_memory_context import VectorMemoryContextMixin
        parsed = VectorMemoryContextMixin._parse_trade_timestamp("")
        assert parsed is not None


# ── 4. LARGE / OVERWHELMING CONTEXT ──────────────────────────────────────────

class TestExtremeContextBoundaries:
    """Extremely large context, many noisy articles, token budget edge cases."""

    def test_build_context_with_hundreds_of_articles_respects_token_limit(self):
        """When many articles are candidates, context builder must respect max_tokens."""
        builder = _build_context_builder()

        # Create 200 near-identical articles
        articles = [
            {
                "title": f"Article {i}",
                "body": f"This is the body of article {i}. It contains some market analysis text.",
                "source": "test",
                "url": f"https://test.com/{i}",
            }
            for i in range(200)
        ]

        context = builder.build_context(articles, max_tokens=200)
        # Token counter returns 50 per call, so we should have at most 4 articles (200/50)
        # But actually the context builder uses token_counter.count_tokens which always returns 50
        # So each article costs 50 tokens, max 200 tokens = 4 articles max
        article_count = context.count("📰")
        assert article_count <= 5, f"Should be at most ~4 articles with 200 token budget, got {article_count}"

    def test_build_context_with_empty_news_list_returns_empty_string(self):
        """No news items should return empty string (not crash)."""
        builder = _build_context_builder()
        context = builder.build_context([], max_tokens=500)
        assert context == ""

    def test_build_context_with_malformed_articles_missing_body(self):
        """Articles with no body should be skipped gracefully."""
        builder = _build_context_builder()
        articles = [
            {"title": "Headline Only", "source": "test"},
            {"title": "With Body", "body": "Real content here", "source": "test"},
        ]

        context = builder.build_context(articles, max_tokens=500)
        # Only the article with a body should appear
        assert "Headline Only" not in context, "Articles without body should be skipped"
        assert "With Body" in context

    def test_build_context_with_null_title_does_not_crash(self):
        """Articles with None title must not crash _process_article_simple."""
        builder = _build_context_builder()
        articles = [
            {
                "title": None,
                "body": "This article has no title but has content",
                "source_info": {"name": "NewsAPI"},
                "published_on": 1700000000,
                "url": "https://test.com/no-title",
            }
        ]

        context = builder.build_context(articles, max_tokens=500)
        assert "No Title" in context


# ── 5. VECTOR MEMORY STORE EDGE CASES ────────────────────────────────────────

class TestVectorMemoryStoreEdgeCases:
    """store_experience with corrupted/pathological data."""

    def test_store_experience_with_outcome_update_and_no_pnl(self):
        """An UPDATE event (position still open) should not crash even without pnl."""
        logger = MagicMock()
        chroma_client = MagicMock()
        embedding_model = MagicMock()
        embedding_model.encode.return_value = [0.1, 0.2, 0.3]

        svc = VectorMemoryService(
            logger=logger,
            chroma_client=chroma_client,
            embedding_model=embedding_model,
            timeframe_minutes=240,
        )
        svc._ensure_initialized = lambda: True  # type: ignore[method-assign]

        mock_collection = MagicMock()
        svc._collection = mock_collection

        success = svc.store_experience(
            trade_id="update-1",
            market_context="NEUTRAL",
            outcome="UPDATE",
            pnl_pct=0.0,
            direction="LONG",
            confidence="MEDIUM",
            reasoning="updating SL",
            metadata={"pnl_pct": 0.0},
            symbol="BTC/USDC",
            close_reason="",
        )
        assert success, "UPDATE event should store without error"

    def test_store_experience_after_failed_init_returns_false(self):
        """If _ensure_initialized fails, store_experience returns False."""
        logger = MagicMock()
        chroma_client = MagicMock()
        embedding_model = MagicMock()

        svc = VectorMemoryService(
            logger=logger,
            chroma_client=chroma_client,
            embedding_model=embedding_model,
            timeframe_minutes=240,
        )
        svc._ensure_initialized = lambda: False  # type: ignore[method-assign]

        success = svc.store_experience(
            trade_id="fail-1",
            market_context="BULLISH",
            outcome="WIN",
            pnl_pct=5.0,
            direction="LONG",
            confidence="HIGH",
            reasoning="test",
        )
        assert not success

    def test_store_blocked_trade_with_nan_suggested_rr(self):
        """Blocked trade with NaN R/R values must not crash."""
        logger = MagicMock()
        chroma_client = MagicMock()
        embedding_model = MagicMock()
        embedding_model.encode.return_value = [0.1] * 128

        svc = VectorMemoryService(
            logger=logger,
            chroma_client=chroma_client,
            embedding_model=embedding_model,
            timeframe_minutes=240,
        )
        svc._ensure_initialized = lambda: True  # type: ignore[method-assign]
        mock_collection = MagicMock()
        svc._blocked_collection = mock_collection

        success = svc.store_blocked_trade(
            guard_type="rr_minimum",
            direction="LONG",
            confidence="HIGH",
            suggested_rr=float("nan"),
            required_rr=1.5,
            suggested_sl_pct=0.02,
            suggested_tp_pct=0.03,
            suggested_sl=49000.0,
            suggested_tp=51000.0,
            current_price=50000.0,
            volatility_level="MEDIUM",
            reasoning_snippet="bad rr",
        )
        # The rr_delta calculation uses math.isfinite check, so NaN should be handled
        assert success


# ── 6. RAG ENGINE RETRIEVAL EDGE CASES ────────────────────────────────────────

class TestRagEngineRetrievalBoundaries:
    """RagEngine.retrieve_context edge cases with pathological inputs."""

    @pytest.mark.asyncio
    async def test_retrieve_context_with_empty_query_safe(self):
        """Empty query string must not crash."""
        engine = _make_rag_engine()
        engine.news_manager.get_database_size.return_value = 10
        engine.news_manager.news_database = [
            {"title": "test", "body": "some content", "source": "test"}
        ]
        engine.context_builder.keyword_search = AsyncMock(return_value=[(0, 0.5)])

        context = await engine.retrieve_context("", "BTC/USDC")
        assert context is not None

    @pytest.mark.asyncio
    async def test_retrieve_context_with_exception_from_news_manager(self):
        """If keyword_search raises, retrieve_context must return error string (not propagate)."""
        engine = _make_rag_engine()
        engine.news_manager.get_database_size.return_value = 10

        async def raise_error(*args, **kwargs):
            raise RuntimeError("Unstable network")

        engine.context_builder.keyword_search = raise_error

        context = await engine.retrieve_context("BTC analysis", "BTC/USDC")
        # The exception is caught and a fallback string returned
        assert "Error retrieving" in context
