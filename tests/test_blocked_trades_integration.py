"""Integration tests for ChromaDB blocked trade persistence layer.

Uses a local ChromaDB instance (not mocked) to validate:
  1. store_blocked_trade() → saves to system_constraints_rejections collection
  2. get_recent_blocked_trades() → retrieves with correct metadata
  3. get_blocked_trade_feedback() → builds formatted feedback string
  4. get_blocked_trade_count() → returns correct count
  5. Schema integrity — no None values cause ChromaDB errors
  6. Multiple trades survive the journey without data corruption
  7. Filtering by guard_type and max_age_hours
"""

import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import chromadb
import pytest
from sentence_transformers import SentenceTransformer

from src.trading.vector_memory import VectorMemoryService


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def embedding_model():
    """Lightweight embedding model shared across all tests."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture
def vector_memory(embedding_model):
    """Fresh VectorMemoryService with ephemeral ChromaDB for each test."""
    client = chromadb.Client(chromadb.config.Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=False,
    ))
    client.reset()

    logger = MagicMock()
    svc = VectorMemoryService(
        logger=logger,
        chroma_client=client,
        embedding_model=embedding_model,
        timeframe_minutes=240,
    )
    initialized = svc._ensure_initialized()
    assert initialized, "VectorMemoryService failed to initialize"
    return svc


# ── Helpers ──────────────────────────────────────────────────────


def _store_test_block(
    svc: VectorMemoryService,
    *,
    guard_type: str = "rr_minimum",
    direction: str = "LONG",
    confidence: str = "HIGH",
    suggested_rr: float = 1.2,
    required_rr: float = 2.0,
    suggested_sl_pct: float = 0.03,
    suggested_tp_pct: float = 0.04,
    suggested_sl: float = 97.0,
    suggested_tp: float = 104.0,
    current_price: float = 100.0,
    volatility_level: str = "MEDIUM",
    reasoning_snippet: str = "Test reasoning for LLM feedback",
    metadata: dict | None = None,
) -> bool:
    return svc.store_blocked_trade(
        guard_type=guard_type,
        direction=direction,
        confidence=confidence,
        suggested_rr=suggested_rr,
        required_rr=required_rr,
        suggested_sl_pct=suggested_sl_pct,
        suggested_tp_pct=suggested_tp_pct,
        suggested_sl=suggested_sl,
        suggested_tp=suggested_tp,
        current_price=current_price,
        volatility_level=volatility_level,
        reasoning_snippet=reasoning_snippet,
        metadata=metadata,
    )


# ── Basic Persistence ────────────────────────────────────────────


class TestBlockedTradePersistence:
    """Save a blocked trade and immediately retrieve it."""

    def test_store_and_retrieve_single_blocked_trade(self, vector_memory):
        """Blocked trade survives round-trip through ChromaDB."""
        stored = _store_test_block(vector_memory)
        assert stored is True

        count = vector_memory.get_blocked_trade_count()
        assert count == 1

        results = vector_memory.get_recent_blocked_trades(n=5)
        assert len(results) == 1

    def test_retrieved_block_has_all_fields(self, vector_memory):
        """Retrieved blocked trade contains all expected metadata keys."""
        _store_test_block(vector_memory)

        result = vector_memory.get_recent_blocked_trades(n=1)[0]

        expected_keys = {
            "id", "document", "guard_type", "direction", "confidence",
            "suggested_rr", "required_rr", "rr_delta",
            "suggested_sl_pct", "suggested_tp_pct",
            "suggested_sl", "suggested_tp",
            "current_price", "volatility_level",
            "reasoning_snippet", "timestamp", "event_type",
        }
        missing = expected_keys - set(result.keys())
        assert missing == set(), f"Missing keys: {missing}"

    def test_rr_delta_calculated_correctly(self, vector_memory):
        """rr_delta = suggested_rr - required_rr."""
        _store_test_block(vector_memory, suggested_rr=1.2, required_rr=2.5)

        result = vector_memory.get_recent_blocked_trades(n=1)[0]
        assert result["rr_delta"] == pytest.approx(1.2 - 2.5)

    def test_event_type_is_system_rejection(self, vector_memory):
        """All blocked trades are tagged with event_type='system_rejection'."""
        _store_test_block(vector_memory)

        result = vector_memory.get_recent_blocked_trades(n=1)[0]
        assert result["event_type"] == "system_rejection"


# ── Collection Name Correctness ──────────────────────────────────


class TestBlockedCollectionName:
    """Verify the blocked trades collection uses the correct name."""

    def test_collection_is_system_constraints_rejections(self, vector_memory):
        """The collection name matches the expected contract."""
        assert vector_memory.BLOCKED_TRADES_COLLECTION == "system_constraints_rejections"

    def test_collection_exists_after_store(self, vector_memory):
        """Collection is created/accessible after storing."""
        _store_test_block(vector_memory)
        assert vector_memory._blocked_collection is not None
        assert vector_memory._blocked_collection.count() >= 1


# ── Schema Integrity (No None Values) ────────────────────────────


class TestBlockedTradeSchemaIntegrity:
    """ChromaDB rejects None values — verify _sanitize_metadata works."""

    def test_none_metadata_not_persisted(self, vector_memory):
        """Adding None via metadata should not cause ChromaDB error."""
        stored = _store_test_block(vector_memory, metadata={"extra_field": None})
        assert stored is True

        result = vector_memory.get_recent_blocked_trades(n=1)[0]
        assert "extra_field" not in result  # None should be sanitized out

    def test_mixed_none_and_valid_metadata(self, vector_memory):
        """Valid metadata persists alongside sanitized None values."""
        stored = _store_test_block(vector_memory, metadata={
            "custom_tag": "important",
            "nullable_field": None,
            "score": 42,
        })
        assert stored is True

        result = vector_memory.get_recent_blocked_trades(n=1)[0]
        assert result.get("custom_tag") == "important"
        assert result.get("score") == 42
        assert "nullable_field" not in result


# ── Multiple Trades ──────────────────────────────────────────────


class TestMultipleBlockedTrades:
    """Block multiple trades and verify they all survive."""

    def test_store_multiple_blocked_trades(self, vector_memory):
        """Each blocked trade is stored and retrievable."""
        for i in range(5):
            stored = _store_test_block(
                vector_memory,
                guard_type=f"test_guard_{i}",
                suggested_rr=1.0 + i * 0.1,
            )
            assert stored is True

        assert vector_memory.get_blocked_trade_count() == 5

    def test_retrieval_returns_newest_first(self, vector_memory):
        """get_recent_blocked_trades returns newest trades first."""
        for i in range(3):
            _store_test_block(vector_memory, guard_type=f"guard_{i}")

        results = vector_memory.get_recent_blocked_trades(n=3)
        assert len(results) == 3

        # Verify newest-first ordering by timestamp
        timestamps = [r.get("timestamp", "") for r in results]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_n_parameter_limits_results(self, vector_memory):
        """n parameter caps the number of returned results."""
        for i in range(10):
            _store_test_block(vector_memory, guard_type=f"guard_{i}")

        results = vector_memory.get_recent_blocked_trades(n=3)
        assert len(results) == 3


# ── Filtering by Guard Type ──────────────────────────────────────


class TestBlockedTradeFiltering:
    """Test guard_type and max_age_hours filters."""

    def test_filter_by_guard_type(self, vector_memory):
        """Only matching guard_type trades are returned."""
        _store_test_block(vector_memory, guard_type="rr_minimum")
        _store_test_block(vector_memory, guard_type="sl_distance_max")
        _store_test_block(vector_memory, guard_type="rr_minimum")

        rr_results = vector_memory.get_recent_blocked_trades(n=10, guard_type="rr_minimum")
        assert len(rr_results) == 2
        assert all(r["guard_type"] == "rr_minimum" for r in rr_results)

        sl_results = vector_memory.get_recent_blocked_trades(n=10, guard_type="sl_distance_max")
        assert len(sl_results) == 1
        assert sl_results[0]["guard_type"] == "sl_distance_max"

    def test_filter_by_nonexistent_guard_type_returns_empty(self, vector_memory):
        """Filtering for a guard type with no entries returns empty list."""
        _store_test_block(vector_memory, guard_type="rr_minimum")

        results = vector_memory.get_recent_blocked_trades(n=10, guard_type="nonexistent")
        assert results == []

    def test_max_age_filter_filters_old_entries(self, vector_memory):
        """max_age_hours parameter excludes entries older than threshold."""
        from unittest.mock import patch
        from datetime import datetime as dt, timezone as tz

        # Store a blocked trade normally (it gets current timestamp)
        _store_test_block(vector_memory)

        # With max_age_hours=0, nothing should be recent enough
        results = vector_memory.get_recent_blocked_trades(n=10, max_age_hours=0)
        assert len(results) == 0

        # With large max_age_hours, everything should be included
        results = vector_memory.get_recent_blocked_trades(n=10, max_age_hours=9999)
        assert len(results) == 1


# ── Feedback String Generation ───────────────────────────────────


class TestBlockedTradeFeedback:
    """Verify get_blocked_trade_feedback produces correct format."""

    def test_feedback_contains_critical_feedback_header(self, vector_memory):
        """Feedback starts with 'CRITICAL FEEDBACK' header."""
        _store_test_block(vector_memory)

        feedback = vector_memory.get_blocked_trade_feedback(n=5)
        assert "CRITICAL FEEDBACK" in feedback

    def test_feedback_includes_guard_type_label(self, vector_memory):
        """Feedback includes the human-readable guard label."""
        _store_test_block(vector_memory, guard_type="rr_minimum")

        feedback = vector_memory.get_blocked_trade_feedback(n=5)
        assert "R/R Minimum Guard" in feedback

    def test_feedback_includes_pre_flight_checklist(self, vector_memory):
        """Feedback ends with mandatory pre-flight checklist."""
        _store_test_block(vector_memory)

        feedback = vector_memory.get_blocked_trade_feedback(n=5)
        assert "PRE-FLIGHT CHECKLIST" in feedback
        assert "R/R >= required minimum" in feedback

    def test_feedback_empty_when_no_blocks(self, vector_memory):
        """Empty string when there are no blocked trades."""
        feedback = vector_memory.get_blocked_trade_feedback(n=5)
        assert feedback == ""

    def test_feedback_groups_by_guard_type(self, vector_memory):
        """Multiple block types are grouped under separate headers."""
        _store_test_block(vector_memory, guard_type="rr_minimum")
        _store_test_block(vector_memory, guard_type="sl_distance_max")

        feedback = vector_memory.get_blocked_trade_feedback(n=10)
        assert "R/R Minimum Guard" in feedback
        assert "SL Too Far (max 10%)" in feedback

    def test_feedback_includes_rr_delta(self, vector_memory):
        """Feedback shows the gap between suggested and required R/R."""
        _store_test_block(vector_memory, suggested_rr=1.2, required_rr=2.5)

        feedback = vector_memory.get_blocked_trade_feedback(n=5)
        assert "1.20" in feedback or "1.2" in feedback  # suggested_rr
        assert "2.50" in feedback or "2.5" in feedback  # required_rr
        assert "gap:" in feedback

    def test_feedback_includes_ai_reasoning(self, vector_memory):
        """Feedback includes the AI reasoning snippet."""
        _store_test_block(vector_memory, reasoning_snippet="Expecting breakout above resistance")

        feedback = vector_memory.get_blocked_trade_feedback(n=5)
        assert "Expecting breakout" in feedback


# ── Edge: Empty State ────────────────────────────────────────────


class TestBlockedTradeEmptyState:
    """Handle empty collection gracefully."""

    def test_count_zero_when_empty(self, vector_memory):
        """get_blocked_trade_count returns 0 for empty collection."""
        assert vector_memory.get_blocked_trade_count() == 0

    def test_recent_returns_empty_when_no_data(self, vector_memory):
        """get_recent_blocked_trades returns [] when empty."""
        results = vector_memory.get_recent_blocked_trades(n=5)
        assert results == []

    def test_feedback_returns_empty_when_no_data(self, vector_memory):
        """get_blocked_trade_feedback returns '' when empty."""
        assert vector_memory.get_blocked_trade_feedback() == ""


# ── Unique IDs ───────────────────────────────────────────────────


class TestBlockedTradeUniqueIds:
    """Each blocked trade gets a unique, timestamp-based ID."""

    def test_ids_are_unique(self, vector_memory):
        """No two blocked trades share the same ID."""
        _store_test_block(vector_memory)
        _store_test_block(vector_memory)
        _store_test_block(vector_memory)

        results = vector_memory.get_recent_blocked_trades(n=10)
        ids = [r["id"] for r in results]
        assert len(ids) == len(set(ids))

    def test_id_starts_with_blocked_prefix(self, vector_memory):
        """Blocked trade IDs start with 'blocked_'."""
        _store_test_block(vector_memory)

        result = vector_memory.get_recent_blocked_trades(n=1)[0]
        assert result["id"].startswith("blocked_")
