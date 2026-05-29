from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from src.trading.vector_memory import VectorMemoryService


class FakeCollection:
    def __init__(self, rows: list[tuple[str, dict]]):
        self._rows = list(rows)

    def count(self):
        return len(self._rows)

    def get(self, include=None):
        return {
            "ids": [row_id for row_id, _ in self._rows],
            "metadatas": [meta for _, meta in self._rows],
        }

    def delete(self, ids):
        remove_ids = set(ids)
        self._rows = [(row_id, meta) for row_id, meta in self._rows if row_id not in remove_ids]


def _make_service() -> VectorMemoryService:
    svc = VectorMemoryService(
        logger=MagicMock(),
        chroma_client=MagicMock(),
        embedding_model=MagicMock(),
        timeframe_minutes=240,
    )
    svc._ensure_initialized = lambda: True  # type: ignore[method-assign]
    return svc


def test_prune_removes_old_experiences_and_blocked_entries():
    svc = _make_service()
    now = datetime.now(timezone.utc)
    old_ts = (now - timedelta(days=200)).isoformat()
    fresh_ts = (now - timedelta(days=5)).isoformat()

    svc._collection = FakeCollection([
        ("exp_old", {"timestamp": old_ts, "outcome": "WIN"}),
        ("exp_fresh", {"timestamp": fresh_ts, "outcome": "LOSS"}),
    ])
    svc._blocked_collection = FakeCollection([
        ("blk_old", {"timestamp": old_ts}),
        ("blk_fresh", {"timestamp": fresh_ts}),
    ])
    svc._semantic_rules_collection = FakeCollection([])

    removed = svc.prune_aged_documents()

    assert removed["trading_experiences"] == 1
    assert removed["system_constraints_rejections"] == 1
    assert svc._collection.count() == 1
    assert svc._blocked_collection.count() == 1


def test_prune_preserves_active_semantic_rules_even_when_old():
    svc = _make_service()
    now = datetime.now(timezone.utc)
    old_ts = (now - timedelta(days=250)).isoformat()

    svc._collection = FakeCollection([])
    svc._blocked_collection = FakeCollection([])
    svc._semantic_rules_collection = FakeCollection([
        ("rule_old_active", {"timestamp": old_ts, "active": True}),
        ("rule_old_inactive", {"timestamp": old_ts, "active": False}),
    ])

    removed = svc.prune_aged_documents()

    assert removed["semantic_rules"] == 1
    remaining = svc._semantic_rules_collection.get()["ids"]
    assert remaining == ["rule_old_active"]


def test_prune_ignores_malformed_timestamps():
    svc = _make_service()
    svc._collection = FakeCollection([
        ("exp_bad", {"timestamp": "not-a-date", "outcome": "WIN"}),
    ])
    svc._blocked_collection = FakeCollection([])
    svc._semantic_rules_collection = FakeCollection([])

    removed = svc.prune_aged_documents()

    assert removed["trading_experiences"] == 0
    assert svc._collection.count() == 1
