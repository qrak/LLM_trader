"""Tests for PostMortemRepository SQLite + FTS5 operations."""

from unittest.mock import MagicMock

from src.managers.post_mortem_repository import PostMortemRepository


class TestPostMortemRepository:
    """Tests for PostMortemRepository SQLite + FTS5 operations."""

    @staticmethod
    def _make_repo(tmp_path, logger=None):
        """Factory: create repo with temp DB."""
        db_path = str(tmp_path / "test_trade_history.db")
        return PostMortemRepository(logger=logger or MagicMock(), db_path=db_path)

    @staticmethod
    def _make_data(**overrides):
        """Factory: create post-mortem insert data."""
        defaults = {
            "trade_id": 1,
            "symbol": "BTC/USDC",
            "direction": "LONG",
            "verdict": "overestimated_breakout",
            "llm_analysis": "Price rejected resistance twice before dropping. Entry was premature.",
            "expected_vs_actual": "Expected breakout above 72.5k, actual rejection and -3% drop.",
            "lesson_learned": "When price rejects a level twice, wait for confirmation before entering.",
            "pnl_pct": -3.2,
            "close_reason": "stop_loss",
        }
        defaults.update(overrides)
        return defaults

    def test_insert_creates_record_in_both_tables(self, tmp_path):
        """Insert should populate both trade_post_mortem and FTS table."""
        repo = self._make_repo(tmp_path)
        data = self._make_data()
        pm_id = repo.insert_post_mortem(**data)
        assert pm_id == 1
        assert repo.get_post_mortem_count() == 1

    def test_get_recent_post_mortems_returns_newest_first(self, tmp_path):
        """Recent post-mortems should be ordered by created_at DESC."""
        import sqlite3
        db_path = str(tmp_path / "test_trade_history.db")
        repo = PostMortemRepository(logger=MagicMock(), db_path=db_path)
        repo.insert_post_mortem(**self._make_data(trade_id=1, verdict="first", pnl_pct=-1.0))
        repo.insert_post_mortem(**self._make_data(trade_id=2, verdict="second", pnl_pct=-2.0))
        # Override created_at with explicit distinct timestamps to avoid
        # relying on datetime('now') 1-second resolution.
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("UPDATE trade_post_mortem SET created_at = '2026-06-17 12:00:00' WHERE verdict = 'first'")
            conn.execute("UPDATE trade_post_mortem SET created_at = '2026-06-18 12:00:00' WHERE verdict = 'second'")
            conn.commit()
        finally:
            conn.close()
        recent = repo.get_recent_post_mortems(limit=5)
        assert len(recent) == 2
        assert recent[0]["verdict"] == "second"

    def test_get_recent_post_mortems_empty_db(self, tmp_path):
        """Empty DB should return empty list."""
        repo = self._make_repo(tmp_path)
        assert repo.get_recent_post_mortems() == []

    def test_search_post_mortems_fts_match(self, tmp_path):
        """FTS5 search should find records by keyword."""
        repo = self._make_repo(tmp_path)
        repo.insert_post_mortem(**self._make_data(
            llm_analysis="Price rejected resistance during a failed breakout. Premature entry.",
        ))
        results = repo.search_post_mortems("breakout")
        assert len(results) == 1
        assert results[0]["verdict"] == "overestimated_breakout"

    def test_search_post_mortems_no_match(self, tmp_path):
        """FTS5 search with no matching query returns empty list."""
        repo = self._make_repo(tmp_path)
        repo.insert_post_mortem(**self._make_data(
            llm_analysis="Price rejected resistance twice. Entry was premature.",
        ))
        results = repo.search_post_mortems("nonexistent_term_xyz")
        assert results == []

    def test_search_post_mortems_invalid_query_returns_empty(self, tmp_path):
        """Malformed FTS query should return empty list, not raise."""
        repo = self._make_repo(tmp_path)
        repo.insert_post_mortem(**self._make_data(
            llm_analysis="Price rejected resistance twice. Entry was premature.",
        ))
        results = repo.search_post_mortems("invalid^^^query!!!")
        assert results == []

    def test_schema_creation_is_idempotent(self, tmp_path):
        """Initializing twice should not error."""
        logger = MagicMock()
        self._make_repo(tmp_path, logger=logger)
        self._make_repo(tmp_path, logger=logger)
        # No assertion — just verifying no exception raised

    def test_get_post_mortem_count(self, tmp_path):
        """Count should reflect inserted records."""
        repo = self._make_repo(tmp_path)
        assert repo.get_post_mortem_count() == 0
        repo.insert_post_mortem(**self._make_data())
        assert repo.get_post_mortem_count() == 1
        repo.insert_post_mortem(**self._make_data(trade_id=2))
        assert repo.get_post_mortem_count() == 2

    def test_get_recent_returns_correct_fields(self, tmp_path):
        """get_recent_post_mortems should return expected field subset."""
        repo = self._make_repo(tmp_path)
        repo.insert_post_mortem(**self._make_data())
        recent = repo.get_recent_post_mortems()
        assert len(recent) == 1
        pm = recent[0]
        assert "id" in pm
        assert pm["symbol"] == "BTC/USDC"
        assert pm["direction"] == "LONG"
        assert pm["verdict"] == "overestimated_breakout"
        assert pm["lesson_learned"]
        assert pm["pnl_pct"] == -3.2
        assert pm["close_reason"] == "stop_loss"
        assert "created_at" in pm

    def test_search_returns_full_record(self, tmp_path):
        """search_post_mortems should return all columns including llm_analysis."""
        repo = self._make_repo(tmp_path)
        repo.insert_post_mortem(**self._make_data(
            llm_analysis="Failed breakout above resistance. Price dropped afterward.",
        ))
        results = repo.search_post_mortems("breakout")
        assert len(results) == 1
        pm = results[0]
        assert "llm_analysis" in pm
        assert "expected_vs_actual" in pm
        assert "rank" in pm
