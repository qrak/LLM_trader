"""SQLite + FTS5 storage for trade post-mortem analyses."""

import sqlite3
import threading
from typing import Any


class PostMortemRepository:
    """Append-only post-mortem storage with FTS5 full-text search.

    Uses the same trade_history.db as SQLiteTradeHistory. Creates its own
    connection (WAL mode + busy_timeout already set by PersistenceManager).
    Existing tables are never modified — only new tables are added.
    """

    def __init__(self, logger: Any, db_path: str) -> None:
        """Initialize the repository and ensure schema exists.

        Args:
            logger: Logger instance.
            db_path: Absolute path to trade_history.db.
        """
        self.logger = logger
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Create a new connection with row factory enabled."""
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """Create tables and indexes if they don't exist (idempotent)."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS trade_post_mortem (
                        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id           INTEGER,
                        symbol             TEXT NOT NULL,
                        direction          TEXT,
                        verdict            TEXT NOT NULL,
                        llm_analysis       TEXT NOT NULL,
                        expected_vs_actual TEXT,
                        lesson_learned     TEXT NOT NULL,
                        pnl_pct            REAL,
                        close_reason       TEXT,
                        created_at         TEXT NOT NULL DEFAULT (datetime('now'))
                    );
                    CREATE INDEX IF NOT EXISTS idx_pm_created_at ON trade_post_mortem(created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_pm_symbol ON trade_post_mortem(symbol);
                    CREATE VIRTUAL TABLE IF NOT EXISTS trade_post_mortem_fts USING fts5(
                        llm_analysis,
                        lesson_learned,
                        post_mortem_id UNINDEXED
                    );
                """)
                conn.commit()
            finally:
                conn.close()

    def insert_post_mortem(
        self,
        trade_id: int | None,
        symbol: str,
        direction: str | None,
        verdict: str,
        llm_analysis: str,
        expected_vs_actual: str | None,
        lesson_learned: str,
        pnl_pct: float | None,
        close_reason: str | None,
    ) -> int:
        """Insert a post-mortem record into both the table and FTS index.

        Args:
            trade_id: trade_history.id of the CLOSE row, or None.
            symbol: Trading symbol (e.g. BTC/USDC).
            direction: LONG or SHORT.
            verdict: Short snake_case tag (e.g. overestimated_breakout).
            llm_analysis: Full LLM analysis text.
            expected_vs_actual: What was expected vs what happened.
            lesson_learned: Concise actionable lesson.
            pnl_pct: P&L percentage of the closed trade.
            close_reason: stop_loss / take_profit / analysis_signal.

        Returns:
            The rowid of the newly inserted post-mortem.
        """
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO trade_post_mortem
                        (trade_id, symbol, direction, verdict, llm_analysis,
                         expected_vs_actual, lesson_learned, pnl_pct, close_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (trade_id, symbol, direction, verdict, llm_analysis,
                     expected_vs_actual, lesson_learned, pnl_pct, close_reason),
                )
                post_mortem_id = cursor.lastrowid
                conn.execute(
                    """
                    INSERT INTO trade_post_mortem_fts
                        (llm_analysis, lesson_learned, post_mortem_id)
                    VALUES (?, ?, ?)
                    """,
                    (llm_analysis, lesson_learned, post_mortem_id),
                )
                conn.commit()
                return post_mortem_id
            finally:
                conn.close()

    def get_recent_post_mortems(self, limit: int = 5) -> list[dict[str, Any]]:
        """Get the most recent post-mortems for brain context injection.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of dicts with keys: id, symbol, direction, verdict,
            lesson_learned, pnl_pct, close_reason, created_at.
        """
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    """
                    SELECT id, symbol, direction, verdict, lesson_learned,
                           pnl_pct, close_reason, created_at
                    FROM trade_post_mortem
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()

    def search_post_mortems(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Full-text search across post-mortem analyses and lessons.

        Uses FTS5 MATCH syntax. The query is passed as-is to FTS5, which
        supports operators like AND, OR, NOT, and prefix matching (term*).

        Args:
            query: FTS5 search query (e.g. "breakout AND resistance").
            limit: Maximum number of results.

        Returns:
            List of dicts with all post-mortem columns, ranked by relevance.
        """
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    """
                    SELECT pm.*, rank
                    FROM trade_post_mortem_fts fts
                    JOIN trade_post_mortem pm ON pm.id = fts.post_mortem_id
                    WHERE trade_post_mortem_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, limit),
                ).fetchall()
                return [dict(row) for row in rows]
            except sqlite3.OperationalError as e:
                self.logger.warning("FTS5 search failed for query '%s': %s", query, e)
                return []
            finally:
                conn.close()

    def get_post_mortem_count(self) -> int:
        """Return total number of stored post-mortems."""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute("SELECT COUNT(*) FROM trade_post_mortem").fetchone()
                return row[0] if row else 0
            finally:
                conn.close()
