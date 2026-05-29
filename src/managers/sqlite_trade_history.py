"""SQLite-backed trade history store replacing JSON file accumulation.

Designed for months-long 24/7 paper trading: O(1) appends via INSERT,
indexed queries without full-file deserialization.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from src.logger.logger import Logger


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trade_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    confidence TEXT,
    price REAL,
    stop_loss REAL,
    take_profit REAL,
    position_size REAL,
    quote_amount REAL,
    quantity REAL,
    fee REAL,
    reasoning TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_th_timestamp ON trade_history(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_th_symbol ON trade_history(symbol);
CREATE INDEX IF NOT EXISTS idx_th_action ON trade_history(action);
"""

# Key fields that map from serialized TradeDecision data to SQLite columns.
_INSERT_COLS = [
    "timestamp", "symbol", "action", "confidence", "price",
    "stop_loss", "take_profit", "position_size", "quote_amount",
    "quantity", "fee", "reasoning",
]


class SQLiteTradeHistory:
    """Thread-safe SQLite store for trade history."""

    def __init__(self, logger: Logger, db_path: str):
        """Initialize the SQLite store.

        Args:
            logger: Logger instance for error logging.
            db_path: Path to the SQLite database file.
        """
        self._logger = logger
        self._db_path = Path(db_path)
        self._lock = threading.Lock()

        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Run schema creation synchronously (called once during DI wiring).
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers (all assume caller holds self._lock)
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.executescript(SCHEMA_SQL)
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _coerce_col(col: str, value: Any) -> Any:
        """Normalize a JSON value to the SQLite column type."""
        if value is None:
            return None
        if col in ("timestamp", "symbol", "action", "confidence", "reasoning"):
            return str(value)
        if col in (
            "price", "stop_loss", "take_profit", "position_size",
            "quote_amount", "quantity", "fee",
        ):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        return value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, decision_dict: dict[str, Any]) -> int:
        """Insert a trade decision and return its row ID."""
        with self._lock:
            conn = self._get_conn()
            try:
                row = tuple(
                    self._coerce_col(col, decision_dict.get(col))
                    for col in _INSERT_COLS
                )
                placeholders = ", ".join(["?"] * len(_INSERT_COLS))
                cols = ", ".join(_INSERT_COLS)
                sql = f"INSERT INTO trade_history ({cols}) VALUES ({placeholders})"
                cursor = conn.execute(sql, row)
                conn.commit()
                return cursor.lastrowid or 0
            except Exception as e:
                self._logger.error("Failed to insert trade decision: %s", e)
                conn.rollback()
                return 0
            finally:
                conn.close()

    def query(
        self,
        limit: int = 50,
        offset: int = 0,
        symbol: str | None = None,
        action: str | None = None,
        since: str | None = None,
        until: str | None = None,
        order: str = "DESC",
    ) -> list[dict[str, Any]]:
        """Query trade history with optional filters.

        Args:
            limit: Max rows to return.
            offset: Pagination offset.
            symbol: Filter by trading pair (e.g. 'BTC/USDC').
            action: Filter by action type ('BUY', 'SELL', 'CLOSE_LONG', etc.).
            since: ISO timestamp — return trades on or after this time.
            until: ISO timestamp — return trades on or before this time.
            order: Sort direction ('DESC' or 'ASC').

        Returns:
            List of trade dicts with all columns.
        """
        safe_order = str(order).upper()
        if safe_order not in {"ASC", "DESC"}:
            raise ValueError(f"Invalid order: {order}. Expected 'ASC' or 'DESC'.")

        safe_limit = max(1, min(int(limit), 1000))
        safe_offset = max(0, int(offset))

        conditions: list[str] = []
        params: list[Any] = []

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if action:
            conditions.append("action = ?")
            params.append(action)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until:
            conditions.append("timestamp <= ?")
            params.append(until)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        order_clause = f"ORDER BY timestamp {safe_order}"
        sql = f"SELECT * FROM trade_history {where_clause} {order_clause} LIMIT ? OFFSET ?"
        params.extend([safe_limit, safe_offset])

        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            except Exception as e:
                self._logger.error("Query failed: %s", e)
                return []
            finally:
                conn.close()

    def get_last_execution_timestamp(self, actions: tuple[str, ...] = ("BUY", "SELL")) -> str | None:
        """Return the newest timestamp for the provided action set.

        Args:
            actions: Action labels to include in the lookup.

        Returns:
            ISO timestamp string if found, else None.
        """
        if not actions:
            return None

        placeholders = ", ".join(["?"] * len(actions))
        sql = (
            f"SELECT timestamp FROM trade_history "
            f"WHERE action IN ({placeholders}) "
            "ORDER BY timestamp DESC LIMIT 1"
        )

        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(sql, list(actions)).fetchone()
                return row[0] if row else None
            finally:
                conn.close()

    def count(
        self,
        symbol: str | None = None,
        action: str | None = None,
    ) -> int:
        """Count rows, optionally filtered."""
        conditions: list[str] = []
        params: list[Any] = []

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if action:
            conditions.append("action = ?")
            params.append(action)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT COUNT(*) FROM trade_history {where_clause}"

        with self._lock:
            conn = self._get_conn()
            try:
                return conn.execute(sql, params).fetchone()[0]
            finally:
                conn.close()

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate statistics for dashboard display."""
        with self._lock:
            conn = self._get_conn()
            try:
                total = conn.execute(
                    "SELECT COUNT(*) FROM trade_history"
                ).fetchone()[0]

                by_action = {
                    r["action"]: r["cnt"]
                    for r in conn.execute(
                        "SELECT action, COUNT(*) as cnt FROM trade_history GROUP BY action"
                    ).fetchall()
                }

                pnl = conn.execute(
                    "SELECT "
                    "  COALESCE(SUM(CASE WHEN action LIKE 'CLOSE%' THEN "
                    "    CAST(SUBSTR(reasoning, INSTR(reasoning, 'P&L: ') + 5, "
                    "      INSTR(SUBSTR(reasoning, INSTR(reasoning, 'P&L: ') + 5), '%') - 1) "
                    "  AS REAL) ELSE 0 END), 0) as total_pnl "
                    "FROM trade_history"
                ).fetchone()["total_pnl"] or 0.0

                first_ts = conn.execute(
                    "SELECT MIN(timestamp) FROM trade_history"
                ).fetchone()[0]
                last_ts = conn.execute(
                    "SELECT MAX(timestamp) FROM trade_history"
                ).fetchone()[0]

                return {
                    "total_trades": total,
                    "by_action": by_action,
                    "total_pnl_pct": round(pnl, 2),
                    "first_trade": first_ts,
                    "last_trade": last_ts,
                }
            except Exception as e:
                self._logger.error("Stats query failed: %s", e)
                return {"total_trades": 0, "error": str(e)}
            finally:
                conn.close()

    def export_json(self) -> list[dict[str, Any]]:
        """Export full history as a JSON-serializable list."""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM trade_history ORDER BY timestamp ASC"
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()
