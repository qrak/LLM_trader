"""SQLite-backed trade history store replacing JSON file accumulation.

Designed for months-long 24/7 paper trading: O(1) appends via INSERT,
indexed queries without full-file deserialization.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta
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
    indicators_json TEXT,
    execution_quality TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_th_timestamp ON trade_history(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_th_symbol ON trade_history(symbol);
CREATE INDEX IF NOT EXISTS idx_th_action ON trade_history(action);
"""

_INSERT_COLS = [
    "timestamp", "symbol", "action", "confidence", "price",
    "stop_loss", "take_profit", "position_size", "quote_amount",
    "quantity", "fee", "reasoning", "indicators_json", "execution_quality",
]
_INSERT_SQL = f"INSERT INTO trade_history ({', '.join(_INSERT_COLS)}) VALUES ({', '.join(['?'] * len(_INSERT_COLS))})"


class SQLiteTradeHistory:
    """Thread-safe SQLite store for trade history."""

    def __init__(self, logger: Logger, db_path: str):
        """Initialize the SQLite store."""
        self._logger = logger
        self._db_path = Path(db_path)
        self._lock = threading.Lock()

        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_schema()


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
                columns = {r[1] for r in conn.execute("PRAGMA table_info(trade_history)")}
                if "execution_quality" not in columns:
                    conn.execute("ALTER TABLE trade_history ADD COLUMN execution_quality TEXT")
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _coerce_col(col: str, value: Any) -> Any:
        """Normalize a JSON value to the SQLite column type."""
        if value is None:
            return None
        if col == "execution_quality" and isinstance(value, dict):
            return json.dumps(value, allow_nan=False, sort_keys=True)
        if col in {"price", "stop_loss", "take_profit", "position_size", "quote_amount", "quantity", "fee"}:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        return value


    def insert(self, decision_dict: dict[str, Any]) -> int:
        """Insert a trade decision and return its row ID."""
        with self._lock:
            conn = self._get_conn()
            try:
                row = tuple(
                    self._coerce_col(col, decision_dict.get(col))
                    for col in _INSERT_COLS
                )
                cursor = conn.execute(_INSERT_SQL, row)
                conn.commit()
                return cursor.lastrowid or 0
            except Exception as e:  # noqa: BLE001
                self._logger.error("Failed to insert trade decision: %s", e)
                conn.rollback()
                return 0
            finally:
                conn.close()

    def update_entry_fill(
        self, *, entry_time: datetime, symbol: str, actions: frozenset[str],
        price: float, quantity: float, fee: float | None,
        tolerance_seconds: float, row_id: int | None = None,
        execution_quality: dict[str, Any] | None = None,
    ) -> int:
        """Update exactly one correlated entry, preserving its timestamp and reasoning."""
        since = (entry_time - timedelta(seconds=tolerance_seconds)).isoformat()
        until = (entry_time + timedelta(seconds=tolerance_seconds)).isoformat()
        placeholders = ", ".join("?" for _ in actions)
        sql = (
            "SELECT * FROM trade_history WHERE symbol = ? AND timestamp >= ? "
            f"AND timestamp <= ? AND action IN ({placeholders})"
        )
        params: list[Any] = [symbol, since, until, *sorted(actions)]
        if row_id is not None:
            sql += " AND id = ?"
            params.append(row_id)
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("BEGIN IMMEDIATE")
                rows = conn.execute(sql, params).fetchall()
                if len(rows) != 1:
                    raise RuntimeError(f"Entry fill requires exactly one history row, found {len(rows)}")
                row = rows[0]
                quote_amount = price * quantity
                allocation = row["quote_amount"]
                size_pct = row["position_size"]
                if allocation and size_pct:
                    size_pct *= quote_amount / allocation
                conn.execute(
                    "UPDATE trade_history SET price = ?, quantity = ?, fee = ?, "
                    "quote_amount = ?, position_size = ?, execution_quality = ? WHERE id = ?",
                    (price, quantity, fee, quote_amount, size_pct, self._coerce_col("execution_quality", execution_quality), row["id"]),
                )
                conn.commit()
                return int(row["id"])
            finally:
                conn.close()

    @staticmethod
    def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        raw = result.get("execution_quality")
        if raw is not None:
            try:
                result["execution_quality"] = json.loads(raw)
            except (ValueError, TypeError):
                result["execution_quality"] = {"eligible": False, "reason": "invalid execution quality"}
        return result

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
                return [self._decode_row(r) for r in rows]
            except Exception as e:  # noqa: BLE001
                self._logger.error("Query failed: %s", e)
                return []
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

    def export_json(self) -> list[dict[str, Any]]:
        """Export full history as a JSON-serializable list."""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM trade_history ORDER BY timestamp ASC"
                ).fetchall()
                return [self._decode_row(r) for r in rows]
            finally:
                conn.close()

