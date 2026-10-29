

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

MAX_LIMIT = 200
DEFAULT_DB = Path("data") / "trading" / "trade_history.db"


class TradeHistoryReader:
    """Read-only queries over one trade-history database.

    The connection is opened per query with ``mode=ro`` and ``PRAGMA query_only``, so a
    write attempt fails at the SQLite level instead of trusting the caller.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def recent_trades(
        self, *, symbol: str | None = None, action: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:

        clauses, params = [], []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if action:
            clauses.append("UPPER(action) = ?")
            params.append(str(action).upper())
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        query = (
            "SELECT timestamp, symbol, action, confidence, price, stop_loss, take_profit, "
            "position_size, quote_amount, quantity, fee "
            f"FROM trade_history {where}ORDER BY timestamp DESC, id DESC LIMIT ?"
        )
        with self._connect_ro() as connection:
            rows = connection.execute(query, (*params, self._clamp(limit))).fetchall()
        return [dict(row) for row in rows]

    def decision_stats(self, *, symbol: str | None = None, days: int = 30) -> dict[str, Any]:

        try:
            window = max(1, int(days))
        except (TypeError, ValueError):
            window = 30
        where = "WHERE timestamp >= datetime('now', ?)"
        params: list[Any] = [f"-{window} days"]
        if symbol:
            where += " AND symbol = ?"
            params.append(symbol)
        with self._connect_ro() as connection:
            counts = connection.execute(
                f"SELECT action, COUNT(*) AS n FROM trade_history {where} GROUP BY action", params
            ).fetchall()
            coverage = connection.execute(
                f"SELECT COUNT(*) AS n, MIN(timestamp) AS first, MAX(timestamp) AS last "
                f"FROM trade_history {where}",
                params,
            ).fetchone()
        return {
            "symbol": symbol,
            "days": window,
            "rows": coverage["n"] if coverage else 0,
            "first": coverage["first"] if coverage else None,
            "last": coverage["last"] if coverage else None,
            "by_action": {str(row["action"]).upper(): row["n"] for row in counts},
        }

    def _connect_ro(self) -> sqlite3.Connection:
        path = self.db_path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"trade history database not found: {path}")
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection

    @staticmethod
    def _clamp(limit: int, maximum: int = MAX_LIMIT) -> int:
        try:
            value = int(limit)
        except (TypeError, ValueError):
            return 20
        return max(1, min(value, maximum))


def build_server(db_path: str | Path):

    from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]

    server = FastMCP("LLM trader history (read-only)")
    reader = TradeHistoryReader(db_path)

    @server.tool(description="Recent trade-history rows: timestamp, symbol, action, prices, size, fee.")
    def recent(symbol: str | None = None, action: str | None = None, limit: int = 20) -> str:
        return json.dumps(reader.recent_trades(symbol=symbol, action=action, limit=limit), ensure_ascii=False)

    @server.tool(description="Decision counts by action for the last N days (coverage, not win rate).")
    def stats(symbol: str | None = None, days: int = 30) -> str:
        return json.dumps(reader.decision_stats(symbol=symbol, days=days), ensure_ascii=False)

    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only MCP server for LLM_trader trade history")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="path to trade_history.db")
    args = parser.parse_args(argv)
    build_server(args.db).run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
