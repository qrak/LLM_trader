"""Read-only MCP server over the bot's trade-history SQLite file.

Why read-only and separate from the bot: an agent asking "what did the bot do last week"
must not be able to place, cancel or rewrite anything. The connection is opened with
``mode=ro`` and ``PRAGMA query_only``; there is no tool that writes, and the database path
is the only input.

Run (after ``pip install -r requirements-mcp.txt``)::

    python -m src.mcp_tools.history_server --db data/trading/trade_history.db

The pure query helpers are importable without the MCP dependency, so the test suite does
not need it installed.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

MAX_LIMIT = 200
DEFAULT_DB = Path("data") / "trading" / "trade_history.db"


def _connect_ro(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"trade history database not found: {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _clamp(limit: int, maximum: int = MAX_LIMIT) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return 20
    return max(1, min(value, maximum))


def recent_trades(
    db_path: str | Path, *, symbol: str | None = None, action: str | None = None, limit: int = 20
) -> list[dict[str, Any]]:
    """Most recent rows, optionally filtered by symbol and action."""
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
    with _connect_ro(db_path) as connection:
        rows = connection.execute(query, (*params, _clamp(limit))).fetchall()
    return [dict(row) for row in rows]


def decision_stats(db_path: str | Path, *, symbol: str | None = None, days: int = 30) -> dict[str, Any]:
    """Counts by action and basic coverage — no win-rate claim (that needs pairing)."""
    try:
        window = max(1, int(days))
    except (TypeError, ValueError):
        window = 30
    where = "WHERE timestamp >= datetime('now', ?)"
    params: list[Any] = [f"-{window} days"]
    if symbol:
        where += " AND symbol = ?"
        params.append(symbol)
    with _connect_ro(db_path) as connection:
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


def build_server(db_path: str | Path):
    """Register the read-only tools on a FastMCP server (import kept inside)."""
    from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]

    server = FastMCP("LLM trader history (read-only)")

    @server.tool()
    def recent(symbol: str | None = None, action: str | None = None, limit: int = 20) -> str:
        """Recent trade-history rows: timestamp, symbol, action, prices, size, fee."""
        return json.dumps(recent_trades(db_path, symbol=symbol, action=action, limit=limit), ensure_ascii=False)

    @server.tool()
    def stats(symbol: str | None = None, days: int = 30) -> str:
        """Decision counts by action for the last N days (coverage, not win rate)."""
        return json.dumps(decision_stats(db_path, symbol=symbol, days=days), ensure_ascii=False)

    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only MCP server for LLM_trader trade history")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="path to trade_history.db")
    args = parser.parse_args(argv)
    build_server(args.db).run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
