#!/usr/bin/env python3
"""Query the LLM_trader trade history SQLite database from the terminal.

Usage:
  python scripts/query_trade_history.py <command> [options]

Commands:
  recent      Show the 20 most recent trades.
  stats       Show aggregate statistics (count, actions, date range).
  search      Search trades by symbol, action, or date range.

Options:
  --symbol SYM     Filter by trading pair (e.g. BTC/USDC).
  --action ACT     Filter by action (e.g. BUY, SELL, CLOSE_LONG).
  --since TS       ISO timestamp — trades on or after this time.
  --until TS       ISO timestamp — trades on or before this time.
  --limit N        Max rows to return (default: 20).
  --offset N       Pagination offset (default: 0).
  --order DESC|ASC Sort direction (default: DESC).

Examples:
  # Show last 50 trades
  python scripts/query_trade_history.py recent --limit 50

  # All BUY trades for BTC/USDC
  python scripts/query_trade_history.py search --symbol BTC/USDC --action BUY

  # Trades in a specific month
  python scripts/query_trade_history.py search --since 2026-01-01 --until 2026-01-31

  # Quick stats
  python scripts/query_trade_history.py stats

  # Open a live SQLite shell
  sqlite3 data/trading/trade_history.db
"""

import argparse
import sys
from pathlib import Path

# Resolve the database path relative to the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "trading" / "trade_history.db"


def _check_db() -> None:
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        print("Run the bot at least once to create the SQLite trade history database.")
        sys.exit(1)


def _format_trade(row: dict) -> str:
    """Format a single trade row for terminal display."""
    ts = row.get("timestamp", "?")[:19]
    sym = row.get("symbol", "?")
    act = row.get("action", "?")
    price = row.get("price")
    size = row.get("position_size")
    fee = row.get("fee")
    conf = row.get("confidence", "?")
    reason = (row.get("reasoning") or "")[:120]

    line = f"[{ts}] {act:12s} {sym:10s}"
    if price:
        line += f" @ ${price:,.2f}"
    if size:
        line += f"  size={size*100:.1f}%"
    if fee:
        line += f"  fee=${fee:.4f}"
    line += f"  conf={conf}"
    if reason:
        line += f"\n    → {reason}"
    return line


def cmd_recent(args: argparse.Namespace) -> None:
    """Show the most recent trades."""
    import sqlite3
    _check_db()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    limit = args.limit or 20
    order = args.order or "DESC"

    sql = f"SELECT * FROM trade_history ORDER BY timestamp {order} LIMIT ?"
    rows = conn.execute(sql, [limit]).fetchall()
    conn.close()

    if not rows:
        print("No trades found.")
        return

    print(f"\n=== Last {len(rows)} trades ===\n")
    for row in rows:
        print(_format_trade(dict(row)))
    print()


def cmd_search(args: argparse.Namespace) -> None:
    """Search trades with filters."""
    import sqlite3
    _check_db()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    conditions = []
    params = []

    if args.symbol:
        conditions.append("symbol = ?")
        params.append(args.symbol)
    if args.action:
        conditions.append("action = ?")
        params.append(args.action)
    if args.since:
        conditions.append("timestamp >= ?")
        params.append(args.since)
    if args.until:
        conditions.append("timestamp <= ?")
        params.append(args.until)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    order = args.order or "DESC"
    limit = args.limit or 20
    offset = args.offset or 0

    sql = f"SELECT * FROM trade_history {where} ORDER BY timestamp {order} LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(sql, params).fetchall()
    total = conn.execute(
        f"SELECT COUNT(*) FROM trade_history {where}",
        params[:-2] if conditions else []
    ).fetchone()[0]
    conn.close()

    if not rows:
        print("No trades match the filter criteria.")
        return

    print(f"\n=== {len(rows)} of {total} matching trades ===\n")
    for row in rows:
        print(_format_trade(dict(row)))
    print()


def cmd_stats(_args: argparse.Namespace) -> None:
    """Show aggregate statistics."""
    import sqlite3
    _check_db()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    total = conn.execute("SELECT COUNT(*) as cnt FROM trade_history").fetchone()["cnt"]
    first = conn.execute("SELECT MIN(timestamp) as ts FROM trade_history").fetchone()["ts"]
    last = conn.execute("SELECT MAX(timestamp) as ts FROM trade_history").fetchone()["ts"]

    by_action = conn.execute(
        "SELECT action, COUNT(*) as cnt FROM trade_history GROUP BY action ORDER BY cnt DESC"
    ).fetchall()

    by_symbol = conn.execute(
        "SELECT symbol, COUNT(*) as cnt FROM trade_history GROUP BY symbol ORDER BY cnt DESC"
    ).fetchall()

    entry_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM trade_history WHERE action IN ('BUY', 'SELL')"
    ).fetchone()["cnt"]
    close_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM trade_history WHERE action LIKE 'CLOSE%'"
    ).fetchone()["cnt"]

    conn.close()

    print("\n=== Trade History Statistics ===\n")
    print(f"  Total records:   {total}")
    print(f"  Entry trades:    {entry_count} (BUY/SELL)")
    print(f"  Close events:    {close_count}")
    print(f"  Date range:      {first[:10] if first else 'N/A'} → {last[:10] if last else 'N/A'}")
    print()

    print("  By action:")
    for row in by_action:
        print(f"    {row['action']:20s}  {row['cnt']:>6d}")
    print()

    print("  By symbol:")
    for row in by_symbol:
        print(f"    {row['symbol']:10s}  {row['cnt']:>6d}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the LLM_trader trade history SQLite database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # recent
    p_recent = sub.add_parser("recent", help="Show recent trades")
    p_recent.add_argument("--limit", type=int)
    p_recent.add_argument("--order", choices=["DESC", "ASC"])

    # search
    p_search = sub.add_parser("search", help="Search trades")
    p_search.add_argument("--symbol")
    p_search.add_argument("--action")
    p_search.add_argument("--since")
    p_search.add_argument("--until")
    p_search.add_argument("--limit", type=int)
    p_search.add_argument("--offset", type=int)
    p_search.add_argument("--order", choices=["DESC", "ASC"])

    # stats
    sub.add_parser("stats", help="Show aggregate statistics")

    args = parser.parse_args()

    if args.command == "recent":
        cmd_recent(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
