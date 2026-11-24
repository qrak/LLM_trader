from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.managers.cycle_metrics import load_cycles

ENTRY_ACTIONS = ("BUY", "SELL")


def load_closed_trades(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "select p.trade_id, p.created_at, p.direction, p.pnl_pct, p.close_reason "
            "from trade_post_mortem p order by p.created_at"
        ).fetchall()
        trades = []
        for trade_id, closed_at, direction, pnl_pct, close_reason in rows:
            entry = con.execute(
                "select timestamp from trade_history where id <= ? and action in (?, ?) "
                "order by id desc limit 1",
                (trade_id, *ENTRY_ACTIONS),
            ).fetchone()
            trades.append(
                {
                    "closed_at": closed_at,
                    "entry_at": entry[0] if entry else None,
                    "direction": direction,
                    "pnl_pct": pnl_pct,
                    "reason": (close_reason or "")[:60],
                }
            )
    finally:
        con.close()
    return trades


def parse_stamp(stamp: str | None) -> datetime | None:
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(str(stamp).strip().replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def cycle_before(cycles: list[dict], moment: datetime) -> dict | None:
    best: dict | None = None
    best_stamp: datetime | None = None
    for cycle in cycles:
        stamp = parse_stamp(cycle.get("ts"))
        if stamp is None or stamp > moment:
            continue
        if best_stamp is None or stamp > best_stamp:
            best, best_stamp = cycle, stamp
    return best


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show the news the model saw at each closed trade's entry."
    )
    parser.add_argument("--data-dir", default="data/trading", type=Path)
    parser.add_argument("--limit", default=10, type=int)
    args = parser.parse_args()

    cycles = [c for c in load_cycles(args.data_dir / "metrics" / "cycles.jsonl") if c.get("news")]
    trades = load_closed_trades(args.data_dir / "trade_history.db")
    if not cycles:
        print("No cycle record carries news yet; the next cycle writes one.")
        return 0
    if not trades:
        print("No closed trade in the ledger.")
        return 0

    print(f"## News at entry ({len(cycles)} cycles with news, {len(trades)} closed trades)\n")
    for trade in trades[-args.limit:]:
        entry = parse_stamp(trade["entry_at"])
        cycle = cycle_before(cycles, entry) if entry else None
        pnl = "-" if trade["pnl_pct"] is None else f"{trade['pnl_pct']:.2f}%"
        print(f"### {trade['entry_at'] or '?'} {trade['direction']} → {pnl} ({trade['reason']})")
        if cycle is None:
            print("- no cycle record before this entry\n")
            continue
        news = cycle["news"]
        print(f"- query: `{news.get('query')}` | signal: {cycle.get('signal')} | action: {cycle.get('action')}")
        for article in news.get("articles") or []:
            print(f"- {article.get('title')}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
