from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.news_feedback import load_closed_trades


def out_of_order_ledger(tmp_path: Path) -> Path:
    db_path = tmp_path / "trade_history.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        "create table trade_history (id integer primary key, timestamp text, symbol text, action text)"
    )
    connection.execute(
        "create table trade_post_mortem "
        "(trade_id integer, created_at text, direction text, pnl_pct real, close_reason text)"
    )
    connection.executemany(
        "insert into trade_history values (?, ?, ?, ?)",
        [
            (1, "2026-09-20T12:04:15+00:00", "BTC/USDC", "BUY"),
            (2, "2026-09-20T20:04:45+00:00", "BTC/USDC", "BUY"),
            (3, "2026-09-20T15:29:25+00:00", "BTC/USDC", "CLOSE_LONG"),
            (4, "2026-09-20T14:00:00+00:00", "ETH/USDC", "BUY"),
        ],
    )
    connection.execute(
        "insert into trade_post_mortem values (3, '2026-09-24 06:59:36', 'LONG', -2.06, 'stop loss')"
    )
    connection.commit()
    connection.close()
    return db_path


def test_a_close_takes_the_entry_that_precedes_it_in_time(tmp_path: Path) -> None:
    trades = load_closed_trades(out_of_order_ledger(tmp_path))

    assert [trade["entry_at"] for trade in trades] == ["2026-09-20T12:04:15+00:00"]
