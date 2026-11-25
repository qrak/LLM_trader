from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from scripts.ledger_facts import ledger_facts


def frozen_ledger(tmp_path: Path) -> Path:
    data = tmp_path / "results" / "2026-11-12"
    data.mkdir(parents=True)
    (data / "statistics.json").write_text(json.dumps({"winning_trades": 6, "losing_trades": 8}), encoding="utf-8")
    connection = sqlite3.connect(data / "trade_history.db")
    connection.execute("create table trade_history (timestamp text)")
    connection.execute("insert into trade_history values ('2026-09-20T00:00:00+00:00')")
    connection.execute("insert into trade_history values ('2026-11-10T00:00:00+00:00')")
    connection.execute(
        "create table trade_post_mortem (created_at text, direction text, pnl_pct real, close_reason text)"
    )
    connection.commit()
    connection.close()
    return data


def test_a_frozen_snapshot_keeps_its_own_date(tmp_path: Path) -> None:
    facts = ledger_facts(frozen_ledger(tmp_path), with_benchmark=False, snapshot_at="2026-11-12")

    assert facts["snapshot_at"] == "2026-11-12"
    assert facts["window_start"] == "2026-09-20"
    assert facts["window_end"] == "2026-11-10"


def test_a_ledger_measured_now_is_stamped_today(tmp_path: Path) -> None:
    facts = ledger_facts(frozen_ledger(tmp_path), with_benchmark=False)

    assert facts["snapshot_at"] == datetime.now(timezone.utc).date().isoformat()


def ledger_with_dropped_pairs(tmp_path: Path) -> Path:
    data = tmp_path / "results" / "2026-11-12"
    data.mkdir(parents=True)
    (data / "statistics.json").write_text(
        json.dumps({"total_trades": 1, "winning_trades": 1, "losing_trades": 0}), encoding="utf-8"
    )
    connection = sqlite3.connect(data / "trade_history.db")
    connection.execute(
        "create table trade_history (id integer primary key, timestamp text, action text, price real,"
        " quantity real, quote_amount real, reasoning text, execution_quality text)"
    )
    connection.executemany(
        "insert into trade_history (timestamp, action, price, quantity, quote_amount, reasoning,"
        " execution_quality) values (?, ?, ?, ?, ?, ?, ?)",
        [
            ("2026-09-20T00:00:00+00:00", "BUY", 100.0, 1.0, 100.0, "entry", None),
            ("2026-09-20T04:00:00+00:00", "CLOSE", 110.0, 1.0, 100.0, "Position closed: take_profit.", None),
            ("2026-09-21T00:00:00+00:00", "BUY", 100.0, 1.0, 100.0, "entry", None),
            ("2026-09-21T04:00:00+00:00", "CLOSE", 105.0, 1.0, 100.0, "EXIT FILL DIVERGENCE: diverged.",
             '{"eligible": false}'),
            ("2026-09-22T00:00:00+00:00", "BUY", 100.0, 1.0, 100.0, "entry", None),
            ("2026-09-22T04:00:00+00:00", "CLOSE", 100.0, 1.0, 100.0, "Position closed: blocked.",
             '{"eligible": false}'),
        ],
    )
    connection.execute(
        "create table trade_post_mortem (created_at text, direction text, pnl_pct real, close_reason text)"
    )
    connection.commit()
    connection.close()
    return data


def test_excluded_trades_counts_the_pairs_the_statistics_dropped(tmp_path: Path) -> None:
    facts = ledger_facts(ledger_with_dropped_pairs(tmp_path), with_benchmark=False, snapshot_at="2026-11-12")

    assert facts["closed_trades"] == 1
    assert facts["counted_trades"] == 1
    assert facts["excluded_trades"] == 2
