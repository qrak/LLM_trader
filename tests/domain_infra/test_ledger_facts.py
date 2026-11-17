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
