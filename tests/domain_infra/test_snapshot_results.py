from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.snapshot_results import LedgerSnapshot, main


def build_ledger(root: Path) -> Path:
    ledger = root / "trading"
    (ledger / "metrics").mkdir(parents=True)
    (ledger / "statistics.json").write_text(json.dumps({"total_trades": 3}), encoding="utf-8")
    (ledger / "api_costs.json").write_text(json.dumps({"deepseek": {"total_cost": 1.5}}), encoding="utf-8")
    (ledger / "metrics" / "cycles.jsonl").write_text('{"signal": "HOLD"}\n', encoding="utf-8")
    with sqlite3.connect(ledger / "trade_history.db") as connection:
        connection.execute("CREATE TABLE trade_history (id INTEGER PRIMARY KEY, action TEXT)")
        connection.execute("INSERT INTO trade_history (action) VALUES ('HOLD')")
    return ledger


def test_snapshot_writes_a_verifiable_copy(tmp_path: Path) -> None:
    source = build_ledger(tmp_path)
    target = tmp_path / "results" / "2026-11-12"

    manifest = LedgerSnapshot(source, target).write()

    assert set(manifest["files"]) == {
        "statistics.json",
        "api_costs.json",
        "metrics/cycles.jsonl",
        "trade_history.db",
    }
    assert all(entry["bytes"] > 0 for entry in manifest["files"].values())
    assert (target / "manifest.json").exists()
    assert LedgerSnapshot(source, target).verify() == []


def test_snapshot_copies_the_database_rows(tmp_path: Path) -> None:
    source = build_ledger(tmp_path)
    target = tmp_path / "snapshot"

    LedgerSnapshot(source, target).write()

    with sqlite3.connect(target / "trade_history.db") as connection:
        assert connection.execute("SELECT action FROM trade_history").fetchall() == [("HOLD",)]


def test_verify_reports_changed_and_extra_files(tmp_path: Path) -> None:
    source = build_ledger(tmp_path)
    target = tmp_path / "snapshot"
    LedgerSnapshot(source, target).write()
    (target / "statistics.json").write_text(json.dumps({"total_trades": 99}), encoding="utf-8")
    (target / "stray.txt").write_text("x", encoding="utf-8")

    problems = LedgerSnapshot(source, target).verify()

    assert "statistics.json: sha256 differs from the manifest" in problems
    assert "stray.txt: not in the manifest" in problems


def test_snapshot_refuses_a_missing_ledger_file(tmp_path: Path) -> None:
    source = build_ledger(tmp_path)
    (source / "api_costs.json").unlink()

    with pytest.raises(FileNotFoundError, match="api_costs.json"):
        LedgerSnapshot(source, tmp_path / "snapshot").write()


def test_main_verify_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = build_ledger(tmp_path)
    target = tmp_path / "snapshot"
    assert main(["--source", str(source), "--target", str(target)]) == 0
    assert main(["--source", str(source), "--verify", str(target)]) == 0

    (target / "api_costs.json").write_text("{}", encoding="utf-8")

    assert main(["--source", str(source), "--verify", str(target)]) == 1
    assert "1 problem(s)" in capsys.readouterr().out
