from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.publish_public_numbers import MARKER_END, MARKER_START, PublicNumbersPublisher

CLAIMS = {
    "ledger": {
        "window_start": "2026-09-20",
        "window_end": "2026-11-10",
        "window_days": 52,
        "pair": "BTC/USDC",
        "timeframe": "4h",
        "capital_usdc": 10000.0,
        "closed_trades": 16,
        "counted_trades": 14,
        "excluded_trades": 2,
        "win_rate_pct": 37.5,
        "return_pct": 0.05,
        "return_usd": 5.27,
        "profit_factor": 1.11,
        "sharpe": 0.11,
        "sortino": 0.21,
        "max_drawdown_pct": -0.25,
        "best_trade_pct": 4.06,
        "worst_trade_pct": -2.6,
        "inference_usd": 7.1156,
        "net_after_inference_usd": -1.84,
        "cycles": 146,
        "cycle_latency_median_s": 275.5,
        "cycle_cost_usd": 0.0114,
        "benchmark_pct": -9.27,
        "snapshot_at": "2026-11-12",
    },
    "tests": {"collected": 2729, "runnable": 2729, "suite_seconds": 119},
}
FACTS = {"tokens_in": 9950171, "tokens_out": 4752047}


def write_snapshot(root: Path) -> Path:
    snapshot = root / "results" / "2026-11-12"
    snapshot.mkdir(parents=True)
    (snapshot / "statistics.json").write_text(
        json.dumps({"winning_trades": 6, "losing_trades": 8}), encoding="utf-8"
    )
    return snapshot


def publisher(tmp_path: Path) -> PublicNumbersPublisher:
    root = tmp_path / "bot"
    (root / "scripts").mkdir(parents=True)
    (tmp_path / "site").mkdir()
    return PublicNumbersPublisher(root, tmp_path / "site", dry_run=False)


def test_the_block_carries_every_number_the_gate_looks_for(tmp_path: Path) -> None:
    pub = publisher(tmp_path)
    snapshot = write_snapshot(pub.root)

    block = pub._block(CLAIMS, FACTS, snapshot)

    for token in ("+0.05%", "$5.27", "-9.27%", "146 cycles", "37.5%", "6W / 8L", "fourteen", "9.3",
                  "10 000 USDC", "2026-11-12"):
        assert token in block, token
    assert block.startswith(MARKER_START) and block.endswith(MARKER_END)


def test_the_block_renders_a_loss_as_a_signed_amount(tmp_path: Path) -> None:
    pub = publisher(tmp_path)
    snapshot = write_snapshot(pub.root)
    claims = json.loads(json.dumps(CLAIMS))
    claims["ledger"]["net_after_inference_usd"] = 3.5

    assert "+$3.50" in pub._block(claims, FACTS, snapshot)


def test_rewriting_the_readme_touches_only_the_marked_block(tmp_path: Path) -> None:
    pub = publisher(tmp_path)
    snapshot = write_snapshot(pub.root)
    readme = pub.root / "README.md"
    readme.write_text(f"before\n{MARKER_START}\nSTALE_SENTINEL\n{MARKER_END}\nafter\n", encoding="utf-8")

    pub._readme(CLAIMS, snapshot, FACTS)

    body = readme.read_text(encoding="utf-8")
    assert body.startswith("before\n") and body.endswith("\nafter\n")
    assert "STALE_SENTINEL" not in body and "+0.05%" in body


def test_a_readme_without_markers_is_refused(tmp_path: Path) -> None:
    pub = publisher(tmp_path)
    snapshot = write_snapshot(pub.root)
    (pub.root / "README.md").write_text("no markers here", encoding="utf-8")

    with pytest.raises(RuntimeError, match="markers"):
        pub._readme(CLAIMS, snapshot, FACTS)


def test_a_second_publication_on_the_same_day_is_skipped(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pub = publisher(tmp_path)
    (pub.root / "results" / pub.today).mkdir(parents=True)
    pub._dirty = lambda: ""

    assert pub.run() == 0
    assert "already holds today's publication" in capsys.readouterr().out


def test_a_dirty_tree_aborts_before_anything_is_written(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pub = publisher(tmp_path)
    pub._dirty = lambda: str(pub.site)

    assert pub.run() == 1
    assert "uncommitted changes" in capsys.readouterr().out
    assert not (pub.root / "results").exists()


def test_dry_run_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pub = publisher(tmp_path)
    pub.dry_run = True

    assert pub.run() == 0
    assert "would snapshot into" in capsys.readouterr().out
    assert not (pub.root / "results").exists()
