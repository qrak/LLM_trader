

from datetime import datetime, timedelta, timezone

from src.managers.cycle_metrics import append_cycle, load_cycles, percentile, summarize

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)


def test_append_and_load_roundtrip(tmp_path):
    path = tmp_path / "metrics" / "cycles.jsonl"

    written = append_cycle(path, {"signal": "HOLD", "latency_ms": 12.5}, now=NOW)
    loaded = load_cycles(path)

    assert written["ts"] == NOW.isoformat(timespec="seconds")
    assert loaded == [written]
    assert loaded[0]["signal"] == "HOLD"


def test_missing_file_is_empty_and_malformed_lines_are_skipped(tmp_path):
    path = tmp_path / "cycles.jsonl"
    assert load_cycles(path) == []

    append_cycle(path, {"signal": "HOLD"}, now=NOW)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{torn line\n")

    assert len(load_cycles(path)) == 1


def test_percentile_is_nearest_rank():
    assert percentile([10, 20, 30, 40], 50) == 20
    assert percentile([10, 20, 30, 40], 95) == 40
    assert percentile([5], 50) == 5
    assert percentile([], 50) is None
    assert percentile([None, "x", 3.0], 50) == 3.0


def test_summarize_counts_and_window(tmp_path):
    records = [
        {"ts": (NOW - timedelta(hours=1)).isoformat(), "signal": "HOLD", "action": "HOLD",
         "latency_ms": 1000.0, "guarded": False, "stale": False, "error": None},
        {"ts": (NOW - timedelta(hours=2)).isoformat(), "signal": "BUY", "action": None,
         "latency_ms": 3000.0, "guarded": True, "stale": False, "error": None},
        {"ts": (NOW - timedelta(hours=3)).isoformat(), "signal": "SELL", "action": None,
         "latency_ms": None, "guarded": False, "stale": True, "error": "provider timeout"},
        {"ts": (NOW - timedelta(days=3)).isoformat(), "signal": "HOLD", "action": "HOLD",
         "latency_ms": 9999.0, "guarded": False, "stale": False, "error": None},
    ]

    summary = summarize(records, days=1, now=NOW)

    assert summary["cycles"] == 3
    assert summary["errors"] == 1
    assert summary["stale"] == 1
    assert summary["guarded"] == 1
    assert summary["signals"] == {"HOLD": 1, "BUY": 1, "SELL": 1}
    assert summary["hold_share"] == 0.3333
    assert summary["latency_ms"] == {"p50": 1000.0, "p95": 3000.0}


def test_summarize_reports_inference_cost_and_coverage():
    records = [
        {"ts": (NOW - timedelta(hours=1)).isoformat(), "inference_cost_usd": 0.25},
        {"ts": (NOW - timedelta(hours=2)).isoformat(), "inference_cost_usd": 0.0},
        {"ts": (NOW - timedelta(hours=3)).isoformat(), "inference_cost_usd": None},
        {"ts": (NOW - timedelta(hours=4)).isoformat(), "inference_cost_usd": "0.5"},
        {"ts": (NOW - timedelta(hours=5)).isoformat(), "inference_cost_usd": float("nan")},
        {"ts": (NOW - timedelta(hours=6)).isoformat(), "inference_cost_usd": float("inf")},
        {"ts": (NOW - timedelta(hours=7)).isoformat(), "inference_cost_usd": -1.0},
        {"ts": (NOW - timedelta(hours=8)).isoformat(), "inference_cost_usd": True},
    ]

    summary = summarize(records, days=1, now=NOW)

    assert summary["inference_cost"] == {
        "analyses": 2,
        "total_usd": 0.25,
        "per_analysis_usd": 0.125,
        "coverage": 0.25,
    }


def test_summarize_without_cycles_is_explicit():
    summary = summarize([], days=1, now=NOW)

    assert summary["cycles"] == 0
    assert summary["hold_share"] is None
    assert summary["latency_ms"] == {"p50": None, "p95": None}
    assert summary["inference_cost"] == {
        "analyses": 0,
        "total_usd": 0.0,
        "per_analysis_usd": None,
        "coverage": None,
    }

async def test_dashboard_endpoint_summarizes_the_journal(tmp_path):

    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.dashboard.routers.performance import PerformanceRouter

    router = PerformanceRouter(config=MagicMock(), logger=MagicMock(),
                               dashboard_state=MagicMock(),
                               persistence=SimpleNamespace(data_dir=tmp_path))
    append_cycle(
        tmp_path / "metrics" / "cycles.jsonl",
        {"signal": "HOLD", "latency_ms": 10.0, "inference_cost_usd": 0.125},
        now=datetime.now(timezone.utc),
    )

    summary = await router.get_cycle_metrics(days=1)

    assert summary["cycles"] == 1
    assert summary["signals"] == {"HOLD": 1}
    assert summary["inference_cost"] == {
        "analyses": 1,
        "total_usd": 0.125,
        "per_analysis_usd": 0.125,
        "coverage": 1.0,
    }

    empty = PerformanceRouter(config=MagicMock(), logger=MagicMock(),
                              dashboard_state=MagicMock(), persistence=None)
    empty_summary = await empty.get_cycle_metrics(days=1)
    assert empty_summary["cycles"] == 0
    assert empty_summary["inference_cost"]["analyses"] == 0
