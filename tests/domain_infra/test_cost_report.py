

import json

from scripts.cost_report import build_report, main


def write_journal(data_dir, records):

    metrics_dir = data_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record) for record in records]
    (metrics_dir / "cycles.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_build_report_separates_gross_and_net(tmp_path):
    write_journal(
        tmp_path,
        [
            {"inference_cost_usd": 0.10},
            {"inference_cost_usd": 0.0},
            {"inference_cost_usd": None},
            {"inference_cost_usd": float("nan")},
            {"inference_cost_usd": float("inf")},
        ],
    )
    (tmp_path / "statistics.json").write_text(
        json.dumps({"total_pnl_quote": 5.0, "total_trades": 10}),
        encoding="utf-8",
    )

    report = build_report(tmp_path)

    assert report is not None
    assert report["cycles"] == 5
    assert report["analyses_with_cost"] == 2
    assert report["inference_total"] == 0.10
    assert report["gross_pnl"] == 5.0
    assert report["net_pnl"] == 4.9
    assert report["cost_per_analysis"] == 0.05
    assert report["cost_per_closed_trade"] == 0.01
    assert report["coverage"] == 0.4


def test_build_report_without_statistics_still_reports_cost(tmp_path):
    write_journal(tmp_path, [{"inference_cost_usd": 0.25}])

    report = build_report(tmp_path)

    assert report is not None
    assert report["inference_total"] == 0.25
    assert report["gross_pnl"] is None
    assert report["net_pnl"] is None
    assert report["statistics_available"] is False


def test_main_exits_one_without_a_journal(tmp_path, capsys):
    assert main(["--data-dir", str(tmp_path)]) == 1
    assert "No cycle metrics found" in capsys.readouterr().out


def test_main_prints_gross_and_net_with_data(tmp_path, capsys):
    write_journal(tmp_path, [{"inference_cost_usd": 0.25}])
    (tmp_path / "statistics.json").write_text(
        json.dumps({"total_pnl_quote": -1.0, "total_trades": 2}),
        encoding="utf-8",
    )

    assert main(["--data-dir", str(tmp_path)]) == 0

    output = capsys.readouterr().out
    assert "Gross P&L: $-1.00" in output
    assert "Net P&L: $-1.25" in output
    assert "net = gross P&L" in output
