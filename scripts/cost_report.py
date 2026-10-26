"""Report gross and net P&L after recorded LLM inference cost.

Usage:
    python scripts/cost_report.py [--data-dir data/trading]

The report reads metrics/cycles.jsonl and statistics.json directly; it does not
import src, so it runs from a checkout without installing the package.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_DATA_DIR = Path("data/trading")


def _finite_non_negative(value: Any) -> float | None:
    """Finite, non-negative number or None; bools and strings are not costs."""
    if isinstance(value, (bool, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _finite_number(value: Any) -> float | None:
    """Finite number or None; a negative P&L is valid, strings are not."""
    if isinstance(value, (bool, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_cycles(path: Path) -> list[dict[str, Any]]:
    """Read the cycle journal, skipping blank and malformed lines."""
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
    except OSError:
        return []
    return records


def load_statistics(path: Path) -> dict[str, Any]:
    """Read statistics.json; missing or malformed data is reported as unavailable."""
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def build_report(data_dir: Path) -> dict[str, Any] | None:
    """Build the cost and P&L report, or None when the cycle journal has no rows."""
    records = load_cycles(data_dir / "metrics" / "cycles.jsonl")
    if not records:
        return None

    cost_values = [
        value
        for record in records
        if (value := _finite_non_negative(record.get("inference_cost_usd"))) is not None
    ]
    statistics = load_statistics(data_dir / "statistics.json")
    gross_pnl = _finite_number(statistics.get("total_pnl_quote"))
    raw_trades = statistics.get("total_trades")
    closed_trades = 0
    if not isinstance(raw_trades, bool):
        try:
            closed_trades = max(0, int(raw_trades))
        except (TypeError, ValueError):
            closed_trades = 0

    inference_total = sum(cost_values)
    analyses = len(cost_values)
    cycles = len(records)
    return {
        "data_dir": str(data_dir),
        "cycles": cycles,
        "analyses_with_cost": analyses,
        "coverage": analyses / cycles,
        "inference_total": inference_total,
        "gross_pnl": gross_pnl,
        "net_pnl": gross_pnl - inference_total if gross_pnl is not None else None,
        "closed_trades": closed_trades,
        "cost_per_analysis": inference_total / analyses if analyses else None,
        "cost_per_closed_trade": inference_total / closed_trades if closed_trades else None,
        "statistics_available": bool(statistics),
    }


def _format_money(value: float | None, *, decimals: int = 2) -> str:
    """Format dollars, or n/a when the value is unavailable."""
    if value is None:
        return "n/a"
    return f"${value:,.{decimals}f}"


def format_report(report: dict[str, Any]) -> str:
    """Render the report as a terminal-friendly table."""
    lines = [
        "",
        "Inference cost report",
        f"Data directory: {report['data_dir']}",
        "",
        f"Cycles: {report['cycles']}",
        f"Analyses with cost: {report['analyses_with_cost']} ({report['coverage']:.1%})",
        f"Inference cost: {_format_money(report['inference_total'], decimals=6)}",
        f"Gross P&L: {_format_money(report['gross_pnl'])}",
        f"Net P&L: {_format_money(report['net_pnl'])}",
        f"Cost per analysis: {_format_money(report['cost_per_analysis'], decimals=6)}",
        f"Cost per closed trade: {_format_money(report['cost_per_closed_trade'], decimals=6)}",
    ]
    if not report["statistics_available"]:
        lines.append("statistics.json is missing or unreadable; gross/net P&L are unavailable.")
    lines.extend([
        "",
        "net = gross P&L - recorded inference cost; compare only overlapping periods",
        "gross P&L is statistics.json total_pnl_quote; exchange fees are not part of it.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; 1 means the cycle journal has no data to report."""
    parser = argparse.ArgumentParser(
        description="Report gross and net P&L after recorded LLM inference cost."
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Trading data directory containing metrics/cycles.jsonl and statistics.json",
    )
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)
    report = build_report(data_dir)
    if report is None:
        print(f"No cycle metrics found under {data_dir} (expected metrics/cycles.jsonl).")
        return 1
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
