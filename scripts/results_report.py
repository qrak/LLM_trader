from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from ledger_facts import (
    benchmark,
    load_costs,
    load_cycles,
    load_json,
    load_period,
    load_trades,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the paper-trading ledger, inference cost and the buy-and-hold benchmark.")
    parser.add_argument("--data-dir", default="data/trading", type=Path)
    parser.add_argument("--symbol", default="BTCUSDC")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    stats = load_json(data_dir / "statistics.json")
    costs = load_costs(data_dir)
    trades = load_trades(data_dir)
    cycles = load_cycles(data_dir)

    print("## Ledger\n")
    print("| metric | value |")
    print("|---|---|")
    for key, label in [
        ("total_trades", "Closed trades"),
        ("winning_trades", "Winners"),
        ("losing_trades", "Losers"),
        ("win_rate", "Win rate %"),
        ("total_pnl_pct", "P&L %"),
        ("total_pnl_quote", "P&L USD"),
        ("profit_factor", "Profit factor"),
        ("sharpe_ratio", "Sharpe"),
        ("sortino_ratio", "Sortino"),
        ("max_drawdown_pct", "Max drawdown %"),
        ("best_trade_pct", "Best trade %"),
        ("worst_trade_pct", "Worst trade %"),
    ]:
        if key in stats:
            value = stats[key]
            if isinstance(value, float):
                value = round(value, 2)
            print(f"| {label} | {value} |")
    if stats.get("last_updated"):
        print(f"\nLedger written at `{stats['last_updated']}`.\n")

    print("## Inference cost\n")
    print("| item | value |")
    print("|---|---|")
    print(f"| Total USD | {costs.get('total', 0)} |")
    for provider, amount in costs.get("providers", {}).items():
        print(f"| {provider} | {amount} |")
    print(f"| Input tokens | {costs.get('tokens_in', 0):,} |")
    print(f"| Output tokens | {costs.get('tokens_out', 0):,} |")
    if stats.get("total_pnl_quote") is not None:
        net = round(stats["total_pnl_quote"] - costs.get("total", 0.0), 2)
        print(f"| Net after inference | {net} |")
    print()

    if cycles:
        print("## Analysis cycles\n")
        print(f"- cycles: {cycles['count']}")
        print(f"- median latency: {cycles['latency_median']} s")
        print(f"- mean inference cost: ${cycles['avg_cost']}")
        print(f"- signals: {cycles['signals']}\n")

    print("## Closed trades\n")
    print("| closed | direction | P&L % | close reason |")
    print("|---|---|---|---|")
    for trade in trades:
        pnl = "-" if trade["pnl_pct"] is None else f"{trade['pnl_pct']:.2f}"
        print(f"| {trade['when']} | {trade['direction']} | {pnl} | {trade['reason']} |")
    print()

    first, last = load_period(data_dir)
    if first and last:
        start = datetime.fromisoformat(first).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(last).replace(tzinfo=timezone.utc)
        bench = benchmark(args.symbol, start, end)
        print("## Buy-and-hold benchmark\n")
        if "error" in bench:
            print(f"unavailable: {bench['error']}")
        else:
            print(f"- {bench['symbol']}, {first} to {last}: {bench['change_pct']}%")
            print(f"- {bench['start']:.2f} -> {bench['end']:.2f} over {bench['bars']} 4h bars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
