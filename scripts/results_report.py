"""Print the paper-trading ledger, inference cost and the buy-and-hold benchmark.

Reads only local runtime files (`data/trading/`) plus one public Binance klines
request, then prints a Markdown block that matches `docs/RESULTS.md`.

Usage:
    python scripts/results_report.py
    python scripts/results_report.py --data-dir data/trading --symbol BTCUSDC
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_costs(data_dir: Path) -> dict:
    raw = load_json(data_dir / "api_costs.json")
    providers = {k: v for k, v in raw.items() if isinstance(v, dict) and "total_cost" in v}
    return {
        "total": round(sum(v.get("total_cost", 0.0) for v in providers.values()), 4),
        "providers": {k: round(v.get("total_cost", 0.0), 4) for k, v in providers.items()},
        "tokens_in": sum(v.get("total_input_tokens", 0) for v in providers.values()),
        "tokens_out": sum(v.get("total_output_tokens", 0) for v in providers.values()),
    }


def load_trades(data_dir: Path) -> list[dict]:
    db = data_dir / "trade_history.db"
    if not db.exists():
        return []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "select created_at, direction, pnl_pct, close_reason from trade_post_mortem order by created_at"
        ).fetchall()
    finally:
        con.close()
    short_reasons = {
        "stop_loss_filled": "stop loss",
        "take_profit_filled": "take profit",
        "close_confirmed_by_executor": "closed",
        "EXIT FILL DIVERGENCE": "excluded (fill divergence)",
    }
    trades = []
    for r in rows:
        reason = r[3] or ""
        label = next((text for key, text in short_reasons.items() if key in reason), reason[:40])
        trades.append({"when": r[0][:10], "direction": r[1], "pnl_pct": r[2], "reason": label})
    return trades


def load_period(data_dir: Path) -> tuple[str | None, str | None]:
    db = data_dir / "trade_history.db"
    if not db.exists():
        return None, None
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        first, last = con.execute("select min(timestamp), max(timestamp) from trade_history").fetchone()
    finally:
        con.close()
    return (first or "")[:10] or None, (last or "")[:10] or None


def load_cycles(data_dir: Path) -> dict:
    path = data_dir / "metrics" / "cycles.jsonl"
    if not path.exists():
        return {}
    cycles = [json.loads(line) for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    latencies = [c["latency_ms"] / 1000.0 for c in cycles if c.get("latency_ms")]
    costs = [c["inference_cost_usd"] for c in cycles if c.get("inference_cost_usd")]
    signals: dict[str, int] = {}
    for cycle in cycles:
        key = cycle.get("signal") or "?"
        signals[key] = signals.get(key, 0) + 1
    return {
        "count": len(cycles),
        "signals": signals,
        "latency_median": round(statistics.median(latencies), 1) if latencies else 0.0,
        "avg_cost": round(statistics.mean(costs), 4) if costs else 0.0,
    }


def klines(symbol: str, start: datetime, end: datetime) -> list[list]:
    bars: list[list] = []
    cursor = int(start.timestamp() * 1000)
    stop = int(end.timestamp() * 1000)
    while cursor < stop:
        url = f"{BINANCE_KLINES}?symbol={symbol}&interval=4h&startTime={cursor}&limit=1000"
        with urllib.request.urlopen(url, timeout=30) as response:
            batch = json.loads(response.read())
        if not batch:
            break
        bars += batch
        cursor = batch[-1][0] + 1
    return bars


def benchmark(symbol: str, first: datetime, last: datetime) -> dict:
    try:
        bars = klines(symbol, first, last + timedelta(days=1))
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        return {"error": str(exc)}
    if not bars:
        return {"error": "no klines returned"}
    start_price = float(bars[0][1])
    end_price = float(bars[-1][4])
    return {
        "symbol": symbol,
        "start": start_price,
        "end": end_price,
        "change_pct": round((end_price / start_price - 1.0) * 100.0, 2),
        "bars": len(bars),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
