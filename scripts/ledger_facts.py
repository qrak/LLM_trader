from __future__ import annotations

import json
import sqlite3
import statistics
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
EXCLUDED_REASON = "EXIT FILL DIVERGENCE"
SHORT_REASONS = {
    "stop_loss_filled": "stop loss",
    "take_profit_filled": "take profit",
    "close_confirmed_by_executor": "closed",
    "EXIT FILL DIVERGENCE": "excluded (fill divergence)",
}


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


def _open_ledger(data_dir: Path) -> sqlite3.Connection | None:
    db = data_dir / "trade_history.db"
    if not db.exists():
        return None
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def load_trades(data_dir: Path) -> list[dict]:
    con = _open_ledger(data_dir)
    if con is None:
        return []
    try:
        rows = con.execute(
            "select created_at, direction, pnl_pct, close_reason from trade_post_mortem order by created_at"
        ).fetchall()
    finally:
        con.close()
    trades = []
    for r in rows:
        reason = r[3] or ""
        label = next((text for key, text in SHORT_REASONS.items() if key in reason), reason[:40])
        trades.append({"when": r[0][:10], "direction": r[1], "pnl_pct": r[2], "reason": label})
    return trades


def load_period(data_dir: Path) -> tuple[str | None, str | None]:
    con = _open_ledger(data_dir)
    if con is None:
        return None, None
    try:
        first, last = con.execute("select min(timestamp), max(timestamp) from trade_history").fetchone()
    finally:
        con.close()
    return (first or "")[:10] or None, (last or "")[:10] or None


def count_excluded(data_dir: Path) -> int:
    con = _open_ledger(data_dir)
    if con is None:
        return 0
    try:
        (count,) = con.execute(
            "select count(*) from trade_post_mortem where close_reason like ?", (f"%{EXCLUDED_REASON}%",)
        ).fetchone()
    finally:
        con.close()
    return int(count)


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
    window_end_ms = int((last + timedelta(days=1)).timestamp() * 1000)
    now_ms = int(time.time() * 1000)
    bars = [b for b in bars if b[6] <= window_end_ms and b[6] <= now_ms]
    if not bars:
        return {"error": "no closed klines in the window"}
    start_price = float(bars[0][1])
    end_price = float(bars[-1][4])
    return {
        "symbol": symbol,
        "start": start_price,
        "end": end_price,
        "change_pct": round((end_price / start_price - 1.0) * 100.0, 2),
        "bars": len(bars),
    }


def ledger_facts(data_dir: Path, symbol: str = "BTCUSDC", with_benchmark: bool = True) -> dict:
    stats = load_json(data_dir / "statistics.json")
    costs = load_costs(data_dir)
    cycles = load_cycles(data_dir)
    first, last = load_period(data_dir)
    if first is None or last is None:
        return {"available": False, "data_dir": str(data_dir)}
    start = datetime.fromisoformat(first).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(last).replace(tzinfo=timezone.utc)
    bench = benchmark(symbol, start, end) if with_benchmark else {}
    excluded = count_excluded(data_dir)
    counted = int(stats.get("winning_trades", 0)) + int(stats.get("losing_trades", 0))
    pnl_quote = stats.get("total_pnl_quote")
    return {
        "available": True,
        "window_start": first,
        "window_end": last,
        "window_days": (end - start).days + 1,
        "pair": "BTC/USDC",
        "timeframe": "4h",
        "capital_usdc": stats.get("initial_capital"),
        "closed_trades": stats.get("total_trades"),
        "counted_trades": counted,
        "excluded_trades": excluded,
        "win_rate_pct": round(float(stats.get("win_rate", 0.0)), 2),
        "return_pct": round(float(stats.get("total_pnl_pct", 0.0)), 2),
        "return_usd": round(float(pnl_quote), 2) if pnl_quote is not None else None,
        "profit_factor": round(float(stats.get("profit_factor", 0.0)), 2),
        "sharpe": round(float(stats.get("sharpe_ratio", 0.0)), 2),
        "sortino": round(float(stats.get("sortino_ratio", 0.0)), 2),
        "max_drawdown_pct": round(float(stats.get("max_drawdown_pct", 0.0)), 2),
        "best_trade_pct": round(float(stats.get("best_trade_pct", 0.0)), 2),
        "worst_trade_pct": round(float(stats.get("worst_trade_pct", 0.0)), 2),
        "inference_usd": costs["total"],
        "tokens_in": costs["tokens_in"],
        "tokens_out": costs["tokens_out"],
        "net_after_inference_usd": round(float(pnl_quote) - costs["total"], 2) if pnl_quote is not None else None,
        "cycles": cycles.get("count", 0),
        "cycle_latency_median_s": cycles.get("latency_median", 0.0),
        "cycle_cost_usd": cycles.get("avg_cost", 0.0),
        "signals": cycles.get("signals", {}),
        "benchmark_pct": bench.get("change_pct"),
        "benchmark_start": bench.get("start"),
        "benchmark_end": bench.get("end"),
        "benchmark_bars": bench.get("bars"),
        "benchmark_error": bench.get("error"),
        "snapshot_at": datetime.now(timezone.utc).date().isoformat(),
    }
