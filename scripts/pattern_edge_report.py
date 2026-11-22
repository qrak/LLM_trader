from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analyzer.pattern_engine.indicator_patterns.candlestick_patterns import (
    PATTERN_IDS,
    detect_candlestick_patterns_numba,
)

SYMBOL = "BTC/USDC"
EXCHANGE_ID = "binance"
TIMEFRAME = "4h"
HORIZONS = (6, 12, 24)
MIN_OCCURRENCES = 30
ALPHA = 0.05
BULLISH = frozenset({
    "hammer", "inverted_hammer", "bullish_engulfing", "morning_star",
    "piercing_line", "three_white_soldiers",
})
BEARISH = frozenset({
    "shooting_star", "hanging_man", "bearish_engulfing", "evening_star",
    "dark_cloud_cover", "three_black_crows",
})


def history(days: int) -> np.ndarray:

    exchange = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    rows: list[list[float]] = []
    cursor = since
    while True:
        batch = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + 1
        if len(batch) < 1000:
            break
    array = np.asarray([[float(value) for value in row[:6]] for row in rows], dtype=float)
    return array[:-1]


def forward_returns(closes: np.ndarray, horizon: int) -> np.ndarray:

    result = np.full(len(closes), np.nan)
    result[:-horizon] = (closes[horizon:] - closes[:-horizon]) / closes[:-horizon]
    return result


def main() -> int:

    parser = argparse.ArgumentParser(description="Measure the forward-return edge of each candlestick pattern.")
    parser.add_argument("--days", type=int, default=1500)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    candles = history(args.days)
    if len(candles) < 500:
        print("not enough history")
        return 1
    opens, highs, lows, closes = candles[:, 1], candles[:, 2], candles[:, 3], candles[:, 4]
    ids, indices = detect_candlestick_patterns_numba(opens, highs, lows, closes, len(closes) - 20)
    print(f"candles: {len(candles)}  patterns: {len(ids)}")

    forward = {horizon: forward_returns(closes, horizon) for horizon in HORIZONS}
    base = {
        horizon: float(np.nanmean(forward[horizon] > 0)) for horizon in HORIZONS
    }
    tests = len(PATTERN_IDS) * len(HORIZONS)
    threshold = ALPHA / tests
    rows: list[dict[str, object]] = []

    for pattern_id, name in PATTERN_IDS.items():
        if name not in BULLISH and name not in BEARISH:
            continue
        bullish = name in BULLISH
        hits = indices[ids == pattern_id]
        for horizon in HORIZONS:
            values = forward[horizon][hits]
            values = values[~np.isnan(values)]
            if len(values) < MIN_OCCURRENCES:
                continue
            wins = int(np.sum(values > 0)) if bullish else int(np.sum(values < 0))
            rate = wins / len(values)
            expected = base[horizon] if bullish else 1.0 - base[horizon]
            p_value = stats.binomtest(wins, len(values), expected, alternative="greater").pvalue
            rows.append({
                "pattern": name,
                "direction": "bullish" if bullish else "bearish",
                "horizon": horizon,
                "occurrences": len(values),
                "hit_rate": round(rate, 4),
                "base_rate": round(expected, 4),
                "edge": round(rate - expected, 4),
                "p_value": round(float(p_value), 5),
                "passes_bonferroni": bool(p_value < threshold),
            })

    rows.sort(key=lambda row: (row["pattern"], row["horizon"]))
    header = f"{'pattern':22s} {'dir':8s} {'h':>3s} {'n':>6s} {'hit':>7s} {'base':>7s} {'edge':>7s} {'p':>8s} pass"
    print(header)
    for row in rows:
        print(
            f"{row['pattern']:22s} {row['direction']:8s} {row['horizon']:3d} {row['occurrences']:6d} "
            f"{row['hit_rate']:7.3f} {row['base_rate']:7.3f} {row['edge']:+7.3f} "
            f"{row['p_value']:8.4f} {'yes' if row['passes_bonferroni'] else 'no'}"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
