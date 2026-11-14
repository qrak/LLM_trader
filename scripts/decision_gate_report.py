

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analyzer.decision_gate import DecisionGate, DecisionsClient
from src.analyzer.technical_calculator import TechnicalCalculator
from src.config.loader import config as live_config
from src.utils.format_utils import FormatUtils
from src.utils.timeframe_validator import TimeframeValidator

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_SYMBOL = "BTCUSDC"
CANDLES = 999
ENTRY_ACTIONS = ("BUY", "SELL")
CLOSE_ACTIONS = ("CLOSE", "CLOSE_LONG", "CLOSE_SHORT")
PAIR_TOLERANCE_SECONDS = 900


def klines(symbol: str, start: datetime, end: datetime, interval: str) -> list[list]:

    bars: list[list] = []
    cursor = int(start.timestamp() * 1000)
    stop = int(end.timestamp() * 1000)
    while cursor < stop:
        url = (
            f"{BINANCE_KLINES}?symbol={symbol}&interval={interval}"
            f"&startTime={cursor}&limit=1000"
        )
        with urllib.request.urlopen(url, timeout=30) as response:
            batch = json.loads(response.read())
        if not batch:
            break
        bars += batch
        cursor = batch[-1][0] + 1
    return bars


def bars_to_array(bars: list[list]) -> np.ndarray:

    return np.asarray(
        [[float(bar[index]) for index in range(6)] for bar in bars], dtype=float
    )


def ranks(values: list[float]) -> list[float]:

    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            result[order[index]] = average
        position = end + 1
    return result


def spearman(pairs: list[tuple[float, float]]) -> float | None:

    if len(pairs) < 3:
        return None
    xs = ranks([pair[0] for pair in pairs])
    ys = ranks([pair[1] for pair in pairs])
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = (
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    ) ** 0.5
    return round(numerator / denominator, 3) if denominator else None


def utc(value: str) -> datetime:

    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


class GateReport:
    def __init__(
        self,
        logger: logging.Logger,
        gate: DecisionGate | None,
        data_dir: Path,
        symbol: str = BINANCE_SYMBOL,
        timeframe: str = "4h",
    ) -> None:

        self.logger = logger
        self.gate = gate
        self.data_dir = data_dir
        self.symbol = symbol
        self.timeframe = timeframe
        self.frame_minutes = TimeframeValidator.to_minutes(timeframe)
        self.calculator = TechnicalCalculator(None, FormatUtils())

    def connect(self) -> sqlite3.Connection:

        con = sqlite3.connect(
            f"file:{self.data_dir / 'trade_history.db'}?mode=ro", uri=True
        )
        con.row_factory = sqlite3.Row
        return con

    def closed_pairs(self) -> list[dict]:

        con = self.connect()
        try:
            closes = [
                dict(row)
                for row in con.execute(
                    "select id, trade_id, direction, pnl_pct, created_at "
                    "from trade_post_mortem where pnl_pct is not null order by created_at"
                )
            ]
            entries = [
                dict(row)
                for row in con.execute(
                    "select id, timestamp, action, confidence, price, stop_loss, "
                    "take_profit, position_size, reasoning, indicators_json "
                    "from trade_history where action in ('BUY', 'SELL') "
                    "order by timestamp"
                )
            ]
            close_rows = {
                row["id"]: row["timestamp"]
                for row in con.execute(
                    "select id, timestamp from trade_history "
                    f"where action in {CLOSE_ACTIONS}"
                )
            }
        finally:
            con.close()
        pairs: list[dict] = []
        for close in closes:
            close_ts = close_rows.get(close["trade_id"]) or close["created_at"]
            close_time = utc(close_ts)
            entry = None
            for candidate in entries:
                if utc(candidate["timestamp"]) < close_time:
                    entry = candidate
                else:
                    break
            if entry is None:
                continue
            pairs.append({"entry": entry, "close": close})
        return pairs

    def state_for(self, entry: dict) -> tuple[dict, SimpleNamespace, dict]:

        entry_time = utc(entry["timestamp"])
        entry_ms = int(entry_time.timestamp() * 1000)
        start = entry_time - timedelta(minutes=CANDLES * self.frame_minutes)
        end = entry_time + timedelta(minutes=8 * self.frame_minutes)
        bars = klines(self.symbol, start, end, self.timeframe)
        closed = [bar for bar in bars if int(bar[6]) < entry_ms]
        forward = [bar for bar in bars if int(bar[0]) >= entry_ms]
        array = bars_to_array(closed[-CANDLES:]) if closed else None
        technical = {}
        if array is not None and len(array) > 20:
            indicators = self.calculator.get_indicators(array)
            for key, values in indicators.items():
                try:
                    if values is not None and not np.isnan(values).all():
                        technical[key] = float(values[-1])
                except (TypeError, ValueError, IndexError):
                    continue
        stored = json.loads(entry["indicators_json"] or "{}")
        if technical.get("adx") is None and stored.get("adx_at_entry") is not None:
            technical["adx"] = float(stored["adx_at_entry"])
        if technical.get("rsi") is None and stored.get("rsi_at_entry") is not None:
            technical["rsi"] = float(stored["rsi_at_entry"])
        analysis = {
            "signal": entry["action"],
            "confidence": entry["confidence"],
            "entry_price": entry["price"],
            "stop_loss": entry["stop_loss"],
            "take_profit": entry["take_profit"],
            "position_size": entry["position_size"],
            "risk_reward_ratio": stored.get("rr_ratio_at_entry"),
            "reasoning": entry["reasoning"],
            "trend": {"direction": stored.get("trend_direction_at_entry")},
        }
        context = SimpleNamespace(
            symbol="BTC/USDC",
            timeframe=self.timeframe,
            current_price=entry["price"],
            technical_data=technical,
            ohlcv_candles=array,
        )
        return analysis, context, self.forward_returns(entry, forward)

    def forward_returns(self, entry: dict, forward: list[list]) -> dict:

        price = float(entry["price"] or 0.0)
        if price <= 0 or not forward:
            return {}
        sign = 1.0 if entry["action"] == "BUY" else -1.0
        returns = {}
        for horizon in (1, 3, 6):
            if len(forward) >= horizon:
                close = float(forward[horizon - 1][4])
                returns[horizon] = round((close - price) / price * 100.0 * sign, 3)
        return returns

    async def replay(self, limit: int | None, dry_run: bool) -> int:

        pairs = self.closed_pairs()
        if limit:
            pairs = pairs[-limit:]
        if not pairs:
            print("no closed trades with a P&L and a matching entry decision")
            return 1
        rows: list[dict] = []
        cost = 0.0
        for pair in pairs:
            entry = pair["entry"]
            analysis, context, forward = self.state_for(entry)
            stamp = entry["timestamp"][:16]
            if dry_run:
                print(f"{stamp} {entry['action']:4} {entry['confidence']:6} "
                      f"pnl {pair['close']['pnl_pct']:+.2f}% fwd {forward}")
                continue
            verdict = await self.gate.review({"analysis": analysis}, context)
            if verdict is None:
                continue
            cost += verdict.cost_usd
            rows.append({
                "stamp": stamp,
                "signal": entry["action"],
                "pnl": float(pair["close"]["pnl_pct"]),
                "quality": float(verdict.quality or 0.0),
                "direction": verdict.direction,
                "edge": verdict.edge_p,
                "multiplier": verdict.multiplier,
                "forward": forward,
            })
            print(
                f"{stamp} {entry['action']:4} {entry['confidence']:6} "
                f"pnl {pair['close']['pnl_pct']:+6.2f}%  quality {verdict.quality:4.2f}  "
                f"gate {verdict.direction!s:5}  edge {verdict.edge_p}  "
                f"fwd3 {forward.get(3)}  x{verdict.multiplier:g}  {verdict.latency_ms} ms"
            )
        if dry_run:
            print(f"\n{len(pairs)} pairs, nothing sent")
            return 0
        if not rows:
            print("the gate produced no verdicts")
            return 1
        print(f"\nspearman(quality, realised pnl) = {self._correlation(rows, 'quality', 'pnl')}")
        for horizon in (1, 3, 6):
            print(f"spearman(quality, forward {horizon} candles) = "
                  f"{self._correlation(rows, 'quality', 'forward', horizon)}")
            print(f"spearman(edge, forward {horizon} candles) = "
                  f"{self._correlation(rows, 'edge', 'forward', horizon)}")
        buckets: dict[float, list[float]] = {}
        for row in rows:
            buckets.setdefault(row["multiplier"], []).append(row["pnl"])
        for multiplier in sorted(buckets):
            values = buckets[multiplier]
            mean = sum(values) / len(values)
            wins = sum(1 for value in values if value > 0)
            print(
                f"  x{multiplier:g}: {len(values)} trades, {wins} wins, "
                f"mean pnl {mean:+.2f}%, sum {sum(values):+.2f}%"
            )
        for signal in sorted({row["signal"] for row in rows}):
            side = [row for row in rows if row["signal"] == signal]
            mean_quality = sum(row["quality"] for row in side) / len(side)
            forward = [row["forward"].get(3) for row in side if row["forward"].get(3) is not None]
            mean_forward = sum(forward) / len(forward) if forward else None
            print(
                f"  {signal}: {len(side)} trades, mean quality {mean_quality:.2f}, "
                f"mean fwd3 {mean_forward:+.2f}%, "
                f"spearman(quality, fwd3) {self._correlation(side, 'quality', 'forward', 3)}"
            )
        print(f"\ngate cost ${cost:.5f}")
        return 0

    @staticmethod
    def _correlation(rows: list[dict], first: str, second: str,
                     horizon: int | None = None) -> float | None:

        pairs: list[tuple[float, float]] = []
        for row in rows:
            left = row.get(first)
            if horizon is None:
                right = row.get(second)
            else:
                right = (row.get(second) or {}).get(horizon)
            if left is None or right is None:
                continue
            pairs.append((float(left), float(right)))
        return spearman(pairs)

    def journal(self) -> int:

        path = self.data_dir / "decisions_gate.jsonl"
        if not path.exists():
            print(f"no journal at {path}")
            return 1
        lines = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not lines:
            print("the journal is empty")
            return 1
        errors = [line for line in lines if line.get("error")]
        quality = [line["quality"] for line in lines if line.get("quality") is not None]
        multipliers: dict[float, int] = {}
        for line in lines:
            multipliers[line["multiplier"]] = multipliers.get(line["multiplier"], 0) + 1
        print(f"{len(lines)} cycles, {len(errors)} failed, "
              f"cost ${sum(line['cost_usd'] for line in lines):.5f}, "
              f"median latency {sorted(line['latency_ms'] for line in lines)[len(lines) // 2]} ms")
        print(f"quality: {self._histogram(quality)}")
        print(f"multipliers: {dict(sorted(multipliers.items()))}")
        print(f"applied: {sum(1 for line in lines if line.get('applied'))}")
        paired = self._paired_outcomes(lines)
        if paired:
            correlation = spearman(paired)
            print(f"spearman(quality, pnl) = {correlation} over {len(paired)} paired trades")
        else:
            print("no journal line could be paired with a closed trade yet")
        return 0

    def _paired_outcomes(self, lines: list[dict]) -> list[tuple[float, float]]:

        con = self.connect()
        try:
            entries = [
                dict(row)
                for row in con.execute(
                    "select id, timestamp from trade_history where action in ('BUY', 'SELL') "
                    "order by timestamp"
                )
            ]
            close_rows = [
                dict(row)
                for row in con.execute(
                    "select id, timestamp from trade_history "
                    f"where action in {CLOSE_ACTIONS} order by timestamp"
                )
            ]
            pnl_by_trade = {
                row["trade_id"]: row["pnl_pct"]
                for row in con.execute(
                    "select trade_id, pnl_pct from trade_post_mortem where pnl_pct is not null"
                )
            }
        finally:
            con.close()
        paired: list[tuple[float, float]] = []
        for line in lines:
            if line.get("quality") is None:
                continue
            stamp = utc(line["ts"])
            entry = next(
                (
                    row for row in entries
                    if abs((utc(row["timestamp"]) - stamp).total_seconds()) <= PAIR_TOLERANCE_SECONDS
                ),
                None,
            )
            if entry is None:
                continue
            close = next(
                (row for row in close_rows if utc(row["timestamp"]) > utc(entry["timestamp"])),
                None,
            )
            if close is None or pnl_by_trade.get(close["id"]) is None:
                continue
            paired.append((float(line["quality"]), float(pnl_by_trade[close["id"]])))
        return paired

    @staticmethod
    def _histogram(values: list[float]) -> str:

        if not values:
            return "none"
        counts = [0, 0, 0, 0]
        for value in values:
            counts[max(0, min(3, round(value)))] += 1
        return " ".join(f"{index}:{count}" for index, count in enumerate(counts))


def build_gate(args: argparse.Namespace, logger: logging.Logger) -> DecisionGate:

    client = DecisionsClient(
        url=args.url or live_config.DECISION_GATE_URL,
        model=args.model or live_config.DECISION_GATE_MODEL,
        api_key=live_config.OPENROUTER_API_KEY or "",
        timeout_seconds=live_config.DECISION_GATE_TIMEOUT_SECONDS,
    )
    return DecisionGate(
        logger=logger,
        client=client,
        mode="shadow",
        min_confidence_pct=live_config.DECISION_GATE_MIN_CONFIDENCE_PCT,
        journal_path=Path(live_config.DATA_DIR) / "trading" / "decisions_gate_replay.jsonl",
    )


def main() -> int:

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Score closed trades and the shadow journal with the decisions gate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Replay writes its own journal (decisions_gate_replay.jsonl); the live one stays clean.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    replay = sub.add_parser("replay", help="score the closed trades the bot already took")
    replay.add_argument("--limit", type=int, default=None)
    replay.add_argument("--dry-run", action="store_true")
    replay.add_argument("--url", default=None)
    replay.add_argument("--model", default=None)

    sub.add_parser("journal", help="summarise the live shadow journal")

    args = parser.parse_args()
    logger = logging.getLogger("decision_gate_report")
    data_dir = Path(live_config.DATA_DIR) / "trading"

    if args.command == "journal":
        report = GateReport(logger, None, data_dir, timeframe=live_config.TIMEFRAME)
        return report.journal()

    gate = build_gate(args, logger)
    report = GateReport(logger, gate, data_dir, timeframe=live_config.TIMEFRAME)
    return asyncio.run(report.replay(args.limit, args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
