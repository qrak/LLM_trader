

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DEFAULT_RELATIVE_PATH = Path("metrics") / "cycles.jsonl"


def append_cycle(path: str | Path, record: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:

    if not isinstance(path, (str, Path)):
        raise TypeError(f"metrics path must be str/Path, got {type(path).__name__}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"), **record}
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    return payload


def load_cycles(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:

    target = Path(path)
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8", errors="replace") as handle:
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
    return records[-limit:] if limit else records


def percentile(values: Iterable[Any], pct: float) -> float | None:

    numbers: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            numbers.append(number)
    numbers.sort()
    if not numbers:
        return None
    rank = max(1, math.ceil(pct / 100.0 * len(numbers)))
    return numbers[min(rank, len(numbers)) - 1]


def _finite_non_negative(value: Any) -> float | None:

    if isinstance(value, (bool, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _parse_ts(raw: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def summarize(
    records: Iterable[dict[str, Any]], *, days: int = 1, now: datetime | None = None
) -> dict[str, Any]:

    moment = now or datetime.now(timezone.utc)
    cutoff = moment - timedelta(days=days)
    window = [record for record in records if (ts := _parse_ts(record.get("ts"))) is not None and ts >= cutoff]

    latencies = [record.get("latency_ms") for record in window]
    signals = Counter(str(record.get("signal") or "UNKNOWN").upper() for record in window)
    actions = Counter(str(record.get("action") or "NONE").upper() for record in window)
    cost_values = [
        value
        for record in window
        if (value := _finite_non_negative(record.get("inference_cost_usd"))) is not None
    ]
    cycles = len(window)
    analyses = len(cost_values)
    total_cost = sum(cost_values)

    return {
        "cycles": cycles,
        "errors": sum(1 for record in window if record.get("error")),
        "stale": sum(1 for record in window if record.get("stale")),
        "guarded": sum(1 for record in window if record.get("guarded")),
        "signals": dict(signals),
        "actions": dict(actions),
        "hold_share": round(signals.get("HOLD", 0) / cycles, 4) if cycles else None,
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
        },
        "inference_cost": {
            "analyses": analyses,
            "total_usd": round(total_cost, 6),
            "per_analysis_usd": round(total_cost / analyses, 6) if analyses else None,
            "coverage": round(analyses / cycles, 4) if cycles else None,
        },
    }
