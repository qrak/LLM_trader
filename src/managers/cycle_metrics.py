"""Per-cycle operational metrics: one JSONL line per analysis cycle.

Why a journal and not counters in memory: the questions asked after an incident are
"how long did cycles take yesterday", "what share of decisions was HOLD", "did the
guards block anything" — all of which need history across restarts. JSONL appends are
crash-safe enough for metrics (a torn last line is skipped on read) and need no schema
migration.

Cost is deliberately absent: the provider usage numbers are not plumbed through the
analyzer yet, and a made-up cost is worse than no cost. Add it here when the usage
fields reach this layer.
"""

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
    """Append one cycle record and return what was written."""
    if not isinstance(path, (str, Path)):
        raise TypeError(f"metrics path must be str/Path, got {type(path).__name__}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"), **record}
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    return payload


def load_cycles(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Read the journal, skipping malformed lines (a torn tail must not break the bot)."""
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
    """Nearest-rank percentile (p50 of [10, 20, 30, 40] is 20)."""
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


def _parse_ts(raw: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def summarize(
    records: Iterable[dict[str, Any]], *, days: int = 1, now: datetime | None = None
) -> dict[str, Any]:
    """Aggregate records from the last ``days`` days into the numbers worth watching."""
    moment = now or datetime.now(timezone.utc)
    cutoff = moment - timedelta(days=days)
    window = [record for record in records if (ts := _parse_ts(record.get("ts"))) is not None and ts >= cutoff]

    latencies = [record.get("latency_ms") for record in window]
    signals = Counter(str(record.get("signal") or "UNKNOWN").upper() for record in window)
    actions = Counter(str(record.get("action") or "NONE").upper() for record in window)
    cycles = len(window)

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
    }
