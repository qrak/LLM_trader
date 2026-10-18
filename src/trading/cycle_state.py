"""Crash-safe checkpoint of one analysis cycle.

The executor owns the durable truth (ledger + journals). This checkpoint answers a
narrower question on the bot side: *did I already send this exact intent?* A process that
dies between "forwarded to the executor" and "confirmed" would otherwise re-send the same
decision after a restart.

Deliberately minimal: one small JSON file, one state at a time, atomic replace. No DAG, no
task queue — the cycle is linear and only ``INTENT_SENT`` changes behaviour (duplicate
suppression). Everything else is diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class CycleState(str, Enum):
    """Where one analysis cycle is."""

    ANALYZING = "analyzing"
    DECIDED = "decided"
    INTENT_SENT = "intent_sent"
    CONFIRMED = "confirmed"
    LEARNED = "learned"
    FAILED = "failed"


@dataclass(frozen=True)
class CycleCheckpoint:
    cycle_id: str
    state: str
    ts: str
    symbol: str | None = None
    decision_hash: str | None = None
    detail: str | None = None


def hash_decision(action: str | None, symbol: str | None, analysis: dict[str, Any] | None) -> str:
    """Stable identity of an intent: two identical decisions hash the same."""
    analysis = analysis or {}
    material = {
        "action": str(action or "").upper(),
        "symbol": symbol,
        "signal": str(analysis.get("signal") or "").upper(),
        "entry": analysis.get("entry_price"),
        "stop_loss": analysis.get("stop_loss"),
        "take_profit": analysis.get("take_profit"),
        "size": analysis.get("quantity"),
    }
    encoded = json.dumps(material, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def is_duplicate_intent(previous: CycleCheckpoint | None, decision_hash: str | None) -> bool:
    """True when the checkpoint says this very intent was already forwarded, unconfirmed."""
    if previous is None or not decision_hash:
        return False
    return previous.state == CycleState.INTENT_SENT.value and previous.decision_hash == decision_hash


class CycleCheckpointStore:
    """One checkpoint file, written atomically, read tolerantly."""

    def __init__(self, path: str | Path) -> None:
        if not isinstance(path, (str, Path)):
            raise TypeError(f"checkpoint path must be str/Path, got {type(path).__name__}")
        self.path = Path(path)

    def load(self) -> CycleCheckpoint | None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict) or not raw.get("state"):
            return None
        return CycleCheckpoint(
            cycle_id=str(raw.get("cycle_id") or ""),
            state=str(raw["state"]),
            ts=str(raw.get("ts") or ""),
            symbol=raw.get("symbol"),
            decision_hash=raw.get("decision_hash"),
            detail=raw.get("detail"),
        )

    def save(self, checkpoint: CycleCheckpoint) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(dir=self.path.parent, prefix=".cycle_state-", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(asdict(checkpoint), stream, ensure_ascii=False)
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def record(self, state: CycleState, *, symbol: str | None = None,
               decision_hash: str | None = None, detail: str | None = None) -> CycleCheckpoint:
        now = datetime.now(timezone.utc)
        checkpoint = CycleCheckpoint(
            cycle_id=now.strftime("%Y%m%dT%H%M%S%f"),
            state=state.value if isinstance(state, CycleState) else str(state),
            ts=now.isoformat(timespec="seconds"),
            symbol=symbol,
            decision_hash=decision_hash,
            detail=detail,
        )
        self.save(checkpoint)
        return checkpoint
