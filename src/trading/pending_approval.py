"""Durable human approval for high-risk decisions.

The bot writes a pending approval and waits (bounded) for a human answer in the admin
console. Rules that matter:

* **Expiry means refusal, never execution.** A stale pending entry is marked ``expired``
  and the decision is dropped; price context is gone by then.
* The file is the interface: the dashboard and the bot are separate objects in the same
  process today, but a file survives a restart and cannot be half-written.
* Disabled by default (``risk_management.require_approval_above_usdc = 0``): with the
  threshold at zero every decision takes the old path.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"
EXPIRED = "expired"


@dataclass
class PendingApproval:
    approval_id: str
    created_at: str
    expires_at: str
    action: str
    status: str = PENDING
    symbol: str | None = None
    decision_hash: str | None = None
    quote_amount: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class PendingApprovalStore:
    """JSON file with a list of approvals; every rule is enforced here, not in the UI."""

    def __init__(self, path: str | Path) -> None:
        if not isinstance(path, (str, Path)):
            raise TypeError(f"approval store path must be str/Path, got {type(path).__name__}")
        self.path = Path(path)

    # ---------------------------------------------------------------- persistence
    def _load(self) -> list[dict[str, Any]]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        items = raw.get("approvals") if isinstance(raw, dict) else raw
        return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []

    def _save(self, items: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(dir=self.path.parent, prefix=".approvals-", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump({"approvals": items}, stream, ensure_ascii=False)
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _now(now: datetime | None) -> datetime:
        moment = now or datetime.now(timezone.utc)
        return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)

    def _expire(self, items: list[dict[str, Any]], moment: datetime) -> bool:
        changed = False
        for item in items:
            if item.get("status") != PENDING:
                continue
            try:
                expires = datetime.fromisoformat(str(item["expires_at"]))
            except (KeyError, ValueError):
                continue
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if moment >= expires:
                item["status"] = EXPIRED
                changed = True
        return changed

    @staticmethod
    def _to_model(item: dict[str, Any]) -> PendingApproval:
        known = {name for name in PendingApproval.__dataclass_fields__}
        return PendingApproval(**{key: value for key, value in item.items() if key in known})

    # ---------------------------------------------------------------- public API
    def submit(
        self,
        *,
        action: str,
        symbol: str | None,
        decision_hash: str | None,
        payload: dict[str, Any],
        quote_amount: float | None = None,
        ttl_minutes: float = 15.0,
        now: datetime | None = None,
    ) -> PendingApproval:
        moment = self._now(now)
        approval = PendingApproval(
            approval_id=uuid.uuid4().hex[:12],
            created_at=moment.isoformat(timespec="seconds"),
            expires_at=(moment + timedelta(minutes=ttl_minutes)).isoformat(timespec="seconds"),
            action=str(action).upper(),
            symbol=symbol,
            decision_hash=decision_hash,
            quote_amount=quote_amount,
            payload=dict(payload),
        )
        items = self._load()
        self._expire(items, moment)
        items.append(asdict(approval))
        self._save(items)
        return approval

    def list(self, *, now: datetime | None = None) -> list[PendingApproval]:
        moment = self._now(now)
        items = self._load()
        if self._expire(items, moment):
            self._save(items)
        return [self._to_model(item) for item in items]

    def pending(self, *, now: datetime | None = None) -> list[PendingApproval]:
        return [approval for approval in self.list(now=now) if approval.status == PENDING]

    def get(self, approval_id: str, *, now: datetime | None = None) -> PendingApproval | None:
        for approval in self.list(now=now):
            if approval.approval_id == approval_id:
                return approval
        return None

    def _decide(self, approval_id: str, status: str, now: datetime | None) -> PendingApproval | None:
        moment = self._now(now)
        items = self._load()
        self._expire(items, moment)
        updated: PendingApproval | None = None
        for item in items:
            if item.get("approval_id") != approval_id:
                continue
            if item.get("status") == PENDING:
                item["status"] = status
            updated = self._to_model(item)
            break
        self._save(items)
        return updated

    def approve(self, approval_id: str, *, now: datetime | None = None) -> PendingApproval | None:
        """Approve only a still-pending entry; an expired one stays refused."""
        current = self.get(approval_id, now=now)
        if current is None or current.status != PENDING:
            return current
        return self._decide(approval_id, APPROVED, now)

    def reject(self, approval_id: str, *, now: datetime | None = None) -> PendingApproval | None:
        current = self.get(approval_id, now=now)
        if current is None or current.status != PENDING:
            return current
        return self._decide(approval_id, REJECTED, now)


def requires_approval(quote_amount: float | None, threshold: float | None) -> bool:
    """A non-positive threshold disables the gate entirely."""
    try:
        limit = float(threshold or 0)
    except (TypeError, ValueError):
        return False
    if limit <= 0:
        return False
    return quote_amount is not None and quote_amount >= limit
