"""Immutable audit records for trade order governance.

Every guard check, state transition, and final approval/rejection
decision is recorded as an immutable AuditRecord. These records are
structured for future telemetry and dashboard exposure.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class AuditRecord(BaseModel):
    """Single immutable entry in the trade governance audit trail.

    Attributes:
        record_id: Unique identifier for this record.
        order_id: Links to the OrderIntent being evaluated.
        event_type: Category of audit event (guard_check, state_transition,
                    approval, rejection, execution, cancellation).
        actor: What produced this event (guard name, system component).
        result: Outcome of the event (passed, failed, approved, rejected, etc.).
        reason: Human-readable explanation.
        metadata: Structured context for downstream consumers.
        timestamp: When the event occurred (UTC).
    """

    model_config = ConfigDict(frozen=True)

    record_id: str
    order_id: str
    event_type: str
    actor: str
    result: str
    reason: str = ""
    metadata: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_telemetry_dict(self) -> dict:
        """Serialize for dashboard/telemetry consumption."""
        return {
            "record_id": self.record_id,
            "order_id": self.order_id,
            "event_type": self.event_type,
            "actor": self.actor,
            "result": self.result,
            "reason": self.reason,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class AuditTrail:
    """Collects and exposes audit records for a trading session.

    Records are immutable once appended. The trail is in-memory by design
    — persistence to disk/dashboard is handled by the caller.
    """

    def __init__(self) -> None:
        self._records: list[AuditRecord] = []
        self._counter: int = 0

    def record(
        self,
        order_id: str,
        event_type: str,
        actor: str,
        result: str,
        reason: str = "",
        metadata: dict | None = None,
    ) -> AuditRecord:
        """Create and append an immutable audit record.

        Returns:
            The created AuditRecord (already appended to trail).
        """
        self._counter += 1
        entry = AuditRecord(
            record_id=f"audit-{self._counter:06d}",
            order_id=order_id,
            event_type=event_type,
            actor=actor,
            result=result,
            reason=reason,
            metadata=metadata or {},
        )
        self._records.append(entry)
        return entry

    def records_for_order(self, order_id: str) -> list[AuditRecord]:
        """Return all audit records for a specific order."""
        return [r for r in self._records if r.order_id == order_id]

    @property
    def all_records(self) -> list[AuditRecord]:
        """Return all records in insertion order (immutable view)."""
        return list(self._records)

    def to_telemetry(self) -> list[dict]:
        """Export all records as a list of dicts for dashboard/telemetry."""
        return [r.to_telemetry_dict() for r in self._records]

    def __len__(self) -> int:
        return len(self._records)
