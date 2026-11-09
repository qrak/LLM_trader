from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

INTENT_PENDING = "pending"
INTENT_UNKNOWN = "unknown"
INTENT_CONFIRMED = "confirmed"
INTENT_REFUSED = "refused"
INTENT_LOCAL_ONLY = "local_only"

INTENT_ACTION_ENTRY = "ENTRY"
INTENT_ACTION_UPDATE = "UPDATE"
INTENT_ACTION_CLOSE = "CLOSE"

RECONCILE_NO_POSITION = "no_position"
RECONCILE_OPEN_EXECUTOR_REPORTED = "open_executor_reported"
RECONCILE_EXIT_BOOKED = "exit_booked"
RECONCILE_DIVERGENCE = "divergence_unproven"
RECONCILE_UNVERIFIED = "unverified"

VERIFICATION_EXCHANGE_VERIFIED = "exchange_verified"
VERIFICATION_EXECUTOR_REPORTED = "executor_reported"
VERIFICATION_UNVERIFIED = "unverified"

SEVERITY_NEUTRAL = "neutral"
SEVERITY_INFO = "info"
SEVERITY_WARN = "warn"
SEVERITY_CRITICAL = "critical"
SEVERITY_OK = "ok"


class CycleState(StrEnum):
    """Where one analysis cycle is; the last terminal state survives the wait for the next candle."""

    ANALYZING = "analyzing"
    DECIDED = "decided"
    SKIPPED = "skipped"
    INTENT_SENT = "intent_sent"
    CONFIRMED = "confirmed"
    FAILED = "failed"

    def can_transition_to(self, target: CycleState) -> bool:
        return target in CYCLE_TRANSITIONS.get(self, ())


CYCLE_TRANSITIONS: dict[CycleState, frozenset[CycleState]] = {
    CycleState.ANALYZING: frozenset({CycleState.DECIDED, CycleState.SKIPPED, CycleState.FAILED}),
    CycleState.DECIDED: frozenset({CycleState.INTENT_SENT, CycleState.FAILED}),
    CycleState.INTENT_SENT: frozenset({CycleState.CONFIRMED, CycleState.FAILED}),
    CycleState.SKIPPED: frozenset(),
    CycleState.CONFIRMED: frozenset(),
    CycleState.FAILED: frozenset(),
}

INTENT_TRANSITIONS: dict[str, frozenset[str]] = {
    INTENT_PENDING: frozenset({INTENT_UNKNOWN, INTENT_CONFIRMED, INTENT_REFUSED}),
    INTENT_UNKNOWN: frozenset({INTENT_CONFIRMED, INTENT_REFUSED}),
    INTENT_CONFIRMED: frozenset({INTENT_UNKNOWN}),
    INTENT_REFUSED: frozenset(),
    INTENT_LOCAL_ONLY: frozenset(),
}

CYCLE_LABELS = {
    CycleState.ANALYZING: "Analyzing",
    CycleState.DECIDED: "Decided",
    CycleState.SKIPPED: "Skipped",
    CycleState.INTENT_SENT: "Intent sent — awaiting proof",
    CycleState.CONFIRMED: "Confirmed",
    CycleState.FAILED: "Failed",
}

CYCLE_SEVERITIES = {
    CycleState.ANALYZING: SEVERITY_INFO,
    CycleState.DECIDED: SEVERITY_INFO,
    CycleState.SKIPPED: SEVERITY_NEUTRAL,
    CycleState.INTENT_SENT: SEVERITY_WARN,
    CycleState.CONFIRMED: SEVERITY_OK,
    CycleState.FAILED: SEVERITY_CRITICAL,
}

SKIP_REASON_LABELS = {
    "analysis_error": "analysis error",
    "stale_position": "position changed during analysis",
    "protection_changed": "protection changed during analysis",
    "duplicate_intent": "duplicate intent",
    "approval_not_granted": "approval not granted",
    "forward_failed": "forward failed",
    "no_decision": "no actionable decision",
    "no_analysis": "no analysis payload",
    "executor_disabled": "executor disabled",
    "hold": "hold",
    "no_verdict": "no executor verdict yet",
    "forwarded": "forwarded, awaiting proof",
    "confirmed": "confirmed by the executor",
    "unknown": "executor did not confirm",
    "refused": "executor refused the command",
    "local_only": "executor confirmation unavailable",
}

INTENT_LABELS = {
    INTENT_PENDING: "Pending",
    INTENT_UNKNOWN: "Unknown",
    INTENT_CONFIRMED: "Confirmed",
    INTENT_REFUSED: "Refused",
    INTENT_LOCAL_ONLY: "Local only",
}

VERIFICATION_LABELS = {
    VERIFICATION_EXCHANGE_VERIFIED: "Exchange-verified",
    VERIFICATION_EXECUTOR_REPORTED: "Executor-reported",
    VERIFICATION_UNVERIFIED: "Unverified",
}

VERIFICATION_SEVERITIES = {
    VERIFICATION_EXCHANGE_VERIFIED: SEVERITY_OK,
    VERIFICATION_EXECUTOR_REPORTED: SEVERITY_INFO,
    VERIFICATION_UNVERIFIED: SEVERITY_WARN,
}


class PositionPhase(StrEnum):
    """Derived view of the bot's belief about one position; never persisted on its own."""

    FLAT = "flat"
    ENTERING = "entering"
    CLOSING = "closing"
    DIVERGENCE = "divergence"
    OPEN_VERIFIED = "open_verified"
    OPEN_REPORTED = "open_reported"
    OPEN_UNVERIFIED = "open_unverified"


POSITION_PHASE_LABELS = {
    PositionPhase.FLAT: "Flat",
    PositionPhase.ENTERING: "Entering — entry unconfirmed",
    PositionPhase.CLOSING: "Closing — fill not proven",
    PositionPhase.DIVERGENCE: "Divergence — local and exchange disagree",
    PositionPhase.OPEN_VERIFIED: "Open — exchange-verified",
    PositionPhase.OPEN_REPORTED: "Open — executor-reported",
    PositionPhase.OPEN_UNVERIFIED: "Open — unverified",
}

POSITION_PHASE_SEVERITIES = {
    PositionPhase.FLAT: SEVERITY_NEUTRAL,
    PositionPhase.ENTERING: SEVERITY_WARN,
    PositionPhase.CLOSING: SEVERITY_WARN,
    PositionPhase.DIVERGENCE: SEVERITY_CRITICAL,
    PositionPhase.OPEN_VERIFIED: SEVERITY_OK,
    PositionPhase.OPEN_REPORTED: SEVERITY_INFO,
    PositionPhase.OPEN_UNVERIFIED: SEVERITY_WARN,
}


class ExecutorQueryState(StrEnum):
    """What the last executor position evidence says; DISABLED is not the same as FLAT."""

    DISABLED = "disabled"
    OPEN = "open"
    FLAT = "flat"
    UNKNOWN = "unknown"


EXECUTOR_STATE_LABELS = {
    ExecutorQueryState.DISABLED: "Disabled",
    ExecutorQueryState.OPEN: "Open",
    ExecutorQueryState.FLAT: "Flat",
    ExecutorQueryState.UNKNOWN: "Unknown",
}


class BrainLifecycle(StrEnum):
    """Dashboard-visible state of the brain rebuild triggered by a closed trade."""

    IDLE = "idle"
    UPDATING = "updating"
    REBUILT = "rebuilt"
    ERROR = "error"


BRAIN_LIFECYCLE_LABELS = {
    BrainLifecycle.IDLE: "Brain idle",
    BrainLifecycle.UPDATING: "Brain updating",
    BrainLifecycle.REBUILT: "Brain rebuilt",
    BrainLifecycle.ERROR: "Brain error",
}

BRAIN_LIFECYCLE_SEVERITIES = {
    BrainLifecycle.IDLE: SEVERITY_NEUTRAL,
    BrainLifecycle.UPDATING: SEVERITY_INFO,
    BrainLifecycle.REBUILT: SEVERITY_OK,
    BrainLifecycle.ERROR: SEVERITY_CRITICAL,
}


class LifecycleProjection:
    """Stateless translation of raw state evidence into labels, severity and position phase."""

    @staticmethod
    def parse_timestamp(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    @staticmethod
    def age_seconds(value: Any, now: datetime) -> float | None:
        parsed = LifecycleProjection.parse_timestamp(value)
        if parsed is None:
            return None
        return max(0.0, (now - parsed).total_seconds())

    @staticmethod
    def cycle_label(state: Any, detail: str | None = None) -> str:
        if state is None or not str(state):
            return "Idle"
        try:
            normalized = CycleState(str(state))
        except ValueError:
            return "Unknown"
        label = CYCLE_LABELS[normalized]
        if normalized in (CycleState.SKIPPED, CycleState.DECIDED, CycleState.FAILED) and detail:
            label = f"{label} — {SKIP_REASON_LABELS.get(detail, detail)}"
        return label

    @staticmethod
    def cycle_severity(state: Any) -> str:
        if state is None or not str(state):
            return SEVERITY_NEUTRAL
        try:
            return CYCLE_SEVERITIES[CycleState(str(state))]
        except ValueError:
            return SEVERITY_NEUTRAL

    @staticmethod
    def intent_label(state: Any) -> str:
        return INTENT_LABELS.get(str(state or ""), "Unknown")

    @staticmethod
    def verification(
        *,
        reconcile_state: Any,
        checked_at: Any,
        exchange_verified_at: Any,
        last_reported_at: Any,
        detail: str | None,
        now: datetime,
        max_age_seconds: float | None,
    ) -> tuple[str, datetime | None, str | None]:
        state = str(reconcile_state) if reconcile_state is not None else None
        checked = LifecycleProjection.parse_timestamp(checked_at)
        verified = LifecycleProjection.parse_timestamp(exchange_verified_at)
        reported = LifecycleProjection.parse_timestamp(last_reported_at)

        if state is None:
            missing = "no exchange verification performed yet" if reported is None else detail
            return VERIFICATION_UNVERIFIED, reported, missing

        if state == RECONCILE_OPEN_EXECUTOR_REPORTED and verified is not None:
            age = (now - verified).total_seconds()
            if max_age_seconds is None or age <= max_age_seconds:
                return VERIFICATION_EXCHANGE_VERIFIED, verified, detail
            return VERIFICATION_UNVERIFIED, verified, "exchange verification is stale"

        if state == RECONCILE_OPEN_EXECUTOR_REPORTED and checked is not None:
            age = (now - checked).total_seconds()
            if max_age_seconds is None or age <= max_age_seconds:
                return VERIFICATION_EXECUTOR_REPORTED, checked, detail
            return (
                VERIFICATION_UNVERIFIED,
                checked,
                "executor tracker report is stale (no recent confirmation)",
            )

        if state in (None, RECONCILE_NO_POSITION) and reported is None:
            return VERIFICATION_UNVERIFIED, None, detail or "no exchange verification performed yet"
        return VERIFICATION_UNVERIFIED, reported, detail

    @staticmethod
    def verification_label(state: Any) -> str:
        return VERIFICATION_LABELS.get(str(state or ""), "Unverified")

    @staticmethod
    def verification_severity(state: Any) -> str:
        return VERIFICATION_SEVERITIES.get(str(state or ""), SEVERITY_WARN)

    @staticmethod
    def executor_state(executor_enabled: bool, reconcile_state: Any) -> ExecutorQueryState:
        if not executor_enabled:
            return ExecutorQueryState.DISABLED
        state = str(reconcile_state or "")
        if state in (RECONCILE_NO_POSITION, RECONCILE_EXIT_BOOKED):
            return ExecutorQueryState.FLAT
        if state == RECONCILE_OPEN_EXECUTOR_REPORTED:
            return ExecutorQueryState.OPEN
        return ExecutorQueryState.UNKNOWN

    @staticmethod
    def executor_state_label(state: Any) -> str:
        try:
            return EXECUTOR_STATE_LABELS[ExecutorQueryState(str(state or ""))]
        except ValueError:
            return "Unknown"

    @staticmethod
    def brain_label(state: Any) -> str:
        try:
            return BRAIN_LIFECYCLE_LABELS[BrainLifecycle(str(state or "idle"))]
        except ValueError:
            return "Brain state unknown"

    @staticmethod
    def brain_severity(state: Any) -> str:
        try:
            return BRAIN_LIFECYCLE_SEVERITIES[BrainLifecycle(str(state or "idle"))]
        except ValueError:
            return SEVERITY_NEUTRAL

    @staticmethod
    def position_phase(
        *,
        has_position: bool,
        intent_action: Any,
        intent_state: Any,
        reconcile_state: Any,
        verification: Any,
    ) -> PositionPhase:
        if not has_position:
            return PositionPhase.FLAT
        action = str(intent_action or "")
        state = str(intent_state or "")
        if action == INTENT_ACTION_CLOSE and state in (INTENT_PENDING, INTENT_UNKNOWN):
            return PositionPhase.CLOSING
        if action == INTENT_ACTION_ENTRY and state in (INTENT_PENDING, INTENT_UNKNOWN):
            return PositionPhase.ENTERING
        if action == INTENT_ACTION_ENTRY and state == INTENT_REFUSED:
            return PositionPhase.DIVERGENCE
        if action == INTENT_ACTION_CLOSE and state == INTENT_CONFIRMED:
            return PositionPhase.DIVERGENCE
        if str(reconcile_state or "") == RECONCILE_DIVERGENCE:
            return PositionPhase.DIVERGENCE
        if str(verification or "") == VERIFICATION_EXCHANGE_VERIFIED:
            return PositionPhase.OPEN_VERIFIED
        if str(verification or "") == VERIFICATION_EXECUTOR_REPORTED:
            return PositionPhase.OPEN_REPORTED
        return PositionPhase.OPEN_UNVERIFIED

    @staticmethod
    def position_phase_label(phase: Any) -> str:
        return POSITION_PHASE_LABELS.get(phase, "Unknown")

    @staticmethod
    def position_phase_severity(phase: Any) -> str:
        return POSITION_PHASE_SEVERITIES.get(phase, SEVERITY_NEUTRAL)
