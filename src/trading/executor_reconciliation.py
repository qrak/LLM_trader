

import asyncio
import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from .data_models import (
    FILL_DIVERGENCE_MARKER,
    FILL_VERIFICATION_DIVERGENT,
    FILL_VERIFICATION_UNAVAILABLE,
    FILL_VERIFICATION_VERIFIED,
    LONG_ENTRY_SIGNALS,
    MARKET_VERIFICATION_FIELDS,
    SHORT_ENTRY_SIGNALS,
    Position,
    TradeDecision,
    UnconfirmedIntentAlert,
    market_verification_is_verified,
)

ENTRY_CONFIRM_ATTEMPTS = 10
ENTRY_CONFIRM_DELAY = 2.5
ENTRY_CONFIRM_MIN_FALSE_REPORTS = 6

INTENT_CONFIRM_ATTEMPTS = 15
INTENT_CONFIRM_DELAY = 2.0

RECONCILE_ENTRY_GRACE_SECONDS = 120.0

UNKNOWN_EXIT_REASONS = frozenset({"", "unknown", "none", "null", "n/a", "na", "-"})

EXIT_EVIDENCE_KINDS = frozenset(
    {
        "exchange_order_fill",
        "exchange_trade_fills",
        "executor_receipt_fill",
    }
)
LEGACY_EXIT_EVIDENCE = "legacy_exit_record"

RECEIPT_EXIT_REASON = "close_confirmed_by_executor"

MAX_BOOKED_EXIT_EVENTS = 256
MAX_LEARNED_EXIT_EVENTS = 256

EXECUTOR_CONFIRMATION_SOURCES = frozenset(
    {
        "executor_verdict_journal",
        "executor_exit_journal",
        "executor_position_tracker",
    }
)

INTENT_PENDING = "pending"
INTENT_UNKNOWN = "unknown"
INTENT_CONFIRMED = "confirmed"
INTENT_REFUSED = "refused"
INTENT_LOCAL_ONLY = "local_only"

INTENT_ACTION_ENTRY = "ENTRY"
INTENT_ACTION_UPDATE = "UPDATE"
INTENT_ACTION_CLOSE = "CLOSE"

NON_BOOKED_INTENT_STATES = frozenset({INTENT_PENDING, INTENT_UNKNOWN})

PHANTOM_ENTRY_GRACE_SECONDS = 900
"""How long an unconfirmed ENTRY may still be in flight before it counts as never sent.

The post-forward check (``confirm_entry_with_executor``) fails OPEN after its short poll
window, so without a later re-check a local position whose entry never reached the
executor is held forever — a "STATE DIVERGENCE" write-out on every start. Fifteen minutes
is far beyond the executor's queue tick plus the confirmation polls.
"""

RECONCILE_NO_POSITION = "no_position"
RECONCILE_OPEN_EXECUTOR_REPORTED = "open_executor_reported"
RECONCILE_EXIT_BOOKED = "exit_booked"
RECONCILE_DIVERGENCE = "divergence_unproven"
RECONCILE_UNVERIFIED = "unverified"

EXECUTOR_TRACKER_EVIDENCE = "executor_position_tracker"
EXECUTOR_VENUE_EVIDENCE = "executor_venue_verification"

VERIFICATION_EXCHANGE_VERIFIED = "exchange_verified"
VERIFICATION_EXECUTOR_REPORTED = "executor_reported"
VERIFICATION_UNVERIFIED = "unverified"

LESSON_SKIPPED_NO_EVIDENCE = (
    "Lesson skipped: no exit evidence — post-mortem and the brain update were NOT "
    "run"
)
LESSON_SKIPPED_NO_POST_MORTEM = (
    "Lesson skipped: no confirmed post-mortem analysis — no brain entry was "
    "written"
)
LESSON_SKIPPED_ALREADY_LEARNED = (
    "Lesson skipped: this exit was already processed (idempotency) — no second "
    "brain entry was written"
)
LESSON_SKIPPED_FILL_DIVERGENCE = (
    "Lesson skipped: the venue fill price is not a price the real market made "
    "(sandbox/venue artifact) — booked for history, excluded from statistics and "
    "from the brain"
)











def quote_asset(symbol: Any) -> str | None:

    text = str(symbol or "")
    if "/" not in text:
        return None
    quote = text.rsplit("/", 1)[-1].strip()
    return quote.upper() or None








def position_identity(position: Position) -> str:

    return f"{position.symbol}|{position.entry_time.isoformat()}"


@dataclass
class PositionIntent:
    """One command the bot SENT, with the local state it is allowed to have.

    ``state`` is derived from executor evidence only (see
    ``EXECUTOR_CONFIRMATION_SOURCES``): ``pending`` while nothing came back,
    ``unknown`` when the source failed or gave no answer, ``confirmed``/``refused``
    when the executor's own journal/receipt said so, ``local_only`` when no command
    was sent at all (executor integration disabled). ``refused`` also covers the
    bot's OWN policy blocking the command before any send (``evidence="bot_policy"``):
    the command did not run, and it never reached the executor.
    """

    key: str
    action: str
    symbol: str
    state: str = INTENT_PENDING
    order_id: str | None = None
    position_id: str | None = None
    evidence: str | None = None
    detail: str | None = None
    created_at: str = ""
    updated_at: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def is_unbooked(self) -> bool:

        return self.state in NON_BOOKED_INTENT_STATES

    def to_dict(self) -> dict[str, Any]:

        return {
            "key": self.key,
            "action": self.action,
            "symbol": self.symbol,
            "state": self.state,
            "order_id": self.order_id,
            "position_id": self.position_id,
            "evidence": self.evidence,
            "detail": self.detail,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "payload": self.payload,
        }


class PositionIntentLedger:
    """Restart-safe store of position intents and booked exit events.

    Append-only JSONL: one line per intent write. ``_load`` replays the file and the
    LAST line per key wins, so a restart reconstructs the states instead of starting
    empty (that is what makes "the same confirmation twice books once" hold across a
    restart). ``event_id`` fields of confirmed lines are replayed into
    ``booked_events`` so an exit already booked before the restart is never booked
    again after it.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._intents: dict[str, PositionIntent] = {}
        self._order_index: dict[str, str] = {}
        self.booked_events: set[str] = set()
        self._event_positions: dict[str, str | None] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            if not self.path.exists():
                return
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            key = record.get("key")
            if not key:
                continue
            event_id = record.get("event_id") or (record.get("payload") or {}).get("event_id")
            if event_id and record.get("state") == INTENT_CONFIRMED:
                self.booked_events.add(str(event_id))
                self._event_positions[str(event_id)] = record.get("position_id") or (
                    record.get("payload") or {}
                ).get("position_id")
            if record.get("event_type") == "exit_event":
                continue
            intent = PositionIntent(
                key=str(key),
                action=str(record.get("action") or ""),
                symbol=str(record.get("symbol") or ""),
                state=str(record.get("state") or INTENT_PENDING),
                order_id=record.get("order_id"),
                position_id=record.get("position_id"),
                evidence=record.get("evidence"),
                detail=record.get("detail"),
                created_at=str(record.get("created_at") or ""),
                updated_at=record.get("updated_at"),
                payload=record.get("payload") or {},
            )
            self._intents[intent.key] = intent
            if intent.order_id:
                self._order_index[str(intent.order_id)] = intent.key

    def _append(self, record: dict[str, Any]) -> None:

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                import os as _os

                _os.fsync(handle.fileno())
        except OSError:
            return

    def record(
        self,
        action: str,
        symbol: str,
        *,
        key: str,
        order_id: str | None = None,
        position_id: str | None = None,
        state: str = INTENT_PENDING,
        evidence: str | None = None,
        detail: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> PositionIntent:

        self._load()
        now = datetime.now(timezone.utc).isoformat()
        existing = self._intents.get(key)
        intent = PositionIntent(
            key=key,
            action=action,
            symbol=symbol,
            state=state,
            order_id=order_id or (existing.order_id if existing else None),
            position_id=position_id or (existing.position_id if existing else None),
            evidence=evidence,
            detail=detail,
            created_at=existing.created_at if existing and existing.created_at else now,
            updated_at=now,
            payload=payload if payload is not None else (existing.payload if existing else {}),
        )
        self._intents[key] = intent
        if intent.order_id:
            self._order_index[str(intent.order_id)] = key
        self._append(intent.to_dict())
        return intent

    def get(self, key: str) -> PositionIntent | None:

        self._load()
        return self._intents.get(key)

    def get_by_order(self, order_id: str | None) -> PositionIntent | None:

        self._load()
        if not order_id:
            return None
        key = self._order_index.get(str(order_id))
        return self._intents.get(key) if key else None

    def pending(self) -> list[PositionIntent]:

        self._load()
        return [intent for intent in self._intents.values() if intent.is_unbooked]

    def recent(self, limit: int = 1) -> list[PositionIntent]:

        self._load()
        ordered = sorted(
            self._intents.values(),
            key=lambda intent: (intent.created_at or "", intent.updated_at or ""),
            reverse=True,
        )
        return ordered[: max(limit, 0)]

    def latest_for_position(self, position_id: str, action: str | None = None) -> PositionIntent | None:

        self._load()
        return max(
            (intent for intent in self._intents.values()
             if intent.position_id == position_id and (action is None or intent.action == action)),
            key=lambda intent: intent.created_at or "",
            default=None,
        )

    def mark_event_booked(self, event_id: str, position_id: str | None = None) -> None:

        self._load()
        self._event_positions.setdefault(event_id, position_id)
        if event_id in self.booked_events:
            return
        self.booked_events.add(event_id)
        self._append(
            {
                "event_type": "exit_event",
                "key": f"exit_event:{event_id}",
                "event_id": event_id,
                "position_id": position_id,
                "state": INTENT_CONFIRMED,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def is_event_booked(self, event_id: str, position_id: str | None = None) -> bool:

        self._load()
        if event_id not in self.booked_events:
            return False
        recorded = self._event_positions.get(event_id)
        if position_id is None or recorded is None:
            return True
        return str(recorded) == str(position_id)


@dataclass(frozen=True)
class VerifiedExit:
    """An exit record that passed validation and may be booked."""

    exit_price: float
    exit_reason: str
    event_id: str
    evidence: str
    position_id: str
    protection_order_id: str | None = None
    quantity: float | None = None
    exit_time: str | None = None
    fees: tuple[float, ...] | None = None
    fee_currency: str | None = None
    protection_version: str | None = None
    fill_verification: str | None = None
    reference_price: float | None = None
    deviation_pct: float | None = None
    market_verification: dict[str, Any] | None = None

    @property
    def exit_fee(self) -> float | None:

        if not self.fees:
            return None
        return math.fsum(self.fees)


@dataclass(frozen=True)
class ExitEvidence:
    """Provenance of a close the bot may LEARN from.

    Wave 5: a booked CLOSE is not automatically a lesson. The trading brain may only
    be fed by a close whose EXIT FACT is proven — a real fill price AND a real filled
    quantity, from a source the evidence semaphore accepts
    (:data:`EXECUTOR_CONFIRMATION_SOURCES`) and carrying a stable event id so the same
    exit can never be learned twice. Everything else is a divergence for a human, not
    training material (the 2026-09-21 incident wrote a lesson the model itself
    produced from an unconfirmed closure).

    ``source`` is a provenance LABEL, never a venue confirmation: it says which
    executor artefact the fact came from, not that the venue was queried.
    """

    source: str
    price: float | None = None
    quantity: float | None = None
    event_id: str | None = None
    protection_version: str | None = None
    fee: float | None = None
    fee_currency: str | None = None
    fill_verification: str | None = None
    reference_price: float | None = None
    deviation_pct: float | None = None
    market_verification: dict[str, Any] | None = None

    @property
    def identity(self) -> str:

        return str(self.event_id or "")

    def to_dict(self) -> dict[str, Any]:

        return {
            "source": self.source,
            "price": self.price,
            "quantity": self.quantity,
            "event_id": self.event_id,
            "protection_version": self.protection_version,
            "fee": self.fee,
            "fee_currency": self.fee_currency,
            "fill_verification": self.fill_verification,
            "reference_price": self.reference_price,
            "deviation_pct": self.deviation_pct,
            "market_verification": self.market_verification,
        }


class ExitRecords:
    """Verification and construction of exit records: fills, fees, evidence and identity.

    The executor reports what the venue did; this class decides whether that report is
    learnable, divergent or unusable, and builds the record the bot books. It is stateless:
    every method reads the plain payload it is handed, so a recorded row can be replayed.
    """

    @staticmethod
    def verify(record: Any) -> tuple[VerifiedExit | None, str | None]:

        if not isinstance(record, dict):
            return None, "exit record is not an object"
        if not record:
            return None, "exit record is empty"

        reason = str(record.get("exit_reason") or "").strip()
        if reason.lower() in UNKNOWN_EXIT_REASONS:
            return None, f"exit reason is not usable ({record.get('exit_reason')!r})"

        raw_evidence = record.get("evidence")
        if raw_evidence is None:
            evidence = LEGACY_EXIT_EVIDENCE
        else:
            evidence = str(raw_evidence).strip()
            if evidence not in EXIT_EVIDENCE_KINDS:
                return None, f"unrecognized evidence kind {raw_evidence!r}"

        raw_average = record.get("average_price")
        if raw_average is not None:
            exit_price = ExitRecords._positive_float(raw_average)
            if exit_price is None:
                return None, f"average price is not usable ({raw_average!r})"
        else:
            exit_price = ExitRecords._positive_float(record.get("exit_price"))
            if exit_price is None:
                return None, "exit price unknown/invalid — booking refused (entry price is not a fill)"

        quantity = None
        raw_quantity = record.get("filled_quantity")
        if raw_quantity is None:
            raw_quantity = record.get("quantity")
        if raw_quantity is None:
            return None, "exit quantity unknown — booking refused (the position size is not a fill)"
        quantity = ExitRecords._positive_float(raw_quantity)
        if quantity is None:
            return None, f"filled quantity is not usable ({raw_quantity!r})"

        raw_remaining = record.get("remaining_quantity")
        if raw_remaining is not None:
            remaining = ExitRecords._non_negative_float(raw_remaining)
            if remaining is None:
                return None, f"remaining quantity is not usable ({raw_remaining!r})"
            if remaining > 0:
                return None, f"exit is partial (remaining_quantity={raw_remaining}) — not a final exit"

        protection_order_id = record.get("protection_order_id") or record.get("exchange_order_id")
        position_id = record.get("position_id") or record.get("entry_order_id") or ""
        return (
            VerifiedExit(
                exit_price=exit_price,
                exit_reason=reason,
                event_id=ExitRecords.identity(record, str(position_id)),
                evidence=evidence,
                position_id=str(position_id),
                protection_order_id=str(protection_order_id) if protection_order_id else None,
                quantity=quantity,
                exit_time=str(record.get("timestamp")) if record.get("timestamp") else None,
                fees=ExitRecords._actual_fees(record),
                fee_currency=ExitRecords._fee_currency(record),
                protection_version=(
                    str(record["protection_version"])
                    if record.get("protection_version") is not None
                    else None
                ),
                fill_verification=(
                    str(record["fill_verification"])
                    if record.get("fill_verification") is not None
                    else None
                ),
                market_verification={key: record.get(key) for key in MARKET_VERIFICATION_FIELDS},
                reference_price=ExitRecords._positive_float(record.get("reference_price")),
                deviation_pct=(
                    record.get("deviation_pct")
                    if isinstance(record.get("deviation_pct"), (int, float))
                    else None
                ),
            ),
            None,
        )

    @staticmethod
    def from_receipt(receipt: Any, *, symbol: str | None = None) -> tuple[dict | None, str]:

        if not isinstance(receipt, dict) or not receipt:
            return None, "no executor receipt for this command"
        if str(receipt.get("verdict") or "").strip().lower() != "executed":
            return None, f"executor verdict is not 'executed' ({receipt.get('verdict')!r})"
        action = str(receipt.get("action") or receipt.get("signal") or "").strip().upper()
        if not action.startswith("CLOSE"):
            return None, f"receipt is not a CLOSE command ({action or 'unknown'})"
        state = str(receipt.get("state") or "").strip().lower()
        if state != "filled":
            return None, (
                f"close is not fully filled (state={state or 'unknown'}) — a partial exit is "
                f"not a final exit"
            )
        if receipt.get("close_confirmed") is not True:
            return None, "executor did not confirm the close by fill evidence"
        if symbol and receipt.get("symbol") and str(receipt["symbol"]) != str(symbol):
            return None, f"receipt is for {receipt['symbol']!r}, not {symbol!r}"
        price = ExitRecords._positive_float(receipt.get("average_price"))
        if price is None:
            return None, f"average price unknown/invalid ({receipt.get('average_price')!r})"
        quantity = ExitRecords._positive_float(receipt.get("filled_quantity"))
        if quantity is None:
            return None, f"filled quantity unknown/invalid ({receipt.get('filled_quantity')!r})"
        fees_known = receipt.get("fees_known")
        fees = receipt.get("fees")
        if fees_known is not True or not isinstance(fees, (list, tuple)):
            fees = []
        identity = receipt.get("command_id") or receipt.get("order_id")
        record: dict[str, Any] = {
            "schema_version": 2,
            "event_id": f"receipt:{identity}",
            "symbol": receipt.get("symbol") or symbol,
            "exit_price": price,
            "exit_reason": RECEIPT_EXIT_REASON,
            "filled_quantity": quantity,
            "fees": list(fees),
            "evidence": "executor_receipt_fill",
            "timestamp": receipt.get("timestamp"),
            "position_id": receipt.get("position_id"),
            "protection_version": receipt.get("protection_version"),
            "receipt_fill_evidence": receipt.get("fill_evidence"),
            "exchange_order_id": receipt.get("exchange_order_id"),
            "source": "executor_verdict_journal",
            **{key: receipt.get(key) for key in MARKET_VERIFICATION_FIELDS},
        }
        verified, rejection = ExitRecords.verify(record)
        if verified is None:
            return None, f"receipt fill evidence is not usable ({rejection})"
        return record, ""

    @staticmethod
    def identity(record: dict, position_id: str) -> str:

        explicit = record.get("event_id")
        if explicit:
            return f"event:{explicit}"
        order_id = (
            record.get("protection_order_id")
            or record.get("exchange_order_id")
            or record.get("order_id")
            or ""
        )
        fill_ids = record.get("fill_ids") or []
        if isinstance(fill_ids, (str, bytes)):
            fill_ids = [fill_ids]
        material = "|".join(
            [
                position_id,
                str(order_id),
                ",".join(sorted(str(fill) for fill in fill_ids)),
                str(record.get("timestamp") or ""),
            ]
        )
        return "compat:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def is_learnable(evidence: Any) -> tuple[bool, str]:

        if evidence is None:
            return False, "no exit evidence was recorded for this close"
        if isinstance(evidence, dict):
            evidence = ExitEvidence(
                source=str(evidence.get("source") or ""),
                price=ExitRecords._positive_float(evidence.get("price")),
                quantity=ExitRecords._positive_float(evidence.get("quantity")),
                event_id=evidence.get("event_id"),
                protection_version=evidence.get("protection_version"),
                fee=ExitRecords._non_negative_float(evidence.get("fee")),
                fee_currency=evidence.get("fee_currency"),
                fill_verification=evidence.get("fill_verification"),
                reference_price=ExitRecords._positive_float(evidence.get("reference_price")),
                deviation_pct=evidence.get("deviation_pct"),
                market_verification=evidence.get("market_verification"),
            )
        if not isinstance(evidence, ExitEvidence):
            return False, f"unrecognized evidence object ({type(evidence).__name__})"
        if ExitRecords.fill_is_divergent(evidence.fill_verification):
            return False, ExitRecords.divergence_reason(evidence)
        if evidence.fill_verification != FILL_VERIFICATION_VERIFIED or not market_verification_is_verified(evidence.market_verification):
            return False, ExitRecords.verification_gap_reason({
                **(evidence.market_verification or {}),
                "fill_verification": evidence.fill_verification,
            })
        source = str(evidence.source or "")
        if source not in EXECUTOR_CONFIRMATION_SOURCES:
            return False, f"source {source!r} is not a confirming source"
        price = ExitRecords._positive_float(evidence.price)
        if price is None:
            return False, f"exit price unknown/invalid ({evidence.price!r})"
        quantity = ExitRecords._positive_float(evidence.quantity)
        if quantity is None:
            return False, f"filled quantity unknown/invalid ({evidence.quantity!r})"
        if not evidence.identity:
            return False, "no stable exit event id — the lesson could not be deduplicated"
        return True, ""

    @staticmethod
    def divergence_reason(evidence: Any) -> str:

        reference = ExitRecords._positive_float(getattr(evidence, "reference_price", None))
        deviation = getattr(evidence, "deviation_pct", None)
        price = ExitRecords._positive_float(getattr(evidence, "price", None))
        measured = ""
        if reference is not None and price is not None:
            measured = f" — the venue filled at {price:,.2f} while the real market traded at {reference:,.2f}"
        crossed = f" ({deviation:+.2f}%)" if isinstance(deviation, (int, float)) else ""
        return (
            f"the venue's fill price is not a price the real market made{measured}{crossed}: "
            f"an exchange-side print the market never made cannot become a lesson, a statistic "
            f"or a chart level"
        )

    @staticmethod
    def verification_gap_reason(quality: Any) -> str:

        status = ExitRecords.verification_status(quality)
        window = quality.get("reference_window_seconds") if isinstance(quality, dict) else None
        span = f"{window:g}s " if isinstance(window, (int, float)) and not isinstance(window, bool) else ""
        if status == FILL_VERIFICATION_UNAVAILABLE:
            return (
                f"no public print on the real market in the {span}reference window before the "
                f"fill — the execution price could not be market-checked"
            )
        if status == FILL_VERIFICATION_VERIFIED:
            return (
                "the fill is marked verified but its reference print is not contemporaneous "
                "with the fill"
            )
        if status == "disabled":
            return "the fill-vs-market check is switched off (FILL_DIVERGENCE_MAX_PCT=0)"
        return f"the fill carries no market check at all ({status or 'missing'})"

    @staticmethod
    def fill_is_divergent(value: Any) -> bool:

        return str(value or "").strip().lower() == FILL_VERIFICATION_DIVERGENT

    @staticmethod
    def close_fill_is_divergent(entry_quality: Any, evidence: Any) -> bool:

        if ExitRecords.verification_status(entry_quality) == FILL_VERIFICATION_DIVERGENT:
            return True
        if isinstance(evidence, dict):
            return ExitRecords.fill_is_divergent(evidence.get("fill_verification"))
        return ExitRecords.fill_is_divergent(getattr(evidence, "fill_verification", None))

    @staticmethod
    def verification_status(quality: Any) -> str:

        if isinstance(quality, str):
            try:
                quality = json.loads(quality)
            except (ValueError, TypeError):
                return ""
        if not isinstance(quality, dict):
            return ""
        return str(quality.get("fill_verification") or "").strip().lower()

    @staticmethod
    def _parse_exit_time(raw: Any) -> datetime | None:

        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(str(raw))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    @staticmethod
    def _positive_float(raw: Any) -> float | None:

        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) and value > 0 else None

    @staticmethod
    def _non_negative_float(raw: Any) -> float | None:

        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) and value >= 0 else None

    @staticmethod
    def _quantity_matches_exchange_amount(reported: Any, local: Any) -> bool:

        reported_value = ExitRecords._positive_float(reported)
        local_value = ExitRecords._positive_float(local)
        if reported_value is None or local_value is None:
            return False
        exponent = Decimal(str(reported_value)).as_tuple().exponent
        if not isinstance(exponent, int):
            return False
        scale = 10 ** -min(exponent, 0)
        units = math.floor(Decimal(str(local_value)) * scale)
        return units == Decimal(str(reported_value)) * scale

    @staticmethod
    def _fee_amount(item: Any, quote: str | None) -> float | None:

        if isinstance(item, bool) or item is None:
            return None
        if isinstance(item, dict):
            currency = item.get("currency", item.get("fee_currency"))
            if currency is None:
                return None
            if quote is None or str(currency).strip().upper() != quote:
                return None
            amount = item.get("cost")
            if amount is None:
                amount = item.get("amount")
            if amount is None:
                amount = item.get("fee")
            return ExitRecords._non_negative_float(amount)
        return ExitRecords._non_negative_float(item)

    @staticmethod
    def _actual_fees(record: dict) -> tuple[float, ...] | None:

        raw = record.get("fees")
        quote = quote_asset(record.get("symbol"))
        collected: list[float] = []
        if isinstance(raw, (list, tuple)):
            for item in raw:
                value = ExitRecords._fee_amount(item, quote)
                if value is None:
                    return None
                collected.append(value)
            if collected:
                return tuple(collected)
        elif isinstance(raw, dict):
            value = ExitRecords._fee_amount(raw, quote)
            return None if value is None else (value,)
        elif isinstance(raw, bool):
            return None
        elif raw is not None:
            value = ExitRecords._non_negative_float(raw)
            if value is None:
                return None
            return (value,)

        tagged = record.get("fee_currency", record.get("currency"))
        if tagged is not None and (quote is None or str(tagged).strip().upper() != quote):
            return None
        for key in ("total_fee", "fee"):
            value = ExitRecords._non_negative_float(record.get(key))
            if value is not None:
                return (value,)
        return None

    @staticmethod
    def _fee_currency(record: dict) -> str | None:

        raw = record.get("fees")
        cur = quote_asset(record.get("symbol"))
        if isinstance(raw, dict):
            value = raw.get("currency", raw.get("fee_currency"))
            return str(value) if value is not None else cur
        if isinstance(raw, (list, tuple)) and raw and all(isinstance(item, dict) for item in raw):
            seen = {str(item.get("currency", item.get("fee_currency"))) for item in raw}
            if len(seen) == 1:
                return seen.pop()
            return None
        tagged = record.get("fee_currency", record.get("currency"))
        return str(tagged) if tagged is not None else None












@dataclass(frozen=True)
class LocalPositionReconciliation:
    """Result of one local-position vs. exchange reconciliation."""

    state: str
    symbol: str | None
    checked_at: datetime
    detail: str | None = None
    exit_booked: bool = False
    exit_event_id: str | None = None
    position_id: str | None = None
    snapshot_token: str = "flat"
    source: str = "periodic"
    evidence: str = "none"
    exchange_verified_at: datetime | None = None







class ExecutorReconciliationMixin:
    """Executor position verification, verdict correlation and ghost rollback."""

    logger: Any
    config: Any
    persistence: Any
    memory_service: Any
    current_position: Position | None
    _http_client: Any
    _last_executor_position_payload: dict | None
    _record_trade_decision: Any
    close_position: Any
    _conditions: Any
    _state_divergence: str | None
    _executor_side_exit_reason: str | None
    _position_transition_lock: asyncio.Lock | None
    _booked_exit_events: set[str]
    _position_intent_ledger: PositionIntentLedger | None
    _unconfirmed_intent_alert: UnconfirmedIntentAlert | None
    _unconfirmed_intent_alert_at: datetime | None
    _rejected_intent_alert: str | None
    _last_reconciliation: LocalPositionReconciliation | None

    def _position_lock(self) -> asyncio.Lock:

        if self._position_transition_lock is None:
            self._position_transition_lock = asyncio.Lock()
        return self._position_transition_lock

    def _booked_exit_event_ids(self) -> set[str]:

        return self._booked_exit_events


    def _intent_journal_path(self) -> Path:

        return Path(self.config.BOT_INTENT_JOURNAL_PATH)

    def position_intents(self) -> PositionIntentLedger:

        if self._position_intent_ledger is None:
            self._position_intent_ledger = PositionIntentLedger(self._intent_journal_path())
        return self._position_intent_ledger

    def intent_identity(self, action: str, symbol: str, *, order_id: str | None = None,
                        position_id: str | None = None) -> str:

        if action == INTENT_ACTION_CLOSE and position_id:
            return f"close:{position_id}"
        if action == INTENT_ACTION_UPDATE and position_id:
            return f"update:{position_id}:{order_id or 'no-order'}"
        return f"{action.lower()}:{symbol}:{order_id or position_id or 'no-order'}"

    def register_position_intent(
        self,
        action: str,
        symbol: str,
        *,
        order_id: str | None = None,
        position_id: str | None = None,
        state: str = INTENT_PENDING,
        detail: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> PositionIntent:

        intent = self.position_intents().record(
            action,
            symbol,
            key=self.intent_identity(action, symbol, order_id=order_id, position_id=position_id),
            order_id=order_id,
            position_id=position_id,
            state=state,
            detail=detail,
            payload=payload,
        )
        self.logger.info(
            "Position intent %s (%s %s) state=%s — local state unchanged until the "
            "executor confirms it (order_id=%s)",
            intent.key, action, symbol, intent.state, order_id,
        )
        return intent

    def _assert_confirmation_source(self, source: str) -> bool:

        if source in EXECUTOR_CONFIRMATION_SOURCES:
            return True
        self.logger.critical(
            "REFUSED confirmation from non-executor source %r — a local state change "
            "needs executor evidence (%s). The local position stays as it is.",
            source, sorted(EXECUTOR_CONFIRMATION_SOURCES),
        )
        return False

    def _resolve_intent(
        self,
        key: str,
        state: str,
        *,
        source: str,
        detail: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> PositionIntent | None:

        if not self._assert_confirmation_source(source):
            return None
        ledger = self.position_intents()
        current = ledger.get(key)
        if current is None:
            return None
        if current.state == state:
            return current
        intent = ledger.record(
            current.action,
            current.symbol,
            key=key,
            order_id=current.order_id,
            position_id=current.position_id,
            state=state,
            evidence=source,
            detail=detail or current.detail,
            payload=payload if payload is not None else current.payload,
        )
        log = self.logger.warning if state == INTENT_UNKNOWN else self.logger.info
        log(
            "Position intent %s -> %s (evidence=%s, detail=%s)",
            key, state, source, detail or "",
        )
        return intent

    def confirm_position_intent(self, key: str, *, source: str, detail: str | None = None,
                                payload: dict[str, Any] | None = None) -> PositionIntent | None:

        return self._resolve_intent(key, INTENT_CONFIRMED, source=source, detail=detail, payload=payload)

    def refuse_position_intent(self, key: str, *, source: str, detail: str | None = None) -> PositionIntent | None:

        return self._resolve_intent(key, INTENT_REFUSED, source=source, detail=detail)

    def mark_position_intent_unknown(self, key: str, *, source: str, detail: str | None = None) -> PositionIntent | None:

        if not self._assert_confirmation_source(source):
            return None
        intent = self._resolve_intent(key, INTENT_UNKNOWN, source=source, detail=detail)
        if intent is None:
            return None
        self._raise_unconfirmed_intent_alert(
            f"{intent.action} {intent.symbol} ({intent.key}) was not confirmed by "
            f"the executor ({source}: {detail or 'no reply'}) — local state: "
            f"{INTENT_UNKNOWN}, local position unchanged",
            intent.key,
        )
        return intent

    def _raise_unconfirmed_intent_alert(self, message: str, intent_key: str | None) -> None:

        self._unconfirmed_intent_alert = UnconfirmedIntentAlert(
            message=message,
            intent_key=intent_key,
            created_at=datetime.now(timezone.utc),
        )
        self._unconfirmed_intent_alert_at = None

    def take_unconfirmed_intent_alert(self) -> str | None:

        alert = self._unconfirmed_intent_alert
        self._unconfirmed_intent_alert = None
        self._unconfirmed_intent_alert_at = None
        if alert is None:
            return None
        if self._intent_is_already_confirmed(alert.intent_key):
            self.logger.debug(
                "Dropped an unconfirmed-intent alert for %s: the intent was CONFIRMED, so "
                "the warning describes a command that already ended",
                alert.intent_key,
            )
            return None
        self._unconfirmed_intent_alert_at = alert.created_at
        return alert.message

    def _intent_is_already_confirmed(self, intent_key: str | None) -> bool:

        if intent_key is None:
            return False
        intent = self.position_intents().get(intent_key)
        return intent is not None and intent.state == INTENT_CONFIRMED

    def unconfirmed_intent_alert_captured_at(self) -> datetime | None:

        return self._unconfirmed_intent_alert_at

    def take_rejected_intent_alert(self) -> str | None:

        message = self._rejected_intent_alert
        self._rejected_intent_alert = None
        return message

    def position_snapshot_token(self) -> str:

        position = self.current_position
        if position is None:
            return "flat"
        direction = position.direction
        return f"open:{position.symbol}:{direction}:{position.entry_time.isoformat()}"

    def protection_snapshot(self) -> tuple[float | None, float | None] | None:

        position = self.current_position
        if position is None:
            return None
        return (position.stop_loss, position.take_profit)

    def protection_loosening_reason(self, decision: TradeDecision | None) -> str | None:

        if decision is None:
            return None
        if decision.action != "UPDATE":
            return None
        position = self.current_position
        if position is None:
            return None
        proposed = decision.stop_loss
        current = position.stop_loss
        if proposed is None or current is None:
            return None
        direction = position.direction.upper()
        widens = (
            (direction == "LONG" and proposed < current)
            or (direction == "SHORT" and proposed > current)
        )
        if not widens:
            return None
        return (
            f"UPDATE loosens protection ({direction} SL {current:,.2f} -> "
            f"{proposed:,.2f}) — refused: the stop may not move away from price on a "
            f"decision whose protection snapshot is not versioned"
        )

    def _get_http_client(self):

        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=3.0)
        return self._http_client

    async def close(self) -> None:

        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    def _executor_query_available(self) -> bool:

        return self.config.EXECUTOR_API_ENABLED and bool(self.config.EXECUTOR_API_URL)

    async def _executor_has_position(self, symbol: str) -> bool | None:

        self._last_executor_position_payload = None
        if not self.config.EXECUTOR_API_ENABLED:
            return True
        url: str = self.config.EXECUTOR_API_URL
        if not url:
            return True
        try:
            base = url.rstrip("/").removesuffix("/decision")
            pos_url = base + "/position"
            client = self._get_http_client()
            resp = await client.get(pos_url, params={"symbol": symbol})
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    self._last_executor_position_payload = data
                    return bool(data.get("open", False))
                return None
            self.logger.warning(
                "Executor returned HTTP %s for position query: %s",
                resp.status_code, resp.text,
            )
            return None
        except httpx.TransportError as exc:
            self.logger.warning(
                "Executor unavailable for %s (%s) — position state remains unverified; "
                "local position unchanged.",
                symbol, type(exc).__name__,
            )
            return None
        except Exception:
            self.logger.error(# noqa: G201
                "CRITICAL: Failed to query executor position for %s — "
                "cannot verify position state.",
                symbol, exc_info=True,
            )
            return None

    @staticmethod
    def _executor_exchange_verified(
        payload: dict | None,
        symbol: str,
        position: Position,
    ) -> bool:

        if not isinstance(payload, dict):
            return False
        positions = payload.get("positions")
        if payload.get("symbol") == symbol:
            reported = payload
        else:
            reported = positions.get(symbol) if isinstance(positions, dict) else None
        if not isinstance(reported, dict):
            return False
        confirmation = reported.get("confirmation")
        if not isinstance(confirmation, dict):
            return False
        reported_quantity = ExitRecords._positive_float(reported.get("quantity"))
        reported_entry = ExitRecords._positive_float(reported.get("entry_price"))
        if reported_quantity is None or reported_entry is None:
            return False
        quantity_matches = ExitRecords._quantity_matches_exchange_amount(reported_quantity, position.size)
        entry_matches = math.isclose(
            reported_entry,
            float(position.entry_price),
            rel_tol=1e-6,
            abs_tol=1e-8,
        )
        return (
            reported.get("symbol") == symbol
            and str(reported.get("side") or "").lower() == position.direction.lower()
            and quantity_matches
            and entry_matches
            and bool(reported.get("sl_order_id"))
            and bool(reported.get("tp_order_id"))
            and confirmation.get("state") == "confirmed"
            and confirmation.get("confirmed") is True
            and confirmation.get("reconcile_status") == "verified"
            and confirmation.get("protection_status") == "active"
            and confirmation.get("protection_verified") is not False
            and confirmation.get("protection_unresolved") is False
            and confirmation.get("protection_duplicated") is False
        )

    async def executor_protection_version(self, symbol: str | None = None) -> str | None:

        payload = await self._executor_position_payload(symbol)
        if not isinstance(payload, dict):
            return None
        for key in ("protection_version", "protection_revision"):
            value = payload.get(key)
            if value is not None:
                text = str(value).strip()
                if text:
                    return text
        return None

    async def _executor_position_payload(self, symbol: str | None = None) -> dict | None:

        if not self.config.EXECUTOR_API_ENABLED:
            return None
        url = self.config.EXECUTOR_API_URL
        if not url:
            return None
        try:
            base = url.rstrip("/").removesuffix("/decision")
            position = self.current_position
            target = symbol or (position.symbol if position is not None else None)
            resp = await self._get_http_client().get(
                base + "/position", params={"symbol": target} if target else None
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            return data if isinstance(data, dict) else None
        except Exception:
            self.logger.debug(
                "Executor /position payload unavailable for %s", symbol, exc_info=True
            )
            return None

    async def confirm_entry_with_executor(self, symbol: str, order_id: str | None = None) -> bool:

        if not order_id:
            false_reports = 0
            polls = 0
            for _ in range(ENTRY_CONFIRM_ATTEMPTS):
                polls += 1
                state = await self._executor_has_position(symbol)
                if state is True:
                    return True
                if state is False:
                    false_reports += 1
                    if false_reports >= ENTRY_CONFIRM_MIN_FALSE_REPORTS:
                        break
                await asyncio.sleep(ENTRY_CONFIRM_DELAY)
            if false_reports >= ENTRY_CONFIRM_MIN_FALSE_REPORTS:
                self.logger.warning(
                    "Executor reports no position for %s after %d polls — entry was likely blocked",
                    symbol, polls,
                )
                return False
            self.logger.warning(
                "Could not verify executor position for %s after %d polls — "
                "keeping local position (fail-open)",
                symbol, polls,
            )
            return True

        for _ in range(ENTRY_CONFIRM_ATTEMPTS):
            verdict_entry = self._read_executor_verdict_entry(order_id)
            verdict = str((verdict_entry or {}).get("verdict") or "").strip().lower()
            if verdict == "executed" or self._entry_fill_is_proven(verdict_entry):
                if verdict != "executed":
                    self.logger.critical(
                        "Executor reported %s for %s after filling the entry — keeping the "
                        "real position; protection requires attention",
                        verdict, order_id,
                    )
                return True
            if self._entry_fill_is_unresolved(verdict_entry):
                self.logger.critical(
                    "Executor entry %s has incomplete fill evidence — keeping local position",
                    order_id,
                )
                return True
            if verdict in ("blocked", "error"):
                self.logger.warning(
                    "Executor verdict for %s: %s — entry was %s",
                    order_id, verdict,
                    "blocked" if verdict == "blocked" else "rejected with error",
                )
                return False
            await asyncio.sleep(ENTRY_CONFIRM_DELAY)

        self.logger.warning(
            "No executor verdict for %s after %d polls — keeping local position (fail-open)",
            order_id, ENTRY_CONFIRM_ATTEMPTS,
        )
        return True

    def _unconfirmed_entry_for(self, position: Position) -> PositionIntent | None:

        identity = position_identity(position)
        for intent in self.position_intents().pending():
            if intent.action == INTENT_ACTION_ENTRY and intent.position_id == identity:
                return intent
        return None

    def _entry_never_reached_the_executor(self, intent: PositionIntent) -> bool:

        if not self.config.EXECUTOR_API_ENABLED:
            return False
        if not self._executor_verdict_path().exists():
            return False
        order_id = intent.order_id or ""
        if not order_id or self._read_executor_verdict(order_id) is not None:
            return False
        try:
            created_at = datetime.fromisoformat(intent.created_at)
        except (TypeError, ValueError):
            return False
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - created_at).total_seconds() >= PHANTOM_ENTRY_GRACE_SECONDS

    def _read_executor_verdict(self, order_id: str) -> str | None:

        path = self._executor_verdict_path()
        try:
            if not path.exists():
                return None
            for line in reversed(path.read_text(encoding="utf-8").splitlines()):
                entry = json.loads(line)
                if entry.get("order_id") == order_id:
                    return entry.get("verdict")
        except (OSError, json.JSONDecodeError):
            self.logger.warning(
                "Failed to read executor verdict journal at %s", path,
            )
        return None

    def _executor_verdict_path(self) -> Path:

        configured = self.config.EXECUTOR_VERDICT_PATH
        if configured:
            return Path(configured)
        return Path("data/trading/executor_verdicts.jsonl")

    def _read_executor_verdict_entry(self, order_id: str) -> dict | None:

        path = self._executor_verdict_path()
        try:
            if not path.exists():
                return None
            for line in reversed(path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("order_id") == order_id:
                    return entry
        except (OSError, json.JSONDecodeError):
            self.logger.warning("Failed to read executor verdict journal at %s", path)
        return None

    @staticmethod
    def _entry_fill_is_proven(entry: dict | None) -> bool:

        if not entry or not entry.get("exposure_possible"):
            return False
        if str(entry.get("state") or "").lower() not in {"filled", "partially_filled"}:
            return False
        value: Any = entry.get("filled_quantity")
        if value is None:
            return False
        try:
            quantity = float(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(quantity) and quantity > 0

    @classmethod
    def _entry_fill_is_unresolved(cls, entry: dict | None) -> bool:

        return bool(
            entry
            and entry.get("exposure_possible")
            and str(entry.get("state") or "").lower() in {"filled", "partially_filled"}
            and not cls._entry_fill_is_proven(entry)
        )

    async def resolve_position_intents_after_forward(
        self,
        order_id: str | None,
        delivered: bool,
        symbol: str | None = None,
    ) -> str | None:

        ledger = self.position_intents()
        intent = ledger.get_by_order(order_id)
        if intent is None:
            return None
        return await self._resolve_forwarded_intent(intent, delivered=delivered)

    async def _resolve_forwarded_intent(
        self, intent: PositionIntent, *, delivered: bool, verdict_entry: dict | None = None,
    ) -> str:

        if intent.state not in NON_BOOKED_INTENT_STATES:
            return intent.state

        if not delivered:
            self.mark_position_intent_unknown(
                intent.key,
                source="executor_verdict_journal",
                detail=(
                    "the command was not delivered to the executor (file fallback / no "
                    "reply) — execution unknown, local state unchanged"
                ),
            )
            return INTENT_UNKNOWN

        if verdict_entry is None:
            verdict_entry = await self._await_forwarded_verdict(intent)
        verdict = str((verdict_entry or {}).get("verdict") or "").strip().lower()

        if intent.action == INTENT_ACTION_UPDATE and verdict in ("executed", "blocked", "error"):
            return await self._settle_update_intent(intent, verdict_entry or {})

        position = self.current_position
        if intent.action == INTENT_ACTION_CLOSE and position is not None and (
            intent.position_id != position_identity(position) or intent.symbol != position.symbol
        ):
            return intent.state

        if verdict == "executed" or (
            intent.action == INTENT_ACTION_ENTRY
            and self._entry_fill_is_proven(verdict_entry)
        ):
            return await self._apply_confirmed_intent(intent, verdict_entry)

        if intent.action == INTENT_ACTION_ENTRY and self._entry_fill_is_unresolved(verdict_entry):
            self.mark_position_intent_unknown(
                intent.key,
                source="executor_verdict_journal",
                detail="executor reports entry exposure but fill evidence is incomplete — local position kept",
            )
            return INTENT_UNKNOWN

        if verdict in ("blocked", "error"):
            self.refuse_position_intent(
                intent.key,
                source="executor_verdict_journal",
                detail=f"executor receipt: {verdict} ({(verdict_entry or {}).get('reason') or ''})",
            )
            self._apply_refused_intent(intent, verdict)
            return INTENT_REFUSED

        self.mark_position_intent_unknown(
            intent.key,
            source="executor_verdict_journal",
            detail=(
                "no executor verdict journal row for this order_id — "
                "the command may have arrived, but execution is unconfirmed"
            ),
        )
        return INTENT_UNKNOWN

    async def _await_forwarded_verdict(self, intent: PositionIntent) -> dict | None:

        order_id = intent.order_id or ""
        if intent.action == INTENT_ACTION_ENTRY or intent.state == INTENT_UNKNOWN:
            return self._read_executor_verdict_entry(order_id)
        for _ in range(INTENT_CONFIRM_ATTEMPTS):
            entry = self._read_executor_verdict_entry(order_id)
            verdict = str((entry or {}).get("verdict") or "").strip().lower()
            if verdict in ("executed", "blocked", "error"):
                return entry
            await asyncio.sleep(INTENT_CONFIRM_DELAY)
        return self._read_executor_verdict_entry(order_id)

    async def _settle_update_intent(self, intent: PositionIntent, receipt: dict) -> str:

        async with self._position_lock():
            current_intent = self.position_intents().get(intent.key)
            if current_intent is None or current_intent.state not in NON_BOOKED_INTENT_STATES:
                return current_intent.state if current_intent else intent.state
            position = self.current_position
            if position is None:
                return current_intent.state
            position_id = position_identity(position)
            if intent.position_id != position_id:
                return current_intent.state
            if intent.symbol != position.symbol or receipt.get("order_id") != intent.order_id:
                return current_intent.state
            if receipt.get("symbol") not in (None, intent.symbol):
                return current_intent.state
            entry = self.position_intents().latest_for_position(position_id, INTENT_ACTION_ENTRY)
            executor_position_id = (entry.payload or {}).get("position_id") if entry else None
            if executor_position_id and receipt.get("position_id") not in (None, executor_position_id):
                return current_intent.state
            latest = self.position_intents().latest_for_position(position_id, INTENT_ACTION_UPDATE)
            superseded = latest is not None and latest.key != intent.key
            verdict = str(receipt.get("verdict") or "").lower()
            if verdict == "executed":
                if not superseded:
                    payload = intent.payload or {}
                    self.current_position = replace(
                        position,
                        stop_loss=payload.get("new_stop_loss", position.stop_loss),
                        take_profit=payload.get("new_take_profit", position.take_profit),
                    )
                    await self.persistence.async_save_position(self.current_position)
                self.confirm_position_intent(
                    intent.key, source="executor_verdict_journal",
                    detail=("executor receipt: executed (UPDATE applied on the exchange)"
                            + ("; superseded by a newer UPDATE, current levels kept" if superseded else "")),
                )
                return INTENT_CONFIRMED
            if not superseded:
                self._apply_refused_intent(intent, verdict)
                await self.persistence.async_save_position(self.current_position)
            self.refuse_position_intent(
                intent.key, source="executor_verdict_journal",
                detail=(f"executor receipt: {verdict} ({receipt.get('reason') or ''})"
                        + ("; superseded by a newer UPDATE, current levels kept" if superseded else "")),
            )
            return INTENT_REFUSED

    async def _settle_entry_fill(self, intent: PositionIntent, receipt: dict) -> bool:

        price = ExitRecords._positive_float(receipt.get("average_price"))
        quantity = ExitRecords._positive_float(receipt.get("filled_quantity"))
        if (price is None or quantity is None
                or isinstance(receipt.get("average_price"), bool)
                or isinstance(receipt.get("filled_quantity"), bool)
                or receipt.get("state") not in {"filled", "partially_filled"}
                or receipt.get("fill_evidence") not in EXIT_EVIDENCE_KINDS):
            return False
        async with self._position_lock():
            position = self.current_position
            current = self.position_intents().get(intent.key)
            if (position is None or current is None
                    or current.state in {INTENT_REFUSED, INTENT_LOCAL_ONLY}
                    or intent.position_id != position_identity(position)
                    or intent.symbol != position.symbol
                    or receipt.get("order_id") != intent.order_id
                    or receipt.get("symbol") != intent.symbol):
                return False
            signals = LONG_ENTRY_SIGNALS if position.direction == "LONG" else SHORT_ENTRY_SIGNALS
            if receipt.get("signal") not in signals or not receipt.get("position_id"):
                return False
            payload = current.payload
            executor_position_id = payload.get("position_id")
            if executor_position_id and receipt["position_id"] != executor_position_id:
                return False
            fees = ExitRecords._actual_fees(receipt) if receipt.get("fees_known") is True else None
            fee = math.fsum(fees) if fees else None
            quality = {key: receipt.get(key) for key in MARKET_VERIFICATION_FIELDS}
            fill = {"price": price, "quantity": quantity, "fee": fee, "execution_quality": quality}
            previous_fill = payload.get("entry_fill")
            if previous_fill == fill:
                return True
            if previous_fill and position.size != previous_fill.get("quantity"):
                self.logger.error("Entry receipt changed after a partial exit for %s — accounting kept", intent.key)
                return False
            quote_amount = price * quantity
            size_pct = position.size_pct
            if position.quote_amount > 0:
                size_pct *= quote_amount / position.quote_amount
            settled = replace(
                position, entry_price=price, size=quantity, entry_fee=fee,
                entry_execution_quality=quality,
                quote_amount=quote_amount, size_pct=size_pct,
            )
            row_id = await asyncio.to_thread(
                self.persistence.apply_entry_fill, settled, row_id=payload.get("entry_history_id"),
                history_time=payload.get("entry_history_time"),
            )
            self.current_position = settled
            await asyncio.to_thread(self.memory_service.refresh_from_history)
            self.position_intents().record(
                current.action, current.symbol, key=current.key, order_id=current.order_id,
                position_id=current.position_id, state=current.state,
                evidence="executor_verdict_journal", detail=current.detail,
                payload={**payload, "entry_history_id": row_id,
                         "position_id": receipt["position_id"], "entry_fill": fill},
            )
            self.logger.info(
                "Entry execution settled %s: price=%s quantity=%s fee=%s (order_id=%s)",
                position.symbol, price, quantity, fee, intent.order_id,
            )
            return True

    async def _reconcile_entry_fill(self) -> None:

        position = self.current_position
        if position is None or not self._executor_query_available():
            return
        intent = self.position_intents().latest_for_position(position_identity(position), INTENT_ACTION_ENTRY)
        if intent is None or intent.state in {INTENT_REFUSED, INTENT_LOCAL_ONLY}:
            return
        receipt = self._read_executor_verdict_entry(intent.order_id or "")
        if (receipt
                and (receipt.get("verdict") == "executed" or self._entry_fill_is_proven(receipt))
                and await self._settle_entry_fill(intent, receipt)):
            self.confirm_position_intent(
                intent.key, source="executor_verdict_journal",
                detail="executor receipt: entry execution facts settled",
            )

    async def _reconcile_late_updates(self) -> None:

        for intent in self.position_intents().pending():
            position = self.current_position
            if position is None:
                return
            if intent.action != INTENT_ACTION_UPDATE:
                continue
            if intent.position_id != position_identity(position) or intent.symbol != position.symbol:
                continue
            receipt = self._read_executor_verdict_entry(intent.order_id or "")
            if str((receipt or {}).get("verdict") or "").lower() not in ("executed", "blocked", "error"):
                continue
            await self._resolve_forwarded_intent(intent, delivered=True, verdict_entry=receipt)

    async def _apply_confirmed_intent(self, intent: PositionIntent, verdict_entry: dict | None) -> str:

        if intent.action == INTENT_ACTION_CLOSE:
            position = self.current_position
            if position is None:
                self.confirm_position_intent(
                    intent.key, source="executor_verdict_journal",
                    detail="executor: executed (local position already flat)",
                )
                return INTENT_CONFIRMED
            conditions = None
            try:
                conditions = self._conditions.build_conditions_from_position(position)
            except Exception as e:  # noqa: BLE001
                self.logger.warning("Could not rebuild entry conditions for the confirmed close: %s", e)
            receipt_record, receipt_rejection = ExitRecords.from_receipt(
                verdict_entry, symbol=intent.symbol
            )
            if receipt_record is None and receipt_rejection:
                self.logger.info(
                    "Executor receipt for the CLOSE of %s carries no bookable fill evidence "
                    "(%s) — waiting for the exit journal instead of inventing numbers.",
                    intent.symbol, receipt_rejection,
                )
            booked = await self.book_executor_side_exit(
                position, position.entry_price, conditions,
                "CLOSE confirmed by executor",
                receipt_record=receipt_record,
            )
            self.confirm_position_intent(
                intent.key, source="executor_verdict_journal",
                detail=f"executor receipt: executed (exit booked={booked})",
            )
            if not booked:
                self._raise_unconfirmed_intent_alert(
                    f"CLOSE {intent.symbol} executed per the executor, but there is no "
                    f"accounting fill record (price/quantity) — the local position STAYS, "
                    f"settlement is on hold until an exit record appears",
                    intent.key,
                )
                self.mark_position_intent_unknown(
                    intent.key, source="executor_exit_journal",
                    detail="executor: executed, but no validated fill record"
                )
                return INTENT_UNKNOWN
            return INTENT_CONFIRMED

        if intent.action == INTENT_ACTION_ENTRY:
            settled = await self._settle_entry_fill(intent, verdict_entry or {})
            verdict = str((verdict_entry or {}).get("verdict") or "").strip().lower()
            detail = ("executor receipt: entry execution facts settled" if settled
                      else "executor receipt: entry filled; execution accounting unavailable")
            if verdict != "executed":
                detail += f"; command verdict={verdict}, protection unresolved"
            self.confirm_position_intent(
                intent.key, source="executor_verdict_journal",
                detail=detail,
            )
            return INTENT_CONFIRMED

        self.confirm_position_intent(
            intent.key, source="executor_verdict_journal",
            detail="executor receipt: executed (UPDATE applied on the exchange)",
        )
        return INTENT_CONFIRMED

    def _apply_refused_intent(self, intent: PositionIntent, verdict: str) -> None:

        if intent.action == INTENT_ACTION_UPDATE:
            payload = intent.payload or {}
            old_sl = payload.get("old_stop_loss")
            old_tp = payload.get("old_take_profit")
            position = self.current_position
            if position is not None and (old_sl is not None or old_tp is not None):
                import dataclasses as _dc

                self.current_position = _dc.replace(
                    position,
                    stop_loss=old_sl if old_sl is not None else position.stop_loss,
                    take_profit=old_tp if old_tp is not None else position.take_profit,
                )
                self.logger.warning(
                    "Executor refused the UPDATE (%s) — local protection mirror restored "
                    "to SL=%s TP=%s", verdict, old_sl, old_tp,
                )
                self._raise_unconfirmed_intent_alert(
                    f"UPDATE {intent.symbol} rejected by the executor ({verdict}) — "
                    f"the local protection mirror was restored; the exchange still enforces the old protection",
                    intent.key,
                )
        elif intent.action == INTENT_ACTION_CLOSE:
            self._raise_unconfirmed_intent_alert(
                f"CLOSE {intent.symbol} REJECTED by the executor ({verdict}) — "
                f"the local position STAYS, check the exchange state",
                intent.key,
            )

    def _executor_exit_path(self) -> Path:

        return Path(self.config.EXECUTOR_EXIT_PATH)

    def _executor_exit_record(self, symbol: str, not_before: datetime | None = None) -> dict | None:

        path = self._executor_exit_path()
        try:
            if not path.exists():
                return None
            for line in reversed(path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("symbol") != symbol:
                    continue
                if not_before is not None:
                    exit_time = ExitRecords._parse_exit_time(entry.get("timestamp"))
                    if exit_time is None or exit_time < not_before:
                        continue
                return entry
        except (OSError, json.JSONDecodeError):
            self.logger.warning(
                "Failed to read executor exit journal at %s", path,
            )
        return None

    async def book_executor_side_exit(
        self,
        position: Position,
        current_price: float,
        market_conditions: Any,
        reason_label: str,
        *,
        receipt_record: dict | None = None,
    ) -> bool:

        async with self._position_lock():
            current = self.current_position
            if current is None:
                self.logger.info(
                    "Exchange-side exit for %s already booked (no local position) — nothing to do",
                    position.symbol,
                )
                return False
            if position_identity(current) != position_identity(position):
                self.logger.warning(
                    "Exchange-side exit booking skipped for %s — the local position changed "
                    "since it was read (stale intent, not this trade's exit).",
                    position.symbol,
                )
                return False

            exit_record = self._executor_exit_record(position.symbol, not_before=position.entry_time)
            scope = f"{self._executor_exit_path()}"
            evidence_source = "executor_exit_journal"
            if not exit_record and receipt_record is None:
                key = self.intent_identity(
                    INTENT_ACTION_CLOSE, position.symbol, position_id=position_identity(position)
                )
                intent = self.position_intents().get(key)
                if intent and intent.state in (INTENT_PENDING, INTENT_UNKNOWN) and intent.order_id:
                    verdict = self._read_executor_verdict_entry(intent.order_id)
                    candidate, rejection = ExitRecords.from_receipt(verdict, symbol=position.symbol)
                    receipt_time = ExitRecords._parse_exit_time(candidate.get("timestamp")) if candidate else None
                    if candidate and receipt_time is not None and receipt_time >= position.entry_time:
                        receipt_record = candidate
                    elif rejection and verdict:
                        self.logger.warning("CLOSE receipt for %s is not bookable: %s", position.symbol, rejection)
            if not exit_record and receipt_record:
                exit_record = dict(receipt_record)
                scope = f"{self._executor_verdict_path()}#receipt"
                evidence_source = "executor_verdict_journal"
                self.logger.info(
                    "No exit-journal line for %s yet — booking the confirmed exit from the "
                    "executor's receipt fill evidence (average price + filled amount).",
                    position.symbol,
                )
            if not exit_record:
                phantom = self._unconfirmed_entry_for(position)
                if phantom is not None and self._entry_never_reached_the_executor(phantom):
                    self.logger.warning(
                        "PHANTOM ROLLBACK %s: entry %s (older than %d min) never executed — "
                        "no executor verdict, no position, no exit record. Local position dropped.",
                        position.symbol, phantom.order_id, PHANTOM_ENTRY_GRACE_SECONDS // 60,
                    )
                    self.current_position = None
                    await self.persistence.async_save_position(None)
                    await self._record_blocked_entry_close(
                        position,
                        reason=(
                            f"The {position.direction} entry from {position.entry_time.isoformat()} "
                            f"never reached the exchange: no executor verdict for {phantom.order_id} "
                            f"after the {PHANTOM_ENTRY_GRACE_SECONDS // 60} minute grace window, the "
                            f"executor reports no position and holds no exit record for it. Local "
                            f"phantom rolled back; compensating CLOSE booked at the entry price."
                        ),
                    )
                    return False
                self.logger.critical(
                    "STATE DIVERGENCE %s: executor flat, no exit record — local position KEPT "
                    "(entry %.2f). Verify the exchange.",
                    position.symbol, position.entry_price,
                )
                self._state_divergence = (
                    f"executor reports no open position for {position.symbol}, but has no exit "
                    f"record — booking is on hold, the local position STAYS "
                    f"(entry {position.entry_price:,.2f})"
                )
                return False

            verified, rejection = ExitRecords.verify(exit_record)
            if verified is None:
                self.logger.critical(
                    "STATE DIVERGENCE %s: exit record is not usable evidence (%s) — local "
                    "position KEPT. Verify the exchange.",
                    position.symbol, rejection,
                )
                self._state_divergence = (
                    f"executor reports no open position for {position.symbol}, but its exit record "
                    f"is unusable ({rejection}) — booking is on hold, the local position STAYS "
                    f"(entry {position.entry_price:,.2f}). The exit price was NOT guessed."
                )
                return False

            event_id = ExitRecords.identity(exit_record, position_identity(position))
            booked = self._booked_exit_event_ids()
            ledger = self.position_intents()
            scoped_event_key = f"{scope}#{event_id}"
            if event_id in booked or ledger.is_event_booked(scoped_event_key, position_identity(position)):
                self.logger.warning(
                    "Exit event %s was already booked for %s — ignoring the duplicate",
                    event_id, position.symbol,
                )
                return False

            if verified.quantity is not None and abs(verified.quantity - position.size) > max(
                1e-9, abs(position.size) * 0.01
            ):
                self.logger.critical(
                    "QUANTITY DIVERGENCE for %s: the executor's fill is %.8f but the local "
                    "position size is %.8f — booking the FILL amount, flagging the mismatch.",
                    position.symbol, verified.quantity, position.size,
                )
                self._state_divergence = (
                    f"quantity divergence for {position.symbol}: executor fill "
                    f"{verified.quantity:.8f} vs local position {position.size:.8f} — "
                    f"the fill quantity was booked, check the exchange balance"
                )

            detail = (
                f"{verified.exit_reason} @ {verified.exit_price} "
                f"(exchange order {verified.protection_order_id}, evidence {verified.evidence})"
            )
            if ExitRecords.fill_is_divergent(verified.fill_verification):
                self.logger.critical(
                    "FILL DIVERGENCE %s: %s — booking the CLOSE for history pairing, "
                    "excluding it from statistics and learning.",
                    position.symbol,
                    ExitRecords.divergence_reason(verified),
                )
                self._state_divergence = (
                    f"fill divergence for {position.symbol}: {ExitRecords.divergence_reason(verified)}"
                    f" (booked, not counted, not learned)"
                )
                detail = (
                    f"{FILL_DIVERGENCE_MARKER}: the venue filled at {verified.exit_price:,.2f}"
                    + (
                        f" while the real market traded at {verified.reference_price:,.2f}"
                        f" ({verified.deviation_pct:+.2f}%)"
                        if verified.reference_price is not None
                        and isinstance(verified.deviation_pct, (int, float))
                        else ""
                    )
                    + f" — NOT a market level. {detail}"
                )
            self.logger.warning(
                "Executor confirmed no open position for %s while a local position was tracked — "
                "booking the exchange-side exit (%s) so trade history stays paired.",
                position.symbol, detail,
            )
            await self.close_position(
                f"{reason_label}: {detail}",
                verified.exit_price,
                market_conditions,
                exit_fee=verified.exit_fee,
                fee_source=(
                    f"{evidence_source} ({verified.fee_currency})"
                    if verified.exit_fee is not None and verified.fee_currency
                    else (evidence_source if verified.exit_fee is not None else None)
                ),
                filled_quantity=verified.quantity,
                exit_time=ExitRecords._parse_exit_time(verified.exit_time),
                evidence=ExitEvidence(
                    source=evidence_source,
                    price=verified.exit_price,
                    quantity=verified.quantity,
                    event_id=scoped_event_key,
                    protection_version=verified.protection_version,
                    fee=verified.exit_fee,
                    fee_currency=verified.fee_currency,
                    fill_verification=verified.fill_verification,
                    reference_price=verified.reference_price,
                    deviation_pct=verified.deviation_pct,
                    market_verification=verified.market_verification,
                ),
            )
            self._executor_side_exit_reason = f"{reason_label}: {detail}"
            booked.add(event_id)
            while len(booked) > MAX_BOOKED_EXIT_EVENTS:
                booked.pop()
            ledger.mark_event_booked(scoped_event_key, position_identity(position))
            ledger.record(
                INTENT_ACTION_CLOSE,
                position.symbol,
                key=self.intent_identity(
                    INTENT_ACTION_CLOSE, position.symbol,
                    position_id=position_identity(position),
                ),
                position_id=position_identity(position),
                state=INTENT_CONFIRMED,
                evidence=evidence_source,
                detail=f"exit booked from the {evidence_source} ({detail})",
                payload={
                    "event_id": scoped_event_key,
                    "exit_price": verified.exit_price,
                    "exit_quantity": verified.quantity,
                },
            )
            self._last_booked_exit = {
                "event_id": event_id,
                "position_id": position_identity(position),
                "symbol": position.symbol,
                "exit_price": verified.exit_price,
                "exit_reason": verified.exit_reason,
                "evidence": verified.evidence,
                "exit_time": verified.exit_time,
                "protection_order_id": verified.protection_order_id,
            }
            return True

    async def reconcile_local_position(
        self,
        symbol: str | None = None,
        *,
        source: str = "periodic",
    ) -> LocalPositionReconciliation:

        await self._reconcile_entry_fill()
        await self._reconcile_late_updates()
        now = datetime.now(timezone.utc)
        position = self.current_position
        symbol = symbol or (position.symbol if position is not None else None)

        if position is None:
            outcome = LocalPositionReconciliation(
                state=RECONCILE_NO_POSITION,
                symbol=symbol,
                checked_at=now,
                detail="no local position tracked",
                snapshot_token="flat",
                source=source,
            )
            self._last_reconciliation = outcome
            return outcome

        if not symbol:
            outcome = LocalPositionReconciliation(
                state=RECONCILE_UNVERIFIED,
                symbol=None,
                checked_at=now,
                detail="symbol unavailable — cannot verify the exchange state",
                snapshot_token=self.position_snapshot_token(),
                source=source,
            )
            self._last_reconciliation = outcome
            return outcome

        query_available = self._executor_query_available()
        executor_state = await self._executor_has_position(symbol)

        if executor_state is True and not query_available:
            outcome = LocalPositionReconciliation(
                state=RECONCILE_UNVERIFIED,
                symbol=symbol,
                checked_at=now,
                detail=(
                    "executor API disabled/unconfigured — no executor report was "
                    "queried; local state only (not executor-reported)"
                ),
                snapshot_token=self.position_snapshot_token(),
                source=source,
                evidence="local_only",
            )
            self._last_reconciliation = outcome
            return outcome

        if executor_state is True:
            exchange_verified = self._executor_exchange_verified(
                self._last_executor_position_payload,
                symbol,
                position,
            )
            outcome = LocalPositionReconciliation(
                state=RECONCILE_OPEN_EXECUTOR_REPORTED,
                symbol=symbol,
                checked_at=now,
                snapshot_token=self.position_snapshot_token(),
                source=source,
                detail=(
                    "executor reports venue reconciliation verified and protection active"
                    if exchange_verified
                    else (
                        "executor tracker reports the position open (not exchange-verified: "
                        "venue confirmation is absent or incomplete)"
                    )
                ),
                evidence=(
                    EXECUTOR_VENUE_EVIDENCE if exchange_verified else EXECUTOR_TRACKER_EVIDENCE
                ),
                exchange_verified_at=now if exchange_verified else None,
            )
            self._last_reconciliation = outcome
            return outcome

        if executor_state is None:
            outcome = LocalPositionReconciliation(
                state=RECONCILE_UNVERIFIED,
                symbol=symbol,
                checked_at=now,
                detail="executor position query failed — state NOT confirmed",
                snapshot_token=self.position_snapshot_token(),
                source=source,
            )
            self._last_reconciliation = outcome
            return outcome

        position_id = position_identity(position)
        exit_record = self._executor_exit_record(symbol, not_before=position.entry_time)
        if not exit_record:
            age = (now - position.entry_time).total_seconds()
            if age < RECONCILE_ENTRY_GRACE_SECONDS:
                outcome = LocalPositionReconciliation(
                    state=RECONCILE_UNVERIFIED,
                    symbol=symbol,
                    checked_at=now,
                    detail=(
                        f"executor does not report the entry yet ({age:.0f}s < "
                        f"{RECONCILE_ENTRY_GRACE_SECONDS:.0f}s grace) and has no exit record — "
                        f"NOT treated as an exit"
                    ),
                    position_id=position_id,
                    snapshot_token=self.position_snapshot_token(),
                    source=source,
                )
                self._last_reconciliation = outcome
                return outcome

        market_conditions = None
        try:
            market_conditions = self._conditions.build_conditions_from_position(position)
        except Exception as e:  # noqa: BLE001
            self.logger.warning("Could not rebuild entry conditions for the exit booking: %s", e)

        booked = await self.book_executor_side_exit(
            position, position.entry_price, market_conditions, "Executor confirmed flat"
        )

        if booked or self.current_position is None:
            outcome = LocalPositionReconciliation(
                state=RECONCILE_EXIT_BOOKED,
                symbol=symbol,
                checked_at=now,
                detail="exchange-side exit booked from the executor exit journal",
                exit_booked=True,
                exit_event_id=(self._last_booked_exit or {}).get("event_id"),
                position_id=position_id,
                snapshot_token=self.position_snapshot_token(),
                source=source,
            )
        else:
            outcome = LocalPositionReconciliation(
                state=RECONCILE_DIVERGENCE,
                symbol=symbol,
                checked_at=now,
                detail=(
                    "executor reports no open position but the exit is unproven "
                    "(no/failed exit record) — local position kept"
                ),
                position_id=position_id,
                snapshot_token=self.position_snapshot_token(),
                source=source,
            )
        self._last_reconciliation = outcome
        return outcome

    def take_executor_side_exit_reason(self) -> str | None:

        reason = self._executor_side_exit_reason
        self._executor_side_exit_reason = None
        return reason

    def take_state_divergence(self) -> str | None:

        message = self._state_divergence
        self._state_divergence = None
        return message

    async def rollback_blocked_entry(self, symbol: str, forward_delivered: bool, order_id: str | None = None) -> None:

        if self.current_position is None:
            return
        if not forward_delivered:
            return
        if await self.confirm_entry_with_executor(symbol, order_id=order_id):
            return
        entry = self.current_position
        self.current_position = None
        await self.persistence.async_save_position(None)
        await self._record_blocked_entry_close(entry)
        self.logger.warning(
            "Executor blocked %s entry for %s (no position after forward) — "
            "rolled back local phantom position and recorded compensating CLOSE.",
            entry.direction, symbol,
        )

    async def _record_blocked_entry_close(self, entry: Position, reason: str | None = None) -> None:

        decision = TradeDecision(
            timestamp=datetime.now(timezone.utc),
            symbol=entry.symbol,
            action="CLOSE",
            confidence=entry.confidence,
            price=entry.entry_price,
            stop_loss=entry.stop_loss,
            take_profit=entry.take_profit,
            position_size=entry.size_pct,
            quote_amount=entry.quote_amount,
            quantity=entry.size,
            fee=0.0,
            reasoning=(
                reason
                or (
                    f"Executor blocked the {entry.direction} entry (no position on exchange). "
                    f"Local phantom rolled back; entry recorded {entry.entry_time.isoformat()}."
                )
            ),
        )
        await self._record_trade_decision(decision)
