"""Executor reconciliation for the trading strategy.

Verifies that an entry the bot recorded locally actually reached the executor, correlates
per-order outcomes through the executor's verdict journal, and rolls back phantom positions
the executor silently blocked. Split out of trading_strategy.py; the methods use
``self`` state provided by TradingStrategy at MRO resolution time.
"""

import asyncio
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from .data_models import Position, TradeDecision

ENTRY_CONFIRM_ATTEMPTS = 10
ENTRY_CONFIRM_DELAY = 2.5
ENTRY_CONFIRM_MIN_FALSE_REPORTS = 6

INTENT_CONFIRM_ATTEMPTS = 5
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

DEFAULT_INTENT_JOURNAL_PATH = "data/trading/bot_position_intents.jsonl"

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



def _parse_exit_time(raw: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp from the exit journal (UTC when naive)."""
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _positive_float(raw: Any) -> float | None:
    """Coerce a journal field to a usable positive price, else None."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _non_negative_float(raw: Any) -> float | None:
    """Coerce a journal field to a finite non-negative number, else None."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _quantity_matches_exchange_amount(reported: Any, local: Any) -> bool:
    """True when the venue's amount is the local size at the market's precision.

    A venue holds an amount rounded to its own precision, while the local position
    carries the paper-computed size the order was sized from. A fixed relative
    tolerance therefore rejected every real fill (2026-09-28: 0.0038412035106648494
    local, 0.00384 on the exchange — one rounding step, and 3000x the old
    tolerance). The reported amount is already rounded to the market's precision,
    so its own decimal places ARE that precision: flooring the local size to them
    must reproduce the reported amount exactly. An amount a full step smaller or
    larger than the position still fails.
    """
    reported_value = _positive_float(reported)
    local_value = _positive_float(local)
    if reported_value is None or local_value is None:
        return False
    exponent = Decimal(str(reported_value)).as_tuple().exponent
    if not isinstance(exponent, int):
        return False
    scale = 10 ** -min(exponent, 0)
    units = math.floor(Decimal(str(local_value)) * scale)
    return units == Decimal(str(reported_value)) * scale


def quote_asset(symbol: Any) -> str | None:
    """Quote asset of ``BTC/USDC``, or None when the symbol carries none.

    Used to decide whether a currency-tagged fee may be summed as a quote-currency
    amount. No FX/stable conversion exists anywhere in this module, so only the
    symbol's OWN quote asset is accepted (wave 5).
    """
    text = str(symbol or "")
    if "/" not in text:
        return None
    quote = text.rsplit("/", 1)[-1].strip()
    return quote.upper() or None


def _fee_amount(item: Any, quote: str | None) -> float | None:
    """Amount of ONE fee entry in quote currency, or None = not usable.

    Wave 5 fee contract: the executor/venue reports fees as currency-tagged objects
    (ccxt shape: ``{"cost": <amount>, "currency": "USDC"}``). Such an entry is real
    evidence and is now summed — but only when its currency IS the symbol's own quote
    asset. A fee charged in the base asset (BTC) or in a third asset (BNB discounts)
    still means UNKNOWN: converting it would invent a number.

    A plain number (the wave-1 journal shape) stays accepted; it carries no currency
    and is by convention already a quote-currency amount.
    """
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
        return _non_negative_float(amount)
    return _non_negative_float(item)


def _actual_fees(record: dict) -> tuple[float, ...] | None:
    """Real fees reported by the executor/venue for one exit, or None (UNKNOWN).

    Wave 4 fee contract: a commission may only be booked from actual fill/order data.
    An empty list, a missing field or an unparsable value all mean UNKNOWN — never 0.0
    and never the configured rate (the 2026-09-21 incident booked a synthesised
    0.075% fee of 2.851875 USDC where the exchange charged 0).

    Wave 5 additive contract: a currency-tagged fee entry is trusted when (and only
    when) its currency is the symbol's own quote asset (see ``_fee_amount``). A fee in
    the base/third asset is UNKNOWN — no conversion is performed and the number is
    never treated as if it were quote currency.
    """
    raw = record.get("fees")
    quote = quote_asset(record.get("symbol"))
    collected: list[float] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            value = _fee_amount(item, quote)
            if value is None:
                return None
            collected.append(value)
        if collected:
            return tuple(collected)
    elif isinstance(raw, dict):
        value = _fee_amount(raw, quote)
        return None if value is None else (value,)
    elif isinstance(raw, bool):
        return None
    elif raw is not None:
        value = _non_negative_float(raw)
        if value is None:
            return None
        return (value,)

    tagged = record.get("fee_currency", record.get("currency"))
    if tagged is not None and (quote is None or str(tagged).strip().upper() != quote):
        return None
    for key in ("total_fee", "fee"):
        value = _non_negative_float(record.get(key))
        if value is not None:
            return (value,)
    return None


def _fee_currency(record: dict) -> str | None:
    """Currency every booked fee is denominated in, or None when it is not uniform.

    Purely descriptive: it is reported next to the commission (``fee_source``) so the
    operator can see WHICH currency the number is in. ``None`` means the record does
    not say (the wave-1 shape: a bare list of amounts, implicitly quote currency).
    """
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


def position_identity(position: Position) -> str:
    """Stable identity of one trade (symbol + entry stamp) used for booking/dedup.

    Mirrors the ``position_id`` already used in the brain's blocked-trade metadata
    (``f"{symbol}|{entry_time.isoformat()}"``) so the two views agree. Wave-2 ledger
    replaces this with the executor's own ``position_id``/``entry_order_id``.
    """
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
        """True while the local state must NOT be presented as an execution."""
        return self.state in NON_BOOKED_INTENT_STATES

    def to_dict(self) -> dict[str, Any]:
        """JSONL payload of this intent (stable key order for diffs)."""
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
        """Append one record, fsync'd so a crash cannot lose a booked state."""
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
        """Create or update one intent (idempotent by ``key``) and persist it."""
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
        """Intent by key, replaying the journal on first use."""
        self._load()
        return self._intents.get(key)

    def get_by_order(self, order_id: str | None) -> PositionIntent | None:
        """Newest intent carrying this executor command id, or None."""
        self._load()
        if not order_id:
            return None
        key = self._order_index.get(str(order_id))
        return self._intents.get(key) if key else None

    def pending(self) -> list[PositionIntent]:
        """Intents whose local state must not be read as an execution."""
        self._load()
        return [intent for intent in self._intents.values() if intent.is_unbooked]

    def recent(self, limit: int = 1) -> list[PositionIntent]:
        """Newest intents first, by last write — resolved ones included.

        ``pending()`` answers "what is still unresolved"; this answers "what did the
        bot ask for last and what became of it", which is what the next analysis
        prompt needs (one cycle, one analysis, no other channel for that context).
        """
        self._load()
        ordered = sorted(
            self._intents.values(),
            key=lambda intent: intent.updated_at or intent.created_at or "",
            reverse=True,
        )
        return ordered[: max(limit, 0)]

    def mark_event_booked(self, event_id: str, position_id: str | None = None) -> None:
        """Persist that one exit event was booked (restart-safe idempotency)."""
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
        """True when THIS trade's exit event was already booked, in this run or before.

        The event id is scoped by the trade it was booked for: the same venue order id
        reused by a later trade must not be treated as an already-booked exit.
        """
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

    @property
    def exit_fee(self) -> float | None:
        """Total actual exit commission, or None when the fee data is unknown."""
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

    @property
    def identity(self) -> str:
        """Stable identity used to learn an exit exactly once ("" when unknown)."""
        return str(self.event_id or "")

    def to_dict(self) -> dict[str, Any]:
        """JSONL payload of this evidence (stable key order for diffs/logs)."""
        return {
            "source": self.source,
            "price": self.price,
            "quantity": self.quantity,
            "event_id": self.event_id,
            "protection_version": self.protection_version,
            "fee": self.fee,
            "fee_currency": self.fee_currency,
        }


def exit_evidence_is_learnable(evidence: Any) -> tuple[bool, str]:
    """May a close with this evidence feed the brain? ``(learnable, reason)``.

    The reason is an operator-facing explanation of the REFUSAL and is empty when the
    evidence is usable. Accepts an :class:`ExitEvidence` or a mapping with the same
    keys (so a caller/version that only produces a dict still gets validated).
    """
    if evidence is None:
        return False, "no exit evidence was recorded for this close"
    if isinstance(evidence, dict):
        evidence = ExitEvidence(
            source=str(evidence.get("source") or ""),
            price=_positive_float(evidence.get("price")),
            quantity=_positive_float(evidence.get("quantity")),
            event_id=evidence.get("event_id"),
            protection_version=evidence.get("protection_version"),
            fee=_non_negative_float(evidence.get("fee")),
            fee_currency=evidence.get("fee_currency"),
        )
    if not isinstance(evidence, ExitEvidence):
        return False, f"unrecognized evidence object ({type(evidence).__name__})"
    source = str(evidence.source or "")
    if source not in EXECUTOR_CONFIRMATION_SOURCES:
        return False, f"source {source!r} is not a confirming source"
    price = _positive_float(evidence.price)
    if price is None:
        return False, f"exit price unknown/invalid ({evidence.price!r})"
    quantity = _positive_float(evidence.quantity)
    if quantity is None:
        return False, f"filled quantity unknown/invalid ({evidence.quantity!r})"
    if not evidence.identity:
        return False, "no stable exit event id — the lesson could not be deduplicated"
    return True, ""


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

    @property
    def is_open_confirmed(self) -> bool:
        """True when the EXECUTOR's tracker reported the position open in this check.

        Not exchange-verified: see ``is_exchange_verified``. Kept as the wave-1 hook
        name the monitor/app use to mean "a positive answer this cycle".
        """
        return self.state == RECONCILE_OPEN_EXECUTOR_REPORTED

    @property
    def is_exchange_verified(self) -> bool:
        """True when the executor reports fresh venue reconciliation and live protection."""
        return self.exchange_verified_at is not None


def exit_event_identity(record: dict, position_id: str) -> str:
    """Stable dedup key for one exit event.

    Uses the executor's ``event_id`` when the record carries it (wave-1 additive
    contract); otherwise derives one from the venue order id / fill ids / exit
    timestamp — never from the discovery time, so re-reading the same journal line
    yields the same key. Wave-2 replaces this with ledger uniqueness in the DB.
    """
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


def verify_exit_record(record: Any) -> tuple[VerifiedExit | None, str | None]:
    """Validate an executor exit-journal record before it may be booked.

    Returns ``(verified, None)`` or ``(None, rejection_detail)``. Unknown reason,
    missing/malformed/non-positive price, unknown evidence kind or a non-final
    (remaining > 0) record are all rejected: an unproven exit must never become a
    booked trade, and the entry price must never stand in for an unknown fill price.
    """
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
        exit_price = _positive_float(raw_average)
        if exit_price is None:
            return None, f"average price is not usable ({raw_average!r})"
    else:
        exit_price = _positive_float(record.get("exit_price"))
        if exit_price is None:
            return None, "exit price unknown/invalid — booking refused (entry price is not a fill)"

    quantity = None
    raw_quantity = record.get("filled_quantity")
    if raw_quantity is None:
        raw_quantity = record.get("quantity")
    if raw_quantity is None:
        return None, "exit quantity unknown — booking refused (the position size is not a fill)"
    quantity = _positive_float(raw_quantity)
    if quantity is None:
        return None, f"filled quantity is not usable ({raw_quantity!r})"

    raw_remaining = record.get("remaining_quantity")
    if raw_remaining is not None:
        remaining = _non_negative_float(raw_remaining)
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
            event_id=exit_event_identity(record, str(position_id)),
            evidence=evidence,
            position_id=str(position_id),
            protection_order_id=str(protection_order_id) if protection_order_id else None,
            quantity=quantity,
            exit_time=str(record.get("timestamp")) if record.get("timestamp") else None,
            fees=_actual_fees(record),
            fee_currency=_fee_currency(record),
            protection_version=(
                str(record["protection_version"])
                if record.get("protection_version") is not None
                else None
            ),
        ),
        None,
    )


def exit_record_from_receipt(receipt: Any, *, symbol: str | None = None) -> tuple[dict | None, str]:
    """Exit-journal-shaped record built from the executor's CONFIRMED receipt.

    Wave 5 (requirement 1): the executor's verdict row now carries the REAL fill
    evidence of the command it processed — ``average_price``, ``filled_quantity``,
    ``fees`` (a list of ``{"currency", "amount"}``), ``fees_known``, ``state``,
    ``close_confirmed``, ``protection_version``. Those are the executor's own field
    names; nothing here is renamed or guessed, and a missing value stays missing.

    This adapter exposes that evidence to the SAME validation/booking path the exit
    journal uses (:func:`verify_exit_record`), so a CLOSE the bot sent is booked from
    the receipt when the journal line has not landed yet — instead of being left
    "unconfirmed" forever. It is strict on purpose:

    * only a verdict of ``executed`` for a CLOSE command qualifies;
    * the command must be FULLY filled (``state == "filled"``): a ``partially_filled``
      close is NOT a final exit and must not be booked as one (the position keeps its
      open state and the operator is alerted, exactly as with a partial journal line);
    * ``close_confirmed`` must be explicitly true (the executor's own statement that
      the close is backed by fill evidence);
    * a positive ``average_price`` AND ``filled_quantity`` must be present — an unknown
      number is never replaced by the requested quantity or the entry price;
    * ``fees_known is False`` (the exchange reported no fees) turns the fee into
      UNKNOWN instead of 0;
    * an explicit ``symbol`` must match the local position.

    Returns ``(record, "")`` or ``(None, rejection_reason)``. The record is shaped like
    an exit-journal line (minus the exchange leg identity the receipt does not have) and
    carries the stable ``event_id`` ``receipt:<command_id|order_id>`` so a replay cannot
    book twice.
    """
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
    price = _positive_float(receipt.get("average_price"))
    if price is None:
        return None, f"average price unknown/invalid ({receipt.get('average_price')!r})"
    quantity = _positive_float(receipt.get("filled_quantity"))
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
        "source": "executor_verdict_journal",
    }
    verified, rejection = verify_exit_record(record)
    if verified is None:
        return None, f"receipt fill evidence is not usable ({rejection})"
    return record, ""


class ExecutorReconciliationMixin:
    """Executor position verification, verdict correlation and ghost rollback."""

    logger: Any
    config: Any
    persistence: Any
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
    _last_reconciliation: LocalPositionReconciliation | None

    def _position_lock(self) -> asyncio.Lock:
        """Shared transition lock: app analysis path and the monitor sync serialize here."""
        lock = getattr(self, "_position_transition_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._position_transition_lock = lock
        return lock

    def _booked_exit_event_ids(self) -> set[str]:
        """Per-process set of already-booked exit event ids (idempotency guard)."""
        booked = getattr(self, "_booked_exit_events", None)
        if booked is None:
            booked = set()
            self._booked_exit_events = booked
        return booked


    def _intent_journal_path(self) -> Path:
        """Filesystem path of the bot's own position-intent journal (JSONL).

        Prefers an explicit ``BOT_INTENT_JOURNAL_PATH``, then the persistence data
        dir (keeps each deployment/temporary test dir isolated), then a repo default.
        """
        configured = getattr(self.config, "BOT_INTENT_JOURNAL_PATH", None)
        if configured:
            return Path(configured)
        data_dir = getattr(self.persistence, "data_dir", None)
        if isinstance(data_dir, (str, Path)):
            return Path(data_dir) / "bot_position_intents.jsonl"
        return Path(DEFAULT_INTENT_JOURNAL_PATH)

    def position_intents(self) -> PositionIntentLedger:
        """Process-wide intent ledger, replaying its journal on first use."""
        ledger = getattr(self, "_position_intent_ledger", None)
        if ledger is None:
            ledger = PositionIntentLedger(self._intent_journal_path())
            self._position_intent_ledger = ledger
        return ledger

    def intent_identity(self, action: str, symbol: str, *, order_id: str | None = None,
                        position_id: str | None = None) -> str:
        """Stable key of one command intent (never the discovery time).

        ``CLOSE`` is keyed by the TRADE (``close:<symbol>|<entry_time>``) so a retry,
        a restart or the app+monitor running in parallel produce the SAME key and the
        exit can only be booked once. ``ENTRY`` is keyed by the bot's ``order_id``
        (generated once per decision and persisted), ``UPDATE`` by trade + order id.
        """
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
        """Record a command the bot sent (or is about to send) — NOT an execution.

        The returned intent's state is ``pending`` (or ``local_only`` when no command
        was sent at all). Local position state may only change on a state the
        executor evidence established afterwards (see
        ``confirm_position_intent`` / ``refuse_position_intent``).
        """
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
        """The semaphore: only executor-side evidence may move a local state.

        A local monitor tick, a ticker price, a retry/debounce or the bot's own
        optimism is refused here — that refusal is the whole point of this method.
        """
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
        """Move one intent to a terminal state using an admitted evidence source."""
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
        self.logger.warning(
            "Position intent %s -> %s (evidence=%s, detail=%s)",
            key, state, source, detail or "",
        )
        return intent

    def confirm_position_intent(self, key: str, *, source: str, detail: str | None = None,
                                payload: dict[str, Any] | None = None) -> PositionIntent | None:
        """Confirm one intent from executor evidence (no-op on a duplicate)."""
        return self._resolve_intent(key, INTENT_CONFIRMED, source=source, detail=detail, payload=payload)

    def refuse_position_intent(self, key: str, *, source: str, detail: str | None = None) -> PositionIntent | None:
        """The executor's receipt says the command did NOT run (blocked/error)."""
        return self._resolve_intent(key, INTENT_REFUSED, source=source, detail=detail)

    def mark_position_intent_unknown(self, key: str, *, source: str, detail: str | None = None) -> PositionIntent | None:
        """No answer / source failure: state is UNKNOWN, never an execution.

        The local position is deliberately left untouched and an operator alert is
        queued (``take_unconfirmed_intent_alert``).
        """
        if not self._assert_confirmation_source(source):
            return None
        intent = self._resolve_intent(key, INTENT_UNKNOWN, source=source, detail=detail)
        if intent is None:
            return None
        self._unconfirmed_intent_alert = (
            f"{intent.action} {intent.symbol} ({intent.key}) was not confirmed by "
            f"the executor ({source}: {detail or 'no reply'}) — local state: "
            f"{INTENT_UNKNOWN}, local position unchanged"
        )
        return intent

    def take_unconfirmed_intent_alert(self) -> str | None:
        """Return and clear the pending one-shot 'unconfirmed command' alert."""
        message = getattr(self, "_unconfirmed_intent_alert", None)
        self._unconfirmed_intent_alert = None
        return message

    def take_rejected_intent_alert(self) -> str | None:
        """Return and clear the pending one-shot 'command refused' alert.

        A refusal is not a failed execution: the bot itself blocked the command, so
        nothing was sent and no exchange state moved — the operator still has to hear
        about it, because the analysis asked for it in the first place.
        """
        message = getattr(self, "_rejected_intent_alert", None)
        self._rejected_intent_alert = None
        return message

    def unbooked_position_intents(self) -> list[PositionIntent]:
        """Intents that must NOT be reported as executed (pending/unknown)."""
        return self.position_intents().pending()

    def position_intent_state(self, action: str | None = None, order_id: str | None = None) -> str | None:
        """State of one intent (by order id, else the newest of *action*), or None."""
        ledger = self.position_intents()
        if order_id:
            intent = ledger.get_by_order(order_id)
            return intent.state if intent else None
        intents = [item for item in ledger.pending() if action is None or item.action == action]
        return intents[-1].state if intents else None

    def position_snapshot_token(self) -> str:
        """Trade-level identity of the CURRENT position.

        Callers capture it before an LLM run and compare afterwards: a different token
        means a DIFFERENT trade (or no trade at all) — an exit was booked, or the
        position was replaced while the model was working.

        Parameter updates on the same trade deliberately keep the token stable: an
        accepted UPDATE changes the SL/TP of the trade it was computed for, and must
        still be forwarded to the executor. Only a change of trade identity (symbol,
        direction, entry stamp) or its disappearance invalidates the result.

        KNOWN GAP (wave-1, explicit): this token carries NO protection version. The
        executor's ``/position`` exposes no revision/``updated_at`` and the bot never
        reads the venue's live SL/TP back, so a protection change on the SAME trade
        (executor restart/resync, exchange-side order replacement, the bot's own
        tightening) is invisible here. That is intentional for the bot's own accepted
        UPDATE (the 15:06 defect: a self-tightening SL cancelled the decision it was
        computed for), but it means a token match cannot prove the protection the
        decision was computed against is still in force. The cheap, safe alternative
        shipped instead of a version is ``protection_loosening_reason`` — see its
        docstring. Wave 2 must add a real revision (executor must return one).
        """
        position = self.current_position
        if position is None:
            return "flat"
        direction = getattr(position, "direction", "") or ""
        return f"open:{position.symbol}:{direction}:{position.entry_time.isoformat()}"

    def protection_snapshot(self) -> tuple[float | None, float | None] | None:
        """``(stop_loss, take_profit)`` of the current position, or None when flat."""
        position = self.current_position
        if position is None:
            return None
        return (
            getattr(position, "stop_loss", None),
            getattr(position, "take_profit", None),
        )

    def protection_loosening_reason(self, decision: Any) -> str | None:
        """Why an UPDATE decision must NOT be forwarded, or None when it is safe.

        Substitute for the missing protection version (see ``position_snapshot_token``).
        The risk of a stale UPDATE is not that it is stale — it is that it can WIDEN
        protection: ``_update_position_parameters`` accepts a widening SL up to 150%
        of the original distance, so a decision computed against protection that no
        longer exists can loosen the live stop. This guard is monotone and local: it
        only refuses to move the stop AWAY from price (LONG: below the current stop;
        SHORT: above it). Tightening and TP changes always pass, so the bot's own
        accepted UPDATE keeps flowing.
        """
        if decision is None:
            return None
        if str(getattr(decision, "action", "") or "").upper() != "UPDATE":
            return None
        position = self.current_position
        if position is None:
            return None
        proposed = getattr(decision, "stop_loss", None)
        current = getattr(position, "stop_loss", None)
        if proposed is None or current is None:
            return None
        direction = str(getattr(position, "direction", "") or "").upper()
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
        """Reuse one httpx client across executor position queries (keeps TCP pools alive)."""
        if self._http_client is None or self._http_client.is_closed:
            import httpx
            self._http_client = httpx.AsyncClient(timeout=3.0)
        return self._http_client

    async def close(self) -> None:
        """Close persistent HTTP client resources."""
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    def _executor_query_available(self) -> bool:
        """True when the executor ``/position`` could actually have been queried.

        With ``EXECUTOR_API_ENABLED=False`` (or no URL) ``_executor_has_position``
        short-circuits to True for the trading logic, but that answer is NOT an
        executor report — there was no query. Reconciliation labels it local-only so
        no card and no state can claim the executor (let alone the exchange) confirmed
        anything.
        """
        enabled = bool(getattr(self.config, "EXECUTOR_API_ENABLED", False))
        url = getattr(self.config, "EXECUTOR_API_URL", "") or ""
        return enabled and bool(url)

    async def _executor_has_position(self, symbol: str) -> bool | None:
        """Query the executor API for its current position and venue confirmation."""
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
        """Validate the executor's venue-backed confirmation for this exact position.

        ``reconcile_status == "verified"`` is the venue's own answer on the exchange's
        protection orders (for a spot long: every tracked leg was queried and is still
        open), and the protection block must report a live, complete, unduplicated set.
        ``protection_verified`` is a separate cache the executor fills only when it
        re-checks legs recovered from durable artifacts, so it stays None for a
        position the running executor placed itself; False — the exchange explicitly
        refused to confirm the legs — is the one value that must never pass.
        """
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
        reported_quantity = _positive_float(reported.get("quantity"))
        reported_entry = _positive_float(reported.get("entry_price"))
        if reported_quantity is None or reported_entry is None:
            return False
        quantity_matches = _quantity_matches_exchange_amount(reported_quantity, position.size)
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
        """Protection revision the executor reports for ``symbol``, or None.

        Wave 5 requirement (3): a decision computed against a protection set that was
        REPLACED during the analysis must not be forwarded. The reliable way to see
        that is a revision from the executor's ``/position``; the bot then compares the
        revision from before the analysis with the one after it (``app`` does that in
        ``_protection_version_guard``).

        CONTRACT (confirmed against the executor in staging, ``src/api.py`` /
        ``src/position_tracker.py``): ``/position`` exposes ``protection_version`` — a
        MONOTONIC, durable counter of protection changes for the position, bumped on a
        new protection set / a swap / a restoration / a loss of protection, and ``0``
        when none was observed. ``0`` is a real value (it is returned verbatim as
        ``"0"``), never treated as "no field". ``confirmation.protection_verified``
        (whether the EXCHANGE confirmed the legs) is not used here yet.

        The field names below are the executor's own; the ``protection_revision``
        fallback stays as a defensive alias. An older executor without the field
        yields None, and the guard in ``app`` is then inert rather than wrong.

        Returns:
            The revision as a string (number or text, verbatim), or None when the
            integration is off, the query failed, or the payload carries no revision.
        """
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
        """The executor ``/position`` payload, or None when it cannot be read.

        Same query as ``_executor_has_position`` (which only keeps the boolean) but the
        whole answer is returned so additive fields (a protection revision) can be read.
        Never raises: no integration, a failed request or a non-object body all mean
        "no answer".
        """
        if not getattr(self.config, "EXECUTOR_API_ENABLED", False):
            return None
        url: str = getattr(self.config, "EXECUTOR_API_URL", "") or ""
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
        """Confirm a forwarded entry executed, using the verdict journal.

        The executor appends one verdict line per processed decision, keyed by
        the bot's ``order_id`` (written on /decision → queue → main loop →
        SafetyGuard / execution). Polling this journal answers "what happened
        to MY order" definitively — unlike polling /position, which only says
        whether ANY position exists and races the executor's 10s queue tick.

        Returns:
            True if the executor reports ``executed`` (or the journal is
            unreadable/absent — fail-open: never roll back a possibly-live
            order because a log file hiccuped).
            False only when the executor explicitly recorded ``blocked`` or
            ``error`` for THIS order_id.
        """
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
        """The still-unconfirmed ENTRY intent of this local position, or None.

        ``None`` also covers "provenance unknown" — no intent was recorded for this
        position (e.g. state restored from an older build). The caller then keeps the
        fail-open behavior instead of discarding a position it cannot explain.
        """
        identity = position_identity(position)
        for intent in self.position_intents().pending():
            if intent.action == INTENT_ACTION_ENTRY and intent.position_id == identity:
                return intent
        return None

    def _entry_never_reached_the_executor(self, intent: PositionIntent) -> bool:
        """True when nothing proves this ENTRY executed, well after it was forwarded.

        Every condition must hold, so a live order is never rolled back:

        * the intent is still unbooked (``pending``/``unknown``) — never confirmed;
        * the executor integration is enabled (otherwise the executor was never asked);
        * the verdict journal EXISTS but holds no line for this order_id — a missing
          journal is a configuration problem, not proof, so it keeps the fail-open;
        * the intent is older than ``PHANTOM_ENTRY_GRACE_SECONDS``.
        """
        if not getattr(self.config, "EXECUTOR_API_ENABLED", False):
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
        """Read the executor's verdict journal for one order_id.

        Returns ``"executed"`` / ``"blocked"`` / ``"error"``, or None when the
        journal has no entry for this order yet (or is unreadable — treated as
        "no verdict yet", the caller fails open).
        """
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
        """Filesystem path of the executor's verdict journal."""
        configured = self.config.EXECUTOR_VERDICT_PATH
        if configured:
            return Path(configured)
        return Path("data/trading/executor_verdicts.jsonl")

    def _read_executor_verdict_entry(self, order_id: str) -> dict | None:
        """Newest executor verdict LINE for this order_id, or None.

        Same source as ``_read_executor_verdict`` (the executor's own receipt
        journal), returned whole so the caller can keep the reason/timestamp.
        """
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
        """True when an executor receipt proves entry exposure despite an error verdict."""
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

    async def resolve_position_intents_after_forward(
        self,
        order_id: str | None,
        delivered: bool,
        symbol: str | None = None,
    ) -> str | None:
        """Resolve the intent of a command the app just forwarded to the executor.

        The ONLY thing that may move the local position state here is the executor's
        own receipt (``executor_verdict_journal``). ``delivered=False`` (file fallback,
        HTTP failure, no response) or a receipt that is missing/unreadable means
        UNKNOWN: the local position is kept and the operator gets an alert — never a
        silent close and never a silent erase.

        Returns the resulting intent state, or None when there is no such intent.
        """
        ledger = self.position_intents()
        intent = ledger.get_by_order(order_id)
        if intent is None:
            return None
        return await self._resolve_forwarded_intent(intent, delivered=delivered)

    async def _resolve_forwarded_intent(self, intent: PositionIntent, *, delivered: bool) -> str:
        """Apply the executor receipt to one pending intent (idempotent)."""
        if intent.state != INTENT_PENDING:
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

        verdict_entry = await self._await_forwarded_verdict(intent)
        verdict = str((verdict_entry or {}).get("verdict") or "").strip().lower()

        if verdict == "executed" or (
            intent.action == INTENT_ACTION_ENTRY
            and self._entry_fill_is_proven(verdict_entry)
        ):
            return await self._apply_confirmed_intent(intent, verdict_entry)

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
        """Executor verdict for one forwarded command, polling briefly when it is due.

        ``/decision`` answers "queued"; the executor then processes the command in its own
        loop, whose tick reaches ~10s (the reason ``confirm_entry_with_executor`` polls).
        A single immediate read races that loop — the CLOSE of 2026-09-28 was recorded 4ms
        after the forward and was still missed, leaving the intent UNKNOWN plus a critical
        operator alert until the reconcile booked the exit two minutes later. UPDATE/CLOSE
        get the same short grace here (``INTENT_CONFIRM_ATTEMPTS`` x
        ``INTENT_CONFIRM_DELAY``, then one last read). An entry is not polled: it already
        waits through ``confirm_entry_with_executor``. Nothing is assumed while waiting —
        a verdict that never arrives still resolves to UNKNOWN with the position kept.
        """
        order_id = intent.order_id or ""
        if intent.action == INTENT_ACTION_ENTRY:
            return self._read_executor_verdict_entry(order_id)
        for _ in range(INTENT_CONFIRM_ATTEMPTS):
            entry = self._read_executor_verdict_entry(order_id)
            verdict = str((entry or {}).get("verdict") or "").strip().lower()
            if verdict in ("executed", "blocked", "error"):
                return entry
            await asyncio.sleep(INTENT_CONFIRM_DELAY)
        return self._read_executor_verdict_entry(order_id)

    async def _apply_confirmed_intent(self, intent: PositionIntent, verdict_entry: dict | None) -> str:
        """Book what the executor receipt confirmed (never anything more)."""
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
            receipt_record, receipt_rejection = exit_record_from_receipt(
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
                self._unconfirmed_intent_alert = (
                    f"CLOSE {intent.symbol} executed per the executor, but there is no "
                    f"accounting fill record (price/quantity) — the local position STAYS, "
                    f"settlement is on hold until an exit record appears"
                )
                self.mark_position_intent_unknown(
                    intent.key, source="executor_exit_journal",
                    detail="executor: executed, but no validated fill record"
                )
                return INTENT_UNKNOWN
            return INTENT_CONFIRMED

        if intent.action == INTENT_ACTION_ENTRY:
            verdict = str((verdict_entry or {}).get("verdict") or "").strip().lower()
            detail = "executor receipt: entry filled"
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
        """The executor refused the command: undo the local mirror, keep the trade.

        ENTRY: the local phantom is rolled back by ``rollback_blocked_entry`` (the
        caller), which needs no extra work here. UPDATE: the local SL/TP mirror is
        restored to the pre-intent values — the exchange never applied the change.
        CLOSE: the local position stays, loudly.
        """
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
                self._unconfirmed_intent_alert = (
                    f"UPDATE {intent.symbol} rejected by the executor ({verdict}) — "
                    f"the local protection mirror was restored; the exchange still enforces the old protection"
                )
        elif intent.action == INTENT_ACTION_CLOSE:
            self._unconfirmed_intent_alert = (
                f"CLOSE {intent.symbol} REJECTED by the executor ({verdict}) — "
                f"the local position STAYS, check the exchange state"
            )

    def _executor_exit_path(self) -> Path:
        """Filesystem path of the executor's exit journal (JSONL)."""
        configured = getattr(self.config, "EXECUTOR_EXIT_PATH", None)
        if configured:
            return Path(configured)
        return Path("data/trading/executor_exits.jsonl")

    def _executor_exit_record(self, symbol: str, not_before: datetime | None = None) -> dict | None:
        """Newest exchange-side exit the executor recorded for this symbol.

        The executor writes one line per position it saw exit ON THE EXCHANGE
        (a stop-loss or take-profit order filled while the bot was between
        cycles — the bot only wakes up every few hours, the exchange's SL/TP
        works continuously). Returns None when there is no usable record, which
        means the exit price is unknown — never invent one.

        ``not_before`` (the local position's entry time) rejects records from an
        EARLIER trade on the same symbol, which would otherwise book a stale
        fill price into this position's P&L.
        """
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
                    exit_time = _parse_exit_time(entry.get("timestamp"))
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
        """Book an exit the EXCHANGE performed instead of silently dropping it.

        The bot still tracks a position but the executor reports nothing open:
        the stop-loss or take-profit order filled on the exchange between
        cycles. Previously the bot just erased its local state — trade history
        kept the entry open forever (an entry with no exit, which reads as a
        duplicate buy), the real P&L was lost, and the brain never learned from
        the outcome.

        The exit price is taken from the executor's exit journal (the VERIFIED
        fill). A record without a usable price is NOT booked: the entry price is
        not a fill price, and inventing one corrupts the statistics, the P&L and
        the learning. Such a record is reported as a state divergence instead.

        The whole transition runs under the shared position lock and is
        idempotent per exit event id, so the app's analysis path and the
        independent monitor sync cannot book the same exit twice.

        Args:
            position: the local position the caller read (re-checked under the lock).
            current_price: unused by the booking path (the fill price rules); kept
                for signature compatibility.

        Returns:
            True when a CLOSE row was booked by this call.
        """
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
                    candidate, rejection = exit_record_from_receipt(verdict, symbol=position.symbol)
                    receipt_time = _parse_exit_time(candidate.get("timestamp")) if candidate else None
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

            verified, rejection = verify_exit_record(exit_record)
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

            event_id = exit_event_identity(exit_record, position_identity(position))
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
                exit_time=_parse_exit_time(verified.exit_time),
                evidence=ExitEvidence(
                    source=evidence_source,
                    price=verified.exit_price,
                    quantity=verified.quantity,
                    event_id=scoped_event_key,
                    protection_version=verified.protection_version,
                    fee=verified.exit_fee,
                    fee_currency=verified.fee_currency,
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
        """Reconcile the LOCAL position with the exchange, before/independently of the LLM.

        This is the single public hook used by both callers:

        * the app, BEFORE the analysis prompt is built, so the model never reasons
          about a position the exchange already closed;
        * the position-status monitor, on its own short cadence (target <= 120s on a
          healthy API), independent of the 4h analysis schedule, the ticker and the
          LLM's answer.

        Never books an exit without a validated exit record, never clears the local
        position on an unverifiable answer, and never invents a price. A confirmed
        exit is booked exactly once (shared lock + per-event idempotency).
        """
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
        """Return and clear the reason of the last booked exchange-side exit (one-shot read)."""
        reason = self._executor_side_exit_reason
        self._executor_side_exit_reason = None
        return reason

    def take_state_divergence(self) -> str | None:
        """Return and clear the last state-divergence warning (one-shot read).

        Set when the executor claims the position is gone but provides no exit
        record — the app turns it into a loud operator alert.
        """
        message = self._state_divergence
        self._state_divergence = None
        return message

    async def rollback_blocked_entry(self, symbol: str, forward_delivered: bool, order_id: str | None = None) -> None:
        """After forwarding an entry, roll back the local position if the
        executor rejected the order.

        The bot persists a Position (and records the BUY/SELL row) BEFORE the
        executor processes the order. If the executor then blocks it (silent
        ``Blocked`` on its console), the bot would manage a phantom position
        forever. This verification runs right after the forward:

        - ``forward_delivered=False`` → the order went to the file fallback and
          may still execute later; never roll back a possibly-live order.
        - executor confirms the position (via verdict journal or /position) →
          nothing to do.
        - executor explicitly reports the order blocked/error → roll back the
          phantom and record a compensating CLOSE so trade history stays
          paired/truthful.
        """
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
        """Record a compensating CLOSE row for an entry that never executed.

        ``reason`` overrides the default wording when the entry was not "blocked" but
        never reached the executor at all (see the phantom rollback).
        """
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
