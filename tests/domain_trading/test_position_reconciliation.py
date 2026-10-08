"""Wave-1 P0-C: local-position reconciliation, exit-record validation and the
independent exchange sync driven by the position-status monitor.

Covers:
* ``verify_exit_record`` — an unknown/unusable exit price or reason never books;
* ``book_executor_side_exit`` — validated booking, idempotency per exit event,
  no double CLOSE when the app and the monitor reconcile concurrently;
* ``reconcile_local_position`` — the single public hook (open verified / unverified
  / exit booked / divergence) used before the analysis and by the monitor loop;
* the monitor's independent sync cadence (<= 120s), its independence from the ticker
  and the LLM, and status cards that are never published stale after a final exit.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.trading.data_models import MarketConditions
from src.trading.executor_reconciliation import (
    EXECUTOR_TRACKER_EVIDENCE,
    EXECUTOR_VENUE_EVIDENCE,
    RECONCILE_DIVERGENCE,
    RECONCILE_EXIT_BOOKED,
    RECONCILE_NO_POSITION,
    RECONCILE_OPEN_EXECUTOR_REPORTED,
    RECONCILE_UNVERIFIED,
    LocalPositionReconciliation,
    exit_event_identity,
    position_identity,
    verify_exit_record,
)
from src.trading.exit_monitor import ExitMonitor
from src.trading.position_status_monitor import (
    DEFAULT_RECONCILE_INTERVAL_SECONDS,
    PositionStatusMonitor,
)
from src.trading.trading_strategy import TradingStrategy
from tests.conftest import (
    make_config,
    make_position,
    mock_brain,
    mock_persistence,
    mock_statistics,
    null_logger,
)

ENTRY_TIME = datetime(2026, 9, 21, 9, 4, 12, tzinfo=timezone.utc)


def _decision(action: str, stop_loss: float | None = None) -> SimpleNamespace:
    """Minimal decision double for the protection guard (no TradeDecision import churn)."""
    return SimpleNamespace(action=action, stop_loss=stop_loss)




def strategy(position: Any = None, **config_overrides: Any) -> tuple[TradingStrategy, MagicMock, MagicMock]:
    """Real TradingStrategy on the shared doubles (returned with logger + persistence).

    The intent journal is ISOLATED per call: without it the suite writes to the default
    ``data/trading/bot_position_intents.jsonl`` under the working directory, and a rerun
    (pytest reuses a ``--basetemp`` directory NAME for the same test) replays those
    bookings as "already booked" — the exit then never books and the test fails for a
    reason that has nothing to do with the code.
    """
    config_overrides.setdefault(
        "BOT_INTENT_JOURNAL_PATH",
        str(Path(tempfile.mkdtemp(prefix="w5_intents_")) / "bot_position_intents.jsonl"),
    )
    logger = null_logger()
    persistence = mock_persistence()
    statistics = mock_statistics()
    statistics.get_current_capital.return_value = 10000.0
    brain = mock_brain()
    brain.get_dynamic_thresholds.return_value = {}
    instance = TradingStrategy(
        logger=logger,
        persistence=persistence,
        brain_service=brain,
        statistics_service=statistics,
        memory_service=MagicMock(),
        risk_manager=MagicMock(),
        config=make_config(**config_overrides),
        position_extractor=MagicMock(),
    )
    instance.current_position = position
    return instance, logger, persistence


def position(**overrides: Any) -> Any:
    """LONG BTC/USDC position as the incident reported it."""
    values: dict[str, Any] = {
        "symbol": "BTC/USDC",
        "direction": "LONG",
        "entry_price": 81200.45,
        "stop_loss": 80550.0,
        "take_profit": 84500.0,
        "size": 0.00554,
        "entry_time": ENTRY_TIME,
        "confidence": "HIGH",
    }
    values.update(overrides)
    return make_position(**values)


def verified_position_payload(subject: Any) -> dict[str, Any]:
    return {
        "open": True,
        "symbol": subject.symbol,
        "side": subject.direction.lower(),
        "quantity": subject.size,
        "entry_price": subject.entry_price,
        "sl_order_id": "2341240",
        "tp_order_id": "2341241",
        "confirmation": {
            "state": "confirmed",
            "confirmed": True,
            "reconcile_status": "verified",
            "protection_status": "active",
            "protection_verified": True,
            "protection_unresolved": False,
            "protection_duplicated": False,
        },
    }


def exit_record(**overrides: Any) -> dict[str, Any]:
    """The real TP exit as the executor's journal carried it (legacy shape)."""
    record: dict[str, Any] = {
        "timestamp": "2026-09-21T09:32:51.956000+00:00",
        "symbol": "BTC/USDC",
        "side": "long",
        "quantity": 0.00554,
        "entry_price": 81200.45,
        "exit_price": 84500.0,
        "exit_reason": "take_profit_filled",
        "protection_order_id": "2012150",
        "source": "reconcile",
    }
    record.update(overrides)
    return record


def exit_journal(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    """Executor exit journal holding the given records."""
    path = tmp_path / "executor_exits.jsonl"
    path.write_text("".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8")
    return path


class MonitorStore:
    """In-memory position-monitor state store."""

    def __init__(self) -> None:
        self.state: dict[str, Any] = {}

    async def async_load_position_monitor_state(self) -> dict[str, Any]:
        return dict(self.state)

    async def async_save_position_monitor_state(self, state: dict[str, Any]) -> None:
        self.state = dict(state)

    async def async_clear_position_monitor_state(self) -> None:
        self.state = {}

    def load_trade_history(self) -> list[dict[str, Any]]:
        return []


def monitor_config(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "STOP_LOSS_TYPE": "soft",
        "STOP_LOSS_CHECK_INTERVAL": "15m",
        "STOP_LOSS_CHECK_INTERVAL_SECONDS": 900,
        "TAKE_PROFIT_TYPE": "soft",
        "TAKE_PROFIT_CHECK_INTERVAL": "15m",
        "TAKE_PROFIT_CHECK_INTERVAL_SECONDS": 900,
        "MAIN_CHANNEL_ID": 123,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def monitor(
    *,
    trading_strategy: Any,
    notifier: Any = None,
    state: dict[str, Any] | None = None,
    is_running: Any = None,
    fetch_current_ticker: Any = None,
    interruptible_sleep: Any = None,
    reconcile_position: Any = None,
    reconcile_interval_seconds: float = DEFAULT_RECONCILE_INTERVAL_SECONDS,
) -> tuple[PositionStatusMonitor, MonitorStore]:
    """PositionStatusMonitor wired to an in-memory state store."""
    config = monitor_config()
    persistence = MonitorStore()
    persistence.state = dict(state or {})
    instance = PositionStatusMonitor(
        logger=null_logger(),
        config=config,
        persistence=persistence,
        trading_strategy=trading_strategy,
        exit_monitor=ExitMonitor(config, "1h", 3600),
        notifier=notifier,
        active_tasks=set(),
        is_running=is_running or (lambda: False),
        fetch_current_ticker=fetch_current_ticker or AsyncMock(return_value=None),
        interruptible_sleep=interruptible_sleep or AsyncMock(return_value=False),
        get_symbol=lambda: "BTC/USDC",
        manages_exits=lambda: False,
        reconcile_position=reconcile_position,
        reconcile_interval_seconds=reconcile_interval_seconds,
    )
    return instance, persistence


def outcome(state: str, **overrides: Any) -> LocalPositionReconciliation:
    values: dict[str, Any] = {
        "state": state,
        "symbol": "BTC/USDC",
        "checked_at": datetime.now(timezone.utc),
    }
    values.update(overrides)
    return LocalPositionReconciliation(**values)




class TestExitRecordValidation:
    """An exit record must prove reason + actual price before any booking happens."""

    def test_legacy_record_is_marked_legacy_and_keeps_its_fill(self) -> None:
        """A record without the v2 evidence field books only as explicitly legacy."""
        verified, rejection = verify_exit_record(exit_record())

        assert rejection is None
        assert verified is not None
        assert verified.exit_price == 84500.0
        assert verified.exit_reason == "take_profit_filled"
        assert verified.evidence == "legacy_exit_record"
        assert verified.protection_order_id == "2012150"

    def test_v2_record_carries_exchange_fill_evidence(self) -> None:
        """Wave-1 additive fields are accepted and keep their evidence kind."""
        verified, rejection = verify_exit_record(
            exit_record(evidence="exchange_order_fill", filled_quantity=0.00554, remaining_quantity=0)
        )

        assert rejection is None
        assert verified is not None
        assert verified.evidence == "exchange_order_fill"
        assert verified.quantity == pytest.approx(0.00554)

    @pytest.mark.parametrize(
        ("record_overrides", "expected_fragment"),
        [
            pytest.param({"exit_price": None}, "exit price unknown/invalid", id="price-missing"),
            pytest.param({"exit_price": "abc"}, "exit price unknown/invalid", id="price-unparseable"),
            pytest.param({"exit_price": 0}, "exit price unknown/invalid", id="price-zero"),
            pytest.param({"exit_price": -84500.0}, "exit price unknown/invalid", id="price-negative"),
            pytest.param({"exit_price": float("inf")}, "exit price unknown/invalid", id="price-infinite"),
            pytest.param({"exit_price": float("nan")}, "exit price unknown/invalid", id="price-nan"),
        ],
    )
    def test_unusable_price_is_rejected(self, record_overrides: dict[str, Any], expected_fragment: str) -> None:
        """An unknown price must never be replaced by the entry price."""
        verified, rejection = verify_exit_record(exit_record(**record_overrides))

        assert verified is None
        assert expected_fragment in rejection

    @pytest.mark.parametrize(
        ("record", "expected_fragment"),
        [
            pytest.param(None, "not an object", id="not-a-dict"),
            pytest.param({}, "exit record is empty", id="empty-record"),
            pytest.param(exit_record(exit_reason=None), "exit reason is not usable", id="reason-missing"),
            pytest.param(exit_record(exit_reason=""), "exit reason is not usable", id="reason-empty"),
            pytest.param(exit_record(exit_reason="unknown"), "exit reason is not usable", id="reason-unknown"),
            pytest.param(exit_record(quantity=0), "filled quantity is not usable", id="quantity-zero"),
            pytest.param(exit_record(remaining_quantity=0.001), "not a final exit", id="partial-exit"),
            pytest.param(exit_record(remaining_quantity="x"), "remaining quantity is not usable", id="remaining-garbage"),
            pytest.param(exit_record(evidence="vibes"), "unrecognized evidence kind", id="bad-evidence"),
        ],
    )
    def test_malformed_or_unknown_fields_are_rejected(self, record: Any, expected_fragment: str) -> None:
        verified, rejection = verify_exit_record(record)

        assert verified is None
        assert expected_fragment in rejection


class TestExitEventIdentity:
    """Event identity: stable per venue fill, never derived from discovery time."""

    def test_identity_is_stable_and_position_scoped(self) -> None:
        record = exit_record()
        first = exit_event_identity(record, "BTC/USDC|entry")
        second = exit_event_identity(dict(record), "BTC/USDC|entry")

        assert first == second
        assert first.startswith("compat:")
        assert first != exit_event_identity(record, "BTC/USDC|other-entry")
        assert first != exit_event_identity({**record, "protection_order_id": "999"}, "BTC/USDC|entry")

    def test_explicit_event_id_wins_and_fill_ids_are_normalized(self) -> None:
        assert exit_event_identity(exit_record(event_id="evt-1"), "pos") == "event:evt-1"

        with_fills = exit_event_identity(exit_record(fill_ids=["b", "a"]), "pos")
        sorted_fills = exit_event_identity(exit_record(fill_ids=["a", "b"]), "pos")
        assert with_fills == sorted_fills

    def test_position_identity_matches_the_brain_metadata_convention(self) -> None:
        assert position_identity(position()) == f"BTC/USDC|{ENTRY_TIME.isoformat()}"




class TestBookExecutorSideExit:
    """Only a validated exit record books; every rejection keeps the local state."""

    async def test_books_the_verified_fill_once(self, tmp_path: Path) -> None:
        subject = position()
        instance, _, persistence = strategy(subject, EXECUTOR_EXIT_PATH=str(exit_journal(tmp_path, [exit_record()])))

        booked = await instance.book_executor_side_exit(subject, 84000.0, MarketConditions(), "Executor confirmed flat")

        assert booked is True
        assert instance.current_position is None
        decision = persistence.async_save_trade_decision.await_args.args[0]
        assert decision.price == 84500.0
        assert "take_profit_filled" in decision.reasoning
        assert "legacy_exit_record" in decision.reasoning

    async def test_late_close_receipt_books_after_initial_unknown(self, tmp_path: Path) -> None:
        """The executor may write its fill receipt after the bot first checks an empty journal."""
        subject = position()
        verdict_path = verdict_journal(tmp_path, [])
        instance, _, persistence = strategy(
            subject,
            EXECUTOR_EXIT_PATH=str(exit_journal(tmp_path, [])),
            EXECUTOR_VERDICT_PATH=str(verdict_path),
        )
        intent = instance.register_position_intent(
            "CLOSE", subject.symbol, order_id="close-late",
            position_id=position_identity(subject),
        )
        instance.mark_position_intent_unknown(
            intent.key, source="executor_verdict_journal", detail="no verdict yet",
        )
        assert await instance.book_executor_side_exit(subject, 83000.0, MarketConditions(), "flat") is False
        assert instance.current_position is subject

        verdict_path.write_text(json.dumps({
            "order_id": "close-late", "signal": "CLOSE", "symbol": subject.symbol,
            "verdict": "executed", "state": "filled", "close_confirmed": True,
            "command_id": "close:late", "average_price": 83755.35,
            "filled_quantity": subject.size, "fees_known": True,
            "fees": [{"currency": "USDC", "amount": 0.0}],
            "timestamp": "2026-09-23T16:04:07+00:00",
        }) + "\n", encoding="utf-8")

        instance._executor_has_position = AsyncMock(return_value=False)
        result = await instance.reconcile_local_position(source="periodic")
        assert result.state == RECONCILE_EXIT_BOOKED
        assert result.exit_booked is True
        assert instance.current_position is None
        assert persistence.async_save_trade_decision.await_args.args[0].price == 83755.35
        assert persistence.async_save_trade_decision.await_args.args[0].timestamp == datetime.fromisoformat("2026-09-23T16:04:07+00:00")
        updated = instance.position_intents().get(intent.key)
        assert updated is not None and updated.state == "confirmed"
        assert await instance.book_executor_side_exit(subject, 83000.0, MarketConditions(), "flat") is False
        assert persistence.async_save_trade_decision.await_count == 1

    @pytest.mark.parametrize("change", [
        {"average_price": None}, {"state": "partially_filled"},
        {"close_confirmed": False}, {"verdict": "blocked"},
        {"timestamp": "2026-09-20T16:04:07+00:00"},
    ])
    async def test_late_close_without_matching_final_fill_keeps_position(
        self, tmp_path: Path, change: dict[str, Any]
    ) -> None:
        subject = position()
        receipt = {
            "order_id": "close-late", "signal": "CLOSE", "symbol": subject.symbol,
            "verdict": "executed", "state": "filled", "close_confirmed": True,
            "command_id": "close:late", "average_price": 83755.35,
            "filled_quantity": subject.size,
            "timestamp": "2026-09-23T16:04:07+00:00",
        }
        receipt.update(change)
        instance, _, persistence = strategy(
            subject,
            EXECUTOR_EXIT_PATH=str(exit_journal(tmp_path, [])),
            EXECUTOR_VERDICT_PATH=str(verdict_journal(tmp_path, [receipt])),
        )
        instance.register_position_intent(
            "CLOSE", subject.symbol, order_id="close-late",
            position_id=position_identity(subject),
        )

        assert await instance.book_executor_side_exit(subject, 83000.0, MarketConditions(), "flat") is False
        assert instance.current_position is subject
        persistence.async_save_trade_decision.assert_not_awaited()

    async def test_second_call_is_idempotent(self, tmp_path: Path) -> None:
        """Re-reading the same journal line must not create another CLOSE."""
        subject = position()
        instance, _, persistence = strategy(subject, EXECUTOR_EXIT_PATH=str(exit_journal(tmp_path, [exit_record()])))

        assert await instance.book_executor_side_exit(subject, 84000.0, MarketConditions(), "flat") is True
        assert await instance.book_executor_side_exit(subject, 84000.0, MarketConditions(), "flat") is False

        assert persistence.async_save_trade_decision.await_count == 1

    @pytest.mark.parametrize(
        "record_overrides",
        [
            pytest.param({"exit_price": None}, id="no-price"),
            pytest.param({"exit_price": "unknown"}, id="garbage-price"),
            pytest.param({"exit_reason": None}, id="no-reason"),
        ],
    )
    async def test_unusable_record_keeps_the_position_and_flags_divergence(
        self, tmp_path: Path, record_overrides: dict[str, Any]
    ) -> None:
        """No verified price/reason → nothing is booked and the state stays put."""
        subject = position()
        instance, _, persistence = strategy(
            subject, EXECUTOR_EXIT_PATH=str(exit_journal(tmp_path, [exit_record(**record_overrides)]))
        )

        booked = await instance.book_executor_side_exit(subject, 84000.0, MarketConditions(), "Executor confirmed flat")

        assert booked is False
        assert instance.current_position is subject
        persistence.async_save_trade_decision.assert_not_awaited()
        persistence.async_save_position.assert_not_awaited()
        divergence = instance.take_state_divergence()
        assert divergence is not None
        assert "booking is on hold" in divergence

    async def test_stale_position_object_cannot_book(self, tmp_path: Path) -> None:
        """A caller holding an older position revision must not book against the new one."""
        instance, _, persistence = strategy(
            position(entry_time=ENTRY_TIME + timedelta(hours=1)),
            EXECUTOR_EXIT_PATH=str(exit_journal(tmp_path, [exit_record()])),
        )

        assert await instance.book_executor_side_exit(position(), 84000.0, MarketConditions(), "flat") is False

        assert persistence.async_save_trade_decision.await_count == 0




def verdict_journal(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    """Executor verdict journal holding the given records."""
    path = tmp_path / "executor_verdicts.jsonl"
    path.write_text("".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8")
    return path


def pending_entry_journal(
    tmp_path: Path, *, minutes_old: float, order_id: str = "order-phantom"
) -> Path:
    """Intent journal holding ONE unconfirmed ENTRY, aged by ``minutes_old``."""
    path = tmp_path / "bot_position_intents.jsonl"
    created = (datetime.now(timezone.utc) - timedelta(minutes=minutes_old)).isoformat()
    path.write_text(
        json.dumps(
            {
                "key": f"entry:BTC/USDC:{order_id}",
                "action": "ENTRY",
                "symbol": "BTC/USDC",
                "state": "pending",
                "order_id": order_id,
                "position_id": position_identity(position()),
                "evidence": None,
                "detail": "entry forwarded to the executor — awaiting its receipt",
                "created_at": created,
                "updated_at": created,
                "payload": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


class TestNeverExecutedEntry:
    """A local position whose entry never reached the executor is dropped, not held forever.

    The post-forward check fails OPEN (a possibly-live order is never rolled back), so
    without a late re-check the phantom was held until a human noticed — every start
    logged "executor flat, no exit record" and the alert card stayed UNVERIFIED.
    """

    async def test_old_unconfirmed_entry_is_rolled_back(self, tmp_path: Path) -> None:
        subject = position()
        instance, _, persistence = strategy(
            subject,
            EXECUTOR_API_ENABLED=True,
            EXECUTOR_EXIT_PATH=str(exit_journal(tmp_path, [])),
            EXECUTOR_VERDICT_PATH=str(verdict_journal(tmp_path, [])),
            BOT_INTENT_JOURNAL_PATH=str(pending_entry_journal(tmp_path, minutes_old=30)),
        )

        booked = await instance.book_executor_side_exit(subject, 84000.0, MarketConditions(), "Executor confirmed flat")

        assert booked is False
        assert instance.current_position is None
        persistence.async_save_position.assert_awaited_once_with(None)
        decision = persistence.async_save_trade_decision.await_args.args[0]
        assert decision.action == "CLOSE"
        assert decision.price == subject.entry_price
        assert "never reached the exchange" in decision.reasoning
        assert instance.take_state_divergence() is None

    async def test_fresh_unconfirmed_entry_keeps_the_position(self, tmp_path: Path) -> None:
        """A forward still inside its grace window may be in flight — never rolled back."""
        subject = position()
        instance, _, persistence = strategy(
            subject,
            EXECUTOR_API_ENABLED=True,
            EXECUTOR_EXIT_PATH=str(exit_journal(tmp_path, [])),
            EXECUTOR_VERDICT_PATH=str(verdict_journal(tmp_path, [])),
            BOT_INTENT_JOURNAL_PATH=str(pending_entry_journal(tmp_path, minutes_old=1)),
        )

        assert await instance.book_executor_side_exit(subject, 84000.0, MarketConditions(), "flat") is False

        assert instance.current_position is subject
        persistence.async_save_position.assert_not_awaited()
        assert "booking is on hold" in (instance.take_state_divergence() or "")

    async def test_executed_verdict_keeps_the_position(self, tmp_path: Path) -> None:
        """A confirmed entry means the exchange held it: a flat answer is a real divergence."""
        subject = position()
        instance, _, _ = strategy(
            subject,
            EXECUTOR_API_ENABLED=True,
            EXECUTOR_EXIT_PATH=str(exit_journal(tmp_path, [])),
            EXECUTOR_VERDICT_PATH=str(
                verdict_journal(
                    tmp_path,
                    [
                        {
                            "order_id": "order-phantom",
                            "verdict": "executed",
                            "reason": "",
                            "timestamp": "2026-09-21T16:05:00",
                            "symbol": "BTC/USDC",
                            "signal": "BUY",
                        }
                    ],
                )
            ),
            BOT_INTENT_JOURNAL_PATH=str(pending_entry_journal(tmp_path, minutes_old=30)),
        )

        assert await instance.book_executor_side_exit(subject, 84000.0, MarketConditions(), "flat") is False

        assert instance.current_position is subject
        assert "booking is on hold" in (instance.take_state_divergence() or "")

    async def test_missing_verdict_journal_keeps_the_position(self, tmp_path: Path) -> None:
        """No verdict journal at all is a config problem, not proof — stay fail-open."""
        subject = position()
        instance, _, _ = strategy(
            subject,
            EXECUTOR_API_ENABLED=True,
            EXECUTOR_EXIT_PATH=str(exit_journal(tmp_path, [])),
            EXECUTOR_VERDICT_PATH=str(tmp_path / "absent_verdicts.jsonl"),
            BOT_INTENT_JOURNAL_PATH=str(pending_entry_journal(tmp_path, minutes_old=30)),
        )

        assert await instance.book_executor_side_exit(subject, 84000.0, MarketConditions(), "flat") is False

        assert instance.current_position is subject


class TestReconcileLocalPosition:
    """The single public hook: verified open, unverified, booked exit, divergence."""

    async def test_flat_locally_is_reported_without_querying(self) -> None:
        instance, _, _ = strategy(None)
        instance._executor_has_position = AsyncMock()

        result = await instance.reconcile_local_position()

        assert result.state == RECONCILE_NO_POSITION
        assert result.snapshot_token == "flat"
        assert instance._executor_has_position.await_count == 0

    async def test_executor_reports_the_position_open_but_never_verified(self) -> None:
        """``/position`` is the executor's TRACKER view — the state must say so."""
        subject = position()
        instance, _, persistence = strategy(
            subject, EXECUTOR_API_ENABLED=True, EXECUTOR_API_URL="http://127.0.0.1:9199/decision"
        )
        instance._executor_has_position = AsyncMock(return_value=True)

        result = await instance.reconcile_local_position(source="pre_analysis")

        assert result.state == RECONCILE_OPEN_EXECUTOR_REPORTED
        assert result.state != "open_verified"
        assert result.is_open_confirmed is True
        assert result.is_exchange_verified is False
        assert result.evidence == EXECUTOR_TRACKER_EVIDENCE
        assert "not exchange-verified" in (result.detail or "")
        assert result.source == "pre_analysis"
        assert instance.current_position is subject
        persistence.async_save_trade_decision.assert_not_awaited()

    async def test_executor_venue_confirmation_marks_the_exact_position_verified(self) -> None:
        subject = position()
        instance, _, _ = strategy(
            subject, EXECUTOR_API_ENABLED=True, EXECUTOR_API_URL="http://127.0.0.1:9199/decision"
        )

        async def reports_open(symbol: str) -> bool | None:
            assert symbol == subject.symbol
            instance._last_executor_position_payload = verified_position_payload(subject)
            return True

        instance._executor_has_position = reports_open

        result = await instance.reconcile_local_position(source="periodic")

        assert result.state == RECONCILE_OPEN_EXECUTOR_REPORTED
        assert result.is_open_confirmed is True
        assert result.is_exchange_verified is True
        assert result.exchange_verified_at is not None
        assert result.evidence == EXECUTOR_VENUE_EVIDENCE
        assert "protection active" in (result.detail or "")

    async def test_executor_venue_confirmation_requires_matching_position_facts(self) -> None:
        subject = position()
        instance, _, _ = strategy(
            subject, EXECUTOR_API_ENABLED=True, EXECUTOR_API_URL="http://127.0.0.1:9199/decision"
        )

        async def reports_open(symbol: str) -> bool | None:
            assert symbol == subject.symbol
            payload = verified_position_payload(subject)
            payload["quantity"] = subject.size * 2
            instance._last_executor_position_payload = payload
            return True

        instance._executor_has_position = reports_open

        result = await instance.reconcile_local_position(source="periodic")

        assert result.is_exchange_verified is False
        assert result.evidence == EXECUTOR_TRACKER_EVIDENCE

    async def test_venue_amount_rounded_to_market_precision_still_verifies(self) -> None:
        """The live 2026-09-28 position: paper size 0.0038412035106648494 held as 0.00384."""
        subject = position(size=0.0038412035106648494, entry_price=83257.62)
        instance, _, _ = strategy(
            subject, EXECUTOR_API_ENABLED=True, EXECUTOR_API_URL="http://127.0.0.1:9199/decision"
        )

        async def reports_open(symbol: str) -> bool | None:
            payload = verified_position_payload(subject)
            payload["quantity"] = 0.00384
            payload["confirmation"]["protection_verified"] = None
            instance._last_executor_position_payload = payload
            return True

        instance._executor_has_position = reports_open

        result = await instance.reconcile_local_position(source="periodic")

        assert result.is_exchange_verified is True
        assert result.evidence == EXECUTOR_VENUE_EVIDENCE

    async def test_venue_amount_one_step_short_of_the_position_never_verifies(self) -> None:
        subject = position(size=0.0038412035106648494, entry_price=83257.62)
        instance, _, _ = strategy(
            subject, EXECUTOR_API_ENABLED=True, EXECUTOR_API_URL="http://127.0.0.1:9199/decision"
        )

        async def reports_open(symbol: str) -> bool | None:
            payload = verified_position_payload(subject)
            payload["quantity"] = 0.00383
            instance._last_executor_position_payload = payload
            return True

        instance._executor_has_position = reports_open

        result = await instance.reconcile_local_position(source="periodic")

        assert result.is_exchange_verified is False
        assert result.evidence == EXECUTOR_TRACKER_EVIDENCE

    async def test_exchange_denied_protection_legs_never_verifies(self) -> None:
        subject = position()
        instance, _, _ = strategy(
            subject, EXECUTOR_API_ENABLED=True, EXECUTOR_API_URL="http://127.0.0.1:9199/decision"
        )

        async def reports_open(symbol: str) -> bool | None:
            payload = verified_position_payload(subject)
            payload["confirmation"]["protection_verified"] = False
            instance._last_executor_position_payload = payload
            return True

        instance._executor_has_position = reports_open

        result = await instance.reconcile_local_position(source="periodic")

        assert result.is_exchange_verified is False

    async def test_disabled_executor_api_is_local_only_not_executor_reported(self) -> None:
        """No query happened, so nothing may be labelled an executor report."""
        subject = position()
        instance, _, _ = strategy(subject)
        assert instance._executor_query_available() is False

        result = await instance.reconcile_local_position()

        assert result.state == RECONCILE_UNVERIFIED
        assert result.evidence == "local_only"
        assert "no executor report" in (result.detail or "")
        assert instance.current_position is subject

    async def test_unverifiable_answer_never_clears_the_position(self) -> None:
        subject = position()
        instance, _, persistence = strategy(subject)
        instance._executor_has_position = AsyncMock(return_value=None)

        result = await instance.reconcile_local_position()

        assert result.state == RECONCILE_UNVERIFIED
        assert instance.current_position is subject
        persistence.async_save_position.assert_not_awaited()

    async def test_fresh_position_inside_the_executor_grace_is_unverified_not_dead(self) -> None:
        """A just-opened position the executor has not processed yet is not an exit."""
        subject = position(entry_time=datetime.now(timezone.utc))
        instance, _, persistence = strategy(subject)
        instance._executor_has_position = AsyncMock(return_value=False)

        result = await instance.reconcile_local_position()

        assert result.state == RECONCILE_UNVERIFIED
        assert "grace" in result.detail
        assert instance.current_position is subject
        assert instance.take_state_divergence() is None
        persistence.async_save_trade_decision.assert_not_awaited()

    async def test_flat_without_any_exit_record_flags_divergence(self, tmp_path: Path) -> None:
        subject = position()
        instance, _, persistence = strategy(subject, EXECUTOR_EXIT_PATH=str(tmp_path / "missing.jsonl"))
        instance._executor_has_position = AsyncMock(return_value=False)

        result = await instance.reconcile_local_position()

        assert result.state == RECONCILE_DIVERGENCE
        assert instance.current_position is subject
        assert instance.take_state_divergence() is not None
        persistence.async_save_trade_decision.assert_not_awaited()

    async def test_confirmed_exit_is_booked_and_snapshot_invalidated(self, tmp_path: Path) -> None:
        subject = position()
        instance, _, persistence = strategy(subject, EXECUTOR_EXIT_PATH=str(exit_journal(tmp_path, [exit_record()])))
        instance._executor_has_position = AsyncMock(return_value=False)
        before = instance.position_snapshot_token()

        result = await instance.reconcile_local_position(source="periodic")

        assert result.state == RECONCILE_EXIT_BOOKED
        assert result.exit_booked is True
        assert result.exit_event_id is not None
        assert instance.current_position is None
        assert instance.position_snapshot_token() != before
        decision = persistence.async_save_trade_decision.await_args.args[0]
        assert decision.price == 84500.0

    async def test_invalid_exit_record_retains_state(self, tmp_path: Path) -> None:
        """Invalid price → divergence, position kept, snapshot token unchanged."""
        subject = position(entry_time=datetime.now(timezone.utc) - timedelta(hours=3))
        instance, _, persistence = strategy(
            subject, EXECUTOR_EXIT_PATH=str(exit_journal(tmp_path, [exit_record(exit_price=None)]))
        )
        instance._executor_has_position = AsyncMock(return_value=False)
        before = instance.position_snapshot_token()

        result = await instance.reconcile_local_position()

        assert result.state == RECONCILE_DIVERGENCE
        assert instance.current_position is subject
        assert instance.position_snapshot_token() == before
        persistence.async_save_trade_decision.assert_not_awaited()

    async def test_concurrent_callers_book_the_exit_exactly_once(self, tmp_path: Path) -> None:
        """App analysis path + monitor sync racing must yield ONE CLOSE row."""
        subject = position()
        instance, _, persistence = strategy(subject, EXECUTOR_EXIT_PATH=str(exit_journal(tmp_path, [exit_record()])))
        instance._executor_has_position = AsyncMock(return_value=False)

        results = await asyncio.gather(
            instance.reconcile_local_position(source="pre_analysis"),
            instance.reconcile_local_position(source="periodic"),
        )

        assert persistence.async_save_trade_decision.await_count == 1
        assert instance.current_position is None
        assert results[0].exit_booked or results[1].exit_booked

    async def test_token_is_stable_when_only_the_protection_changes(self) -> None:
        """Regression (15:06): an accepted UPDATE on the SAME trade must still forward.

        The strategy rebinds ``current_position`` through ``dataclasses.replace`` when
        it (or the tightening policy) applies new SL/TP. If the token moved with the
        parameters, the recommendation computed for those parameters would be thrown
        away by the app's staleness check and never reach the executor.
        """
        instance, _, _ = strategy(position())
        before = instance.position_snapshot_token()

        instance.current_position = position(stop_loss=82400.0, take_profit=85000.0)

        assert instance.position_snapshot_token() == before

    async def test_token_changes_when_the_trade_is_replaced(self) -> None:
        """A real replacement (new entry stamp) invalidates the old recommendation."""
        instance, _, _ = strategy(position())
        before = instance.position_snapshot_token()

        instance.current_position = position(entry_time=ENTRY_TIME + timedelta(minutes=1))

        assert instance.position_snapshot_token() != before

    async def test_token_changes_when_the_direction_flips(self) -> None:
        instance, _, _ = strategy(position())
        before = instance.position_snapshot_token()

        instance.current_position = position(direction="SHORT")

        assert instance.position_snapshot_token() != before

    async def test_token_is_flat_when_the_trade_is_gone(self) -> None:
        instance, _, _ = strategy(position())
        before = instance.position_snapshot_token()

        instance.current_position = None

        assert before != "flat"
        assert instance.position_snapshot_token() == "flat"

    async def test_protection_guard_refuses_a_loosening_update(self) -> None:
        """No protection version exists, so the monotone guard is the safety net."""
        instance, _, _ = strategy(position(stop_loss=82400.0))

        assert instance.protection_snapshot() == (82400.0, 84500.0)
        assert instance.protection_loosening_reason(_decision(action="UPDATE", stop_loss=80550.0)) is not None
        assert instance.protection_loosening_reason(_decision(action="UPDATE", stop_loss=83000.0)) is None
        assert instance.protection_loosening_reason(_decision(action="HOLD", stop_loss=80550.0)) is None
        assert instance.protection_loosening_reason(None) is None

    async def test_protection_guard_is_direction_aware_and_safe_when_flat(self) -> None:
        short, _, _ = strategy(position(direction="SHORT", stop_loss=82400.0, take_profit=80550.0))
        assert short.protection_loosening_reason(_decision(action="UPDATE", stop_loss=83000.0)) is not None
        assert short.protection_loosening_reason(_decision(action="UPDATE", stop_loss=82000.0)) is None

        flat, _, _ = strategy(None)
        assert flat.protection_snapshot() is None
        assert flat.protection_loosening_reason(_decision(action="UPDATE", stop_loss=1.0)) is None




class TestMonitorIndependentSync:
    """The monitor reconciles on its own cadence, without ticker or LLM help."""

    async def test_cadence_is_the_reconcile_interval_not_the_status_tick(self) -> None:
        """With a 1h status cadence the loop still wakes at the reconcile interval."""
        opened = datetime.now(timezone.utc)
        state = {
            "last_status_sent_at": opened.isoformat(),
            "last_stop_loss_check_at": opened.isoformat(),
            "last_take_profit_check_at": opened.isoformat(),
        }
        subject = position(entry_time=opened - timedelta(minutes=5))
        strategy_double = SimpleNamespace(current_position=subject)
        reconcile = AsyncMock(return_value=outcome(RECONCILE_OPEN_EXECUTOR_REPORTED))
        sleeps: list[float] = []
        checks = {"count": 0}

        async def fake_sleep(seconds: float, respect_force_analysis: bool = True) -> bool:
            sleeps.append(seconds)
            return False

        def is_running() -> bool:
            checks["count"] += 1
            return checks["count"] <= 3

        instance, _ = monitor(
            trading_strategy=strategy_double,
            state=state,
            is_running=is_running,
            interruptible_sleep=fake_sleep,
            reconcile_position=reconcile,
        )

        await instance._loop()

        assert reconcile.await_count >= 1
        assert sleeps, "the loop must wake for the reconciliation cadence"
        assert max(sleeps) <= DEFAULT_RECONCILE_INTERVAL_SECONDS + 1


    async def test_sync_books_the_exit_without_any_ticker_or_llm(self) -> None:
        """A dead ticker (and no analysis cycle at all) cannot block the sync."""
        subject = position(entry_time=datetime.now(timezone.utc) - timedelta(hours=3))
        strategy_double = MagicMock()
        strategy_double.current_position = subject
        strategy_double.take_executor_side_exit_reason = MagicMock(return_value="Executor confirmed flat: take_profit_filled @ 84500.0")
        notifier = MagicMock()
        notifier.send_position_status = AsyncMock()
        notifier.send_performance_stats = AsyncMock()
        reconcile = AsyncMock(return_value=outcome(RECONCILE_EXIT_BOOKED, exit_booked=True))
        sleeping = {"done": False}

        async def fake_sleep(seconds: float, respect_force_analysis: bool = True) -> bool:
            sleeping["done"] = True
            return False

        def is_running() -> bool:
            return True

        async def dead_ticker() -> dict[str, Any] | None:
            raise RuntimeError("ticker unavailable")

        instance, _ = monitor(
            trading_strategy=strategy_double,
            notifier=notifier,
            is_running=is_running,
            fetch_current_ticker=dead_ticker,
            interruptible_sleep=fake_sleep,
            reconcile_position=reconcile,
        )

        await instance._loop()

        assert reconcile.await_count >= 1
        assert notifier.send_performance_stats.await_count == 1
        notifier.send_position_status.assert_not_called()
        strategy_double.take_executor_side_exit_reason.assert_called_once()

    async def test_failed_reconcile_is_unverified_not_flat(self) -> None:
        subject = position()
        strategy_double = SimpleNamespace(current_position=subject)
        reconcile = AsyncMock(side_effect=RuntimeError("executor down"))
        instance, store = monitor(trading_strategy=strategy_double, reconcile_position=reconcile)

        result = await instance.sync_position_state(force=True)

        assert result is None
        assert instance.verification()[0] == "unverified"
        assert store.state == {}

    async def test_verification_freshness_downgrades_a_stale_report(self) -> None:
        strategy_double = SimpleNamespace(current_position=position())
        fresh = outcome(RECONCILE_OPEN_EXECUTOR_REPORTED)
        stale = outcome(
            RECONCILE_OPEN_EXECUTOR_REPORTED,
            checked_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        reconcile = AsyncMock(return_value=fresh)
        instance, _ = monitor(
            trading_strategy=strategy_double, reconcile_position=reconcile, reconcile_interval_seconds=120.0
        )

        await instance.sync_position_state(force=True)
        reported_state, reported_at, _ = instance.verification()
        assert reported_state == "executor_reported"
        assert reported_state != "verified"
        assert reported_at is not None

        instance._last_reconcile_outcome = stale
        state, verified_at, detail = instance.verification()
        assert state == "unverified"
        assert verified_at == stale.checked_at
        assert "stale" in detail

    async def test_monitor_publishes_fresh_venue_backed_confirmation(self) -> None:
        subject = position()
        verified_at = datetime.now(timezone.utc)
        verified = outcome(
            RECONCILE_OPEN_EXECUTOR_REPORTED,
            exchange_verified_at=verified_at,
            evidence=EXECUTOR_VENUE_EVIDENCE,
        )
        instance, _ = monitor(
            trading_strategy=SimpleNamespace(current_position=subject),
            reconcile_position=AsyncMock(return_value=verified),
            reconcile_interval_seconds=120.0,
        )

        await instance.sync_position_state(force=True)

        state, reported_at, _ = instance.verification()
        assert state == "exchange_verified"
        assert reported_at == verified_at

    async def test_no_open_card_after_the_sync_detected_the_final_exit(self) -> None:
        subject = position(entry_time=datetime.now(timezone.utc) - timedelta(hours=3))
        strategy_double = MagicMock()
        strategy_double.current_position = subject
        strategy_double.take_executor_side_exit_reason = MagicMock(return_value="Executor confirmed flat: take_profit_filled @ 84500.0")
        notifier = MagicMock()
        notifier.send_position_status = AsyncMock()
        notifier.send_performance_stats = AsyncMock()

        async def sync() -> LocalPositionReconciliation:
            strategy_double.current_position = None
            return outcome(RECONCILE_EXIT_BOOKED, exit_booked=True)

        state = {"last_status_sent_at": datetime.now(timezone.utc).isoformat()}
        instance, _ = monitor(
            trading_strategy=strategy_double,
            notifier=notifier,
            state=state,
            is_running=lambda: True,
            reconcile_position=AsyncMock(side_effect=sync),
        )

        await instance._loop()

        notifier.send_position_status.assert_not_called()
        assert notifier.send_performance_stats.await_count == 1

    async def test_unverified_status_card_says_so(self) -> None:
        subject = position()
        strategy_double = SimpleNamespace(current_position=subject)
        notifier = MagicMock()
        notifier.send_position_status = AsyncMock()
        instance, _ = monitor(
            trading_strategy=strategy_double,
            notifier=notifier,
            reconcile_position=AsyncMock(return_value=outcome(RECONCILE_UNVERIFIED, detail="query failed")),
        )

        await instance.handle_new_position(100.0)

        notifier.send_position_status.assert_awaited_once()
        kwargs = notifier.send_position_status.await_args.kwargs
        assert kwargs["verification"] == "unverified"
        assert kwargs["verification_detail"] == "query failed"

    async def test_open_card_reports_executor_not_exchange(self) -> None:
        subject = position()
        strategy_double = SimpleNamespace(current_position=subject)
        notifier = MagicMock()
        notifier.send_position_status = AsyncMock()
        instance, _ = monitor(
            trading_strategy=strategy_double,
            notifier=notifier,
            reconcile_position=AsyncMock(return_value=outcome(RECONCILE_OPEN_EXECUTOR_REPORTED)),
        )

        await instance.handle_new_position(100.0)

        kwargs = notifier.send_position_status.await_args.kwargs
        assert kwargs["verification"] == "executor_reported"
        assert kwargs["verification"] != "verified"
        assert instance._task is not None
        await instance.stop()
