"""Wave-2 bookkeeping discipline: intent != execution, evidence-only confirmation.

Local accounting rules under test:

* a command the bot SENDS (entry/UPDATE/CLOSE) is an intent — the local position is
  NOT changed until the executor supplies evidence, and an unanswered/failed command is
  ``unknown`` (position kept + operator alert), never a silent close or erase;
* only executor-side evidence (verdict journal / exit journal) may confirm anything —
  the local monitor, a ticker price or a repeated report may not;
* the same confirmation processed twice (app + monitor, retry, restart) books ONCE;
* an exit without a real fill price AND quantity is not booked (position kept + alarm);
* a write failure is NOT an execution.

Every test uses temporary journals, so nothing here touches the live repo or data.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.trading import executor_reconciliation
from src.trading.data_models import MarketConditions, Position
from src.trading.executor_reconciliation import (
    INTENT_ACTION_CLOSE,
    INTENT_CONFIRMED,
    INTENT_PENDING,
    INTENT_UNKNOWN,
    exit_record_from_receipt,
    position_identity,
    verify_exit_record,
)
from src.trading.trading_strategy import TradingStrategy
from tests.conftest import (
    make_config,
    make_market_conditions,
    make_position,
    mock_brain,
    mock_persistence,
    mock_statistics,
    null_logger,
)

ENTRY_TIME = datetime(2026, 9, 21, 8, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _instant_intent_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    """The forwarded-command verdict poll waits seconds in production; the suite must not."""
    monkeypatch.setattr(executor_reconciliation, "INTENT_CONFIRM_DELAY", 0.0)


def _position(**overrides: Any) -> Position:
    values: dict[str, Any] = {
        "symbol": "BTC/USDT",
        "direction": "LONG",
        "entry_price": 81200.45,
        "stop_loss": 78800.0,
        "take_profit": 84500.0,
        "size": 0.00554,
        "entry_time": ENTRY_TIME,
        "conditions_at_entry": make_market_conditions(),
    }
    values.update(overrides)
    return make_position(**values)


def _strategy(
    tmp_path: Path,
    position: Position | None = None,
    **config_overrides: Any,
) -> tuple[TradingStrategy, MagicMock, MagicMock]:
    """Real strategy on shared doubles with an isolated intent journal."""
    logger = null_logger()
    persistence = mock_persistence()
    statistics = mock_statistics()
    statistics.get_current_capital.return_value = 10000.0
    brain = mock_brain()
    brain.get_dynamic_thresholds.return_value = {}
    config = make_config(
        **{
            "EXECUTOR_API_ENABLED": True,
            "EXECUTOR_API_URL": "http://127.0.0.1:9199/decision",
            "BOT_INTENT_JOURNAL_PATH": str(tmp_path / "bot_position_intents.jsonl"),
            **config_overrides,
        }
    )
    strategy = TradingStrategy(
        logger=logger,
        persistence=persistence,
        brain_service=brain,
        statistics_service=statistics,
        memory_service=MagicMock(),
        risk_manager=MagicMock(),
        config=config,
        position_extractor=MagicMock(),
    )
    if position is not None:
        strategy.current_position = position
    return strategy, logger, persistence


def _exit_record(**overrides: Any) -> dict[str, Any]:
    """One executor exit-journal line (price AND quantity from the fill)."""
    record: dict[str, Any] = {
        "timestamp": "2026-09-21T09:32:51.956000+00:00",
        "symbol": "BTC/USDT",
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


def _journal(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8")
    return path


def _verdicts(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")
    return path


async def _close_signal(strategy: TradingStrategy) -> Any:
    """Run the CLOSE-signal branch of the existing-position handler."""
    return await strategy._handle_existing_position(
        signal="CLOSE",
        confidence="HIGH",
        stop_loss=None,
        take_profit=None,
        current_price=84500.0,
        symbol="BTC/USDT",
        reasoning="Take profit reached, closing.",
        market_conditions=MarketConditions(),
    )


class TestCloseIntentIsNotAnExecution:
    """A CLOSE the bot sends stays pending until the executor's evidence arrives."""

    async def test_close_signal_without_executor_evidence_keeps_the_position(self, tmp_path: Path) -> None:
        """No receipt -> pending/unknown, position intact, no CLOSE row, no stats run."""
        position = _position()
        strategy, _, persistence = _strategy(tmp_path, position)
        strategy._executor_has_position = AsyncMock(return_value=True)

        decision = await _close_signal(strategy)

        assert decision is not None and decision.action == "CLOSE"
        assert strategy.current_position is position, "an unconfirmed CLOSE must not erase the position"
        persistence.async_save_position.assert_not_awaited()
        persistence.async_save_trade_decision.assert_not_awaited()
        strategy.statistics_service.recalculate.assert_not_called()

        intent = strategy.position_intents().get_by_order(decision.order_id)
        assert intent is not None and intent.state == INTENT_PENDING
        assert intent.action == INTENT_ACTION_CLOSE

        state = await strategy.resolve_position_intents_after_forward(
            order_id=decision.order_id, delivered=True
        )
        assert state == INTENT_UNKNOWN
        assert strategy.current_position is position
        assert strategy.take_unconfirmed_intent_alert() is not None
        assert persistence.async_save_trade_decision.await_count == 0

    async def test_close_verdict_that_lands_after_the_forward_still_confirms(self, tmp_path: Path) -> None:
        """The executor's queue tick can reach ~10s: the forwarding read is retried."""
        position = _position()
        strategy, _, persistence = _strategy(
            tmp_path,
            position,
            EXECUTOR_EXIT_PATH=str(_journal(tmp_path / "exits.jsonl", [_exit_record()])),
        )
        strategy._executor_has_position = AsyncMock(return_value=True)
        decision = await _close_signal(strategy)
        _verdicts(
            tmp_path / "verdicts.jsonl",
            [{"order_id": decision.order_id, "verdict": "executed", "reason": ""}],
        )
        strategy.config.EXECUTOR_VERDICT_PATH = str(tmp_path / "verdicts.jsonl")
        reads = {"count": 0}
        real_read = strategy._read_executor_verdict_entry

        def late_verdict(order_id: str) -> dict | None:
            reads["count"] += 1
            return None if reads["count"] == 1 else real_read(order_id)

        strategy._read_executor_verdict_entry = late_verdict

        state = await strategy.resolve_position_intents_after_forward(
            order_id=decision.order_id, delivered=True
        )

        assert reads["count"] == 2, "the journal must be read again, not abandoned after one look"
        assert state == INTENT_CONFIRMED
        assert strategy.current_position is None
        persistence.async_save_trade_decision.assert_awaited_once()

    async def test_close_without_a_verdict_polls_every_attempt_then_stays_unknown(self, tmp_path: Path) -> None:
        """A short grace window, then UNKNOWN: an unanswered command is never a close."""
        position = _position()
        strategy, _, persistence = _strategy(tmp_path, position)
        strategy._executor_has_position = AsyncMock(return_value=True)
        decision = await _close_signal(strategy)
        reads = {"count": 0}

        def never_a_verdict(order_id: str) -> dict | None:
            reads["count"] += 1
            return None

        strategy._read_executor_verdict_entry = never_a_verdict

        state = await strategy.resolve_position_intents_after_forward(
            order_id=decision.order_id, delivered=True
        )

        assert reads["count"] == executor_reconciliation.INTENT_CONFIRM_ATTEMPTS + 1
        assert state == INTENT_UNKNOWN
        assert strategy.current_position is position
        assert persistence.async_save_position.await_count == 0
        assert persistence.async_save_trade_decision.await_count == 0
        assert strategy.take_unconfirmed_intent_alert() is not None

    async def test_undelivered_close_is_unknown_and_never_erases_the_position(self, tmp_path: Path) -> None:
        """A failed forward (file fallback / no response) is UNKNOWN, not a close."""
        position = _position()
        strategy, _, persistence = _strategy(tmp_path, position)
        strategy._executor_has_position = AsyncMock(return_value=True)
        decision = await _close_signal(strategy)

        state = await strategy.resolve_position_intents_after_forward(
            order_id=decision.order_id, delivered=False
        )

        assert state == INTENT_UNKNOWN
        assert strategy.current_position is position
        assert persistence.async_save_position.await_count == 0
        assert strategy.take_unconfirmed_intent_alert() is not None

    async def test_confirmed_close_books_once_from_the_executor_receipt(self, tmp_path: Path) -> None:
        """The receipt + a validated fill book the trade exactly once."""
        position = _position()
        strategy, _, persistence = _strategy(
            tmp_path,
            position,
            EXECUTOR_EXIT_PATH=str(_journal(tmp_path / "exits.jsonl", [_exit_record()])),
        )
        strategy._executor_has_position = AsyncMock(return_value=True)
        decision = await _close_signal(strategy)
        _verdicts(
            tmp_path / "verdicts.jsonl",
            [{"order_id": decision.order_id, "verdict": "executed", "reason": ""}],
        )
        strategy.config.EXECUTOR_VERDICT_PATH = str(tmp_path / "verdicts.jsonl")

        state = await strategy.resolve_position_intents_after_forward(
            order_id=decision.order_id, delivered=True
        )

        assert state == INTENT_CONFIRMED
        assert strategy.current_position is None
        persistence.async_save_trade_decision.assert_awaited_once()
        booked = persistence.async_save_trade_decision.await_args.args[0]
        assert booked.action == "CLOSE_LONG"
        assert booked.price == 84500.0, "the booked price is the fill, never the entry/ticker"
        assert strategy.statistics_service.recalculate.call_count == 1

        again = await strategy.resolve_position_intents_after_forward(
            order_id=decision.order_id, delivered=True
        )
        assert again == INTENT_CONFIRMED
        assert persistence.async_save_trade_decision.await_count == 1

    async def test_refused_close_keeps_the_position_and_alerts(self, tmp_path: Path) -> None:
        """An executor receipt of blocked/error leaves the trade alone."""
        position = _position()
        strategy, _, persistence = _strategy(
            tmp_path, position, EXECUTOR_EXIT_PATH=str(tmp_path / "missing.jsonl")
        )
        strategy._executor_has_position = AsyncMock(return_value=True)
        decision = await _close_signal(strategy)
        _verdicts(
            tmp_path / "verdicts.jsonl",
            [{"order_id": decision.order_id, "verdict": "blocked", "reason": "safety guard"}],
        )
        strategy.config.EXECUTOR_VERDICT_PATH = str(tmp_path / "verdicts.jsonl")

        state = await strategy.resolve_position_intents_after_forward(
            order_id=decision.order_id, delivered=True
        )

        assert state == "refused"
        assert strategy.current_position is position
        assert persistence.async_save_trade_decision.await_count == 0
        alert = strategy.take_unconfirmed_intent_alert()
        assert alert is not None and "CLOSE" in alert


class TestEvidenceSemaphore:
    """Only executor-side evidence may move a local state."""

    async def test_local_monitor_source_is_refused(self, tmp_path: Path) -> None:
        """A local-monitor/repeat confirmation is rejected and changes nothing."""
        position = _position()
        strategy, logger, _ = _strategy(tmp_path, position)
        intent = strategy.register_position_intent(
            INTENT_ACTION_CLOSE, position.symbol, position_id=position_identity(position)
        )

        for source in ("local_monitor", "soft_exit_check", "ticker", "repeat", "app_cycle"):
            assert strategy.confirm_position_intent(intent.key, source=source) is None
            assert strategy.refuse_position_intent(intent.key, source=source) is None
            assert strategy.mark_position_intent_unknown(intent.key, source=source) is None

        assert strategy.position_intents().get(intent.key).state == INTENT_PENDING
        assert any("REFUSED confirmation" in str(call) for call in logger.critical.call_args_list)

    async def test_executor_source_confirms(self, tmp_path: Path) -> None:
        """The executor verdict journal is an admitted source."""
        position = _position()
        strategy, _, _ = _strategy(tmp_path, position)
        intent = strategy.register_position_intent(
            "ENTRY", position.symbol, order_id="order-1", position_id=position_identity(position)
        )

        confirmed = strategy.confirm_position_intent(
            intent.key, source="executor_verdict_journal", detail="executed"
        )

        assert confirmed is not None and confirmed.state == INTENT_CONFIRMED
        assert strategy.unbooked_position_intents() == []


class TestExitBookingNeedsAFill:
    """No booking without the real exit price AND quantity from the fill."""

    async def test_missing_exit_price_keeps_the_position_and_alarms(self, tmp_path: Path) -> None:
        position = _position()
        strategy, _, persistence = _strategy(
            tmp_path,
            position,
            EXECUTOR_EXIT_PATH=str(_journal(tmp_path / "exits.jsonl", [_exit_record(exit_price=None)])),
        )

        booked = await strategy.book_executor_side_exit(
            position, 84500.0, MarketConditions(), "Executor confirmed flat"
        )

        assert booked is False
        assert strategy.current_position is position
        assert persistence.async_save_trade_decision.await_count == 0
        assert strategy.take_state_divergence() is not None

    async def test_missing_fill_quantity_keeps_the_position_and_alarms(self, tmp_path: Path) -> None:
        """Wave-2 rule: the local size is not a fill amount either."""
        position = _position()
        record = _exit_record()
        record.pop("quantity")
        strategy, _, persistence = _strategy(
            tmp_path, position, EXECUTOR_EXIT_PATH=str(_journal(tmp_path / "exits.jsonl", [record]))
        )

        booked = await strategy.book_executor_side_exit(
            position, 84500.0, MarketConditions(), "Executor confirmed flat"
        )

        assert booked is False
        assert strategy.current_position is position
        assert persistence.async_save_trade_decision.await_count == 0
        assert strategy.take_state_divergence() is not None

    async def test_quantity_mismatch_is_booked_and_flagged(self, tmp_path: Path) -> None:
        """A fill amount differing from the local size is booked from the FILL + flagged."""
        position = _position()
        strategy, _, persistence = _strategy(
            tmp_path,
            position,
            EXECUTOR_EXIT_PATH=str(
                _journal(tmp_path / "exits.jsonl", [_exit_record(quantity=0.0099)])
            ),
        )

        booked = await strategy.book_executor_side_exit(
            position, 84500.0, MarketConditions(), "Executor confirmed flat"
        )

        assert booked is True
        assert persistence.async_save_trade_decision.await_count == 1
        assert strategy.take_state_divergence() is not None
        decision = persistence.async_save_trade_decision.await_args.args[0]
        assert decision.quantity == pytest.approx(0.0099)
        assert decision.quantity != pytest.approx(position.size)
        assert "0.00990000" in decision.reasoning
        assert "0.00554000" in decision.reasoning


class TestLocalExitNeverBooksWithoutFillEvidence:
    """Wave 3: a local SL/TP condition REQUESTS a close and never fabricates one.

    The old path booked a CLOSE at the ticker price. Nothing here may book anything:
    with the executor live the condition becomes a pending CLOSE intent (position
    kept, statistics untouched) and without it the state is explicitly ``unknown``
    with a divergence alarm — never a silent reset and never a zeroed trade.
    """

    async def test_local_condition_with_executor_live_requests_and_keeps_the_position(
        self, tmp_path: Path
    ) -> None:
        """Hit -> CLOSE intent + position kept + no CLOSE row + no statistics entry."""
        position = _position()
        strategy, _, persistence = _strategy(tmp_path, position)

        reason = await strategy.check_take_profit(84500.0)

        assert reason is None, "a ticker price must never book a local close"
        assert strategy.current_position is position
        persistence.async_save_trade_decision.assert_not_awaited()
        strategy.statistics_service.recalculate.assert_not_called()

        request = strategy.take_local_exit_request()
        assert request is not None
        assert (request.reason, request.state) == ("take_profit", INTENT_PENDING)
        assert request.order_id is not None
        intent = strategy.position_intents().get(request.intent_key)
        assert intent is not None and intent.action == INTENT_ACTION_CLOSE
        assert intent.state == INTENT_PENDING
        assert intent.payload["source"] == "local_monitor"

        recommendation = strategy.take_pending_local_close_decision()
        assert recommendation is not None and recommendation.action == "CLOSE"
        assert strategy.take_pending_local_close_decision() is None, "one-shot read"
        assert strategy.take_unconfirmed_intent_alert() is not None

    async def test_local_condition_without_integration_is_unknown_and_alarms(
        self, tmp_path: Path
    ) -> None:
        """Integration off -> no exit may be invented: position kept + unknown + alarm."""
        position = _position()
        strategy, _, persistence = _strategy(
            tmp_path, position, EXECUTOR_API_ENABLED=False, EXECUTOR_API_URL=""
        )

        reason = await strategy.check_stop_loss(78000.0)

        assert reason is None
        assert strategy.current_position is position
        persistence.async_save_trade_decision.assert_not_awaited()
        strategy.statistics_service.recalculate.assert_not_called()

        request = strategy.take_local_exit_request()
        assert request is not None
        assert request.reason == "stop_loss"
        assert request.state == INTENT_UNKNOWN
        assert request.order_id is None
        assert strategy.position_intents().get(request.intent_key).state == INTENT_UNKNOWN
        assert strategy.take_state_divergence() is not None, "operator divergence alarm"
        assert strategy.take_unconfirmed_intent_alert() is not None

    async def test_repeated_local_condition_keeps_one_intent_per_trade(self, tmp_path: Path) -> None:
        """Double processing (app + monitor, retry) books nothing and keeps one intent."""
        position = _position()
        strategy, _, persistence = _strategy(tmp_path, position)

        await strategy.check_take_profit(84500.0)
        first = strategy.take_local_exit_request()
        await strategy.check_take_profit(84600.0)
        second = strategy.take_local_exit_request()

        assert first is not None and second is not None
        assert first.intent_key == second.intent_key
        assert first.order_id == second.order_id, "the same command id is reused"
        assert strategy.current_position is position
        assert persistence.async_save_trade_decision.await_count == 0
        assert len(strategy.unbooked_position_intents()) == 1

    async def test_fill_evidence_booked_once_at_the_fill_price(self, tmp_path: Path) -> None:
        """Only a real fill (price AND quantity) books the CLOSE, exactly once."""
        position = _position()
        strategy, _, persistence = _strategy(
            tmp_path,
            position,
            EXECUTOR_EXIT_PATH=str(_journal(tmp_path / "exits.jsonl", [_exit_record()])),
        )

        await strategy.check_take_profit(84500.0)
        recommendation = strategy.take_pending_local_close_decision()
        _verdicts(
            tmp_path / "verdicts.jsonl",
            [{"order_id": recommendation.order_id, "verdict": "executed", "reason": ""}],
        )
        strategy.config.EXECUTOR_VERDICT_PATH = str(tmp_path / "verdicts.jsonl")

        state = await strategy.resolve_position_intents_after_forward(
            order_id=recommendation.order_id, delivered=True
        )
        replay = await strategy.resolve_position_intents_after_forward(
            order_id=recommendation.order_id, delivered=True
        )

        assert state == INTENT_CONFIRMED and replay == INTENT_CONFIRMED
        assert strategy.current_position is None
        persistence.async_save_trade_decision.assert_awaited_once()
        booked = persistence.async_save_trade_decision.await_args.args[0]
        assert booked.action == "CLOSE_LONG"
        assert booked.price == 84500.0, "the fill price, never the ticker price"

    async def test_confirmed_close_without_a_usable_fill_keeps_the_trade_alarmed(
        self, tmp_path: Path
    ) -> None:
        """Receipt executed but no fill price -> position kept + alarm (never guessed)."""
        position = _position()
        record = _exit_record(exit_price=None)
        strategy, _, persistence = _strategy(
            tmp_path, position, EXECUTOR_EXIT_PATH=str(_journal(tmp_path / "exits.jsonl", [record]))
        )

        await strategy.check_take_profit(84500.0)
        recommendation = strategy.take_pending_local_close_decision()
        _verdicts(
            tmp_path / "verdicts.jsonl",
            [{"order_id": recommendation.order_id, "verdict": "executed", "reason": ""}],
        )
        strategy.config.EXECUTOR_VERDICT_PATH = str(tmp_path / "verdicts.jsonl")

        state = await strategy.resolve_position_intents_after_forward(
            order_id=recommendation.order_id, delivered=True
        )

        assert state == INTENT_UNKNOWN
        assert strategy.current_position is position
        assert persistence.async_save_trade_decision.await_count == 0
        assert strategy.take_unconfirmed_intent_alert() is not None


class TestIdempotencyAcrossRepeatsAndRestart:
    """The same event books once, in this run and after a restart."""

    async def test_the_same_exit_event_books_once_in_one_run(self, tmp_path: Path) -> None:
        position = _position()
        exit_path = _journal(tmp_path / "exits.jsonl", [_exit_record()])
        strategy, _, persistence = _strategy(tmp_path, position, EXECUTOR_EXIT_PATH=str(exit_path))

        first = await strategy.book_executor_side_exit(
            position, 84500.0, MarketConditions(), "Executor confirmed flat"
        )
        second = await strategy.book_executor_side_exit(
            position, 84500.0, MarketConditions(), "Executor confirmed flat"
        )

        assert (first, second) == (True, False)
        assert persistence.async_save_trade_decision.await_count == 1

    async def test_restart_replays_the_journal_without_double_booking(self, tmp_path: Path) -> None:
        """A fresh process (new strategy) reconstructs the booked event from the journal."""
        position = _position()
        exit_path = _journal(tmp_path / "exits.jsonl", [_exit_record()])
        first_strategy, _, first_persistence = _strategy(
            tmp_path, position, EXECUTOR_EXIT_PATH=str(exit_path)
        )
        assert await first_strategy.book_executor_side_exit(
            position, 84500.0, MarketConditions(), "Executor confirmed flat"
        ) is True
        assert first_persistence.async_save_trade_decision.await_count == 1

        restarted, _, restarted_persistence = _strategy(
            tmp_path, _position(), EXECUTOR_EXIT_PATH=str(exit_path)
        )
        booked_again = await restarted.book_executor_side_exit(
            _position(), 84500.0, MarketConditions(), "Executor confirmed flat"
        )

        assert booked_again is False
        assert restarted_persistence.async_save_trade_decision.await_count == 0
        recalled = restarted.position_intents().get(
            restarted.intent_identity(INTENT_ACTION_CLOSE, position.symbol,
                                      position_id=position_identity(position))
        )
        assert recalled is not None and recalled.state == INTENT_CONFIRMED

    async def test_a_write_failure_is_not_an_execution(self, tmp_path: Path) -> None:
        """A failed CLOSE-row write must not count as booked, and must not clear state."""
        position = _position()
        strategy, _, persistence = _strategy(
            tmp_path,
            position,
            EXECUTOR_EXIT_PATH=str(_journal(tmp_path / "exits.jsonl", [_exit_record()])),
        )
        persistence.async_save_trade_decision = AsyncMock(
            side_effect=[RuntimeError("disk full"), None]
        )

        with pytest.raises(RuntimeError):
            await strategy.book_executor_side_exit(
                position, 84500.0, MarketConditions(), "Executor confirmed flat"
            )

        assert strategy.current_position is position, "a failed write booked nothing"
        assert strategy.position_intents().pending() or strategy.position_intents().booked_events == set()

        assert await strategy.book_executor_side_exit(
            position, 84500.0, MarketConditions(), "Executor confirmed flat"
        ) is True
        assert strategy.current_position is None


class TestAnalysisCloseIsNotAnExecution:
    """Wave 4: the last unproven booking path — the analysis ``CLOSE`` signal.

    With the executor integration OFF there is no command and no confirmation channel
    at all (the bot never places exchange orders itself: ``ExchangeManager`` only reads
    markets and the only order path is ``ExecutorHandler``, disabled when
    ``EXECUTOR_API_ENABLED=False``). The old branch booked a CLOSE at the analysis price
    and marked it ``local_only`` — a trade nobody verified. It must now stay UNKNOWN:
    position kept, divergence alarm, zero statistics entries.
    """

    async def test_close_signal_without_integration_books_nothing(self, tmp_path: Path) -> None:
        """Integration off -> no execution, position kept, UNKNOWN + divergence alarm."""
        position = _position()
        strategy, logger, persistence = _strategy(
            tmp_path, position, EXECUTOR_API_ENABLED=False, EXECUTOR_API_URL=""
        )

        decision = await _close_signal(strategy)

        assert decision is None, "nothing may be forwarded or booked as an execution"
        assert strategy.current_position is position, "an unproven CLOSE must keep the position"
        persistence.async_save_position.assert_not_awaited()
        persistence.async_save_trade_decision.assert_not_awaited()
        strategy.statistics_service.recalculate.assert_not_called()
        strategy.statistics_service.get_statistics.assert_not_called()

        request = strategy.take_local_exit_request()
        assert request is not None
        assert (request.reason, request.state) == ("analysis_signal", INTENT_UNKNOWN)
        assert request.order_id is None
        intent = strategy.position_intents().get(request.intent_key)
        assert intent is not None and intent.action == INTENT_ACTION_CLOSE
        assert intent.state == INTENT_UNKNOWN
        assert intent.evidence == "analysis_signal"
        assert strategy.take_state_divergence() is not None, "operator divergence alarm"
        assert strategy.take_unconfirmed_intent_alert() is not None
        assert any("NOTHING booked locally" in str(call) for call in logger.critical.call_args_list)

    async def test_repeated_close_signal_keeps_one_intent_and_books_nothing(
        self, tmp_path: Path
    ) -> None:
        """App + monitor + retry on the same trade: one intent key, still no booking."""
        position = _position()
        strategy, _, persistence = _strategy(
            tmp_path, position, EXECUTOR_API_ENABLED=False, EXECUTOR_API_URL=""
        )

        await _close_signal(strategy)
        first = strategy.take_local_exit_request()
        await _close_signal(strategy)
        second = strategy.take_local_exit_request()

        assert first is not None and second is not None
        assert first.intent_key == second.intent_key
        assert strategy.current_position is position
        assert persistence.async_save_trade_decision.await_count == 0
        assert strategy.statistics_service.recalculate.call_count == 0
        assert len(strategy.unbooked_position_intents()) == 1
        assert first.intent_key == strategy.intent_identity(
            INTENT_ACTION_CLOSE, position.symbol, position_id=position_identity(position)
        ), "the analysis CLOSE shares the monitor's per-trade key"

    async def test_close_signal_with_integration_is_a_command_only(self, tmp_path: Path) -> None:
        """Integration live -> pending CLOSE command, still nothing booked locally."""
        position = _position()
        strategy, _, persistence = _strategy(tmp_path, position)
        strategy._executor_has_position = AsyncMock(return_value=True)

        decision = await _close_signal(strategy)

        assert decision is not None and decision.action == "CLOSE"
        assert strategy.current_position is position
        persistence.async_save_position.assert_not_awaited()
        persistence.async_save_trade_decision.assert_not_awaited()
        strategy.statistics_service.recalculate.assert_not_called()
        intent = strategy.position_intents().get_by_order(decision.order_id)
        assert intent is not None and intent.state == INTENT_PENDING
        assert strategy.take_local_exit_request() is None

    async def test_confirmed_close_books_once_at_the_fill_price(self, tmp_path: Path) -> None:
        """Only the executor's receipt + a validated fill book the analysis CLOSE."""
        position = _position()
        strategy, _, persistence = _strategy(
            tmp_path,
            position,
            EXECUTOR_EXIT_PATH=str(_journal(tmp_path / "exits.jsonl", [_exit_record(fees=[0.0])])),
        )
        strategy._executor_has_position = AsyncMock(return_value=True)
        decision = await _close_signal(strategy)
        _verdicts(
            tmp_path / "verdicts.jsonl",
            [{"order_id": decision.order_id, "verdict": "executed", "reason": ""}],
        )
        strategy.config.EXECUTOR_VERDICT_PATH = str(tmp_path / "verdicts.jsonl")

        state = await strategy.resolve_position_intents_after_forward(
            order_id=decision.order_id, delivered=True
        )
        replay = await strategy.resolve_position_intents_after_forward(
            order_id=decision.order_id, delivered=True
        )

        assert state == INTENT_CONFIRMED and replay == INTENT_CONFIRMED
        assert strategy.current_position is None
        persistence.async_save_trade_decision.assert_awaited_once()
        booked = persistence.async_save_trade_decision.await_args.args[0]
        assert booked.action == "CLOSE_LONG"
        assert (booked.price, booked.quantity) == (84500.0, 0.00554)
        assert strategy.statistics_service.recalculate.call_count == 1
        assert booked.fee == 0.0

    async def test_unconfirmed_close_signal_survives_a_restart(self, tmp_path: Path) -> None:
        """The UNKNOWN state is persisted: after a restart the trade is still unbooked."""
        position = _position()
        strategy, _, _ = _strategy(
            tmp_path, position, EXECUTOR_API_ENABLED=False, EXECUTOR_API_URL=""
        )
        await _close_signal(strategy)
        intent_key = strategy.take_local_exit_request().intent_key

        restarted, _, restarted_persistence = _strategy(
            tmp_path, _position(), EXECUTOR_API_ENABLED=False, EXECUTOR_API_URL=""
        )
        await _close_signal(restarted)

        recalled = restarted.position_intents().get(intent_key)
        assert recalled is not None and recalled.state == INTENT_UNKNOWN
        assert restarted_persistence.async_save_trade_decision.await_count == 0
        assert restarted.current_position is not None


class TestCommissionIsEvidenceOnly:
    """Wave 4: a commission comes from real fill/order data — never from a rate.

    The incident booked a synthesised 0.075% fee (2.851875 USDC) for a trade whose real
    fee was 0. Missing fee data is UNKNOWN: ``None`` in the booked row, stated as unknown
    in the reasoning, and it never blocks a validated exit.
    """

    async def test_exit_without_fee_data_books_an_unknown_commission(self, tmp_path: Path) -> None:
        """A validated exit with no fee data books UNKNOWN, not 0.0 and not the rate."""
        position = _position()
        strategy, _, persistence = _strategy(
            tmp_path,
            position,
            EXECUTOR_EXIT_PATH=str(_journal(tmp_path / "exits.jsonl", [_exit_record()])),
        )

        booked = await strategy.book_executor_side_exit(
            position, 84500.0, MarketConditions(), "Executor confirmed flat"
        )

        assert booked is True
        decision = persistence.async_save_trade_decision.await_args.args[0]
        assert decision.fee is None
        assert "unknown" in decision.reasoning.lower()
        rate_fee = decision.price * decision.quantity * strategy.config.TRANSACTION_FEE_PERCENT
        assert rate_fee > 0
        assert decision.fee != pytest.approx(rate_fee)
        assert f"${rate_fee:.4f}" not in decision.reasoning

    @pytest.mark.parametrize(
        ("fees", "expected_total"),
        [
            pytest.param([0.0], 0.0, id="real-zero-fee-from-the-exchange"),
            pytest.param([0.0, 0.0], 0.0, id="two-fills-both-zero"),
            pytest.param([0.00012, 0.00034], pytest.approx(0.00046), id="two-fills-summed"),
            pytest.param(0.00075, pytest.approx(0.00075), id="scalar-total-fee"),
        ],
    )
    async def test_exit_with_real_fee_data_books_the_actual_fees(
        self, tmp_path: Path, fees: Any, expected_total: Any
    ) -> None:
        """Actual fee data (including a real 0) is what gets booked."""
        position = _position()
        strategy, _, persistence = _strategy(
            tmp_path,
            position,
            EXECUTOR_EXIT_PATH=str(
                _journal(tmp_path / "exits.jsonl", [_exit_record(fees=fees)])
            ),
        )

        assert await strategy.book_executor_side_exit(
            position, 84500.0, MarketConditions(), "Executor confirmed flat"
        ) is True

        decision = persistence.async_save_trade_decision.await_args.args[0]
        assert decision.fee == expected_total
        assert "executor_exit_journal" in decision.reasoning

    @pytest.mark.parametrize(
        "fees",
        [
            pytest.param([], id="empty-list-is-unknown"),
            pytest.param(None, id="missing-field-is-unknown"),
            pytest.param(["not-a-number"], id="malformed-entry-is-unknown"),
            pytest.param([-1.0], id="negative-fee-is-unknown"),
            pytest.param(float("nan"), id="nan-fee-is-unknown"),
            pytest.param([True], id="bool-fee-is-unknown"),
            pytest.param(
                [{"fee": 0.0002, "currency": "USDC"}],
                id="currency-tagged-fee-object-is-unknown",
            ),
        ],
    )
    async def test_unusable_fee_data_is_unknown_and_never_blocks_the_exit(
        self, tmp_path: Path, fees: Any
    ) -> None:
        """The fee never gates the exit: it is booked with an UNKNOWN commission."""
        position = _position()
        record = _exit_record(fees=fees)
        strategy, _, persistence = _strategy(
            tmp_path, position, EXECUTOR_EXIT_PATH=str(_journal(tmp_path / "exits.jsonl", [record]))
        )

        assert await strategy.book_executor_side_exit(
            position, 84500.0, MarketConditions(), "Executor confirmed flat"
        ) is True

        decision = persistence.async_save_trade_decision.await_args.args[0]
        assert decision.fee is None
        assert strategy.current_position is None, "an unusable FEE must not block a real exit"

    def test_exit_record_fee_is_parsed_from_the_journal_only(self) -> None:
        """``verify_exit_record`` carries real fees through and reports the absence."""
        verified, rejection = verify_exit_record(_exit_record(fees=[0.0001, 0.0002]))
        assert rejection is None and verified is not None
        assert verified.exit_fee == pytest.approx(0.0003)

        no_fees, rejection = verify_exit_record(_exit_record())
        assert rejection is None and no_fees is not None
        assert no_fees.exit_fee is None, "no ``fees`` field means UNKNOWN, never 0.0"


class TestNoRateDerivedCommissionSurvives:
    """The 0.075% ``TRANSACTION_FEE_PERCENT`` may not appear in a booked row anywhere."""

    async def test_entry_and_exit_rows_never_carry_the_configured_rate(
        self, tmp_path: Path
    ) -> None:
        """A real RiskManager entry + a fee-less exit: both commissions are UNKNOWN."""
        from src.managers.risk_manager import RiskManager

        strategy, _, persistence = _strategy(
            tmp_path,
            _position(),
            TRANSACTION_FEE_PERCENT=0.00075,
            EXECUTOR_API_ENABLED=False,
            EXECUTOR_API_URL="",
            EXECUTOR_EXIT_PATH=str(_journal(tmp_path / "exits.jsonl", [_exit_record()])),
        )
        config = strategy.config
        strategy.risk_manager = RiskManager(logger=null_logger(), config=config)

        assessment = strategy.risk_manager.calculate_entry_parameters(
            signal="BUY",
            current_price=81200.45,
            capital=10000.0,
            confidence="HIGH",
            market_conditions=make_market_conditions(),
        )
        assert assessment.entry_fee is None
        assert assessment.quote_amount * config.TRANSACTION_FEE_PERCENT > 0

        await strategy.book_executor_side_exit(
            strategy.current_position, 84500.0, MarketConditions(), "Executor confirmed flat"
        )
        decision = persistence.async_save_trade_decision.await_args.args[0]
        assert decision.fee is None


class TestQuoteCurrencyFeesAreEvidence:
    """Wave 5: a currency-tagged fee is REAL evidence when it is the symbol's quote asset.

    The executor emits ccxt fee objects (``{"cost": ..., "currency": ...}``). Such an
    entry may now be booked — but only when its currency IS the quote asset of the
    symbol. A fee charged in the base asset (BTC) or any third asset still means UNKNOWN:
    no conversion exists here, and summing it as quote currency would invent a number.
    """

    @pytest.mark.parametrize(
        ("record_overrides", "expected"),
        [
            pytest.param({"fees": [{"cost": 0.0025, "currency": "USDT"}]}, (0.0025,), id="quote-currency-object"),
            pytest.param(
                {"symbol": "BTC/USDC", "fees": [{"cost": 0.0, "currency": "USDC"}]},
                (0.0,),
                id="real-incident-symbol-zero-fee",
            ),
            pytest.param(
                {"fees": [{"cost": 0.001, "currency": "USDT"}, {"cost": 0.0005, "currency": "USDT"}]},
                (0.001, 0.0005),
                id="two-fill-fees-summed",
            ),
            pytest.param({"fee": 0.004, "fee_currency": "USDT"}, (0.004,), id="scalar-fee-with-matching-tag"),
            pytest.param({"fees": [{"cost": 0.0001, "currency": "BTC"}]}, None, id="base-asset-fee-is-unknown"),
            pytest.param({"fees": [{"cost": 0.0001, "currency": "BNB"}]}, None, id="third-asset-fee-is-unknown"),
            pytest.param({"fees": [{"cost": 0.0001}]}, None, id="fee-without-a-currency-is-unknown"),
            pytest.param(
                {"symbol": "BTC/USDC", "fees": [{"cost": 0.0001, "currency": "USDT"}]},
                None,
                id="other-stable-is-not-converted",
            ),
            pytest.param({"fee": 0.004, "fee_currency": "BTC"}, None, id="scalar-fee-tagged-in-base-asset"),
        ],
    )
    def test_fee_currency_decides_whether_a_fee_is_bookable(
        self, record_overrides: dict[str, Any], expected: tuple[float, ...] | None
    ) -> None:
        verified, rejection = verify_exit_record(_exit_record(**record_overrides))

        assert rejection is None and verified is not None
        if expected is None:
            assert verified.fees is None
            assert verified.exit_fee is None, "a foreign-currency fee is never summed as USDT"
        else:
            assert verified.fees == expected
            assert verified.exit_fee == pytest.approx(sum(expected))

    async def test_a_quote_currency_fee_is_booked_and_labelled_with_its_currency(
        self, tmp_path: Path
    ) -> None:
        position = _position()
        strategy, _, persistence = _strategy(
            tmp_path,
            position,
            EXECUTOR_EXIT_PATH=str(
                _journal(
                    tmp_path / "exits.jsonl",
                    [_exit_record(fees=[{"cost": 0.0025, "currency": "USDT"}])],
                )
            ),
        )

        assert await strategy.book_executor_side_exit(
            position, 84500.0, MarketConditions(), "Executor confirmed flat"
        ) is True

        decision = persistence.async_save_trade_decision.await_args.args[0]
        assert decision.fee == pytest.approx(0.0025)
        assert "executor_exit_journal (USDT)" in decision.reasoning

    async def test_a_base_asset_fee_still_books_an_unknown_commission(self, tmp_path: Path) -> None:
        position = _position()
        strategy, _, persistence = _strategy(
            tmp_path,
            position,
            EXECUTOR_EXIT_PATH=str(
                _journal(
                    tmp_path / "exits.jsonl",
                    [_exit_record(fees=[{"cost": 0.0000001, "currency": "BTC"}])],
                )
            ),
        )

        assert await strategy.book_executor_side_exit(
            position, 84500.0, MarketConditions(), "Executor confirmed flat"
        ) is True

        decision = persistence.async_save_trade_decision.await_args.args[0]
        assert decision.fee is None, "a BTC fee is not a USDT fee — never converted"
        assert "executor_exit_journal (BTC)" not in decision.reasoning


class TestAveragePriceEvidence:
    """Wave 5: an explicit ``average_price`` is accepted — and never papered over."""

    def test_average_price_is_used_when_exit_price_is_absent(self) -> None:
        record = _exit_record(exit_price=None, average_price=84500.0)

        verified, rejection = verify_exit_record(record)

        assert rejection is None and verified is not None
        assert verified.exit_price == pytest.approx(84500.0)

    def test_average_price_takes_precedence_over_exit_price(self) -> None:
        record = _exit_record(exit_price=1.0, average_price=84500.0)

        verified, _ = verify_exit_record(record)

        assert verified is not None and verified.exit_price == pytest.approx(84500.0)

    @pytest.mark.parametrize(
        "raw", ["not-a-number", 0, -1.0, float("nan"), float("inf")], ids=["text", "zero", "negative", "nan", "inf"]
    )
    def test_an_unusable_average_price_is_a_rejection_not_a_fallback(self, raw: Any) -> None:
        """A malformed evidence field must not be ignored while a valid ``exit_price`` exists."""
        record = _exit_record(exit_price=84500.0, average_price=raw)

        verified, rejection = verify_exit_record(record)

        assert verified is None
        assert "average price" in str(rejection)

    async def test_a_bad_average_price_keeps_the_position_and_alarms(self, tmp_path: Path) -> None:
        position = _position()
        strategy, _, persistence = _strategy(
            tmp_path,
            position,
            EXECUTOR_EXIT_PATH=str(
                _journal(
                    tmp_path / "exits.jsonl",
                    [_exit_record(exit_price=84500.0, average_price="oops")],
                )
            ),
        )

        booked = await strategy.book_executor_side_exit(
            position, 84500.0, MarketConditions(), "Executor confirmed flat"
        )

        assert booked is False
        assert strategy.current_position is position
        assert persistence.async_save_trade_decision.await_count == 0
        assert strategy.take_state_divergence() is not None


class TestAProvenExitTeachesTheBrainOnce:
    """Wave 5 end-to-end: only a proven exit + a validated post-mortem update the brain."""

    async def test_a_proven_executor_exit_updates_the_brain_with_its_evidence(self, tmp_path: Path) -> None:
        position = _position()
        strategy, _, persistence = _strategy(
            tmp_path,
            position,
            EXECUTOR_EXIT_PATH=str(
                _journal(
                    tmp_path / "exits.jsonl",
                    [_exit_record(fees=[{"cost": 0.0025, "currency": "USDT"}], protection_version="rev-4")],
                )
            ),
        )
        entry = MagicMock()
        entry.reasoning = "Breakout continuation."
        persistence.get_entry_decision_for_position = MagicMock(return_value=entry)
        post_mortem = MagicMock()
        post_mortem.analyze_closed_trade = AsyncMock(return_value=MagicMock())
        strategy.post_mortem_service = post_mortem
        brain = strategy.brain_service

        assert await strategy.book_executor_side_exit(
            position, 84500.0, MarketConditions(), "Executor confirmed flat"
        ) is True

        post_mortem.analyze_closed_trade.assert_awaited_once()
        assert brain.update_from_closed_trade.call_count == 1
        evidence = brain.update_from_closed_trade.call_args.kwargs["evidence"]
        assert evidence.source == "executor_exit_journal"
        assert evidence.price == pytest.approx(84500.0)
        assert evidence.quantity == pytest.approx(0.00554)
        assert evidence.protection_version == "rev-4"
        decision = persistence.async_save_trade_decision.await_args.args[0]
        assert decision.quantity == pytest.approx(0.00554)
        assert decision.fee == pytest.approx(0.0025)

    async def test_an_unproven_executor_exit_teaches_nothing(self, tmp_path: Path) -> None:
        """No usable exit record = no booking, no post-mortem, no lesson."""
        position = _position()
        strategy, _, persistence = _strategy(
            tmp_path,
            position,
            EXECUTOR_EXIT_PATH=str(_journal(tmp_path / "exits.jsonl", [_exit_record(exit_price=None)])),
        )
        persistence.get_entry_decision_for_position = MagicMock(return_value=MagicMock())
        post_mortem = MagicMock()
        post_mortem.analyze_closed_trade = AsyncMock(return_value=MagicMock())
        strategy.post_mortem_service = post_mortem

        assert await strategy.book_executor_side_exit(
            position, 84500.0, MarketConditions(), "Executor confirmed flat"
        ) is False

        post_mortem.analyze_closed_trade.assert_not_awaited()
        strategy.brain_service.update_from_closed_trade.assert_not_called()
        assert persistence.async_save_trade_decision.await_count == 0


def _receipt(**overrides: Any) -> dict[str, Any]:
    """One executor verdict-journal row WITH the wave-5 fill-evidence block.

    Field names are the executor's own (``average_price``, ``filled_quantity``,
    ``fees`` as ``{"currency", "amount"}``, ``fees_known``, ``state``,
    ``close_confirmed``, ``protection_version``) — nothing is renamed here.
    """
    row: dict[str, Any] = {
        "order_id": "order-close-1",
        "timestamp": "2026-09-21T09:32:51.956000+00:00",
        "symbol": "BTC/USDT",
        "signal": "CLOSE",
        "action": "CLOSE",
        "verdict": "executed",
        "reason": "",
        "command_id": "cmd-1",
        "position_id": "pos-1",
        "average_price": 84500.0,
        "filled_quantity": 0.00554,
        "fees": [{"currency": "USDT", "amount": 1.5}],
        "fees_known": True,
        "state": "filled",
        "close_confirmed": True,
        "protection_version": 7,
        "fill_evidence": "exchange_order_fill",
    }
    row.update(overrides)
    return row


async def test_filled_entry_with_protection_error_is_confirmed_not_refused(
    tmp_path: Path,
) -> None:
    position = _position(symbol="BTC/USDC")
    strategy, _, persistence = _strategy(tmp_path, position)
    order_id = "order-entry-filled"
    intent = strategy.register_position_intent(
        "ENTRY",
        "BTC/USDC",
        order_id=order_id,
        position_id=position_identity(position),
    )
    _verdicts(
        tmp_path / "verdicts.jsonl",
        [{
            "order_id": order_id,
            "verdict": "error",
            "reason": "entry filled but protection unresolved",
            "state": "filled",
            "filled_quantity": 0.00698,
            "average_price": 85987.62,
            "exposure_possible": True,
        }],
    )
    strategy.config.EXECUTOR_VERDICT_PATH = str(tmp_path / "verdicts.jsonl")

    state = await strategy.resolve_position_intents_after_forward(
        order_id=order_id, delivered=True
    )

    assert state == INTENT_CONFIRMED
    assert strategy.current_position is position
    resolved = strategy.position_intents().get(intent.key)
    assert resolved is not None and resolved.state == INTENT_CONFIRMED
    persistence.async_save_position.assert_not_awaited()


class TestReceiptFillEvidenceBooking:
    """Wave-5 req. 1: the executor's RECEIPT itself carries the fill evidence.

    When the exit journal has not caught up yet, the confirmed CLOSE is booked from the
    receipt's ``average_price`` / ``filled_quantity`` / ``fees`` — never from the local
    size, the requested amount or the ticker. Anything incomplete keeps the local
    position and alerts, exactly like an unusable journal line. Backwards compatible:
    without those fields the receipt can only confirm the intent, and booking keeps
    waiting for the exit journal (see ``test_receipt_without_a_fill_price_never_invents_one``).
    """

    async def test_books_the_close_from_the_receipt_when_the_journal_is_empty(
        self, tmp_path: Path
    ) -> None:
        position = _position()
        strategy, _, persistence = _strategy(
            tmp_path,
            position,
            EXECUTOR_EXIT_PATH=str(tmp_path / "exits.jsonl"),
        )
        strategy._executor_has_position = AsyncMock(return_value=True)
        decision = await _close_signal(strategy)
        _verdicts(
            tmp_path / "verdicts.jsonl",
            [_receipt(order_id=decision.order_id)],
        )
        strategy.config.EXECUTOR_VERDICT_PATH = str(tmp_path / "verdicts.jsonl")

        state = await strategy.resolve_position_intents_after_forward(
            order_id=decision.order_id, delivered=True
        )

        assert state == INTENT_CONFIRMED
        assert strategy.current_position is None
        persistence.async_save_trade_decision.assert_awaited_once()
        booked = persistence.async_save_trade_decision.await_args.args[0]
        assert booked.action == "CLOSE_LONG"
        assert booked.price == pytest.approx(84500.0), "the fill average from the receipt"
        assert booked.quantity == pytest.approx(0.00554), "the filled amount, not the size"
        assert booked.fee == pytest.approx(1.5)
        assert "executor_receipt_fill" in booked.reasoning
        assert "close_confirmed_by_executor" in booked.reasoning

    async def test_receipt_without_a_fill_price_never_invents_one(self, tmp_path: Path) -> None:
        """Unknown average price: no booking, position kept, alarm — like the journal."""
        position = _position()
        strategy, _, persistence = _strategy(
            tmp_path, position, EXECUTOR_EXIT_PATH=str(tmp_path / "exits.jsonl")
        )
        strategy._executor_has_position = AsyncMock(return_value=True)
        decision = await _close_signal(strategy)
        _verdicts(
            tmp_path / "verdicts.jsonl",
            [_receipt(order_id=decision.order_id, average_price=None)],
        )
        strategy.config.EXECUTOR_VERDICT_PATH = str(tmp_path / "verdicts.jsonl")

        state = await strategy.resolve_position_intents_after_forward(
            order_id=decision.order_id, delivered=True
        )

        assert state == INTENT_UNKNOWN
        assert strategy.current_position is position
        assert persistence.async_save_trade_decision.await_count == 0
        assert strategy.take_unconfirmed_intent_alert() is not None
        assert strategy.take_state_divergence() is not None

    async def test_partial_receipt_is_not_a_final_exit(self, tmp_path: Path) -> None:
        """A partially filled close is NOT a final exit: nothing is booked."""
        position = _position()
        strategy, _, persistence = _strategy(
            tmp_path, position, EXECUTOR_EXIT_PATH=str(tmp_path / "exits.jsonl")
        )
        strategy._executor_has_position = AsyncMock(return_value=True)
        decision = await _close_signal(strategy)
        _verdicts(
            tmp_path / "verdicts.jsonl",
            [_receipt(order_id=decision.order_id, state="partially_filled")],
        )
        strategy.config.EXECUTOR_VERDICT_PATH = str(tmp_path / "verdicts.jsonl")

        state = await strategy.resolve_position_intents_after_forward(
            order_id=decision.order_id, delivered=True
        )

        assert state == INTENT_UNKNOWN
        assert strategy.current_position is position
        assert persistence.async_save_trade_decision.await_count == 0

    async def test_unknown_receipt_fees_are_booked_as_unknown(self, tmp_path: Path) -> None:
        """``fees_known`` false = the exchange reported nothing: UNKNOWN, never 0."""
        position = _position()
        strategy, _, persistence = _strategy(
            tmp_path, position, EXECUTOR_EXIT_PATH=str(tmp_path / "exits.jsonl")
        )
        strategy._executor_has_position = AsyncMock(return_value=True)
        decision = await _close_signal(strategy)
        _verdicts(
            tmp_path / "verdicts.jsonl",
            [_receipt(order_id=decision.order_id, fees_known=False, fees=[])],
        )
        strategy.config.EXECUTOR_VERDICT_PATH = str(tmp_path / "verdicts.jsonl")

        await strategy.resolve_position_intents_after_forward(
            order_id=decision.order_id, delivered=True
        )

        booked = persistence.async_save_trade_decision.await_args.args[0]
        assert booked.fee is None
        assert "unknown" in booked.reasoning.lower()

    async def test_receipt_quantity_divergence_is_booked_from_the_fill(self, tmp_path: Path) -> None:
        """A >1% mismatch books the FILL amount and raises the loud alarm."""
        position = _position()
        strategy, logger, persistence = _strategy(
            tmp_path, position, EXECUTOR_EXIT_PATH=str(tmp_path / "exits.jsonl")
        )
        strategy._executor_has_position = AsyncMock(return_value=True)
        decision = await _close_signal(strategy)
        _verdicts(
            tmp_path / "verdicts.jsonl",
            [_receipt(order_id=decision.order_id, filled_quantity=0.006)],
        )
        strategy.config.EXECUTOR_VERDICT_PATH = str(tmp_path / "verdicts.jsonl")

        await strategy.resolve_position_intents_after_forward(
            order_id=decision.order_id, delivered=True
        )

        booked = persistence.async_save_trade_decision.await_args.args[0]
        assert booked.quantity == pytest.approx(0.006)
        assert booked.quantity != pytest.approx(position.size)
        assert strategy.take_state_divergence() is not None
        assert any(
            "QUANTITY DIVERGENCE" in str(call) for call in logger.critical.call_args_list
        )

    async def test_the_exit_journal_wins_when_both_sources_exist(self, tmp_path: Path) -> None:
        """Backwards compatible: a real journal line keeps priority over the receipt."""
        position = _position()
        strategy, _, persistence = _strategy(
            tmp_path,
            position,
            EXECUTOR_EXIT_PATH=str(_journal(tmp_path / "exits.jsonl", [_exit_record()])),
        )
        strategy._executor_has_position = AsyncMock(return_value=True)
        decision = await _close_signal(strategy)
        _verdicts(
            tmp_path / "verdicts.jsonl",
            [_receipt(order_id=decision.order_id, average_price=83000.0)],
        )
        strategy.config.EXECUTOR_VERDICT_PATH = str(tmp_path / "verdicts.jsonl")

        await strategy.resolve_position_intents_after_forward(
            order_id=decision.order_id, delivered=True
        )

        booked = persistence.async_save_trade_decision.await_args.args[0]
        assert booked.price == pytest.approx(84500.0), "the journal fill, not the receipt"
        assert "take_profit_filled" in booked.reasoning

    async def test_receipt_proven_close_feeds_one_lesson(self, tmp_path: Path) -> None:
        """Proven receipt fill + a real post-mortem = exactly one lesson (idempotent)."""
        position = _position()
        strategy, _, persistence = _strategy(
            tmp_path, position, EXECUTOR_EXIT_PATH=str(tmp_path / "exits.jsonl")
        )
        entry = MagicMock()
        entry.reasoning = "Breakout continuation."
        persistence.get_entry_decision_for_position = MagicMock(return_value=entry)
        post_mortem = MagicMock()
        post_mortem.analyze_closed_trade = AsyncMock(return_value=MagicMock())
        strategy.post_mortem_service = post_mortem
        brain = strategy.brain_service
        strategy._executor_has_position = AsyncMock(return_value=True)
        decision = await _close_signal(strategy)
        _verdicts(tmp_path / "verdicts.jsonl", [_receipt(order_id=decision.order_id)])
        strategy.config.EXECUTOR_VERDICT_PATH = str(tmp_path / "verdicts.jsonl")

        await strategy.resolve_position_intents_after_forward(
            order_id=decision.order_id, delivered=True
        )
        await strategy.resolve_position_intents_after_forward(
            order_id=decision.order_id, delivered=True
        )

        post_mortem.analyze_closed_trade.assert_awaited_once()
        assert brain.update_from_closed_trade.call_count == 1
        evidence = brain.update_from_closed_trade.call_args.kwargs["evidence"]
        assert evidence.source == "executor_verdict_journal"
        assert evidence.price == pytest.approx(84500.0)
        assert evidence.quantity == pytest.approx(0.00554)
        assert evidence.identity, "the lesson is deduplicated by a stable event id"


class TestReceiptEvidenceAdapter:
    """The receipt → exit-record adapter rejects anything it cannot prove."""

    def test_a_full_receipt_becomes_a_validated_exit_record(self) -> None:
        record, rejection = exit_record_from_receipt(_receipt(), symbol="BTC/USDT")

        assert rejection == ""
        assert record is not None
        verified, problem = verify_exit_record(record)
        assert problem is None and verified is not None
        assert verified.exit_price == pytest.approx(84500.0)
        assert verified.quantity == pytest.approx(0.00554)
        assert verified.exit_fee == pytest.approx(1.5)
        assert verified.event_id.endswith("receipt:cmd-1"), "stable, replay-safe event id"
        assert record["evidence"] == "executor_receipt_fill"

    @pytest.mark.parametrize(
        ("receipt", "expected"),
        [
            pytest.param(None, "no executor receipt", id="no-receipt"),
            pytest.param({"verdict": "blocked"}, "not 'executed'", id="not-executed"),
            pytest.param(_receipt(signal="UPDATE", action="UPDATE"), "not a CLOSE", id="update"),
            pytest.param(_receipt(state="accepted"), "not fully filled", id="accepted-not-filled"),
            pytest.param(
                _receipt(state="partially_filled"), "not fully filled", id="partial-fill"
            ),
            pytest.param(
                _receipt(close_confirmed=False),
                "did not confirm the close",
                id="close-unconfirmed",
            ),
            pytest.param(_receipt(average_price=None), "average price unknown", id="no-price"),
            pytest.param(_receipt(average_price=0.0), "average price unknown", id="zero-price"),
            pytest.param(
                _receipt(filled_quantity=None), "filled quantity unknown", id="no-quantity"
            ),
            pytest.param(_receipt(filled_quantity=float("nan")), "filled quantity", id="nan"),
        ],
    )
    def test_unusable_receipts_are_refused_with_a_reason(self, receipt: Any, expected: str) -> None:
        record, rejection = exit_record_from_receipt(receipt, symbol="BTC/USDT")

        assert record is None
        assert expected in rejection

    def test_a_receipt_for_another_symbol_is_refused(self) -> None:
        record, rejection = exit_record_from_receipt(_receipt(symbol="ETH/USDT"), symbol="BTC/USDT")

        assert record is None
        assert "not 'BTC/USDT'" in rejection

    def test_receipt_without_a_signal_falls_back_to_the_action(self) -> None:
        row = _receipt()
        row.pop("signal")

        record, rejection = exit_record_from_receipt(row, symbol="BTC/USDT")

        assert rejection == "" and record is not None

    def test_unknown_fees_are_not_converted_to_zero(self) -> None:
        record, _ = exit_record_from_receipt(_receipt(fees_known=False), symbol="BTC/USDT")

        assert record is not None and record["fees"] == []

    def test_a_foreign_currency_fee_is_not_summed_into_the_quote_currency(self) -> None:
        """A USDC fee on a USDT symbol cannot be added to a USDT total: UNKNOWN."""
        record, _ = exit_record_from_receipt(
            _receipt(fees=[{"currency": "USDC", "amount": 1.5}]), symbol="BTC/USDT"
        )

        assert record is not None
        verified, problem = verify_exit_record(record)
        assert problem is None and verified is not None
        assert verified.exit_fee is None, "no conversion is invented"
