"""Wave-1 P0-C/P1: the trading check syncs BEFORE the prompt and publishes honest cards.

Covers the acceptance scenarios:
* the reconciliation hook runs before the analysis context is built;
* an exit detected during the LLM run invalidates the intent produced from the stale
  snapshot — it is discarded, never converted into a new entry;
* an analysis error or a HOLD answer does not stop the independent sync;
* a recommendation the bot did NOT execute is published as
  "RECOMMENDATION NOT EXECUTED" with its reason instead of a bare UPDATE.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from src.app import BotServices, CryptoTradingBot
from src.trading.data_models import TradeDecision
from src.trading.executor_reconciliation import (
    RECONCILE_EXIT_BOOKED,
    RECONCILE_OPEN_EXECUTOR_REPORTED,
    RECONCILE_UNVERIFIED,
    LocalPositionReconciliation,
)
from tests.conftest import make_position

ENTRY_SNAPSHOT = "open:BTC/USDC:LONG:2026-09-21T07:04:12+00:00"
TIGHTENED_PROTECTION = (80550.0, 84500.0)
CHANGED_PROTECTION = (79550.0, 84500.0)
FLAT_SNAPSHOT = "flat"


class FakeStrategy:
    """TradingStrategy double exposing the wave-1 reconciliation contract."""

    def __init__(
        self,
        *,
        position: Any = None,
        decision: Any = None,
        tokens: list[str] | None = None,
        outcome: LocalPositionReconciliation | None = None,
        book_exit_on_reconcile: bool = False,
        book_exit_on_analysis: bool = False,
        book_entry_on_analysis: bool = False,
        events: list[str] | None = None,
        protections: list[tuple[float | None, float | None] | None] | None = None,
        guard_reason: str | None = None,
        protection_versions: list[str | None] | None = None,
    ) -> None:
        self.current_position = position
        self._decision = decision
        self._tokens = list(tokens or [])
        self._protections = list(protections) if protections is not None else None
        self.guard_reason = guard_reason
        self.guard_calls: list[Any] = []
        self._protection_versions = (
            list(protection_versions) if protection_versions is not None else None
        )
        self.protection_version_calls = 0
        self._outcome = outcome
        self.book_exit_on_reconcile = book_exit_on_reconcile
        self.book_exit_on_analysis = book_exit_on_analysis
        self.book_entry_on_analysis = book_entry_on_analysis
        self.events = events if events is not None else []
        self.reconcile_sources: list[str] = []
        self.rollback_calls: list[tuple] = []
        self.take_executor_side_exit_reason = MagicMock(return_value=None)
        self.take_state_divergence = MagicMock(return_value=None)

    def position_snapshot_token(self) -> str:
        if self._tokens:
            return self._tokens.pop(0)
        return FLAT_SNAPSHOT if self.current_position is None else ENTRY_SNAPSHOT

    def protection_snapshot(self) -> tuple[float | None, float | None] | None:
        """SL/TP pair: a scripted sequence when given, else the standing position."""
        if self._protections:
            return self._protections.pop(0)
        if self.current_position is None:
            return None
        return (self.current_position.stop_loss, self.current_position.take_profit)

    def protection_loosening_reason(self, decision: Any) -> str | None:
        """Record the guard consultation; report the scripted refusal (or None)."""
        self.guard_calls.append(decision)
        return self.guard_reason

    async def executor_protection_version(self) -> str | None:
        """Wave-5 hook: the executor's protection revision (scripted, unset = None)."""
        self.protection_version_calls += 1
        if self._protection_versions:
            return self._protection_versions.pop(0)
        return None

    async def reconcile_local_position(self, symbol: str | None = None, *, source: str = "periodic") -> Any:
        self.reconcile_sources.append(source)
        self.events.append(f"reconcile:{source}")
        if self.book_exit_on_reconcile:
            self.current_position = None
        return self._outcome

    async def process_analysis(self, result: dict, symbol: str, market_price: float | None = None) -> Any:
        self.events.append("process_analysis")
        if self.book_exit_on_analysis:
            self.current_position = None
        if self.book_entry_on_analysis:
            self.current_position = make_position()
        return self._decision

    async def rollback_blocked_entry(self, symbol: str, forward_delivered: bool, order_id: str | None = None) -> None:
        self.rollback_calls.append((symbol, forward_delivered, order_id))

    def take_pending_local_close_decision(self) -> TradeDecision | None:
        """No local exit queued in these scenarios."""
        return None

    def take_unconfirmed_intent_alert(self) -> str | None:
        """Nothing unconfirmed in these scenarios."""
        return None

    def take_rejected_intent_alert(self) -> str | None:
        """Nothing refused in these scenarios."""
        return None

    async def resolve_position_intents_after_forward(
        self, order_id: str | None, delivered: bool, symbol: str | None = None
    ) -> str | None:
        """No intent layer in this double: the state stays unknown."""
        return None


def open_outcome(state: str = RECONCILE_OPEN_EXECUTOR_REPORTED, **overrides: Any) -> LocalPositionReconciliation:
    values: dict[str, Any] = {
        "state": state,
        "symbol": "BTC/USDC",
        "checked_at": datetime.now(timezone.utc),
    }
    values.update(overrides)
    return LocalPositionReconciliation(**values)


def update_decision(**overrides: Any) -> TradeDecision:
    values: dict[str, Any] = {
        "timestamp": datetime(2026, 9, 21, 12, 4, tzinfo=timezone.utc),
        "symbol": "BTC/USDC",
        "action": "UPDATE",
        "confidence": "MEDIUM",
        "price": 81198.56,
        "stop_loss": 79550.0,
        "take_profit": 84500.0,
        "reasoning": "Tighten the stop under the recent swing",
        "order_id": "update-20260921120400000000",
    }
    values.update(overrides)
    return TradeDecision(**values)


def harness(
    strategy: FakeStrategy,
    *,
    analysis: dict[str, Any] | None = None,
    analyze_result: dict[str, Any] | None = None,
    events: list[str] | None = None,
) -> Any:
    """Bot wired for _execute_trading_check with recording collaborators."""
    events = events if events is not None else strategy.events
    config = SimpleNamespace(
        MAIN_CHANNEL_ID=123,
        RAG_UPDATE_TIMEOUT=1,
        EXECUTOR_API_ENABLED=True,
        EXECUTOR_API_URL="http://127.0.0.1:9199/decision",
        RESEARCH_TEAM_ENABLED=False,
        MIN_RR_ENTRY=1.0,
        EXECUTOR_MAX_POSITION_USDC=0.0,
        DEMO_QUOTE_CAPITAL=10000.0,
        SOCIAL_SENTIMENT_ENABLED=False,
        EXECUTOR_VERDICT_PATH="data/trading/executor_verdicts.jsonl",
        REQUIRE_APPROVAL_ABOVE_USDC=0.0,
        APPROVAL_TTL_MINUTES=15.0,
        APPROVAL_POLL_SECONDS=5.0,
    )
    discord_notifier = MagicMock()
    discord_notifier.send_analysis_notification = AsyncMock()
    market_analyzer = MagicMock()
    payload = analyze_result if analyze_result is not None else {
        "analysis": analysis if analysis is not None else {"signal": "UPDATE", "confidence": 75, "reasoning": "Structure intact."},
        "raw_response": 'Reasoning text {"analysis": {}}',
    }
    market_analyzer.analyze_market = AsyncMock(return_value=payload)
    market_analyzer.last_chart_buffer = None

    persistence = MagicMock()
    persistence.data_dir = Path(tempfile.mkdtemp(prefix="llm_trader_sync_state_"))
    persistence.async_save_last_analysis_time = AsyncMock()
    persistence.async_load_previous_response = AsyncMock(return_value={})
    persistence.get_last_analysis_time = MagicMock(return_value=None)

    statistics_service = MagicMock()
    statistics_service.get_current_capital = MagicMock(return_value=10000.0)
    statistics_service.get_context = MagicMock(return_value="")
    memory_service = MagicMock()
    memory_service.get_context_summary = MagicMock(return_value="")
    brain_service = MagicMock()
    brain_service.get_dynamic_thresholds = MagicMock(return_value={})

    position_monitor = MagicMock()
    position_monitor.check_soft_exit_status = AsyncMock()
    position_monitor.handle_new_position = AsyncMock()
    position_monitor.handle_position_closed = AsyncMock()

    executor_handler = MagicMock()
    executor_handler.handle = AsyncMock(return_value=True)

    services: dict[str, Any] = {
        "logger": MagicMock(),
        "config": config,
        "shutdown_manager": None,
        "exchange_manager": MagicMock(),
        "market_analyzer": market_analyzer,
        "trading_strategy": strategy,
        "discord_notifier": discord_notifier,
        "keyboard_handler": MagicMock(),
        "rag_engine": MagicMock(),
        "coingecko_api": MagicMock(),
        "market_api": MagicMock(),
        "alternative_me_api": MagicMock(),
        "http_session": MagicMock(),
        "persistence": persistence,
        "model_manager": MagicMock(),
        "brain_service": brain_service,
        "statistics_service": statistics_service,
        "memory_service": memory_service,
        "exit_monitor": MagicMock(),
        "dashboard_state": None,
        "discord_task": None,
        "executor_handler": executor_handler,
        "position_monitor_factory": lambda _bot: position_monitor,
    }
    bot = CryptoTradingBot(BotServices(**services))
    bot.current_symbol = "BTC/USDC"
    bot.current_timeframe = "4h"
    bot._fetch_ticker_data = AsyncMock(return_value=({"last": 81198.56}, 81198.56))
    bot._execute_market_knowledge_update = AsyncMock()

    async def build_context(current_price: float | None, ticker: Any) -> dict[str, Any]:
        events.append("build_context")
        return {}

    bot._build_analysis_context = build_context
    bot._save_analysis_data = AsyncMock()

    return SimpleNamespace(
        bot=bot,
        notifier=discord_notifier,
        market_analyzer=market_analyzer,
        position_monitor=position_monitor,
        executor_handler=executor_handler,
        events=events,
    )


def note_for(harness_object: Any) -> str | None:
    """The execution_note the analysis card was published with."""
    return harness_object.notifier.send_analysis_notification.await_args.kwargs["execution_note"]


async def test_reconciliation_runs_before_the_analysis_context_is_built() -> None:
    """The prompt is only built after the local position was checked against the exchange."""
    events: list[str] = []
    strategy = FakeStrategy(position=make_position(), outcome=open_outcome(), events=events)
    fixture = harness(strategy, events=events)

    await fixture.bot._execute_trading_check(check_count=1)

    assert strategy.reconcile_sources == ["pre_analysis"]
    assert events.index("reconcile:pre_analysis") < events.index("build_context")


async def test_prior_tp_leaves_no_stale_context_and_blocks_the_update() -> None:
    """The 2026-09-21 replay: the exit is booked before the prompt, the UPDATE dies."""
    position = make_position()
    strategy = FakeStrategy(
        position=position,
        decision=update_decision(),
        tokens=[ENTRY_SNAPSHOT, FLAT_SNAPSHOT],
        outcome=open_outcome(RECONCILE_EXIT_BOOKED, exit_booked=True, detail="exchange-side exit booked"),
        book_exit_on_reconcile=True,
    )
    fixture = harness(strategy, analysis={"signal": "UPDATE", "confidence": 75, "reasoning": "Tighten SL."})

    await fixture.bot._execute_trading_check(check_count=1)

    assert strategy.current_position is None
    fixture.executor_handler.handle.assert_not_awaited()
    fixture.position_monitor.handle_new_position.assert_not_awaited()
    note = note_for(fixture)
    assert note is not None
    assert "UPDATE" in note
    assert "not executed" in note.lower()
    assert "during the analysis" in note


async def test_exit_during_analysis_discards_the_stale_intent() -> None:
    """A position that changed while the model worked invalidates its UPDATE."""
    strategy = FakeStrategy(
        position=make_position(),
        decision=update_decision(),
        tokens=[ENTRY_SNAPSHOT, FLAT_SNAPSHOT],
        outcome=open_outcome(),
        book_exit_on_analysis=True,
    )
    fixture = harness(strategy, analysis={"signal": "UPDATE", "confidence": 75, "reasoning": "Tighten SL."})

    await fixture.bot._execute_trading_check(check_count=1)

    fixture.executor_handler.handle.assert_not_awaited()
    fixture.position_monitor.handle_new_position.assert_not_awaited()
    assert fixture.executor_handler.handle.await_count == 0
    assert "not executed" in (note_for(fixture) or "").lower()


async def test_unverified_exchange_state_is_disclosed_instead_of_claimed() -> None:
    """A failing exchange check downgrades the card to "not executed / unverified"."""
    strategy = FakeStrategy(
        position=make_position(),
        decision=None,
        outcome=open_outcome(RECONCILE_UNVERIFIED, detail="executor position query failed"),
    )
    fixture = harness(strategy, analysis={"signal": "UPDATE", "confidence": 60, "reasoning": "Manage the trade."})

    await fixture.bot._execute_trading_check(check_count=1)

    note = note_for(fixture)
    assert note is not None
    assert "did not confirm" in note
    assert "executor position query failed" in note


async def test_sync_still_runs_when_the_analysis_errors() -> None:
    """The exchange sync never depends on the LLM answering."""
    events: list[str] = []
    strategy = FakeStrategy(position=make_position(), decision=None, outcome=open_outcome(), events=events)
    fixture = harness(
        strategy, analyze_result={"error": "model unavailable"}, events=events
    )

    await fixture.bot._execute_trading_check(check_count=1)

    assert strategy.reconcile_sources == ["pre_analysis"]
    assert "process_analysis" not in events
    fixture.notifier.send_analysis_notification.assert_not_awaited()


async def test_hold_answer_still_syncs_and_needs_no_not_executed_note() -> None:
    """HOLD is not an actionable recommendation: sync yes, disclaimer no."""
    strategy = FakeStrategy(
        position=make_position(),
        decision=None,
        outcome=open_outcome(),
    )
    fixture = harness(strategy, analysis={"signal": "HOLD", "confidence": 55, "reasoning": "No edge."})

    await fixture.bot._execute_trading_check(check_count=1)

    assert strategy.reconcile_sources == ["pre_analysis"]
    assert note_for(fixture) is None


async def test_executed_recommendation_carries_no_note() -> None:
    """A forwarded BUY is a performed action — no "not executed" banner."""
    strategy = FakeStrategy(
        position=make_position(),
        decision=update_decision(action="CLOSE"),
        outcome=open_outcome(),
    )
    fixture = harness(strategy, analysis={"signal": "CLOSE", "confidence": 80, "reasoning": "Exit now."})

    await fixture.bot._execute_trading_check(check_count=1)

    assert note_for(fixture) is None
    fixture.executor_handler.handle.assert_awaited_once()


async def test_new_entry_is_not_mistaken_for_a_stale_recommendation() -> None:
    """The strategy's own flat-to-open mutation must still reach the executor."""
    decision = update_decision(action="BUY", order_id="entry-1")
    strategy = FakeStrategy(
        position=None,
        decision=decision,
        outcome=open_outcome(),
        book_entry_on_analysis=True,
    )
    fixture = harness(
        strategy,
        analysis={"signal": "BUY", "confidence": 80, "reasoning": "Enter now."},
    )

    await fixture.bot._execute_trading_check(check_count=1)

    fixture.executor_handler.handle.assert_awaited_once()
    assert fixture.executor_handler.handle.await_args.args[1] is decision
    assert note_for(fixture) is None


async def test_a_stale_exchange_exit_does_not_stop_the_monitor_of_a_new_position() -> None:
    """The exit summary belongs to the position that closed, not to the one opened after it.

    Contract (`src/trading/AGENTS.md:13`): the real-market check on
    `PositionStatusMonitor.check_reference_exits()` "is the stop that actually runs"; ADR
    0005 forbids leaving a live position protected only by the exchange backstop. When the
    pre-analysis reconcile books the previous exit and the same cycle opens a new position,
    `handle_position_closed` for the OLD trade must be delivered before the monitor is
    handed the new one — otherwise it clears state and stops the loop for the new position.
    """
    decision = update_decision(action="BUY", order_id="entry-1")
    strategy = FakeStrategy(
        position=make_position(),
        decision=decision,
        outcome=open_outcome(
            RECONCILE_EXIT_BOOKED, exit_booked=True, detail="exchange-side exit booked"
        ),
        book_exit_on_reconcile=True,
        book_entry_on_analysis=True,
    )
    strategy.take_executor_side_exit_reason = MagicMock(
        return_value="Executor confirmed flat: stop_loss @ 78,000.00"
    )
    fixture = harness(
        strategy,
        analysis={"signal": "BUY", "confidence": 80, "reasoning": "Enter now."},
    )

    await fixture.bot._execute_trading_check(check_count=1)

    assert strategy.current_position is not None
    fixture.position_monitor.handle_new_position.assert_awaited_once()
    names = [call[0] for call in fixture.position_monitor.mock_calls]
    assert "handle_position_closed" in names, "the booked exit is still reported"
    assert names.index("handle_position_closed") < names.index("handle_new_position")


async def test_strategy_rejected_recommendation_is_reported_as_not_executed() -> None:
    """A guard-rejected BUY is published with its rejection reason."""
    strategy = FakeStrategy(
        position=None,
        decision=update_decision(action="HOLD", reasoning="Entry blocked: R/R 1.1 below minimum 2.0."),
        outcome=open_outcome(RECONCILE_OPEN_EXECUTOR_REPORTED),
    )
    fixture = harness(strategy, analysis={"signal": "BUY", "confidence": 90, "reasoning": "Breakout."})

    await fixture.bot._execute_trading_check(check_count=1)

    note = note_for(fixture)
    assert note is not None
    assert "BUY" in note
    assert "R/R 1.1 below minimum 2.0" in note
    forwarded = fixture.executor_handler.handle.await_args.args[1]
    assert forwarded.action == "HOLD"


async def test_accepted_update_on_the_same_trade_is_forwarded_after_a_tightening() -> None:
    """Regression (15:06): protection changed under the LLM, but the trade did not.

    The bot's own tightening policy may move the stop while the model is running. The
    trade token stays stable, the guard sees a NON-loosening UPDATE and the decision
    still reaches the executor — exactly what the token must not break.
    """
    strategy = FakeStrategy(
        position=make_position(),
        decision=update_decision(stop_loss=79550.0),
        tokens=[ENTRY_SNAPSHOT, ENTRY_SNAPSHOT],
        outcome=open_outcome(),
        protections=[
            TIGHTENED_PROTECTION,
            CHANGED_PROTECTION,
            CHANGED_PROTECTION,
            CHANGED_PROTECTION,
        ],
        guard_reason=None,
    )
    fixture = harness(strategy, analysis={"signal": "UPDATE", "confidence": 75, "reasoning": "Tighten SL."})

    await fixture.bot._execute_trading_check(check_count=1)

    assert strategy.guard_calls, "the guard must be consulted when the protection moved"
    assert note_for(fixture) is None
    forwarded = fixture.executor_handler.handle.await_args.args[1]
    assert forwarded.action == "UPDATE"
    assert forwarded.stop_loss == 79550.0


async def test_update_that_loosens_protection_is_refused_and_disclosed() -> None:
    """A stale UPDATE may not widen the stop: it is discarded, and the card says why."""
    strategy = FakeStrategy(
        position=make_position(),
        decision=update_decision(stop_loss=78800.0),
        tokens=[ENTRY_SNAPSHOT, ENTRY_SNAPSHOT],
        outcome=open_outcome(),
        protections=[
            TIGHTENED_PROTECTION,
            CHANGED_PROTECTION,
            CHANGED_PROTECTION,
            CHANGED_PROTECTION,
        ],
        guard_reason="UPDATE loosens protection (LONG SL 79,550.00 -> 78,800.00) — refused",
    )
    fixture = harness(strategy, analysis={"signal": "UPDATE", "confidence": 75, "reasoning": "Loosen the stop."})

    await fixture.bot._execute_trading_check(check_count=1)

    assert strategy.guard_calls, "the guard must be consulted before forwarding"
    fixture.executor_handler.handle.assert_not_awaited()
    note = note_for(fixture) or ""
    assert "loosen" in note
    assert "loosens protection" in note


async def test_protection_guard_is_not_consulted_when_nothing_changed() -> None:
    """Unchanged protection keeps the old path: no extra guard, no new refusal."""
    strategy = FakeStrategy(
        position=make_position(),
        decision=update_decision(stop_loss=79550.0),
        tokens=[ENTRY_SNAPSHOT, ENTRY_SNAPSHOT],
        outcome=open_outcome(),
        guard_reason="should never be reached",
    )
    fixture = harness(strategy, analysis={"signal": "UPDATE", "confidence": 75, "reasoning": "Tighten SL."})

    await fixture.bot._execute_trading_check(check_count=1)

    assert strategy.guard_calls == []
    fixture.executor_handler.handle.assert_awaited_once()
    assert note_for(fixture) is None


async def test_a_protection_version_change_during_the_analysis_refuses_the_update() -> None:
    """Wave-5 requirement (3): a protection REPLACED under the model invalidates the UPDATE.

    The executor revision is read before and after the analysis; a different value means
    the decision was computed for protection that no longer exists, so it is discarded
    and disclosed — never forwarded, whatever the SL/TP numbers say.
    """
    strategy = FakeStrategy(
        position=make_position(),
        decision=update_decision(stop_loss=79550.0),
        tokens=[ENTRY_SNAPSHOT, ENTRY_SNAPSHOT],
        outcome=open_outcome(),
        protection_versions=["rev-7", "rev-8"],
    )
    fixture = harness(strategy, analysis={"signal": "UPDATE", "confidence": 75, "reasoning": "Tighten SL."})

    await fixture.bot._execute_trading_check(check_count=1)

    assert strategy.protection_version_calls == 2, "the revision is read before AND after"
    fixture.executor_handler.handle.assert_not_awaited()
    fixture.position_monitor.handle_new_position.assert_not_awaited()
    note = note_for(fixture) or ""
    assert "UPDATE" in note
    assert "outdated state" in note
    assert "rev-7" in note and "rev-8" in note


async def test_an_unchanged_protection_version_still_forwards_the_update() -> None:
    """Same revision = the protection the decision was computed against still stands."""
    strategy = FakeStrategy(
        position=make_position(),
        decision=update_decision(stop_loss=79550.0),
        tokens=[ENTRY_SNAPSHOT, ENTRY_SNAPSHOT],
        outcome=open_outcome(),
        protection_versions=["rev-7", "rev-7"],
    )
    fixture = harness(strategy, analysis={"signal": "UPDATE", "confidence": 75, "reasoning": "Tighten SL."})

    await fixture.bot._execute_trading_check(check_count=1)

    assert strategy.protection_version_calls == 2
    forwarded = fixture.executor_handler.handle.await_args.args[1]
    assert forwarded.action == "UPDATE"
    assert note_for(fixture) is None


async def test_without_a_protection_version_the_guard_stays_inert() -> None:
    """An older executor without the revision: nothing changes (wave-1 guard only)."""
    strategy = FakeStrategy(
        position=make_position(),
        decision=update_decision(stop_loss=79550.0),
        tokens=[ENTRY_SNAPSHOT, ENTRY_SNAPSHOT],
        outcome=open_outcome(),
    )
    fixture = harness(strategy, analysis={"signal": "UPDATE", "confidence": 75, "reasoning": "Tighten SL."})

    await fixture.bot._execute_trading_check(check_count=1)

    assert strategy.protection_version_calls == 2
    fixture.executor_handler.handle.assert_awaited_once()
    assert note_for(fixture) is None


async def test_the_executor_counter_0_to_1_refuses_the_update() -> None:
    """The executor's real contract: ``protection_version`` 0 = none observed.

    ``0`` is a VALUE, not "no field": a position whose protection was registered while
    the model was working goes 0 -> 1, and the UPDATE computed against the unversioned
    state must be discarded.
    """
    strategy = FakeStrategy(
        position=make_position(),
        decision=update_decision(stop_loss=79550.0),
        tokens=[ENTRY_SNAPSHOT, ENTRY_SNAPSHOT],
        outcome=open_outcome(),
        protection_versions=["0", "1"],
    )
    fixture = harness(strategy, analysis={"signal": "UPDATE", "confidence": 75, "reasoning": "Tighten SL."})

    await fixture.bot._execute_trading_check(check_count=1)

    assert strategy.protection_version_calls == 2
    fixture.executor_handler.handle.assert_not_awaited()
    note = note_for(fixture) or ""
    assert "0 -> 1" in note


async def test_a_broken_protection_version_query_never_blocks_the_cycle() -> None:
    """A failing revision read is reported, not fatal: the cycle still runs."""
    strategy = FakeStrategy(
        position=make_position(),
        decision=update_decision(stop_loss=79550.0),
        tokens=[ENTRY_SNAPSHOT, ENTRY_SNAPSHOT],
        outcome=open_outcome(),
    )
    strategy.executor_protection_version = AsyncMock(side_effect=RuntimeError("executor down"))
    fixture = harness(strategy, analysis={"signal": "UPDATE", "confidence": 75, "reasoning": "Tighten SL."})

    await fixture.bot._execute_trading_check(check_count=1)

    fixture.executor_handler.handle.assert_awaited_once()
    assert note_for(fixture) is None
