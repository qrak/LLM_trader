"""Dense domain tests for the application runtime: reload command, keyboard
shutdown/reload commands, timeframe scheduling, trading-check flow and the
crash-logging hooks.
"""

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.app import BotServices, CryptoTradingBot
from src.logger.logger import Logger, _write_fallback_crash
from src.utils.graceful_shutdown_manager import GracefulShutdownManager
from src.utils.keyboard_handler import KeyboardHandler
from src.utils.timeframe_validator import TimeframeValidator


class NaiveUtcDateTime(datetime):
    """Naive datetime that explodes if it is reinterpreted as local time."""

    def astimezone(self, tz=None):
        raise AssertionError("naive UTC timestamps must not be reinterpreted as local time")


def shutdown_manager() -> GracefulShutdownManager:
    """GracefulShutdownManager with a disposable loop."""
    return GracefulShutdownManager(loop=asyncio.new_event_loop(), logger=MagicMock())


def wait_bot() -> CryptoTradingBot:
    """Bot with mocked services, enough to exercise the wait helpers."""
    services = MagicMock()
    services.logger = MagicMock()
    services.config = MagicMock()
    services.dashboard_state = None
    services.discord_notifier = None
    services.shutdown_manager = None
    for name in (
        "exchange_manager",
        "market_analyzer",
        "trading_strategy",
        "keyboard_handler",
        "rag_engine",
        "persistence",
        "model_manager",
        "brain_service",
        "statistics_service",
        "memory_service",
        "exit_monitor",
    ):
        setattr(services, name, MagicMock())
    for name in ("coingecko_api", "market_api", "alternative_me_api", "http_session", "executor_handler",
                 "position_monitor_factory", "force_analysis_event", "discord_task"):
        setattr(services, name, None)

    bot = CryptoTradingBot(services)
    bot.current_timeframe = "15m"
    bot._interruptible_sleep = AsyncMock(return_value=False)
    return bot


def trading_check_bot(**overrides: Any) -> SimpleNamespace:
    """Bot wired for _execute_trading_check with a recording notifier."""
    config = SimpleNamespace(
        MAIN_CHANNEL_ID=123,
        RAG_UPDATE_TIMEOUT=1,
        EXECUTOR_API_ENABLED=False,
        EXECUTOR_API_URL="http://127.0.0.1:9199/decision",
        RESEARCH_TEAM_ENABLED=False,
        MIN_RR_ENTRY=1.0,
        EXECUTOR_MAX_POSITION_USDC=0.0,
        DEMO_QUOTE_CAPITAL=10000.0,
        SOCIAL_SENTIMENT_ENABLED=False,
        EXECUTOR_VERDICT_PATH="data/trading/executor_verdicts.jsonl",
    )
    discord_notifier = MagicMock()
    discord_notifier.send_trading_decision = AsyncMock()
    discord_notifier.send_analysis_notification = AsyncMock()
    market_analyzer = MagicMock()
    market_analyzer.analyze_market = AsyncMock(
        return_value={
            "analysis": {"signal": "SELL", "confidence": 82, "reasoning": "Bearish structure confirmed."},
            "raw_response": 'AI reasoning text {"analysis": {}}',
        }
    )
    market_analyzer.last_chart_buffer = None
    persistence = MagicMock()
    persistence.async_save_last_analysis_time = AsyncMock()
    persistence.get_last_analysis_time = MagicMock(return_value=datetime.now(timezone.utc))
    persistence.async_load_previous_response = AsyncMock(return_value={})
    trading_strategy = MagicMock()
    trading_strategy.process_analysis = AsyncMock(return_value=SimpleNamespace(action="SELL"))
    trading_strategy.current_position = object()
    trading_strategy.get_position_context = MagicMock(return_value="position context")
    memory_service = MagicMock()
    memory_service.get_context_summary = MagicMock(return_value="memory")
    statistics_service = MagicMock()
    statistics_service.get_context = MagicMock(return_value="stats")
    brain_service = MagicMock()
    brain_service.get_dynamic_thresholds = MagicMock(return_value={})
    position_monitor = MagicMock()
    position_monitor.check_soft_exit_status = AsyncMock()
    position_monitor.handle_new_position = AsyncMock()

    services: dict[str, Any] = {
        "logger": MagicMock(),
        "config": config,
        "shutdown_manager": None,
        "exchange_manager": MagicMock(),
        "market_analyzer": market_analyzer,
        "trading_strategy": trading_strategy,
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
        "position_monitor_factory": lambda _bot: position_monitor,
    }
    services.update(overrides)

    bot = CryptoTradingBot(BotServices(**services))
    bot.current_symbol = "BTC/USDC"
    bot.current_timeframe = "4h"
    bot._fetch_ticker_data = AsyncMock(return_value=({"last": 77163.94}, 77163.94))
    bot._execute_market_knowledge_update = AsyncMock()
    bot._build_analysis_context = AsyncMock(return_value={})
    bot._save_analysis_data = AsyncMock()
    return SimpleNamespace(
        bot=bot,
        discord_notifier=discord_notifier,
        market_analyzer=market_analyzer,
        position_monitor=position_monitor,
        trading_strategy=trading_strategy,
    )


async def test_exchange_side_exit_reports_through_the_position_monitor() -> None:
    """A flagged exchange-side exit sends the closing summary the monitor never saw."""
    harness = trading_check_bot()
    harness.position_monitor.handle_position_closed = AsyncMock()
    harness.trading_strategy.take_executor_side_exit_reason = MagicMock(
        return_value="Executor confirmed flat — stop_loss_filled @ 78800.0"
    )

    await harness.bot._report_executor_side_exit()

    harness.position_monitor.handle_position_closed.assert_awaited_once_with(
        "Executor confirmed flat — stop_loss_filled @ 78800.0"
    )


async def test_no_closing_summary_without_a_booked_exchange_exit() -> None:
    """Nothing is reported when no exit was booked, or when the flag is not a reason string."""
    harness = trading_check_bot()
    harness.position_monitor.handle_position_closed = AsyncMock()

    harness.trading_strategy.take_executor_side_exit_reason = MagicMock(return_value=None)
    await harness.bot._report_executor_side_exit()
    harness.position_monitor.handle_position_closed.assert_not_awaited()

    harness.trading_strategy.take_executor_side_exit_reason = MagicMock(return_value=MagicMock())
    await harness.bot._report_executor_side_exit()
    harness.position_monitor.handle_position_closed.assert_not_awaited()


async def test_state_divergence_is_alerted_to_the_operator() -> None:
    """When the executor claims flat without an exit record, the operator gets told."""
    harness = trading_check_bot()
    harness.discord_notifier.send_message = AsyncMock()
    harness.trading_strategy.take_state_divergence = MagicMock(
        return_value="executor reports no open position for BTC/USDC, but has no exit record"
    )

    await harness.bot._report_state_divergence()

    calls = harness.discord_notifier.send_message.await_args_list
    assert len(calls) == 1
    assert "no exit record" in calls[0].args[0]


async def test_local_exit_recommendation_is_forwarded_never_closed_locally() -> None:
    """A local SL/TP request leaves the bot as a COMMAND to the executor.

    Wave 3: the monitor detects the bracket but cannot close on the exchange, so the
    queued CLOSE recommendation is drained here and forwarded. Nothing local is booked
    — the intent stays pending until the executor's fill evidence arrives.
    """
    from src.trading.data_models import TradeDecision

    executor_handler = MagicMock()
    executor_handler.handle = AsyncMock(return_value=True)
    harness = trading_check_bot(executor_handler=executor_handler)
    decision = TradeDecision(
        timestamp=datetime.now(timezone.utc),
        symbol="BTC/USDC",
        action="CLOSE",
        confidence="HIGH",
        price=78000.0,
        fee=0.0,
        reasoning="the local exit condition stop_loss",
        order_id="close-1",
    )
    harness.trading_strategy.take_pending_local_close_decision = MagicMock(return_value=decision)
    harness.trading_strategy.resolve_position_intents_after_forward = AsyncMock(
        return_value="pending"
    )

    state = await harness.bot._forward_local_exit_request()

    assert state == "pending"
    executor_handler.handle.assert_awaited_once()
    assert executor_handler.handle.await_args.args == ({"signal": "CLOSE"}, decision, "BTC/USDC")
    harness.trading_strategy.resolve_position_intents_after_forward.assert_awaited_once_with(
        order_id="close-1", delivered=True, symbol="BTC/USDC"
    )


async def test_local_exit_recommendation_is_queued_when_no_executor_handler_exists() -> None:
    """No handler -> nothing is sent AND the recommendation is not consumed."""
    executor_handler_missing = trading_check_bot()
    executor_handler_missing.trading_strategy.take_pending_local_close_decision = MagicMock(
        return_value=SimpleNamespace(action="CLOSE", order_id="close-1")
    )

    assert await executor_handler_missing.bot._forward_local_exit_request() is None
    executor_handler_missing.trading_strategy.take_pending_local_close_decision.assert_not_called()


def test_reload_flag_lifecycle():
    manager = shutdown_manager()
    try:
        assert manager.reload_requested is False
        assert manager.request_reload() is True
        assert manager.reload_requested is True

        manager._shutting_down = True
        assert manager.request_reload() is False
    finally:
        manager.loop.close()


@pytest.mark.parametrize(
    ("launcher_supported", "manager_accepts", "expected_requests", "expected_running"),
    [(False, True, 0, True), (True, True, 1, False), (True, False, 1, True)],
    ids=["no-launcher-support", "reload-accepted", "shutdown-already-running"],
)
async def test_request_reload_command(monkeypatch, launcher_supported, manager_accepts, expected_requests, expected_running):
    if launcher_supported:
        monkeypatch.setenv("LLM_TRADER_RELOAD_SUPPORTED", "1")
    else:
        monkeypatch.delenv("LLM_TRADER_RELOAD_SUPPORTED", raising=False)

    bot = object.__new__(CryptoTradingBot)
    bot.logger = MagicMock()
    bot.running = True
    manager = MagicMock()
    manager.request_reload.return_value = manager_accepts
    bot.shutdown_manager = manager

    await bot._request_reload()

    assert manager.request_reload.call_count == expected_requests
    assert bot.running is expected_running


async def test_keyboard_commands_are_case_sensitive(monkeypatch):
    handler = KeyboardHandler(logger=None)
    hits: list[str] = []

    async def callback():
        hits.append("R")

    handler.register_command("R", callback, "Reload")
    monkeypatch.setattr(handler, "_has_input", MagicMock(side_effect=[True, False, True, False]))
    monkeypatch.setattr(handler, "_read_key", MagicMock(side_effect=["R", "r"]))

    await handler._process_keyboard_input()
    assert hits == ["R"]

    await handler._process_keyboard_input()
    assert hits == ["R"]


def test_timeframe_validator_boundaries():
    for timeframe, expected_minutes in [("5m", 5), ("15m", 15), ("30m", 30)]:
        assert TimeframeValidator.validate(timeframe) is True
        assert TimeframeValidator.is_ccxt_compatible(timeframe) is True
        assert TimeframeValidator.to_minutes(timeframe) == expected_minutes
        assert TimeframeValidator.validate_and_normalize(timeframe.upper()) == timeframe

    current_time = datetime(2026, 4, 30, 13, 43, 28, tzinfo=timezone.utc)
    for timeframe, expected_next in [
        ("5m", datetime(2026, 4, 30, 13, 45, tzinfo=timezone.utc)),
        ("15m", datetime(2026, 4, 30, 13, 45, tzinfo=timezone.utc)),
        ("30m", datetime(2026, 4, 30, 14, 0, tzinfo=timezone.utc)),
    ]:
        next_candle = TimeframeValidator.calculate_next_candle_time(
            int(current_time.timestamp() * 1000), timeframe
        )
        assert datetime.fromtimestamp(next_candle / 1000, timezone.utc) == expected_next

    with pytest.raises(ValueError, match="Unsupported timeframe"):
        TimeframeValidator.validate_and_normalize("3m")
    with pytest.raises(ValueError, match="Unrecognized timeframe"):
        TimeframeValidator.to_minutes("3m")


def test_next_candle_time_stays_on_utc_boundaries_across_dst_change():
    warsaw = ZoneInfo("Europe/Warsaw")
    before_dst = datetime(2026, 3, 29, 1, 30, tzinfo=warsaw)
    after_dst = datetime(2026, 3, 29, 6, 30, tzinfo=warsaw)

    next_before = TimeframeValidator.calculate_next_candle_time(int(before_dst.timestamp() * 1000), "4h")
    next_after = TimeframeValidator.calculate_next_candle_time(int(after_dst.timestamp() * 1000), "4h")

    assert datetime.fromtimestamp(next_before / 1000, timezone.utc) == datetime(2026, 3, 29, 4, 0, tzinfo=timezone.utc)
    assert datetime.fromtimestamp(next_after / 1000, timezone.utc) == datetime(2026, 3, 29, 8, 0, tzinfo=timezone.utc)


def test_bot_formats_analysis_times_in_utc_and_local():
    bot = CryptoTradingBot.__new__(CryptoTradingBot)
    bot.persistence = MagicMock()
    bot.persistence.get_last_analysis_time.return_value = NaiveUtcDateTime(2026, 4, 14, 4, 0, 48)

    assert bot._get_formatted_last_analysis_time() == "2026-04-14 04:00:48"
    assert (
        CryptoTradingBot._format_utc_and_local(datetime(2026, 4, 14, 8, 0, tzinfo=timezone.utc), ZoneInfo("Europe/Warsaw"))
        == "2026-04-14 08:00:00 UTC / 2026-04-14 10:00:00 CEST"
    )


def test_calculate_next_check_delay_and_guard():
    bot = wait_bot()

    with patch("src.app.time.time", return_value=1_000_000.0), patch("src.app.TimeframeValidator") as validator:
        validator.calculate_next_candle_time.return_value = int((1_000_000.0 + 200) * 1000)
        delay_seconds, next_check_time = bot._calculate_next_check(int((1_000_000.0 - 100) * 1000))

    assert delay_seconds > 0
    assert type(next_check_time) is datetime
    assert next_check_time.tzinfo == timezone.utc

    with patch("src.app.TimeframeValidator") as validator:
        validator.calculate_next_candle_time.return_value = 1_747_000_000_000
        assert bot._calculate_next_check(1_746_000_000_000)[0] == 0.0

    bot.current_timeframe = None
    with pytest.raises(ValueError, match="current timeframe is not set"):
        bot._calculate_next_check(1_000_000_000_000)


async def test_wait_for_next_timeframe_logs_updates_dashboard_and_recovers():
    bot: Any = wait_bot()
    bot.dashboard_state = MagicMock()
    bot.dashboard_state.update_next_check = AsyncMock()

    with patch("src.app.TimeframeValidator") as validator:
        validator.calculate_next_candle_time.return_value = 1_750_000_000_000
        assert await bot._wait_for_next_timeframe() is False

    bot._interruptible_sleep.assert_awaited_once()
    assert "Next check" in bot.logger.info.call_args[0][0]
    bot.dashboard_state.update_next_check.assert_awaited_once()

    bot.current_timeframe = None
    bot._interruptible_sleep.reset_mock()
    with patch("src.app.ERROR_WAIT_LONG", 1):
        assert await bot._wait_for_next_timeframe() is False
    bot._interruptible_sleep.assert_awaited_with(1)


async def test_wait_until_next_timeframe_after_short_circuits_and_recovers():
    bot: Any = wait_bot()
    last_analysis = datetime.now(timezone.utc) - timedelta(hours=1)

    with patch("src.app.TimeframeValidator") as validator:
        validator.calculate_next_candle_time.return_value = 1
        assert await bot._wait_until_next_timeframe_after(last_analysis) is None
    bot._interruptible_sleep.assert_not_called()

    with patch("src.app.TimeframeValidator") as validator:
        validator.calculate_next_candle_time.return_value = 9_999_999_999_999_999
        validator.is_same_candle.return_value = True
        await bot._wait_until_next_timeframe_after(last_analysis)
    bot._interruptible_sleep.assert_awaited_once()

    bot.current_timeframe = None
    bot._interruptible_sleep.reset_mock()
    with patch("src.app.ERROR_WAIT_SHORT", 1):
        await bot._wait_until_next_timeframe_after(last_analysis)
    bot._interruptible_sleep.assert_awaited_with(1)


async def test_resume_logs_a_single_resuming_line_with_the_next_check():
    """A restart must not print two "Resuming ..." lines for the same timestamp."""
    bot: Any = wait_bot()
    last_analysis = datetime.now(timezone.utc) - timedelta(hours=1)
    next_candle_ms = int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp() * 1000)

    with patch("src.app.TimeframeValidator") as validator:
        validator.calculate_next_candle_time.return_value = next_candle_ms
        validator.is_same_candle.return_value = True
        await bot._wait_until_next_timeframe_after(last_analysis)

    resuming = [
        call[0][0] for call in bot.logger.info.call_args_list if "Resuming" in call[0][0]
    ]

    assert len(resuming) == 1
    assert "next check at" in resuming[0]


async def test_execute_trading_check_notifies_with_analysis_and_hands_position_to_monitor():
    fixture = trading_check_bot()

    await fixture.bot._execute_trading_check(check_count=1, force_news_update=True, is_candle_close=True)

    fixture.discord_notifier.send_trading_decision.assert_not_awaited()
    fixture.discord_notifier.send_analysis_notification.assert_awaited_once()
    fixture.position_monitor.handle_new_position.assert_awaited_once()
    fixture.trading_strategy.process_analysis.assert_awaited_once()


def test_crash_hooks_write_unhandled_errors_into_errors_log(tmp_path, monkeypatch):
    logger = Logger(logger_name="TestBot", log_dir=str(tmp_path / "logs"))
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    logger.install_crash_handler()

    import src.logger.logger as logger_module

    crash_path = Path(logger_module._CRASH_LOG_PATH)
    assert crash_path == tmp_path / "logs" / "TestBot" / datetime.now(timezone.utc).strftime("%Y_%m_%d") / "errors.log"

    try:
        raise RuntimeError("TEST: fallback writer")
    except RuntimeError as exc:
        _write_fallback_crash(type(exc), exc, exc.__traceback__, "TEST: fallback writer")

    written = crash_path.read_text(encoding="utf-8")
    assert "TEST: fallback writer" in written
    assert "RuntimeError" in written

    for handler in list(logger.handlers):
        base = getattr(handler, "baseFilename", None)
        if base is not None and Path(base) == crash_path:
            handler.close()
            logger.removeHandler(handler)

    crash_path.unlink()
    try:
        raise asyncio.CancelledError("TEST: cancelled from task")
    except asyncio.CancelledError as exc:
        sys.excepthook(type(exc), exc, exc.__traceback__)

    assert "TEST: cancelled from task" in crash_path.read_text(encoding="utf-8")


def test_crash_writer_is_a_noop_without_a_configured_path(monkeypatch):
    monkeypatch.setattr("src.logger.logger._CRASH_LOG_PATH", "")

    try:
        raise RuntimeError("must not be written anywhere")
    except RuntimeError as exc:
        _write_fallback_crash(type(exc), exc, exc.__traceback__)


async def test_unconfirmed_command_is_alerted_to_the_operator() -> None:
    """A forwarded command without executor evidence raises a loud operator alert."""
    harness = trading_check_bot()
    harness.discord_notifier.send_message = AsyncMock()
    harness.trading_strategy.take_unconfirmed_intent_alert = MagicMock(
        return_value="CLOSE BTC/USDC was not confirmed by the executor"
    )

    await harness.bot._report_unconfirmed_intent()

    calls = harness.discord_notifier.send_message.await_args_list
    assert len(calls) == 1
    assert "was not confirmed by the executor" in calls[0].args[0]
    assert "no proof it was executed" in calls[0].args[0]


async def test_no_unconfirmed_alert_without_an_unconfirmed_intent() -> None:
    """Nothing is sent when the strategy reports no pending/unknown command."""
    harness = trading_check_bot()
    harness.discord_notifier.send_message = AsyncMock()

    harness.trading_strategy.take_unconfirmed_intent_alert = MagicMock(return_value=None)
    await harness.bot._report_unconfirmed_intent()
    harness.discord_notifier.send_message.assert_not_awaited()

    harness.trading_strategy.take_unconfirmed_intent_alert = MagicMock(return_value=MagicMock())
    await harness.bot._report_unconfirmed_intent()
    harness.discord_notifier.send_message.assert_not_awaited()


async def test_refused_command_is_alerted_to_the_operator() -> None:
    """A command the bot's own policy refused is reported, never swallowed silently."""
    harness = trading_check_bot()
    harness.discord_notifier.send_message = AsyncMock()
    harness.trading_strategy.take_rejected_intent_alert = MagicMock(
        return_value="UPDATE BTC/USDC was refused by the bot's own policy (progress 14.6% < 15.0%)"
    )

    await harness.bot._report_rejected_intent()

    calls = harness.discord_notifier.send_message.await_args_list
    assert len(calls) == 1
    assert "refused by the bot's own policy" in calls[0].args[0]
    assert "never reached the executor" in calls[0].args[0]


async def test_no_refused_alert_without_a_refused_command() -> None:
    """Nothing is sent while the strategy reports no refusal."""
    harness = trading_check_bot()
    harness.discord_notifier.send_message = AsyncMock()
    harness.trading_strategy.take_rejected_intent_alert = MagicMock(return_value=None)

    await harness.bot._report_rejected_intent()

    harness.discord_notifier.send_message.assert_not_awaited()


async def test_forwarded_command_is_resolved_through_the_strategy_hook() -> None:
    """The app hands the forwarded command's intent to the strategy (evidence required)."""
    harness = trading_check_bot()
    harness.trading_strategy.resolve_position_intents_after_forward = AsyncMock(
        return_value="unknown"
    )
    decision = SimpleNamespace(action="CLOSE", order_id="close-1")

    state = await harness.bot._resolve_position_intents(decision, True)

    assert state == "unknown"
    harness.trading_strategy.resolve_position_intents_after_forward.assert_awaited_once_with(
        order_id="close-1", delivered=True, symbol="BTC/USDC"
    )


async def test_resolution_is_none_without_a_working_strategy_hook() -> None:
    """A failing/absent hook is not guessed into a state — None is returned instead."""
    harness = trading_check_bot()
    harness.trading_strategy.resolve_position_intents_after_forward = MagicMock(
        side_effect=AttributeError("no hook")
    )
    harness.trading_strategy.position_intent_state = MagicMock(return_value="pending")

    state = await harness.bot._resolve_position_intents(
        SimpleNamespace(action="BUY", order_id="order-1"), True
    )

    assert state is None

    strategy_without_resolver = MagicMock(spec=["position_intent_state"])
    strategy_without_resolver.position_intent_state = MagicMock(return_value="pending")
    harness.bot.trading_strategy = strategy_without_resolver

    assert (
        await harness.bot._resolve_position_intents(
            SimpleNamespace(action="BUY", order_id="order-1"), True
        )
        == "pending"
    )


def test_execution_note_reports_a_pending_command_as_not_confirmed() -> None:
    """The card must say the command is unconfirmed instead of showing an execution."""
    harness = trading_check_bot()
    result = {"analysis": {"signal": "CLOSE"}}
    decision = SimpleNamespace(action="CLOSE", order_id="close-1")

    pending_note = harness.bot._build_execution_note(
        result, decision, False, None, intent_state="pending"
    )
    unknown_note = harness.bot._build_execution_note(
        result, decision, False, None, intent_state="unknown"
    )
    confirmed_note = harness.bot._build_execution_note(
        result, decision, False, None, intent_state="confirmed"
    )
    missing_note = harness.bot._build_execution_note(result, decision, False, None)

    assert pending_note is not None and "UNCONFIRMED" in pending_note
    assert unknown_note is not None and "UNCONFIRMED" in unknown_note
    assert "local state" in pending_note.lower() or "do NOT count" in pending_note
    assert confirmed_note is None, "a confirmed command is an execution — no disclaimer"
    assert missing_note is None, "no intent layer (test double) keeps the legacy behaviour"
