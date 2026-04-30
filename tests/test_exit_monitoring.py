import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.trading.data_models import Position
from src.trading.exit_monitor import ExitMonitor
from src.trading.position_status_monitor import PositionStatusMonitor
from src.trading.trading_strategy import TradingStrategy


class MonitorPersistence:
    def __init__(self, state=None):
        self.state = dict(state or {})

    async def async_load_position_monitor_state(self):
        return dict(self.state)

    async def async_save_position_monitor_state(self, state):
        self.state = dict(state)

    async def async_clear_position_monitor_state(self):
        self.state = {}

    def load_trade_history(self):
        return []


def _make_position(direction="LONG"):
    return Position(
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        size=1.0,
        entry_time=datetime(2026, 4, 30, tzinfo=timezone.utc),
        confidence="HIGH",
        direction=direction,
        symbol="BTC/USDC",
    )


def _make_strategy(position=None):
    strategy = TradingStrategy.__new__(TradingStrategy)
    strategy.current_position = position or _make_position()
    strategy.persistence = MagicMock()
    strategy.persistence.async_save_position = AsyncMock()
    strategy.close_position = AsyncMock()
    return strategy


async def _noop_sleep(_seconds, respect_force_analysis=True):
    return False


def _make_monitor_context(state=None, stop_type="hard", take_profit_type="hard", stop_interval="5m", take_profit_interval="15m"):
    config = SimpleNamespace(
        STOP_LOSS_TYPE=stop_type,
        STOP_LOSS_CHECK_INTERVAL=stop_interval,
        STOP_LOSS_CHECK_INTERVAL_SECONDS=300,
        TAKE_PROFIT_TYPE=take_profit_type,
        TAKE_PROFIT_CHECK_INTERVAL=take_profit_interval,
        TAKE_PROFIT_CHECK_INTERVAL_SECONDS=900,
        MAIN_CHANNEL_ID=123,
    )
    exit_monitor = ExitMonitor(config, "1h", 3600)
    persistence = MonitorPersistence(state)
    strategy = MagicMock()
    strategy.current_position = object()
    position_monitor = PositionStatusMonitor(
        logger=MagicMock(),
        config=config,
        persistence=persistence,
        trading_strategy=strategy,
        exit_monitor=exit_monitor,
        notifier=None,
        active_tasks=set(),
        is_running=lambda: False,
        fetch_current_ticker=AsyncMock(return_value=None),
        interruptible_sleep=_noop_sleep,
        get_symbol=lambda: "BTC/USDC",
    )
    return SimpleNamespace(
        config=config,
        exit_monitor=exit_monitor,
        persistence=persistence,
        trading_strategy=strategy,
        position_monitor=position_monitor,
    )


def test_exit_monitoring_rejects_interval_greater_than_timeframe():
    config = _make_monitor_context(stop_interval="1h", take_profit_interval="5m").config
    monitor = ExitMonitor(config, "15m", 3600)

    with pytest.raises(ValueError, match="stop loss interval"):
        monitor.validate()


def test_position_monitor_delay_uses_persisted_timestamp():
    now = datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc)
    state = {
        "last_status_sent_at": (now - timedelta(minutes=2)).isoformat(),
        "last_stop_loss_check_at": (now - timedelta(minutes=2)).isoformat(),
        "last_take_profit_check_at": (now - timedelta(minutes=2)).isoformat(),
    }
    context = _make_monitor_context(state=state, stop_type="hard", take_profit_type="soft")

    delay = context.exit_monitor.seconds_until_next_tick(state, now)

    assert delay == 180


@pytest.mark.asyncio
async def test_position_status_loop_waits_then_runs_due_hard_check_and_status():
    now = datetime.now(timezone.utc)
    state = {
        "last_status_sent_at": (now - timedelta(minutes=2)).isoformat(),
        "last_stop_loss_check_at": (now - timedelta(minutes=2)).isoformat(),
        "last_take_profit_check_at": (now - timedelta(minutes=2)).isoformat(),
    }
    context = _make_monitor_context(state=state, stop_type="hard", take_profit_type="hard")
    strategy = MagicMock()
    strategy.current_position = object()
    strategy.check_stop_loss = AsyncMock(return_value=None)
    strategy.check_take_profit = AsyncMock(return_value=None)
    notifier = MagicMock()
    notifier.send_position_status = AsyncMock()
    loop_checks = {"count": 0}
    sleep_calls = []

    def is_running():
        loop_checks["count"] += 1
        return loop_checks["count"] <= 2

    async def fake_sleep(seconds, respect_force_analysis=True):
        sleep_calls.append((seconds, respect_force_analysis))
        due_time = datetime.now(timezone.utc)
        context.persistence.state.update({
            "last_status_sent_at": (due_time - timedelta(minutes=5, seconds=1)).isoformat(),
            "last_stop_loss_check_at": (due_time - timedelta(minutes=5, seconds=1)).isoformat(),
            "last_take_profit_check_at": (due_time - timedelta(minutes=2)).isoformat(),
        })
        return False

    monitor = PositionStatusMonitor(
        logger=MagicMock(),
        config=context.config,
        persistence=context.persistence,
        trading_strategy=strategy,
        exit_monitor=context.exit_monitor,
        notifier=notifier,
        active_tasks=set(),
        is_running=is_running,
        fetch_current_ticker=AsyncMock(return_value={"last": 100.0}),
        interruptible_sleep=fake_sleep,
        get_symbol=lambda: "BTC/USDC",
    )

    await monitor._loop()

    assert len(sleep_calls) == 1
    assert 170 <= sleep_calls[0][0] <= 190
    assert sleep_calls[0][1] is False
    strategy.check_stop_loss.assert_awaited_once_with(100.0)
    strategy.check_take_profit.assert_not_called()
    notifier.send_position_status.assert_awaited_once()
    assert context.persistence.state["last_stop_loss_check_at"] != state["last_stop_loss_check_at"]
    assert context.persistence.state["last_status_sent_at"] != state["last_status_sent_at"]


@pytest.mark.asyncio
async def test_hard_monitor_checks_only_due_exit():
    now = datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc)
    state = {
        "last_stop_loss_check_at": (now - timedelta(minutes=5)).isoformat(),
        "last_take_profit_check_at": (now - timedelta(minutes=10)).isoformat(),
    }
    context = _make_monitor_context(state=state)
    strategy = MagicMock()
    strategy.current_position = object()
    strategy.check_stop_loss = AsyncMock(return_value=None)
    strategy.check_take_profit = AsyncMock(return_value=None)
    context.position_monitor.trading_strategy = strategy

    close_reason = await context.position_monitor.run_hard_exit_checks(
        100.0,
        now,
        state,
    )

    assert close_reason is None
    strategy.check_stop_loss.assert_awaited_once_with(100.0)
    strategy.check_take_profit.assert_not_called()
    assert context.persistence.state["last_stop_loss_check_at"] == now.isoformat()


@pytest.mark.asyncio
async def test_hard_monitor_stops_after_first_exit_closes_position():
    now = datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc)
    state = {
        "last_stop_loss_check_at": (now - timedelta(minutes=5)).isoformat(),
        "last_take_profit_check_at": (now - timedelta(minutes=15)).isoformat(),
    }
    context = _make_monitor_context(state=state)
    strategy = MagicMock()
    strategy.current_position = object()

    async def close_on_stop(_current_price):
        strategy.current_position = None
        return "stop_loss"

    strategy.check_stop_loss = AsyncMock(side_effect=close_on_stop)
    strategy.check_take_profit = AsyncMock(return_value=None)
    context.position_monitor.trading_strategy = strategy

    close_reason = await context.position_monitor.run_hard_exit_checks(
        94.0,
        now,
        state,
    )

    assert close_reason == "stop_loss"
    strategy.check_stop_loss.assert_awaited_once_with(94.0)
    strategy.check_take_profit.assert_not_called()
    assert context.persistence.state["last_stop_loss_check_at"] == now.isoformat()


@pytest.mark.asyncio
async def test_check_stop_loss_ignores_take_profit_hit():
    strategy = _make_strategy()

    close_reason = await strategy.check_stop_loss(111.0)

    assert close_reason is None
    strategy.close_position.assert_not_called()
    strategy.persistence.async_save_position.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_stop_loss_closes_only_stop_loss():
    strategy = _make_strategy()

    close_reason = await strategy.check_stop_loss(94.0)

    assert close_reason == "stop_loss"
    strategy.close_position.assert_awaited_once()
    assert strategy.close_position.await_args.args[0] == "stop_loss"


@pytest.mark.asyncio
async def test_check_take_profit_closes_only_take_profit():
    strategy = _make_strategy()

    close_reason = await strategy.check_take_profit(111.0)

    assert close_reason == "take_profit"
    strategy.close_position.assert_awaited_once()
    assert strategy.close_position.await_args.args[0] == "take_profit"


@pytest.mark.asyncio
async def test_check_take_profit_ignores_stop_loss_hit():
    strategy = _make_strategy()

    close_reason = await strategy.check_take_profit(94.0)

    assert close_reason is None
    strategy.close_position.assert_not_called()

