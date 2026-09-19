"""Dense domain tests for the notification layer.

Covers the Discord notifier send path (spacing, tracking, transient retry), the
file-handler lifecycle, ticker fetch retry policy, stop/target distance guards,
position-status notification suppression and the performance-stats math.
"""

import asyncio
import io
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import ccxt
import discord
import pytest

from src.notifiers.base_notifier import BaseNotifier
from src.notifiers.filehandler import DiscordFileHandler
from src.notifiers.notifier import DiscordNotifier
from src.utils.decorators import retry_async


class FakeDiscordHTTPError(discord.HTTPException):
    """Discord HTTP error double exposing the status code."""

    def __init__(self, status: int, message: str = "discord send failure"):
        super().__init__(MagicMock(status=status), message)
        self.status = status


class DummyChannel:
    """Channel double recording every send."""

    def __init__(self):
        self.calls: list[dict] = []

    async def send(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(id=len(self.calls))


class DummyBot:
    """Bot double resolving a single channel."""

    def __init__(self, channel):
        self._channel = channel
        self.user = SimpleNamespace(name="DummyBot")

    def event(self, fn):
        return fn

    def get_channel(self, _channel_id):
        return self._channel


class StatsNotifier(BaseNotifier):
    """Minimal concrete notifier used to exercise the shared stats/distance math."""

    async def start(self) -> None:
        return None

    async def wait_until_ready(self) -> None:
        return None

    async def send_message(self, message: str, channel_id: int, expire_after: int | None = None) -> Any:
        return None

    async def send_trading_decision(self, decision: Any, channel_id: int) -> None:
        return None

    async def send_analysis_notification(
        self,
        result: dict[str, Any],
        symbol: str,
        timeframe: str,
        channel_id: int,
        chart_image: Any = None,
    ) -> None:
        return None

    async def send_position_status(self, position: Any, current_price: float, channel_id: int) -> None:
        return None

    async def send_performance_stats(self, trade_history: list[dict[str, Any]], symbol: str, channel_id: int) -> None:
        return None


def retrying_exchange(
    errors_to_raise: int,
    max_retries: int,
    error_factory=None,
    symbol: str = "BTC/USDC",
):
    """Exchange double whose _fetch_current_ticker fails N times then succeeds."""
    calls = {"count": 0}
    payload = {"last": 66500.0, "close": 66500.0}

    class FakeExchange:
        def __init__(self) -> None:
            self.logger = MagicMock()
            self.current_exchange = MagicMock()
            self.current_symbol = symbol

        @retry_async(max_retries=max_retries, initial_delay=0.01, backoff_factor=2, max_delay=1)
        async def _fetch_current_ticker(self) -> dict[str, Any] | None:
            calls["count"] += 1
            if calls["count"] <= errors_to_raise:
                raise error_factory()
            return payload

    return FakeExchange(), calls


@pytest.fixture
def notifier_bundle():
    """DiscordNotifier wired to a recording channel and a file-handler double."""
    unified_parser = MagicMock()
    unified_parser.extract_text_before_json.return_value = "Reasoning text"
    formatter = MagicMock()
    formatter.fmt.side_effect = lambda value: str(value)
    channel = DummyChannel()
    file_handler = MagicMock()
    file_handler.track_message = AsyncMock(return_value=True)
    notifier = DiscordNotifier(
        logger=MagicMock(),
        config=SimpleNamespace(FILE_MESSAGE_EXPIRY=120, TRANSACTION_FEE_PERCENT=0.001, QUOTE_CURRENCY="USDT"),
        unified_parser=unified_parser,
        formatter=formatter,
        bot=DummyBot(channel),
        file_handler=file_handler,
    )
    return notifier, channel, file_handler


def test_performance_stats_use_capital_not_sum_of_trade_returns():
    notifier = StatsNotifier(None, SimpleNamespace(DEMO_QUOTE_CAPITAL=10000.0, TRANSACTION_FEE_PERCENT=0.0), None, None)
    trade_history = [
        {"action": "BUY", "price": 100.0, "quantity": 1.0, "fee": 0.0},
        {"action": "CLOSE_LONG", "price": 110.0, "quantity": 1.0, "fee": 0.0},
        {"action": "BUY", "price": 100.0, "quantity": 20.0, "fee": 0.0},
        {"action": "CLOSE_LONG", "price": 99.0, "quantity": 20.0, "fee": 0.0},
    ]

    stats = notifier.calculate_performance_stats(trade_history)

    assert stats is not None
    assert stats["total_pnl_quote"] == pytest.approx(-10.0)
    assert stats["total_pnl_pct"] == pytest.approx(-0.1)
    assert stats["avg_pnl_pct"] == pytest.approx(4.5)
    assert stats["winning_trades"] == 1
    assert stats["closed_trades"] == 2


def test_stop_and_target_distances_guard_invalid_prices():
    notifier = StatsNotifier(None, MagicMock(), None, None)
    position = MagicMock()
    position.entry_price = 65000.0
    position.stop_loss = 66000.0
    position.take_profit = 63000.0
    position.size = 0.1

    for direction, price, expected_stop, expected_target in [
        ("LONG", 0.0, 0.0, 0.0),
        ("SHORT", 0.0, 0.0, 0.0),
        ("LONG", None, 0.0, 0.0),
        ("SHORT", -1.0, 0.0, 0.0),
        ("SHORT", 65000.0, -1.53846, 3.07692),
        ("LONG", 64500.0, 2.32558, -2.32558),
    ]:
        position.direction = direction
        stop_pct, target_pct = notifier.calculate_stop_target_distances(position, price)
        assert stop_pct == pytest.approx(expected_stop, rel=1e-3), (direction, price)
        assert target_pct == pytest.approx(expected_target, rel=1e-3), (direction, price)


async def test_fetch_current_ticker_retry_policy():
    exchange, calls = retrying_exchange(
        errors_to_raise=2, max_retries=3, error_factory=lambda: ccxt.RequestTimeout("binance GET /ticker")
    )
    assert await exchange._fetch_current_ticker() == {"last": 66500.0, "close": 66500.0}
    assert calls["count"] == 3

    exchange, calls = retrying_exchange(
        errors_to_raise=2, max_retries=3, error_factory=lambda: ccxt.RateLimitExceeded("DDoS protection triggered")
    )
    assert await exchange._fetch_current_ticker() == {"last": 66500.0, "close": 66500.0}
    assert calls["count"] == 3

    exchange, calls = retrying_exchange(
        errors_to_raise=99,
        max_retries=2,
        error_factory=lambda: aiohttp.ClientConnectorError(connection_key=MagicMock(), os_error=OSError("refused")),
    )
    with pytest.raises(aiohttp.ClientConnectorError):
        await exchange._fetch_current_ticker()
    assert calls["count"] == 3

    exchange, calls = retrying_exchange(
        errors_to_raise=99, max_retries=3, error_factory=lambda: ccxt.BadSymbol("Symbol not found")
    )
    with pytest.raises(ccxt.BadSymbol):
        await exchange._fetch_current_ticker()
    assert calls["count"] == 1


async def test_fetch_current_ticker_returns_none_without_exchange_or_symbol():
    class FakeExchange:
        def __init__(self) -> None:
            self.logger = MagicMock()
            self.current_exchange: Any = None
            self.current_symbol: Any = None

        @retry_async(max_retries=3, initial_delay=0.01, backoff_factor=2, max_delay=1)
        async def _fetch_current_ticker(self) -> dict[str, Any] | None:
            if self.current_exchange is None or self.current_symbol is None:
                return None
            return {"last": 66500.0}

    exchange = FakeExchange()
    fetch: Any = exchange._fetch_current_ticker
    assert await fetch() is None

    exchange.current_exchange = MagicMock()
    assert await fetch() is None

    exchange.current_symbol = "BTC/USDC"
    assert await fetch() == {"last": 66500.0}


async def test_position_status_monitor_skips_notification_when_price_is_unavailable():
    from src.trading.position_status_monitor import PositionStatusMonitor

    notifier = MagicMock()
    monitor = PositionStatusMonitor(
        logger=MagicMock(),
        config=MagicMock(),
        persistence=MagicMock(),
        trading_strategy=MagicMock(),
        exit_monitor=MagicMock(),
        notifier=notifier,
        active_tasks=set(),
        is_running=MagicMock(return_value=False),
        fetch_current_ticker=AsyncMock(return_value=None),
        interruptible_sleep=AsyncMock(),
        get_symbol=MagicMock(return_value="BTC/USDC"),
    )
    monitor.trading_strategy.current_position = MagicMock()
    monitor.exit_monitor.is_status_due = MagicMock(return_value=True)
    monitor.exit_monitor.seconds_until_next_tick = MagicMock(return_value=0)
    monitor.exit_monitor.check_hard_exits = AsyncMock(return_value=(None, {}))
    monitor.exit_monitor.due_hard_exits = MagicMock(return_value=[])
    monitor.exit_monitor.load_state = AsyncMock(return_value={})
    monitor.save_state = AsyncMock()

    ticks = {"count": 0}

    def is_running() -> bool:
        ticks["count"] += 1
        return ticks["count"] < 2

    monitor.is_running = is_running
    await monitor._loop()
    notifier.send_position_status.assert_not_called()

    monitor.is_running = MagicMock(return_value=True)
    await monitor.handle_new_position(current_price=None)
    notifier.send_position_status.assert_not_called()


def test_discord_file_handler_lifecycle(tmp_path):
    bot = MagicMock()
    bot.loop = MagicMock()
    handler = DiscordFileHandler(
        bot=bot,
        logger=MagicMock(),
        config=MagicMock(FILE_MESSAGE_EXPIRY=120),
        tracking_file=str(tmp_path / "tracked_messages.json"),
    )

    assert handler.is_initialized is False
    assert handler.cleanup_task is None

    tasks = iter([MagicMock(), MagicMock()])
    coroutines = []

    def create_task_stub(coro, **kwargs):
        coroutines.append(coro)
        return next(tasks)

    handler.bot.loop.create_task = create_task_stub

    handler.initialize()
    first_task = handler.cleanup_task
    handler.initialize()

    for coro in coroutines:
        coro.close()

    assert handler.is_initialized is True
    assert handler.cleanup_task is not first_task
    first_task.cancel.assert_called_once_with()


async def test_track_message_fails_without_bot_notifier_and_on_ready_timeout(tmp_path):
    handler = DiscordFileHandler(
        bot=SimpleNamespace(),
        logger=MagicMock(),
        config=MagicMock(FILE_MESSAGE_EXPIRY=120),
        tracking_file=str(tmp_path / "tracked_messages.json"),
    )
    assert await handler.track_message(message_id=1, channel_id=2, user_id=3) is False

    bot = MagicMock()
    bot.loop = MagicMock()
    handler = DiscordFileHandler(
        bot=bot,
        logger=MagicMock(),
        config=MagicMock(FILE_MESSAGE_EXPIRY=120),
        tracking_file=str(tmp_path / "tracked_messages.json"),
    )

    async def wait_until_ready() -> None:
        return None

    wait_coro = wait_until_ready()
    handler.bot.discord_notifier.wait_until_ready.return_value = wait_coro

    with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
        assert await handler.track_message(message_id=1, channel_id=2, user_id=3) is False

    wait_coro.close()


async def test_notifier_spacing_waits_between_messages(notifier_bundle):
    notifier, _, _ = notifier_bundle
    notifier._discord_send_interval_seconds = 0.4
    notifier._last_send_timestamp = asyncio.get_running_loop().time()
    sleep_mock = AsyncMock()

    with patch("src.notifiers.notifier.asyncio.sleep", sleep_mock):
        message = await notifier._send_with_spacing(AsyncMock(return_value=SimpleNamespace(id=42)))

    assert message.id == 42
    sleep_mock.assert_awaited_once()
    assert 0 < sleep_mock.await_args.args[0] <= 0.4


@pytest.mark.parametrize("kind", ["embed", "chart"], ids=["embed", "chart"])
async def test_notifier_tracks_sent_embed_and_chart(notifier_bundle, kind):
    notifier, _, file_handler = notifier_bundle
    notifier._send_with_spacing = AsyncMock(return_value=SimpleNamespace(id=777))

    if kind == "embed":
        result = await notifier._send_embed(discord.Embed(title="Test"), channel_id=123, expire_after=10.0)
        expected_type = "embed"
        expected_expiry = 10
    else:
        result = await notifier._send_analysis_chart(
            chart_image=io.BytesIO(b"fake_png_bytes"),
            symbol="BTC/USDT",
            timeframe="1h",
            channel_id=123,
            expire_after=15.0,
        )
        expected_type = "chart"
        expected_expiry = 15

    assert result.id == 777
    notifier._send_with_spacing.assert_awaited_once()
    file_handler.track_message.assert_awaited_once_with(
        message_id=777,
        channel_id=123,
        user_id=None,
        message_type=expected_type,
        expire_after=expected_expiry,
    )


async def test_send_analysis_notification_fans_out_text_embed_and_chart(notifier_bundle):
    notifier, _, _ = notifier_bundle
    notifier.send_message = AsyncMock()
    notifier._send_embed = AsyncMock()
    notifier._send_analysis_chart = AsyncMock()
    notifier._create_analysis_embed = MagicMock(return_value=discord.Embed(title="Analysis"))

    await notifier.send_analysis_notification(
        result={"analysis": {"signal": "BUY", "confidence": 80}, "raw_response": 'Reasoning text {"analysis":{}}'},
        symbol="BTC/USDT",
        timeframe="1h",
        channel_id=123,
        chart_image=io.BytesIO(b"fake_png_bytes"),
    )

    notifier.send_message.assert_awaited_once()
    notifier._send_embed.assert_awaited_once()
    notifier._send_analysis_chart.assert_awaited_once()


@pytest.mark.parametrize(
    ("status", "expected_attempts", "expected_sleeps", "succeeds"),
    [(503, 2, 1, True), (400, 1, 0, False)],
    ids=["transient-503-retried", "non-transient-400-fails-fast"],
)
async def test_transient_retry_policy(notifier_bundle, status, expected_attempts, expected_sleeps, succeeds):
    notifier, _, _ = notifier_bundle
    outcomes = [FakeDiscordHTTPError(status, "discord error"), SimpleNamespace(id=999)]
    notifier._send_with_spacing = AsyncMock(side_effect=outcomes if succeeds else outcomes[0])

    with patch("src.notifiers.notifier.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        if succeeds:
            result = await notifier._send_with_transient_retry(
                send_operation=AsyncMock(), operation_name="sending embed"
            )
            assert result.id == 999
        else:
            with pytest.raises(FakeDiscordHTTPError):
                await notifier._send_with_transient_retry(
                    send_operation=AsyncMock(), operation_name="sending embed"
                )

    assert notifier._send_with_spacing.await_count == expected_attempts
    assert sleep_mock.await_count == expected_sleeps
