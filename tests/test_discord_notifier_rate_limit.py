import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from src.notifiers.notifier import DiscordNotifier


class FakeDiscordHTTPError(Exception):
    def __init__(self, status: int, message: str = "discord send failure"):
        super().__init__(message)
        self.status = status


class DummyChannel:
    def __init__(self):
        self.calls = []

    async def send(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(id=len(self.calls))


class DummyBot:
    def __init__(self, channel):
        self._channel = channel
        self.user = SimpleNamespace(name="DummyBot")

    def event(self, fn):
        return fn

    def get_channel(self, _channel_id):
        return self._channel


@pytest.fixture
def notifier_fixture():
    logger = MagicMock()
    config = SimpleNamespace(
        FILE_MESSAGE_EXPIRY=120,
        TRANSACTION_FEE_PERCENT=0.001,
        QUOTE_CURRENCY="USDT",
    )
    unified_parser = MagicMock()
    unified_parser.extract_text_before_json.return_value = "Reasoning text"
    formatter = MagicMock()
    formatter.fmt.side_effect = lambda value: str(value)

    channel = DummyChannel()
    bot = DummyBot(channel)

    file_handler = MagicMock()
    file_handler.track_message = AsyncMock(return_value=True)

    notifier = DiscordNotifier(
        logger=logger,
        config=config,
        unified_parser=unified_parser,
        formatter=formatter,
        bot=bot,
        file_handler=file_handler,
    )
    return notifier, channel, file_handler


@pytest.mark.asyncio
async def test_send_with_spacing_waits_when_called_too_quickly(notifier_fixture):
    notifier, _, _ = notifier_fixture
    notifier._discord_send_interval_seconds = 0.4
    notifier._last_send_timestamp = asyncio.get_running_loop().time()

    operation = AsyncMock(return_value=SimpleNamespace(id=42))
    sleep_mock = AsyncMock()

    with patch("src.notifiers.notifier.asyncio.sleep", sleep_mock):
        message = await notifier._send_with_spacing(operation)

    assert message.id == 42
    operation.assert_awaited_once()
    sleep_mock.assert_awaited_once()
    waited_seconds = sleep_mock.await_args.args[0]
    assert 0 < waited_seconds <= 0.4


@pytest.mark.asyncio
async def test_send_embed_uses_spacing_and_tracks_message(notifier_fixture):
    notifier, _, file_handler = notifier_fixture
    sent_message = SimpleNamespace(id=777)
    notifier._send_with_spacing = AsyncMock(return_value=sent_message)

    embed = discord.Embed(title="Test", description="Embed")
    result = await notifier._send_embed(embed, channel_id=123, expire_after=10.0)

    assert result is sent_message
    notifier._send_with_spacing.assert_awaited_once()
    file_handler.track_message.assert_awaited_once_with(
        message_id=777,
        channel_id=123,
        user_id=None,
        message_type="embed",
        expire_after=10,
    )


@pytest.mark.asyncio
async def test_send_analysis_chart_uses_spacing_and_tracks_message(notifier_fixture):
    notifier, _, file_handler = notifier_fixture
    sent_message = SimpleNamespace(id=888)
    notifier._send_with_spacing = AsyncMock(return_value=sent_message)

    chart_image = io.BytesIO(b"fake_png_bytes")
    result = await notifier._send_analysis_chart(
        chart_image=chart_image,
        symbol="BTC/USDT",
        timeframe="1h",
        channel_id=123,
        expire_after=15.0,
    )

    assert result is sent_message
    notifier._send_with_spacing.assert_awaited_once()
    file_handler.track_message.assert_awaited_once_with(
        message_id=888,
        channel_id=123,
        user_id=None,
        message_type="chart",
        expire_after=15,
    )


@pytest.mark.asyncio
async def test_send_analysis_notification_sends_text_embed_and_chart(notifier_fixture):
    notifier, _, _ = notifier_fixture
    notifier.send_message = AsyncMock()
    notifier._send_embed = AsyncMock()
    notifier._send_analysis_chart = AsyncMock()
    notifier._create_analysis_embed = MagicMock(return_value=discord.Embed(title="Analysis"))

    result = {
        "analysis": {
            "signal": "BUY",
            "confidence": 80,
            "reasoning": "Structured analysis",
        },
        "raw_response": "Reasoning text {\"analysis\":{}}",
    }

    await notifier.send_analysis_notification(
        result=result,
        symbol="BTC/USDT",
        timeframe="1h",
        channel_id=123,
        chart_image=io.BytesIO(b"fake_png_bytes"),
    )

    notifier.send_message.assert_awaited_once()
    notifier._send_embed.assert_awaited_once()
    notifier._send_analysis_chart.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_with_transient_retry_retries_and_succeeds(notifier_fixture):
    notifier, _, _ = notifier_fixture
    notifier._send_with_spacing = AsyncMock(side_effect=[
        FakeDiscordHTTPError(503, "no healthy upstream"),
        SimpleNamespace(id=999),
    ])

    with patch("src.notifiers.notifier.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        result = await notifier._send_with_transient_retry(
            send_operation=AsyncMock(),
            operation_name="sending embed",
        )

    assert result.id == 999
    assert notifier._send_with_spacing.await_count == 2
    sleep_mock.assert_awaited_once_with(1.0)


@pytest.mark.asyncio
async def test_send_with_transient_retry_raises_for_non_transient(notifier_fixture):
    notifier, _, _ = notifier_fixture
    notifier._send_with_spacing = AsyncMock(side_effect=FakeDiscordHTTPError(400, "bad request"))

    with patch("src.notifiers.notifier.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        with pytest.raises(FakeDiscordHTTPError):
            await notifier._send_with_transient_retry(
                send_operation=AsyncMock(),
                operation_name="sending embed",
            )

    sleep_mock.assert_not_awaited()
