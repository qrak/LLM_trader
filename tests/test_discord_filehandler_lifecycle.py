"""Lifecycle tests for DiscordFileHandler."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.notifiers.filehandler import DiscordFileHandler


def _make_handler(tmp_path, *, bot=None) -> DiscordFileHandler:
    if bot is None:
        bot = MagicMock()
        bot.loop = MagicMock()
    return DiscordFileHandler(
        bot=bot,
        logger=MagicMock(),
        config=MagicMock(FILE_MESSAGE_EXPIRY=120),
        tracking_file=str(tmp_path / "tracked_messages.json"),
    )


def test_handler_starts_uninitialized(tmp_path) -> None:
    handler = _make_handler(tmp_path)

    assert handler.is_initialized is False
    assert handler.cleanup_task is None


def test_initialize_sets_initialized_and_starts_cleanup_task(tmp_path) -> None:
    handler = _make_handler(tmp_path)
    task_mock = MagicMock()
    captured_coroutines = []

    def create_task_stub(coro, **kwargs):
        captured_coroutines.append(coro)
        return task_mock

    handler.bot.loop.create_task = create_task_stub

    handler.initialize()

    for coro in captured_coroutines:
        coro.close()

    assert handler.is_initialized is True
    assert handler.cleanup_task is task_mock



def test_reinitialize_cancels_previous_cleanup_task(tmp_path) -> None:
    handler = _make_handler(tmp_path)
    first_task = MagicMock()
    second_task = MagicMock()
    task_iter = iter([first_task, second_task])
    captured_coroutines = []

    def create_task_stub(coro, **kwargs):
        captured_coroutines.append(coro)
        return next(task_iter)

    handler.bot.loop.create_task = create_task_stub

    handler.initialize()
    handler.initialize()

    for coro in captured_coroutines:
        coro.close()

    first_task.cancel.assert_called_once_with()
    assert handler.cleanup_task is second_task


@pytest.mark.asyncio
async def test_track_message_returns_false_when_not_initialized_and_notifier_missing(tmp_path) -> None:
    handler = _make_handler(tmp_path, bot=SimpleNamespace())

    success = await handler.track_message(message_id=1, channel_id=2, user_id=3)

    assert success is False


@pytest.mark.asyncio
async def test_track_message_returns_false_when_ready_wait_times_out(tmp_path) -> None:
    handler = _make_handler(tmp_path)
    handler.bot.discord_notifier = MagicMock()

    async def wait_until_ready_stub() -> None:
        return None

    wait_coro = wait_until_ready_stub()
    handler.bot.discord_notifier.wait_until_ready.return_value = wait_coro

    with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
        success = await handler.track_message(message_id=1, channel_id=2, user_id=3)

    wait_coro.close()

    assert success is False
