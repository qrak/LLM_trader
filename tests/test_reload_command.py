"""Tests for the in-place reload command (SHIFT+R) and its shutdown-manager flag."""

import asyncio
from unittest.mock import MagicMock

import pytest

from src.app import CryptoTradingBot
from src.utils.graceful_shutdown_manager import GracefulShutdownManager
from src.utils.keyboard_handler import KeyboardHandler


def _make_manager() -> GracefulShutdownManager:
    return GracefulShutdownManager(loop=asyncio.new_event_loop(), logger=MagicMock())


class TestReloadFlag:
    def test_reload_not_requested_by_default(self):
        manager = _make_manager()
        assert manager.reload_requested is False

    def test_request_reload_sets_flag(self):
        manager = _make_manager()
        assert manager.request_reload() is True
        assert manager.reload_requested is True

    def test_request_reload_refused_when_shutdown_in_progress(self):
        manager = _make_manager()
        manager._shutting_down = True
        assert manager.request_reload() is False
        assert manager.reload_requested is False


class TestRequestReloadCommand:
    def _bot(self) -> tuple[CryptoTradingBot, MagicMock]:
        # Minimal instance: only the attributes _request_reload touches.
        bot = object.__new__(CryptoTradingBot)
        bot.logger = MagicMock()
        bot.running = True
        manager = MagicMock()
        manager.request_reload.return_value = True
        bot.shutdown_manager = manager
        return bot, manager

    @pytest.mark.asyncio
    async def test_reload_refused_without_launcher_support(self, monkeypatch):
        monkeypatch.delenv("LLM_TRADER_RELOAD_SUPPORTED", raising=False)
        bot, manager = self._bot()
        await bot._request_reload()
        manager.request_reload.assert_not_called()
        assert bot.running is True

    @pytest.mark.asyncio
    async def test_reload_requests_graceful_shutdown(self, monkeypatch):
        monkeypatch.setenv("LLM_TRADER_RELOAD_SUPPORTED", "1")
        bot, manager = self._bot()
        await bot._request_reload()
        manager.request_reload.assert_called_once_with()
        assert bot.running is False

    @pytest.mark.asyncio
    async def test_reload_ignored_when_shutdown_already_running(self, monkeypatch):
        monkeypatch.setenv("LLM_TRADER_RELOAD_SUPPORTED", "1")
        bot, manager = self._bot()
        manager.request_reload.return_value = False
        await bot._request_reload()
        assert bot.running is True


class TestKeyboardShiftR:
    @pytest.mark.asyncio
    async def test_shift_r_triggers_uppercase_command(self):
        handler = KeyboardHandler(logger=None)
        hits: list[str] = []

        async def callback():
            hits.append("R")

        handler.register_command("R", callback, "Reload")
        handler._has_input = MagicMock(side_effect=[True, False])  # type: ignore[method-assign]
        handler._read_key = MagicMock(return_value="R")  # type: ignore[method-assign]
        await handler._process_keyboard_input()
        assert hits == ["R"]

    @pytest.mark.asyncio
    async def test_plain_lowercase_r_does_not_trigger_shift_command(self):
        handler = KeyboardHandler(logger=None)
        hits: list[str] = []

        async def callback():
            hits.append("R")

        handler.register_command("R", callback, "Reload")
        handler._has_input = MagicMock(side_effect=[True, False])  # type: ignore[method-assign]
        handler._read_key = MagicMock(return_value="r")  # type: ignore[method-assign]
        await handler._process_keyboard_input()
        assert hits == []
