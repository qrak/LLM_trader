"""Contract tests for ExchangeManager symbol preloading used by startup ticker validation."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.platforms.exchange_manager import ExchangeManager


def _make_manager() -> ExchangeManager:
    config = MagicMock()
    config.SUPPORTED_EXCHANGES = ["binance", "kucoin", "gate"]
    return ExchangeManager(logger=MagicMock(), config=config)


class TestEnsureSymbolsLoaded:
    @pytest.mark.asyncio
    async def test_loads_only_the_first_reachable_exchange(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = _make_manager()
        loader = AsyncMock(return_value=MagicMock())
        monkeypatch.setattr(manager, "_ensure_exchange_loaded", loader)

        await manager.ensure_symbols_loaded()

        assert [call.args[0] for call in loader.await_args_list] == ["binance"]

    @pytest.mark.asyncio
    async def test_falls_through_to_next_exchange_when_first_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = _make_manager()
        loader = AsyncMock(side_effect=[None, MagicMock()])
        monkeypatch.setattr(manager, "_ensure_exchange_loaded", loader)

        await manager.ensure_symbols_loaded()

        assert [call.args[0] for call in loader.await_args_list] == ["binance", "kucoin"]

    @pytest.mark.asyncio
    async def test_no_load_when_symbols_already_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = _make_manager()
        manager.symbols_by_exchange["binance"] = {"BTC/USDT"}
        loader = AsyncMock(return_value=MagicMock())
        monkeypatch.setattr(manager, "_ensure_exchange_loaded", loader)

        await manager.ensure_symbols_loaded()

        loader.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_gives_up_when_no_exchange_loads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = _make_manager()
        loader = AsyncMock(return_value=None)
        monkeypatch.setattr(manager, "_ensure_exchange_loaded", loader)

        await manager.ensure_symbols_loaded()

        assert [call.args[0] for call in loader.await_args_list] == ["binance", "kucoin", "gate"]
