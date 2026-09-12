"""Contract tests for TickerManager category extraction and validation policy.

Regression guards for the 2026-09-12 RAG audit:
- categories are pipe-separated; the old comma split made category-coin
  extraction dead and let feed section names ("Technology", "Business", ...)
  through as pseudo-tickers;
- ticker candidates are only added when validated against live exchange
  symbols; the old "add by default when no exchange data" fallback polluted
  data/known_tickers.json permanently (no cleanup path exists).
"""
from unittest.mock import MagicMock

import pytest

from src.rag.ticker_manager import TickerManager


class _FakeExchangeManager:
    def __init__(self, symbols: set[str]):
        self._symbols = symbols

    def get_all_symbols(self) -> set[str]:
        return set(self._symbols)


def _make_manager(
    symbols: set[str] | None = None,
    with_exchange: bool = True,
    logger: MagicMock | None = None,
) -> TickerManager:
    return TickerManager(
        logger=logger or MagicMock(),
        file_handler=MagicMock(),
        exchange_manager=_FakeExchangeManager(symbols or set()) if with_exchange else None,
    )


class TestCategoryExtractionUsesPipeSeparator:
    def test_pipe_separated_categories_yield_real_tickers(self):
        manager = _make_manager(symbols={"BTC/USDT", "ETH/USDT"})

        coins = manager._extract_category_coins([{"categories": "BTC|ETH"}])

        assert coins == {"BTC", "ETH"}

    def test_whole_pipe_joined_string_is_not_a_single_token(self):
        manager = _make_manager(symbols={"BTC/USDT"})

        coins = manager._extract_category_coins([{"categories": "Markets|BTC|DeFi"}])

        # "Markets" and "DeFi" are skipped by the category filter; BTC is found.
        assert coins == {"BTC"}


class TestValidationPolicy:
    @pytest.mark.asyncio
    async def test_validated_base_asset_is_added(self):
        manager = _make_manager(symbols={"BTC/USDT", "ETH/USDT"})

        await manager.update_known_tickers([{"categories": "BTC"}])

        assert "BTC" in manager.known_tickers

    @pytest.mark.asyncio
    async def test_junk_category_not_added_when_exchange_data_present(self):
        manager = _make_manager(symbols={"BTC/USDT", "ETH/USDT"})

        await manager.update_known_tickers([{"categories": "Technology|BTC"}])

        assert "TECHNOLOGY" not in manager.known_tickers
        assert "BTC" in manager.known_tickers

    @pytest.mark.asyncio
    async def test_no_add_by_default_when_exchange_symbols_empty(self):
        logger = MagicMock()
        manager = _make_manager(symbols=set(), logger=logger)

        await manager.update_known_tickers([{"categories": "Technology|BTC"}])

        assert manager.known_tickers == set()
        warnings = [str(call) for call in logger.warning.call_args_list]
        assert any("Exchange symbol data unavailable" in warning for warning in warnings)

    @pytest.mark.asyncio
    async def test_no_add_when_exchange_manager_missing(self):
        logger = MagicMock()
        manager = _make_manager(with_exchange=False, logger=logger)

        await manager.update_known_tickers([{"categories": "BTC"}])

        assert manager.known_tickers == set()
        warnings = [str(call) for call in logger.warning.call_args_list]
        assert any("Exchange symbol data unavailable" in warning for warning in warnings)
