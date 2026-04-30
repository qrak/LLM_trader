from unittest.mock import AsyncMock, MagicMock

import pytest

from src.platforms.ccxt_market_api import CCXTMarketAPI
from src.rag.market_components.market_data_fetcher import MarketDataFetcher


@pytest.mark.asyncio
async def test_fetch_price_data_uses_ccxt_result_without_market_api_fallback():
    logger = MagicMock()
    market_api = MagicMock()
    market_api.get_multi_price_data = AsyncMock(return_value={"RAW": {"BTC": {"USDT": {}}}})

    fetcher = MarketDataFetcher(logger=logger, market_api=market_api)
    fetcher._try_ccxt_price_data = AsyncMock(return_value={"RAW": {"BTC": {"USDT": {}}}})

    result = await fetcher.fetch_price_data(["BTC"])

    assert result == {"RAW": {"BTC": {"USDT": {}}}}
    market_api.get_multi_price_data.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_price_data_does_not_fallback_when_ccxt_returns_none():
    logger = MagicMock()
    market_api = MagicMock()
    market_api.get_multi_price_data = AsyncMock(return_value={"RAW": {"BTC": {"USDT": {}}}})

    fetcher = MarketDataFetcher(logger=logger, market_api=market_api)
    fetcher._try_ccxt_price_data = AsyncMock(return_value=None)

    result = await fetcher.fetch_price_data(["BTC"])

    assert result is None
    market_api.get_multi_price_data.assert_not_called()


@pytest.mark.asyncio
async def test_ccxt_market_api_coin_details_fallback_when_symbol_missing():
    logger = MagicMock()
    exchange_manager = MagicMock()
    exchange_manager.find_symbol_exchange = AsyncMock(return_value=(None, None))

    provider = CCXTMarketAPI(logger=logger, exchange_manager=exchange_manager)

    details = await provider.get_coin_details("ABC")

    assert details["symbol"] == "ABC"
    assert details["full_name"] == "ABC"
    assert details["description"] == ""


@pytest.mark.asyncio
async def test_ccxt_market_api_coin_details_maps_market_metadata():
    logger = MagicMock()
    exchange = MagicMock()
    exchange.markets = {
        "BTC/USDT": {
            "baseName": "Bitcoin",
            "base": "BTC",
            "active": True,
            "info": {"description": "Peer-to-peer electronic cash."},
        }
    }

    exchange_manager = MagicMock()
    exchange_manager.find_symbol_exchange = AsyncMock(return_value=(exchange, "binance"))

    provider = CCXTMarketAPI(logger=logger, exchange_manager=exchange_manager)

    details = await provider.get_coin_details("BTC")

    assert details["full_name"] == "Bitcoin"
    assert details["coin_name"] == "BTC"
    assert details["description"] == "Peer-to-peer electronic cash."
    assert details["is_trading"] is True