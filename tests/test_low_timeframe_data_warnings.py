from unittest.mock import AsyncMock, MagicMock

import pytest

from src.analyzer.data_fetcher import DataFetcher
from src.analyzer.market_data_collector import MarketDataCollector


def _ohlcv_rows(count: int):
    return [
        [index * 300000, 100.0, 101.0, 99.0, 100.5, 1000.0]
        for index in range(count)
    ]


def test_market_data_collector_warns_with_timeframe_aware_coverage():
    logger = MagicMock()
    collector = MarketDataCollector(logger=logger, rag_engine=MagicMock())
    collector.timeframe = "5m"

    collector._warn_if_insufficient_history(999)

    warning_calls = [call.args for call in logger.warning.call_args_list]
    assert any("Timeframe %s with %s closed candles covers" in args[0] for args in warning_calls)
    assert any(args[2] == "5m" and args[3] == 999 for args in warning_calls)


def test_market_data_collector_does_not_warn_when_coverage_is_complete():
    logger = MagicMock()
    collector = MarketDataCollector(logger=logger, rag_engine=MagicMock())
    collector.timeframe = "1h"

    collector._warn_if_insufficient_history(720)

    logger.warning.assert_not_called()


@pytest.mark.asyncio
async def test_data_fetcher_uses_timeframe_aware_expected_candles():
    exchange = MagicMock()
    exchange.id = "binance"
    exchange.timeframes = {"30m": "30m"}
    exchange.fetch_ohlcv = AsyncMock(return_value=_ohlcv_rows(401))
    logger = MagicMock()
    fetcher = DataFetcher(exchange=exchange, logger=logger)

    result = await fetcher.fetch_candlestick_data("BTC/USDT", "30m", 999)

    assert result is not None
    closed_candles, current_price = result
    assert len(closed_candles) == 400
    assert current_price == 100.5

    warning_calls = [call.args for call in logger.warning.call_args_list]
    assert any("for %s target coverage" in args[0] for args in warning_calls)
    assert any(args[1] == 400 and args[2] == 998 and args[3] == "30m" for args in warning_calls)