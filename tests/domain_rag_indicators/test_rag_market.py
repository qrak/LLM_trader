"""Dense domain tests for the RAG market-data path.

Covers MarketDataManager's cache and refresh contract, MarketOverviewBuilder's
structure and finalisation rules, the CCXT market-metadata and price seams, and
the analyzer's timeframe-aware candle coverage warnings.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import numpy as np
import pytest

from src.analyzer.data_fetcher import DataFetcher
from src.analyzer.market_data_collector import MarketDataCollector
from src.platforms.ccxt_market_api import CCXTMarketAPI
from src.rag.market_components import MarketDataCache
from src.rag.market_components.market_data_fetcher import MarketDataFetcher
from src.rag.market_components.market_overview_builder import MarketOverviewBuilder
from src.rag.market_data_manager import MarketDataManager

OVERVIEW_KEYS = {"timestamp", "summary", "coin_data", "published_on", "data_sources"}
INSUFFICIENT_HISTORY = (
    "Insufficient data for full %s-day analysis. Timeframe %s with %s closed "
    "candles covers ~%.1f days; need %s candles. Multi-day metrics may be partial."
)
COVERAGE_WARNING = (
    "Received fewer closed candles (%s) than expected (%s) for %s target coverage "
    "(~%.1f days available)"
)
COVERAGE_UNKNOWN = "Could not evaluate historical coverage for timeframe %s: %s"
EXCHANGE_LIMIT_WARNING = (
    "Requested limit %s exceeds exchange standard limits, may be truncated"
)
NOT_ENOUGH_CANDLES = "Not enough candles to exclude incomplete one. Received: %s"
NO_CANDLES = "No data returned for %s on %s"
TIMEFRAME_UNSUPPORTED = "Timeframe %s not supported by %s. Supported: %s"
TIMEFRAME_UNLISTED = (
    "Timeframe %s may not be supported by %s. Attempting fetch anyway..."
)
TIMESTAMPS_UNUSABLE = "Could not extract timestamps from OHLCV data: %s"
SECONDARY_FAILED = "Secondary market data fetcher %d failed: %s"
NO_OVERVIEW_DATA = "No market overview data was available from data sources"
FETCH_OVERVIEW_ERROR = "Error fetching market overview: %s"
BUILD_ERROR = "Error building overview structure: %s"
STALE_OVERVIEW = "Market overview data is older than %s hours, refreshing"
FETCHING_OVERVIEW = "Fetching market overview data"
OVERVIEW_UPDATED = "Market overview updated successfully."
DERIVED_LIMIT = "Calculated candle limit: %s for %s timeframe (~%s days of data)"

COINGECKO_TOP_COINS = {
    "data": {
        "top_coins": [
            {
                "symbol": "BTC",
                "current_price": 40000,
                "price_change_percentage_24h": 0.0,
                "total_volume": 0,
            }
        ]
    }
}
BTC_PRICE_DATA = {"BTC/USDT": {"price": 50000, "change_24h": 2.0, "volume": 1e6}}
ABC_DETAILS = {
    "description": "",
    "full_name": "ABC",
    "coin_name": "ABC",
    "symbol": "ABC",
    "is_trading": True,
}


class _StubProcessor:
    """Deterministic stand-in for MarketDataProcessor.process_coin_data.

    Echoes price, change_24h and volume for any non-empty payload and returns
    None otherwise, so the builder's own mapping logic is what gets exercised.
    """

    def process_coin_data(self, values: dict[str, Any]) -> dict[str, Any] | None:
        if not values:
            return None
        return {
            "price": values.get("price", 0),
            "change_24h": values.get("change_24h", 0),
            "volume": values.get("volume", 0),
        }


def rich_coin(
    symbol: str, rank: int, price: float = 0, change: float = 0, volume: float = 0
):
    """Top-coin entry shaped the way build_overview_structure emits bare symbols."""
    return {
        "symbol": symbol,
        "name": symbol,
        "market_cap_rank": rank,
        "current_price": price,
        "price_change_percentage_24h": change,
        "total_volume": volume,
    }


def ohlcv_rows(count: int):
    """Ascending OHLCV rows whose close price encodes the row index."""
    return [
        [index * 300000, 100.0, 101.0, 99.0, 100.5 + index, 1000.0]
        for index in range(count)
    ]


@pytest.fixture
def builder(logger_double) -> MarketOverviewBuilder:
    return MarketOverviewBuilder(logger=logger_double, processor=_StubProcessor())


@pytest.fixture
def manager(logger_double) -> MarketDataManager:
    cache = MarketDataCache(logger=logger_double, file_handler=MagicMock())
    return MarketDataManager(
        logger=logger_double,
        file_handler=MagicMock(),
        unified_parser=SimpleNamespace(
            format_utils=SimpleNamespace(parse_timestamp=MagicMock())
        ),
        fetcher=MagicMock(),
        processor=MagicMock(),
        cache=cache,
        overview_builder=MagicMock(),
    )


@pytest.fixture
def collector(logger_double) -> MarketDataCollector:
    return MarketDataCollector(logger=logger_double, rag_engine=MagicMock())


def test_build_overview_structure_emits_contract_and_keeps_price_data_first(
    builder,
):
    price_data = {
        "BTC/USDT": {"price": 50000, "change_24h": 2.0, "volume": 1e6},
        "BAD": {},
    }
    empty = builder.build_overview_structure(None, None)
    delegated = builder.build_overview({"data": {"market_cap": 2e12}}, price_data)

    assert type(empty) is dict
    assert set(empty) == OVERVIEW_KEYS
    assert empty["summary"] == "CRYPTO MARKET OVERVIEW - 0 coins tracked"
    assert empty["coin_data"] == {}
    assert empty["data_sources"] == ["price_data"]
    assert type(empty["published_on"]) is float
    assert datetime.fromisoformat(empty["timestamp"]).tzinfo is timezone.utc

    assert delegated["market_cap"] == 2e12
    assert delegated["coin_data"] == {
        "BTC/USDT": {"price": 50000, "change_24h": 2.0, "volume": 1000000.0}
    }
    assert delegated["summary"] == "CRYPTO MARKET OVERVIEW - 1 coins tracked"


@pytest.mark.parametrize(
    ("coingecko_data", "expected_extra", "expected_warning"),
    [
        (
            {"data": {"market_cap": 2e12, "dominance": {"btc": 42.5}}},
            {"market_cap": 2e12, "dominance": {"btc": 42.5}},
            [],
        ),
        (
            {"market_cap": 2e12, "volume": 100e9, "dominance": {}, "stats": {}},
            {"market_cap": 2e12, "volume": 100e9, "dominance": {}, "stats": {}},
            [],
        ),
        (
            {"unknown_key": "unexpected_value"},
            {},
            [call("Unexpected CoinGecko data format: %s", ["unknown_key"])],
        ),
    ],
    ids=["wrapped-data", "flat-global", "unknown-format"],
)
def test_coingecko_global_payloads_are_flattened_or_reported(
    builder, logger_double, coingecko_data, expected_extra, expected_warning
):
    result = builder.build_overview_structure(None, coingecko_data)

    assert set(result) == OVERVIEW_KEYS | set(expected_extra)
    for key, value in expected_extra.items():
        assert result[key] == value
    assert logger_double.warning.call_args_list == expected_warning


@pytest.mark.parametrize(
    ("price_data", "top_coins", "expected"),
    [
        (
            BTC_PRICE_DATA,
            ["BTC"],
            [rich_coin("BTC", 1, 50000, 2.0, 1000000.0)],
        ),
        (
            BTC_PRICE_DATA,
            ["btc"],
            [rich_coin("btc", 1, 50000, 2.0, 1000000.0)],
        ),
        (
            {},
            [{"symbol": "ETH", "current_price": 3000}, "SOL"],
            [{"symbol": "ETH", "current_price": 3000}, rich_coin("SOL", 2)],
        ),
    ],
    ids=["quoted-pair", "lowercase-symbol", "dict-entry-passthrough"],
)
def test_top_coins_are_ranked_from_bare_symbols_and_rich_entries_pass_through(
    builder, price_data, top_coins, expected
):
    result = builder.build_overview_structure(price_data, None, top_coins=top_coins)

    assert result["top_coins"] == expected


@pytest.mark.parametrize(
    ("price_data", "expected_coin"),
    [
        (
            BTC_PRICE_DATA,
            {
                "symbol": "BTC",
                "current_price": 50000,
                "price_change_percentage_24h": 2.0,
                "total_volume": 1000000.0,
            },
        ),
        (
            {"BTC/USDT": {"price": 0, "change_24h": 0, "volume": 0}},
            {
                "symbol": "BTC",
                "current_price": 40000,
                "price_change_percentage_24h": 0.0,
                "total_volume": 0,
            },
        ),
    ],
    ids=["positive-fresh-price", "zero-fresh-price"],
)
def test_coingecko_top_coins_are_refreshed_in_place_only_by_a_positive_price(
    builder, price_data, expected_coin
):
    coin = {
        "symbol": "BTC",
        "current_price": 40000,
        "price_change_percentage_24h": 0.0,
        "total_volume": 0,
    }

    result = builder.build_overview_structure(
        price_data, {"data": {"top_coins": [coin]}}
    )

    assert result["top_coins"] == [expected_coin]
    assert result["top_coins"][0] is coin


def test_finalize_overview_stamps_published_on_sources_and_coin_count(builder):
    before = datetime.now(timezone.utc).timestamp()
    result = builder._finalize_overview(
        {"summary": "CRYPTO MARKET OVERVIEW", "coin_data": {"BTC": {}, "ETH": {}}}
    )
    after = datetime.now(timezone.utc).timestamp()

    assert before <= result["published_on"] <= after
    assert result["summary"] == "CRYPTO MARKET OVERVIEW - 2 coins tracked"
    assert result["data_sources"] == ["price_data"]
    assert builder._finalize_overview({})["data_sources"] == []
    assert builder._finalize_overview({"global_data": {}})["data_sources"] == [
        "coingecko_global"
    ]
    unshaped = builder._finalize_overview(
        {"summary": "TEST", "coin_data": "not-a-dict"}
    )

    assert type(unshaped) is dict
    assert unshaped["summary"] == "TEST - 10 coins tracked"
    repeated = builder._finalize_overview(
        builder._finalize_overview({"summary": "S", "coin_data": {"BTC": {}}})
    )
    assert repeated["summary"] == "S - 1 coins tracked - 1 coins tracked"


def test_finalize_overview_prefers_a_parsable_data_timestamp(builder):
    before = datetime.now(timezone.utc).timestamp()
    stamped = builder._finalize_overview(
        {"data_timestamp": "2026-01-01T00:00:00+00:00"}
    )
    corrupt = builder._finalize_overview({"data_timestamp": "not-a-date"})
    after = datetime.now(timezone.utc).timestamp()

    assert stamped["published_on"] == 1767225600.0
    assert before <= corrupt["published_on"] <= after


def test_build_overview_returns_a_partial_structure_when_processing_raises(
    logger_double,
):
    error = RuntimeError("boom")
    broken = MarketOverviewBuilder(
        logger=logger_double,
        processor=MagicMock(process_coin_data=MagicMock(side_effect=error)),
    )

    result = broken.build_overview(
        None, {"price": 1, "volume": 2, "dominance": {}, "stats": {}}
    )

    assert set(result) == {"timestamp", "summary", "coin_data"}
    assert result["summary"] == "CRYPTO MARKET OVERVIEW"
    assert result["coin_data"] == {}
    assert logger_double.error.call_args_list == [call(BUILD_ERROR, error)]
    assert logger_double.exception.call_args_list == [call("Traceback:")]


@pytest.mark.parametrize(
    ("age_hours", "max_age_hours", "expected_updated", "expected_debug"),
    [
        (
            None,
            24,
            True,
            [call(FETCHING_OVERVIEW), call(OVERVIEW_UPDATED)],
        ),
        (0.5, 24, False, []),
        (
            30,
            24,
            True,
            [call(STALE_OVERVIEW, 24), call(FETCHING_OVERVIEW), call(OVERVIEW_UPDATED)],
        ),
    ],
    ids=["no-cache", "fresh", "stale"],
)
async def test_update_market_overview_if_needed_refreshes_only_a_stale_cache(
    manager, logger_double, age_hours, max_age_hours, expected_updated, expected_debug
):
    now = datetime.now(timezone.utc)
    fetched = {
        "timestamp": "now",
        "published_on": (now + timedelta(hours=1)).timestamp(),
    }
    current = None
    if age_hours is not None:
        published_on = (now - timedelta(hours=age_hours)).timestamp()
        current = {"published_on": published_on}
        manager.current_market_overview = current
        manager.unified_parser.format_utils.parse_timestamp.return_value = published_on
    manager.fetch_market_overview = AsyncMock(return_value=fetched)

    updated = await manager.update_market_overview_if_needed(
        max_age_hours=max_age_hours
    )

    assert updated is expected_updated
    assert manager.get_current_overview() == (fetched if expected_updated else current)
    assert manager.fetch_market_overview.await_count == int(expected_updated)
    assert logger_double.debug.call_args_list == expected_debug


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("no-cache", True),
        ("unparseable", True),
        ("recent", False),
        ("future", False),
    ],
    ids=["no-cache", "unparseable", "recent", "future-timestamp"],
)
def test_is_overview_stale_covers_missing_unparseable_and_future_cache(
    manager, state, expected
):
    parser = manager.unified_parser.format_utils.parse_timestamp
    now = datetime.now(timezone.utc)
    if state == "unparseable":
        manager.current_market_overview = {"published_on": "invalid"}
        parser.return_value = None
    elif state == "recent":
        recent = (now - timedelta(minutes=10)).timestamp()
        manager.current_market_overview = {"published_on": recent}
        parser.return_value = recent
    elif state == "future":
        future = (now + timedelta(hours=1)).timestamp()
        manager.current_market_overview = {"published_on": future}
        parser.return_value = future

    assert manager.is_overview_stale(max_age_hours=1) is expected

    if state == "no-cache":
        assert manager.current_market_overview is None
        assert parser.call_count == 0


@pytest.mark.parametrize(
    ("secondary_sources", "expected"),
    [
        (True, {"base": "ok", "macro": {"m": 2}, "fundamentals": {"d": 3}}),
        (False, {"base": "ok"}),
    ],
    ids=["all-sources", "primary-only"],
)
async def test_fetch_market_overview_merges_the_secondary_sources_it_received(
    manager, secondary_sources, expected
):
    manager.fetcher.fetch_global_market_data = AsyncMock(return_value={"cg": 1})
    manager.fetcher.fetch_macro_data = AsyncMock(
        return_value=(
            SimpleNamespace(model_dump=lambda: {"m": 2}) if secondary_sources else None
        )
    )
    manager.fetcher.fetch_defi_fundamentals = AsyncMock(
        return_value=(
            SimpleNamespace(model_dump=lambda: {"d": 3}) if secondary_sources else None
        )
    )
    manager.processor.extract_top_coins = MagicMock(return_value=["BTC"])
    manager.fetcher.fetch_price_data = AsyncMock(return_value={"BTC": 90000})
    manager.overview_builder.build_overview = MagicMock(return_value={"base": "ok"})

    result = await manager.fetch_market_overview()

    assert result == expected
    assert manager.processor.extract_top_coins.call_args == call({"cg": 1})
    assert manager.fetcher.fetch_price_data.await_args == call(["BTC"])
    assert manager.overview_builder.build_overview.call_args == call(
        {"cg": 1}, {"BTC": 90000}, ["BTC"]
    )


@pytest.mark.parametrize(
    ("failure", "expected_warning", "expects_error"),
    [
        ("empty-overview", [call(NO_OVERVIEW_DATA)], False),
        ("gather-raises", [call(NO_OVERVIEW_DATA)], True),
    ],
    ids=["empty-overview", "gather-raises"],
)
async def test_overview_failures_are_reported_as_no_update(
    manager, logger_double, failure, expected_warning, expects_error
):
    error = RuntimeError("boom")
    if failure == "gather-raises":
        manager.fetcher.fetch_global_market_data = AsyncMock(side_effect=error)
        manager.fetcher.fetch_macro_data = AsyncMock(return_value=None)
        manager.fetcher.fetch_defi_fundamentals = AsyncMock(return_value=None)
    else:
        manager.fetch_market_overview = AsyncMock(return_value={})

    updated = await manager.update_market_overview_if_needed()

    assert updated is False
    assert logger_double.warning.call_args_list == expected_warning
    expected_error = [call(FETCH_OVERVIEW_ERROR, error)] if expects_error else []
    assert logger_double.error.call_args_list == expected_error


@pytest.mark.parametrize(
    ("symbol", "market", "expected"),
    [
        ("ABC", None, ABC_DETAILS),
        (
            "BTC",
            {
                "baseName": "Bitcoin",
                "base": "BTC",
                "active": True,
                "info": {"description": "Peer-to-peer electronic cash."},
            },
            {
                "description": "Peer-to-peer electronic cash.",
                "full_name": "Bitcoin",
                "coin_name": "BTC",
                "symbol": "BTC",
                "is_trading": True,
            },
        ),
        (
            "COIN",
            {"baseName": "Unknown Coin", "info": "raw-string"},
            {
                "description": "",
                "full_name": "Unknown Coin",
                "coin_name": "COIN",
                "symbol": "COIN",
                "is_trading": True,
            },
        ),
        (
            "SOL",
            {"base": "SOL", "active": False},
            {
                "description": "",
                "full_name": "SOL",
                "coin_name": "SOL",
                "symbol": "SOL",
                "is_trading": False,
            },
        ),
    ],
    ids=["no-market", "rich-metadata", "unusable-info", "inactive-market"],
)
async def test_get_coin_details_maps_loaded_market_metadata_or_falls_back(
    logger_double, symbol, market, expected
):
    exchange = MagicMock()
    exchange.markets = {f"{symbol}/USDT": market} if market is not None else {}
    exchange_manager = MagicMock()
    exchange_manager.find_symbol_exchange = AsyncMock(
        return_value=(exchange, "binance") if market is not None else (None, None)
    )
    provider = CCXTMarketAPI(logger=logger_double, exchange_manager=exchange_manager)

    assert await provider.get_coin_details(symbol) == expected


async def test_get_coin_details_walks_the_quote_ladder_and_skips_venues_without_markets(
    logger_double,
):
    ladder = MagicMock()

    async def find_pair(pair):
        if pair == "XYZ/USDC":
            market = {"base": "XYZ", "info": {"description": "gate way"}}
            return SimpleNamespace(markets={"XYZ/USDC": market}), "gateio"
        return None, None

    ladder.find_symbol_exchange = AsyncMock(side_effect=find_pair)

    provider = CCXTMarketAPI(logger=logger_double, exchange_manager=ladder)

    details = await provider.get_coin_details("XYZ")

    assert details == {
        "description": "gate way",
        "full_name": "XYZ",
        "coin_name": "XYZ",
        "symbol": "XYZ",
        "is_trading": True,
    }
    assert [entry.args[0] for entry in ladder.find_symbol_exchange.await_args_list] == [
        "XYZ/USDT",
        "XYZ/USD",
        "XYZ/USDC",
    ]

    blank = MagicMock()
    blank.find_symbol_exchange = AsyncMock(
        return_value=(SimpleNamespace(markets=None), "binance")
    )

    blank_provider = CCXTMarketAPI(logger=logger_double, exchange_manager=blank)

    assert await blank_provider.get_coin_details("ABC") == ABC_DETAILS
    assert [entry.args[0] for entry in blank.find_symbol_exchange.await_args_list] == [
        "ABC/USDT",
        "ABC/USD",
        "ABC/USDC",
        "ABC/BTC",
    ]


@pytest.mark.parametrize(
    ("mock_ccxt", "ccxt_result", "expected"),
    [
        (
            True,
            {"RAW": {"BTC": {"USDT": {"last": 90000}}}},
            {"RAW": {"BTC": {"USDT": {"last": 90000}}}},
        ),
        (True, None, None),
        (False, None, None),
    ],
    ids=["ccxt-payload", "ccxt-none", "no-exchange-configured"],
)
async def test_fetch_price_data_never_falls_back_to_the_legacy_market_api(
    mock_ccxt, ccxt_result, expected
):
    market_api = MagicMock()
    market_api.get_multi_price_data = AsyncMock(
        return_value={"RAW": {"fallback": True}}
    )
    fetcher = MarketDataFetcher(
        logger=MagicMock(),
        market_api=market_api,
        exchange_manager=(
            MagicMock(exchanges={"binance": MagicMock()}) if mock_ccxt else None
        ),
    )
    if mock_ccxt:
        fetcher._try_ccxt_price_data = AsyncMock(return_value=ccxt_result)

    result = await fetcher.fetch_price_data(["BTC"])

    assert result == expected
    assert market_api.get_multi_price_data.await_count == 0
    if mock_ccxt:
        assert fetcher._try_ccxt_price_data.await_args == call(["BTC"])


@pytest.mark.parametrize(
    ("capabilities", "expected_id"),
    [
        ({"kucoin": True, "binance": False}, "binance"),
        ({"kucoin": True, "gateio": False}, "kucoin"),
    ],
    ids=["binance-preferred", "fetch-tickers-fallback"],
)
def test_select_exchange_prefers_binance_then_a_ticker_capable_venue(
    capabilities, expected_id
):
    exchanges = {
        name: MagicMock(has={"fetchTickers": capability})
        for name, capability in capabilities.items()
    }
    fetcher = MarketDataFetcher(
        logger=MagicMock(), exchange_manager=MagicMock(exchanges=exchanges)
    )

    selected = fetcher._select_exchange()

    assert selected is exchanges[expected_id]
    expected_debug = (
        [call("Using Binance exchange for market data")]
        if expected_id == "binance"
        else [call("Using %s exchange for market data", "kucoin")]
    )
    assert fetcher.logger.debug.call_args_list == expected_debug


@pytest.mark.parametrize(
    ("timeframe", "candles", "expected_warning"),
    [
        ("5m", 999, call(INSUFFICIENT_HISTORY, 30, "5m", 999, 3.46875, 8640)),
        ("1h", 720, None),
        ("1w", 4, None),
    ],
    ids=["5m-short", "1h-exact-target", "1w-floor-target"],
)
def test_warn_if_insufficient_history_compares_against_timeframe_coverage(
    collector, logger_double, timeframe, candles, expected_warning
):
    collector.timeframe = timeframe

    collector._warn_if_insufficient_history(candles)

    assert logger_double.warning.call_args_list == (
        [] if expected_warning is None else [expected_warning]
    )


def test_warn_if_insufficient_history_reports_an_unknown_timeframe(
    collector, logger_double
):
    collector.timeframe = "3m"

    collector._warn_if_insufficient_history(100)

    assert len(logger_double.warning.call_args_list) == 1
    args = logger_double.warning.call_args_list[0].args
    assert args[0] == COVERAGE_UNKNOWN
    assert args[1] == "3m"
    assert type(args[2]) is ValueError
    assert str(args[2]) == "Unrecognized timeframe: 3m"


@pytest.mark.parametrize(
    ("timeframe", "limit", "row_count", "expected_closed", "expected_warnings"),
    [
        (
            "30m",
            999,
            401,
            400,
            [call(COVERAGE_WARNING, 400, 998, "30m", 8.333333333333334)],
        ),
        ("1d", 1001, 1002, 1001, [call(EXCHANGE_LIMIT_WARNING, 1001)]),
        ("30m", 999, 1, None, [call(NOT_ENOUGH_CANDLES, 1)]),
    ],
    ids=["timeframe-aware-expected", "over-limit", "single-row"],
)
async def test_fetch_candlestick_data_returns_closed_candles_and_warns_on_coverage(
    timeframe, limit, row_count, expected_closed, expected_warnings
):
    exchange = MagicMock()
    exchange.id = "binance"
    exchange.timeframes = {timeframe: timeframe}
    exchange.fetch_ohlcv = AsyncMock(return_value=ohlcv_rows(row_count))
    logger = MagicMock()

    result = await DataFetcher(exchange=exchange, logger=logger).fetch_candlestick_data(
        "BTC/USDT", timeframe, limit
    )

    assert exchange.fetch_ohlcv.await_args == call(
        "BTC/USDT", timeframe, since=None, limit=limit + 1
    )
    assert logger.warning.call_args_list == expected_warnings
    if expected_closed is None:
        assert result is None
        return
    assert type(result) is tuple
    closed, current_price = result
    assert type(closed) is np.ndarray
    assert closed.dtype == np.float64
    assert closed.shape == (expected_closed, 6)
    assert closed[0][4] == 100.5
    assert closed[-1][4] == 100.5 + expected_closed - 1
    assert current_price == 100.5 + row_count - 1


async def test_fetch_candlestick_data_turns_missing_cells_into_nan():
    gap = [
        [0, 100.0, 101.0, 99.0, None, 1000.0],
        [300000, 100.0, 101.0, 99.0, 101.5, 1000.0],
        [600000, 100.0, 101.0, 99.0, 102.5, 1000.0],
    ]
    tail_gap = [list(row) for row in gap]
    tail_gap[2][4] = None

    def fetcher_for(rows):
        exchange = MagicMock()
        exchange.id = "binance"
        exchange.timeframes = {"30m": "30m"}
        exchange.fetch_ohlcv = AsyncMock(return_value=rows)
        return DataFetcher(exchange=exchange, logger=MagicMock())

    params = ("BTC/USDT", "30m", 999)
    first = fetcher_for(gap)
    second = fetcher_for(tail_gap)

    closed, current_price = await first.fetch_candlestick_data(*params)
    tail_closed, tail_price = await second.fetch_candlestick_data(*params)

    assert closed.shape == (2, 6)
    assert math.isnan(closed[0][4])
    assert closed[-1][4] == 101.5
    assert current_price == 102.5
    assert tail_closed.shape == (2, 6)
    assert math.isnan(tail_price)


async def test_fetch_candlestick_data_handles_exchange_timeframe_advertising():
    listed = MagicMock()
    listed.id = "binance"
    listed.timeframes = {"30m": "30m"}
    listed.fetch_ohlcv = AsyncMock(return_value=ohlcv_rows(401))
    listed_logger = MagicMock()

    rejecter = DataFetcher(exchange=listed, logger=listed_logger)

    rejected = await rejecter.fetch_candlestick_data("BTC/USDT", "5m", 400)

    assert rejected is None
    assert listed.fetch_ohlcv.await_count == 0
    assert listed_logger.error.call_args_list == [
        call(TIMEFRAME_UNSUPPORTED, "5m", "binance", "30m")
    ]

    unlisted = SimpleNamespace(id="binance")
    unlisted.fetch_ohlcv = AsyncMock(return_value=ohlcv_rows(401))
    unlisted_logger = MagicMock()

    accepter = DataFetcher(exchange=unlisted, logger=unlisted_logger)

    accepted = await accepter.fetch_candlestick_data("BTC/USDT", "3m", 400)

    assert accepted[0].shape == (400, 6)
    assert unlisted_logger.warning.call_args_list == [
        call(TIMEFRAME_UNLISTED, "3m", "binance")
    ]


async def test_fetch_ohlcv_stores_candles_and_fans_out_secondary_fetches(collector):
    candles = np.array(ohlcv_rows(100))
    collector.symbol = "BTC/USDC"
    collector.exchange = MagicMock()
    collector.data_fetcher = MagicMock()
    collector.data_fetcher.fetch_candlestick_data = AsyncMock(
        return_value=(candles, 100.5)
    )
    collector.fetch_long_term_historical_data = AsyncMock(return_value=True)
    collector.fetch_weekly_macro_data = AsyncMock(return_value=True)
    collector.fetch_and_process_sentiment_data = AsyncMock(return_value=True)
    context = SimpleNamespace()

    assert await collector.fetch_ohlcv(context) is True

    assert context.ohlcv_candles is candles
    assert context.current_price == 100.5
    assert context.timestamps[0] == datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert context.timestamps[-1] == datetime(1970, 1, 1, 8, 15, tzinfo=timezone.utc)
    assert collector.data_fetcher.fetch_candlestick_data.await_args == call(
        pair="BTC/USDC", timeframe="1h", limit=None
    )
    assert collector.fetch_long_term_historical_data.await_args == call(context)
    assert collector.fetch_weekly_macro_data.await_args == call(
        context, target_weeks=300
    )
    assert collector.fetch_and_process_sentiment_data.await_args == call(context)


async def test_fetch_ohlcv_degrades_when_secondary_fetches_or_candles_are_unusable(
    collector, logger_double
):
    error = RuntimeError("lt boom")
    collector.symbol = "BTC/USDT"
    collector.exchange = MagicMock()
    collector.data_fetcher = MagicMock()
    collector.data_fetcher.fetch_candlestick_data = AsyncMock(
        return_value=(np.array(ohlcv_rows(100)), 100.5)
    )
    collector.fetch_long_term_historical_data = AsyncMock(side_effect=error)
    collector.fetch_weekly_macro_data = AsyncMock(return_value=True)
    collector.fetch_and_process_sentiment_data = AsyncMock(return_value=True)

    assert await collector.fetch_ohlcv(SimpleNamespace()) is True
    assert logger_double.error.call_args_list == [call(SECONDARY_FAILED, 0, error)]

    broken_logger = MagicMock()
    broken = MarketDataCollector(logger=broken_logger, rag_engine=MagicMock())
    broken.symbol = "BTC/USDT"
    broken.exchange = MagicMock()
    broken.data_fetcher = MagicMock()
    broken.data_fetcher.fetch_candlestick_data = AsyncMock(
        return_value=(ohlcv_rows(100), 100.5)
    )
    broken.fetch_long_term_historical_data = AsyncMock(return_value=True)
    broken.fetch_weekly_macro_data = AsyncMock(return_value=True)
    broken.fetch_and_process_sentiment_data = AsyncMock(return_value=True)
    broken_context = SimpleNamespace()

    assert await broken.fetch_ohlcv(broken_context) is True

    timestamp_warning = broken_logger.warning.call_args_list[0]
    assert timestamp_warning.args[0] == TIMESTAMPS_UNUSABLE
    assert type(timestamp_warning.args[1]) is TypeError
    assert broken_context.timestamps is None


def test_initialize_derives_the_default_candle_limit_from_the_timeframe(logger_double):
    derived = MarketDataCollector(logger=logger_double, rag_engine=MagicMock())
    explicit = MarketDataCollector(logger=logger_double, rag_engine=MagicMock())

    derived.initialize(
        data_fetcher=MagicMock(),
        symbol="BTC/USDT",
        exchange=MagicMock(),
        timeframe="5m",
    )
    explicit.initialize(
        data_fetcher=MagicMock(),
        symbol="BTC/USDT",
        exchange=MagicMock(),
        timeframe="1h",
        limit=999,
    )

    assert derived.limit == 8640
    assert derived.timeframe == "5m"
    assert explicit.limit == 999
    assert logger_double.debug.call_args_list == [call(DERIVED_LIMIT, 8640, "5m", 30)]
