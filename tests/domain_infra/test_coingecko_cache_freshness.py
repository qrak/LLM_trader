"""CoinGecko cache freshness guards.

The HTTP cache serves an entry indefinitely when it was stored without an expiry,
so a fresh fetch timestamp can sit on year-old upstream data. These tests pin the
guards that make that visible: CoinGecko's own ``updated_at`` travels with the
processed payload, an old source timestamp is reported, a stored entry without an
expiry is reported once per endpoint, and the market overview block shows the
source timestamp when it is available.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.analyzer.formatters.market_overview_formatter import MarketOverviewFormatter
from src.platforms.coingecko import CoinGeckoAPI
from tests.conftest import null_logger

_SOURCE_EPOCH = 1757191107
_STALE_SOURCE_TS = "2025-09-06T20:38:27+00:00"
_TOTAL_MARKET_CAP = 3879827856621.05


def make_client(tmp_path, logger=None) -> CoinGeckoAPI:
    return CoinGeckoAPI(
        logger=logger or null_logger(),
        cache_backend=None,
        cache_dir=str(tmp_path),
    )


def global_payload(updated_at: object = _SOURCE_EPOCH) -> dict:
    data: dict = {
        "total_market_cap": {"usd": _TOTAL_MARKET_CAP},
        "market_cap_change_percentage_24h_usd": -1.29,
        "total_volume": {"usd": 1000.0},
        "market_cap_percentage": {"btc": 56.52, "eth": 13.27},
        "active_cryptocurrencies": 14078,
        "markets": 1200,
    }
    if updated_at is not None:
        data["updated_at"] = updated_at
    return {"data": data}


def test_processed_global_data_carries_the_source_updated_at(tmp_path):
    client = make_client(tmp_path)

    processed = client._process_global_data(global_payload())

    assert processed["source_updated_at"] == _STALE_SOURCE_TS
    assert processed["market_cap"]["total_usd"] == _TOTAL_MARKET_CAP
    assert processed["dominance"] == {"btc": 56.52, "eth": 13.27}


def test_processed_global_data_omits_the_source_timestamp_when_it_is_absent(tmp_path):
    logger = null_logger()
    client = make_client(tmp_path, logger)

    processed = client._process_global_data(global_payload(updated_at=None))

    assert "source_updated_at" not in processed
    logger.warning.assert_not_called()


def test_processed_global_data_reports_an_unusable_updated_at(tmp_path):
    logger = null_logger()
    client = make_client(tmp_path, logger)

    processed = client._process_global_data(global_payload(updated_at="not-an-epoch"))

    assert "source_updated_at" not in processed
    logger.warning.assert_called_once_with("Unusable CoinGecko updated_at value: %r", "not-an-epoch")


def test_stale_source_data_is_reported(tmp_path):
    logger = null_logger()
    client = make_client(tmp_path, logger)

    client._warn_if_source_data_is_stale({"source_updated_at": _STALE_SOURCE_TS})

    assert logger.warning.call_count == 1
    assert "h old according to the API" in logger.warning.call_args.args[0]


@pytest.mark.parametrize(
    "processed_global",
    [
        {"source_updated_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()},
        {"source_updated_at": "not-a-timestamp"},
        {},
    ],
    ids=["fresh", "unparsable", "missing"],
)
def test_source_data_within_the_threshold_stays_silent(tmp_path, processed_global):
    logger = null_logger()
    client = make_client(tmp_path, logger)

    client._warn_if_source_data_is_stale(processed_global)

    logger.warning.assert_not_called()


def test_entry_without_an_expiry_is_reported_once_per_endpoint(tmp_path):
    logger = null_logger()
    client = make_client(tmp_path, logger)
    response = SimpleNamespace(expires=None, created_at="2025-11-07T15:16:37+00:00", from_cache=True)

    client._warn_if_response_has_no_expiry(response, "coins/list")
    client._warn_if_response_has_no_expiry(response, "coins/list")

    assert logger.warning.call_count == 1
    assert logger.warning.call_args.args[1] == "coins/list"


def test_entry_with_an_expiry_stays_silent(tmp_path):
    logger = null_logger()
    client = make_client(tmp_path, logger)
    response = SimpleNamespace(expires=datetime.now(timezone.utc), from_cache=True)

    client._warn_if_response_has_no_expiry(response, "coins/list")

    logger.warning.assert_not_called()


def test_response_fresh_from_the_network_stays_silent(tmp_path):
    """A network response carries expires=None and created_at=None by construction,
    not because its cache entry was stored without a TTL."""
    logger = null_logger()
    client = make_client(tmp_path, logger)
    response = SimpleNamespace(expires=None, created_at=None, from_cache=False)

    client._warn_if_response_has_no_expiry(response, "coins/list")

    logger.warning.assert_not_called()


def test_market_overview_block_shows_the_source_timestamp(format_utils):
    formatter = MarketOverviewFormatter(logger=null_logger(), format_utils=format_utils)

    text = formatter.format_market_overview(
        {"source_updated_at": _STALE_SOURCE_TS, "market_cap": {"total_usd": _TOTAL_MARKET_CAP}}
    )

    assert f"- Market data source updated: {_STALE_SOURCE_TS}" in text
    assert "- Total Market Cap: $" in text


def test_market_overview_block_omits_the_source_timestamp_when_absent(format_utils):
    formatter = MarketOverviewFormatter(logger=null_logger(), format_utils=format_utils)

    text = formatter.format_market_overview({"market_cap": {"total_usd": _TOTAL_MARKET_CAP}})

    assert "Market data source updated" not in text
