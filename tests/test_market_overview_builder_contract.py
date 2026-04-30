"""
Contract tests for MarketOverviewBuilder.

Locks build_overview_structure and _finalize_overview behavior before any
structural changes.  Uses a real processor stub so we test the builder's own
logic, not a mock.
"""
from datetime import datetime
from unittest.mock import MagicMock
from typing import Any, Dict, Optional

import pytest

from src.rag.market_components.market_overview_builder import MarketOverviewBuilder


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

class _StubProcessor:
    """Minimal processor that echoes processed coin data deterministically."""

    def process_coin_data(self, values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not values:
            return None
        return {
            "price": values.get("price", 0),
            "change_24h": values.get("change_24h", 0),
            "volume": values.get("volume", 0),
        }


@pytest.fixture
def builder() -> MarketOverviewBuilder:
    return MarketOverviewBuilder(logger=MagicMock(), processor=_StubProcessor())


# ---------------------------------------------------------------------------
# build_overview_structure – basic contract
# ---------------------------------------------------------------------------

class TestBuildOverviewStructureBasics:
    def test_always_returns_dict(self, builder):
        result = builder.build_overview_structure(None, None)
        assert isinstance(result, dict)

    def test_timestamp_key_present(self, builder):
        result = builder.build_overview_structure(None, None)
        assert "timestamp" in result

    def test_summary_key_present(self, builder):
        result = builder.build_overview_structure(None, None)
        assert "summary" in result

    def test_published_on_key_present(self, builder):
        result = builder.build_overview_structure(None, None)
        assert "published_on" in result

    def test_coin_data_key_present(self, builder):
        result = builder.build_overview_structure(None, None)
        assert "coin_data" in result

    def test_coin_data_empty_when_no_price_data(self, builder):
        result = builder.build_overview_structure(None, None)
        assert result["coin_data"] == {}


class TestBuildOverviewStructurePriceData:
    def test_price_data_populates_coin_data(self, builder):
        price_data = {"BTC/USDT": {"price": 50000, "change_24h": 2.5, "volume": 1_000_000}}
        result = builder.build_overview_structure(price_data, None)
        assert "BTC/USDT" in result["coin_data"]

    def test_coin_data_values_mapped_correctly(self, builder):
        price_data = {"ETH/USDT": {"price": 3000, "change_24h": -1.0, "volume": 500_000}}
        result = builder.build_overview_structure(price_data, None)
        coin = result["coin_data"]["ETH/USDT"]
        assert coin["price"] == 3000
        assert coin["change_24h"] == -1.0
        assert coin["volume"] == 500_000

    def test_processor_returning_none_skips_entry(self, builder):
        # Empty dict triggers None from _StubProcessor
        price_data = {"BAD": {}}
        result = builder.build_overview_structure(price_data, None)
        assert "BAD" not in result["coin_data"]


class TestBuildOverviewStructureCoinGeckoData:
    def test_wrapped_data_key_is_flattened(self, builder):
        cg_data = {"data": {"market_cap": 2e12, "dominance": {"btc": 42.5}}}
        result = builder.build_overview_structure(None, cg_data)
        assert result.get("market_cap") == 2e12

    def test_direct_global_data_is_applied(self, builder):
        cg_data = {"market_cap": 2e12, "volume": 100e9, "dominance": {}, "stats": {}}
        result = builder.build_overview_structure(None, cg_data)
        assert result.get("market_cap") == 2e12

    def test_unexpected_format_does_not_raise(self, builder):
        # Should log a warning, but not raise
        cg_data = {"unknown_key": "unexpected_value"}
        result = builder.build_overview_structure(None, cg_data)
        assert isinstance(result, dict)


class TestBuildOverviewStructureTopCoins:
    def test_top_coins_from_arg_when_no_coingecko(self, builder):
        price_data = {"BTC/USDT": {"price": 50000, "change_24h": 2.0, "volume": 1e6}}
        result = builder.build_overview_structure(price_data, None, top_coins=["BTC"])
        assert "top_coins" in result
        assert len(result["top_coins"]) == 1

    def test_top_coins_symbol_matches(self, builder):
        price_data = {"BTC/USDT": {"price": 50000, "change_24h": 2.0, "volume": 1e6}}
        result = builder.build_overview_structure(price_data, None, top_coins=["BTC"])
        assert result["top_coins"][0]["symbol"] == "BTC"

    def test_top_coins_price_sourced_from_price_data(self, builder):
        price_data = {"BTC/USDT": {"price": 50000, "change_24h": 2.0, "volume": 1e6}}
        result = builder.build_overview_structure(price_data, None, top_coins=["BTC"])
        assert result["top_coins"][0]["current_price"] == 50000

    def test_top_coins_rich_coingecko_updated_with_fresh_price(self, builder):
        cg_data = {
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
        price_data = {"BTC/USDT": {"price": 50000, "change_24h": 2.0, "volume": 1e6}}
        result = builder.build_overview_structure(price_data, cg_data)
        assert result["top_coins"][0]["current_price"] == 50000

    def test_dict_items_in_top_coins_passed_through(self, builder):
        price_data = {}
        rich_coin = {"symbol": "ETH", "current_price": 3000}
        result = builder.build_overview_structure(price_data, None, top_coins=[rich_coin])
        assert result["top_coins"][0]["symbol"] == "ETH"


# ---------------------------------------------------------------------------
# _finalize_overview contract
# ---------------------------------------------------------------------------

class TestFinalizeOverview:
    def test_adds_published_on(self, builder):
        before = datetime.now().timestamp()
        result = builder._finalize_overview({})
        after = datetime.now().timestamp()
        assert before <= result["published_on"] <= after

    def test_adds_data_sources_list(self, builder):
        result = builder._finalize_overview({})
        assert "data_sources" in result
        assert isinstance(result["data_sources"], list)

    def test_price_data_source_tracked(self, builder):
        result = builder._finalize_overview({"coin_data": {"BTC": {}}})
        assert "price_data" in result["data_sources"]

    def test_summary_coin_count_appended(self, builder):
        overview = {"summary": "CRYPTO MARKET OVERVIEW", "coin_data": {"BTC": {}, "ETH": {}}}
        result = builder._finalize_overview(overview)
        assert "2 coins tracked" in result["summary"]

    def test_returns_dict_on_exception(self, builder):
        # Pass something that would blow up iteration – e.g. non-dict coin_data
        # _finalize_overview has a try/except that should still return overview
        overview = {"summary": "TEST", "coin_data": "not-a-dict"}
        result = builder._finalize_overview(overview)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# build_overview (main entry point) delegates to build_overview_structure
# ---------------------------------------------------------------------------

class TestBuildOverview:
    def test_returns_dict(self, builder):
        result = builder.build_overview(None, None)
        assert isinstance(result, dict)

    def test_error_fallback_contains_timestamp(self, builder):
        # Force an error by making processor raise
        broken_builder = MarketOverviewBuilder(
            logger=MagicMock(),
            processor=MagicMock(process_coin_data=MagicMock(side_effect=RuntimeError("boom"))),
        )
        result = broken_builder.build_overview(None, {"price": 1, "volume": 2, "dominance": {}, "stats": {}})
        assert "timestamp" in result
