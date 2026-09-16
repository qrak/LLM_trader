"""
Market Overview Builder
Handles building and structuring market overview data.
"""
from datetime import datetime, timezone
from typing import Any

from src.logger.logger import Logger


class MarketOverviewBuilder:
    """Handles building structured market overview data."""

    def __init__(self, logger: Logger, processor):
        self.logger = logger
        self.processor = processor

    def build_overview_structure(self, price_data: dict | None, coingecko_data: dict | None, top_coins: list | None = None) -> dict[str, Any]:
        """Build the complete market overview structure."""
        overview: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": "CRYPTO MARKET OVERVIEW"
        }

        try:
            if coingecko_data:
                if "data" in coingecko_data:
                    overview.update(coingecko_data["data"])
                elif any(key in coingecko_data for key in ["market_cap", "volume", "dominance", "stats"]):
                    overview.update(coingecko_data)
                else:
                    self.logger.warning("Unexpected CoinGecko data format: %s", list(coingecko_data.keys()))

            overview["coin_data"] = {}
            if price_data:
                for symbol, values in price_data.items():
                    processed_coin = self.processor.process_coin_data(values)
                    if processed_coin:
                        overview["coin_data"][symbol] = processed_coin


            existing_top_coins = overview.get("top_coins", [])

            if existing_top_coins:
                for coin in existing_top_coins:
                    symbol = coin.get("symbol", "").upper()

                    fresh_data = None
                    for key, data in overview.get("coin_data", {}).items():
                        if key.upper().startswith(symbol + "/") or key.upper() == symbol:
                            fresh_data = data
                            break

                    if fresh_data:
                        fresh_price = fresh_data.get("price", 0)
                        if fresh_price > 0:
                            coin["current_price"] = fresh_price
                            coin["price_change_percentage_24h"] = fresh_data.get("change_24h", coin.get("price_change_percentage_24h", 0))
                            coin["total_volume"] = fresh_data.get("volume", coin.get("total_volume", 0))

                overview["top_coins"] = existing_top_coins

            elif top_coins:
                rich_top_coins = []
                for i, item in enumerate(top_coins):
                    if isinstance(item, dict):
                        rich_top_coins.append(item)
                        continue

                    symbol = item
                    coin_info = None
                    for key, data in overview.get("coin_data", {}).items():
                        if key.upper().startswith(symbol.upper() + "/") or key.upper() == symbol.upper():
                            coin_info = data
                            break

                    rich_coin = {
                        "symbol": symbol,
                        "name": symbol,
                        "market_cap_rank": i + 1,
                        "current_price": coin_info.get("price", 0) if coin_info else 0,
                        "price_change_percentage_24h": coin_info.get("change_24h", 0) if coin_info else 0,
                        "total_volume": coin_info.get("volume", 0) if coin_info else 0
                    }
                    rich_top_coins.append(rich_coin)

                overview["top_coins"] = rich_top_coins

            return self._finalize_overview(overview)

        except Exception as e:
            self.logger.error("Error building overview structure: %s", e)
            self.logger.exception("Traceback:")
            return overview

    def build_overview(self, coingecko_data: dict | None, price_data: dict | None, top_coins: list | None = None) -> dict[str, Any]:
        """Build market overview from fetched data - main entry point.

        ``build_overview_structure`` already returns a partial overview instead of
        raising, so there is nothing for a second guard to catch here.
        """
        return self.build_overview_structure(price_data, coingecko_data, top_coins)

    def _finalize_overview(self, overview: dict) -> dict[str, Any]:
        """Finalize and validate the overview structure."""
        try:
            source_ts = overview.get("data_timestamp")
            if source_ts:
                try:
                    overview["published_on"] = datetime.fromisoformat(str(source_ts)).timestamp()
                except (ValueError, TypeError, OverflowError):
                    overview["published_on"] = datetime.now(timezone.utc).timestamp()
            else:
                overview["published_on"] = datetime.now(timezone.utc).timestamp()
            overview["data_sources"] = []

            if "global_data" in overview:
                overview["data_sources"].append("coingecko_global")
            if "coin_data" in overview:
                overview["data_sources"].append("price_data")

            if "coin_data" in overview:
                coin_count = len(overview["coin_data"])
                overview["summary"] += f" - {coin_count} coins tracked"

            return overview

        except Exception as e:  # noqa: BLE001
            self.logger.error("Error finalizing overview: %s", e)
            return overview

