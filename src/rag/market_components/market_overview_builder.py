"""
Market Overview Builder
Handles building and structuring market overview data.
"""
from datetime import datetime
from typing import Dict, Any, Optional

from src.logger.logger import Logger


class MarketOverviewBuilder:
    """Handles building structured market overview data."""

    def __init__(self, logger: Logger, processor):
        self.logger = logger
        self.processor = processor

    def build_overview_structure(self, price_data: Optional[Dict], coingecko_data: Optional[Dict], top_coins: Optional[list] = None) -> Dict[str, Any]:
        """Build the complete market overview structure."""
        overview = {
            "timestamp": datetime.now().isoformat(),
            "summary": "CRYPTO MARKET OVERVIEW"
        }

        try:
            # 1. Add CoinGecko global data if available (flattened for formatter compatibility)
            if coingecko_data:
                # Handle both direct global data and wrapped data
                if 'data' in coingecko_data:
                    overview.update(coingecko_data['data'])
                elif any(key in coingecko_data for key in ['market_cap', 'volume', 'dominance', 'stats']):
                    # Direct global data format
                    overview.update(coingecko_data)
                else:
                    self.logger.warning(f"Unexpected CoinGecko data format: {list(coingecko_data.keys())}")

            # 2. Add price data if available (Process BEFORE top_coins to use it for enrichment)
            overview['coin_data'] = {}
            if price_data:
                for symbol, values in price_data.items():
                    processed_coin = self.processor.process_coin_data(values)
                    if processed_coin:
                        overview['coin_data'][symbol] = processed_coin

            # 3. Add top coins list if available
            # Strategy: Prefer existing rich data from CoinGecko, update with fresh prices if available.
            # If no CoinGecko data, build from scratch using the symbols list.

            existing_top_coins = overview.get('top_coins', [])

            if existing_top_coins and isinstance(existing_top_coins[0], dict):
                # We have rich CoinGecko data. Update it with fresh stats if available.
                for coin in existing_top_coins:
                    symbol = coin.get('symbol', '').upper()

                    # Find matching fresh data
                    fresh_data = None
                    for key, data in overview.get('coin_data', {}).items():
                        if key.upper().startswith(symbol + "/") or key.upper() == symbol:
                            fresh_data = data
                            break

                    if fresh_data:
                        # Only update if we have valid non-zero data, or if we really trust the fresh source
                        # Here we prioritize the fresh source if it has a price
                        fresh_price = fresh_data.get('price', 0)
                        if fresh_price > 0:
                            coin['current_price'] = fresh_price
                            coin['price_change_percentage_24h'] = fresh_data.get('change_24h', coin.get('price_change_percentage_24h', 0))
                            coin['total_volume'] = fresh_data.get('volume', coin.get('total_volume', 0))

                # Update the overview with the potentially updated list
                overview['top_coins'] = existing_top_coins

            elif top_coins:
                 # Fallback: We only have a list of symbols (top_coins arg) and no rich CoinGecko data
                 # We must build the objects from scratch using coin_data
                rich_top_coins = []
                for i, symbol in enumerate(top_coins):
                    # Try to find corresponding data in coin_data
                    coin_info = None
                    if isinstance(symbol, str):
                        # Find matching data (e.g. "BTC" matches "BTC/USDT")
                        for key, data in overview.get('coin_data', {}).items():
                            if key.upper().startswith(symbol.upper() + "/") or key.upper() == symbol.upper():
                                coin_info = data
                                break

                        # Create rich coin object
                        rich_coin = {
                            "symbol": symbol,
                            "name": symbol,
                            "market_cap_rank": i + 1,
                            "current_price": coin_info.get('price', 0) if coin_info else 0,
                            "price_change_percentage_24h": coin_info.get('change_24h', 0) if coin_info else 0,
                            "total_volume": coin_info.get('volume', 0) if coin_info else 0
                        }
                        rich_top_coins.append(rich_coin)
                    elif isinstance(symbol, dict):
                         rich_top_coins.append(symbol)

                overview['top_coins'] = rich_top_coins

            return self._finalize_overview(overview)

        except Exception as e:
            self.logger.error(f"Error building overview structure: {e}")
            self.logger.exception("Traceback:")
            return overview

    def build_overview(self, coingecko_data: Optional[Dict], price_data: Optional[Dict], top_coins: Optional[list] = None) -> Dict[str, Any]:
        """Build market overview from fetched data - main entry point."""
        try:
            return self.build_overview_structure(price_data, coingecko_data, top_coins)
        except Exception as e:
            self.logger.error(f"Error building market overview: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "summary": "CRYPTO MARKET OVERVIEW - Error occurred",
                "published_on": datetime.now().timestamp()
            }

    def _finalize_overview(self, overview: Dict) -> Dict[str, Any]:
        """Finalize and validate the overview structure."""
        try:
            # Add metadata
            overview['published_on'] = datetime.now().timestamp()
            overview['data_sources'] = []

            # Track data sources
            if 'global_data' in overview:
                overview['data_sources'].append('coingecko_global')
            if 'coin_data' in overview:
                overview['data_sources'].append('price_data')

            # Add summary statistics
            if 'coin_data' in overview:
                coin_count = len(overview['coin_data'])
                overview['summary'] += f" - {coin_count} coins tracked"

            return overview

        except Exception as e:
            self.logger.error(f"Error finalizing overview: {e}")
            return overview
