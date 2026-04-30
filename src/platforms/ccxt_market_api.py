from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

import ccxt.async_support as ccxt

from src.logger.logger import Logger

if TYPE_CHECKING:
    from src.factories.data_fetcher_factory import DataFetcherFactory
    from src.platforms.exchange_manager import ExchangeManager


class CCXTMarketAPI:
    """CCXT-backed market provider for prices and best-effort coin metadata."""

    def __init__(
        self,
        logger: Logger,
        exchange_manager: "ExchangeManager",
        data_fetcher_factory: Optional["DataFetcherFactory"] = None,
    ) -> None:
        self.logger = logger
        self.exchange_manager = exchange_manager
        self.data_fetcher_factory = data_fetcher_factory

    async def get_multi_price_data(
        self,
        coins: Optional[list[str]] = None,
        _vs_currencies: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Fetch multi-coin prices and normalize to RAW/DISPLAY shape."""
        if not coins:
            return {}

        exchange = self._select_exchange()
        if not exchange:
            self.logger.warning("No exchange available for CCXT price data")
            return {}
        if self.data_fetcher_factory is None:
            self.logger.warning("DataFetcherFactory is not configured for CCXT price data")
            return {}

        quote_currencies = [quote.upper() for quote in (_vs_currencies or ["USDT"]) if quote]
        symbols = [f"{coin}/{quote}" for coin in coins for quote in quote_currencies]
        data_fetcher = self.data_fetcher_factory.create(exchange)
        return await data_fetcher.fetch_multiple_tickers(symbols)

    async def get_coin_details(self, symbol: str) -> dict[str, Any]:
        """Return best-effort coin details from loaded CCXT market metadata."""
        market = await self._find_market_for_symbol(symbol)
        if not market:
            return {
                "description": "",
                "full_name": symbol,
                "coin_name": symbol,
                "symbol": symbol,
                "is_trading": True,
            }

        raw_info = market.get("info")
        info: dict[str, Any] = raw_info if isinstance(raw_info, dict) else {}
        full_name = (
            market.get("baseName")
            or info.get("fullName")
            or info.get("fullname")
            or info.get("name")
            or market.get("base")
            or symbol
        )

        description = info.get("description") or info.get("desc") or ""

        return {
            "description": description,
            "full_name": str(full_name),
            "coin_name": str(market.get("base") or symbol),
            "symbol": str(market.get("base") or symbol),
            "is_trading": bool(market.get("active", True)),
        }

    def _select_exchange(self) -> Optional[ccxt.Exchange]:
        if not (self.exchange_manager and self.exchange_manager.exchanges):
            return None

        if "binance" in self.exchange_manager.exchanges:
            return self.exchange_manager.exchanges["binance"]

        for exch in self.exchange_manager.exchanges.values():
            if isinstance(exch.has, dict) and exch.has.get("fetchTickers", False):
                return exch

        return None

    async def _find_market_for_symbol(self, symbol: str) -> Optional[dict[str, Any]]:
        for quote in ("USDT", "USD", "USDC", "BTC"):
            pair = f"{symbol}/{quote}"
            exchange, _ = await self.exchange_manager.find_symbol_exchange(pair)
            if not exchange:
                continue

            markets = exchange.markets if isinstance(exchange.markets, dict) else {}
            market = markets.get(pair)
            if market:
                return market

        return None