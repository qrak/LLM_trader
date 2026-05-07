from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from src.logger.logger import Logger

if TYPE_CHECKING:
    from src.platforms.exchange_manager import ExchangeManager


class CCXTMarketAPI:
    """CCXT-backed market provider for prices and best-effort coin metadata."""

    def __init__(
        self,
        logger: Logger,
        exchange_manager: "ExchangeManager",
    ) -> None:
        self.logger = logger
        self.exchange_manager = exchange_manager

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