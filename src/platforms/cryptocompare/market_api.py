from typing import Dict, List, Any, TYPE_CHECKING

import aiohttp

from src.utils.timeframe_validator import TimeframeValidator
from src.logger.logger import Logger
from src.utils.decorators import retry_async

if TYPE_CHECKING:
    from src.config.protocol import ConfigProtocol


class CryptoCompareMarketAPI:
    """
    Handles CryptoCompare market data API operations including price data and OHLCV data
    """

    def __init__(self, logger: Logger, config: "ConfigProtocol") -> None:
        """Initialize CryptoCompareMarketAPI with logger and self.config.

        Args:
            logger: Logger instance
            config: ConfigProtocol instance for API key
        """
        self.logger = logger
        self.config = config
        api_key_param = f"&api_key={self.config.CRYPTOCOMPARE_API_KEY}" if self.config.CRYPTOCOMPARE_API_KEY else ""
        self.OHLCV_API_URL_TEMPLATE = f"https://min-api.cryptocompare.com/data/v2/histo{{timeframe}}?fsym={{base}}&tsym={{quote}}&limit={{limit}}{api_key_param}"

    @retry_async(max_retries=3, initial_delay=2, backoff_factor=2, max_delay=30)
    async def get_multi_price_data(
        self,
        coins: List[str] = None,
        vs_currencies: List[str] = None
    ) -> Dict[str, Any]:
        """
        Get price data for multiple coins

        Args:
            coins: List of coin symbols (default: BTC,ETH,XRP,LTC,BCH,BNB,ADA,DOT,LINK)
            vs_currencies: List of fiat currencies (default: USD)

        Returns:
            Dictionary with price data
        """
        default_coins = ["BTC", "ETH", "XRP", "LTC", "BCH", "BNB", "ADA", "DOT", "LINK"]
        default_currencies = ["USD"]

        # Use defaults if not provided
        fsyms = coins if coins else default_coins
        tsyms = vs_currencies if vs_currencies else default_currencies

        # Build URL: use canonical config URL when caller doesn't provide coins or currencies
        if coins is None and vs_currencies is None:
            url = self.config.RAG_PRICE_API_URL
        else:
            url = f"https://min-api.cryptocompare.com/data/pricemultifull?fsyms={','.join(fsyms)}&tsyms={','.join(tsyms)}"

        # Append API key if available
        if self.config.CRYPTOCOMPARE_API_KEY and "api_key=" not in url:
             connector = "&" if "?" in url else "?"
             url = f"{url}{connector}api_key={self.config.CRYPTOCOMPARE_API_KEY}"

        client_timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=client_timeout) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data and "RAW" in data:
                        return data
                    self.logger.warning("Price data response missing RAW field")
                    return {}
                self.logger.error("Price API request failed with status %s", resp.status)
                return {}

    @retry_async(max_retries=3, initial_delay=2, backoff_factor=2, max_delay=30)
    async def get_coin_details(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed coin information including description, taxonomy, and Weiss ratings

        Args:
            symbol: Cryptocurrency symbol (e.g., 'LINK', 'BTC')

        Returns:
            Dictionary with coin details including:
            - Description: Project description text
            - Algorithm: Consensus algorithm (e.g., "Proof of Work", "N/A")
            - ProofType: Proof mechanism type
            - Sponsored: Whether coin is sponsored on CryptoCompare
            - Taxonomy: Regulatory classifications (Access, FCA, FINMA, Industry, etc.)
            - Rating: Weiss ratings including overall, technology adoption, and market performance
        """
        url = f"https://min-api.cryptocompare.com/data/all/coinlist?fsym={symbol}"

        # Append API key if available
        if self.config.CRYPTOCOMPARE_API_KEY:
             url = f"{url}&api_key={self.config.CRYPTOCOMPARE_API_KEY}"

        client_timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=client_timeout) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data and data.get("Response") == "Success" and "Data" in data:
                        coin_data = data["Data"].get(symbol)
                        if coin_data:
                            return {
                                "description": coin_data.get("Description", ""),
                                "algorithm": coin_data.get("Algorithm", "N/A"),
                                "proof_type": coin_data.get("ProofType", "N/A"),
                                "sponsored": coin_data.get("Sponsored", False),
                                "taxonomy": coin_data.get("Taxonomy", {}),
                                "rating": coin_data.get("Rating", {}),
                                "full_name": coin_data.get("FullName", ""),
                                "coin_name": coin_data.get("CoinName", ""),
                                "symbol": coin_data.get("Symbol", symbol),
                                "is_trading": coin_data.get("IsTrading", True)
                            }
                        self.logger.warning("No data found for symbol %s", symbol)
                        return {}
                    self.logger.warning("Coin details API response unsuccessful: %s", data.get('Message', 'Unknown error'))
                    return {}
                self.logger.error("Coin details API request failed with status %s", resp.status)
                return {}

    def get_ohlcv_url_template(self) -> str:
        """Get the OHLCV API URL template"""
        return self.OHLCV_API_URL_TEMPLATE

    def build_ohlcv_url(self, base: str, quote: str, timeframe: str, limit: int) -> str:
        """
        Build CryptoCompare OHLCV API URL with proper endpoint format.

        CryptoCompare uses different endpoints for different timeframes:
        - histohour for hourly data (1h, 2h, 4h, etc.)
        - histoday for daily data (1d)

        For multi-hour timeframes (2h, 4h, etc.), we use the 'aggregate' parameter.

        Args:
            base: Base currency (e.g., "BTC")
            quote: Quote currency (e.g., "USDT")
            timeframe: Our timeframe format (e.g., "1h", "4h", "1d")
            limit: Number of candles to fetch

        Returns:
            Complete API URL

        Raises:
            ValueError: If timeframe is not supported by CryptoCompare API
        """
        try:
            endpoint_type, multiplier = TimeframeValidator.to_cryptocompare_format(timeframe)
        except ValueError as e:
            self.logger.error("Failed to convert timeframe %s for CryptoCompare API: %s", timeframe, e)
            raise


        # Build aggregate parameter if multiplier > 1
        aggregate_param = f"&aggregate={multiplier}" if multiplier > 1 else ""

        base_url = (
            f"https://min-api.cryptocompare.com/data/v2/histo{endpoint_type}"
            f"?fsym={base}&tsym={quote}&limit={limit}"
            f"{aggregate_param}"
        )

        # Append API key if available
        if self.config.CRYPTOCOMPARE_API_KEY:
            url = f"{base_url}&api_key={self.config.CRYPTOCOMPARE_API_KEY}"
        else:
            url = base_url

        # Note: We do NOT log the full URL here to avoid leaking the API key
        self.logger.debug("Built CryptoCompare OHLCV URL: endpoint=%s, multiplier=%s, timeframe=%s, limit=%s", endpoint_type, multiplier, timeframe, limit)

        return url
