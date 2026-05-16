from typing import Any
import time

import numpy as np
from numpy.typing import NDArray

from src.logger.logger import Logger
from src.utils.decorators import retry_async
from src.utils.timeframe_validator import TimeframeValidator
from src.utils.profiler import profile_performance


class DataFetcher:
    def __init__(self, exchange, logger: Logger):
        self.exchange = exchange
        self.logger: Logger = logger

    @retry_async()
    @profile_performance
    async def fetch_candlestick_data(self,
                                     pair: str,
                                     timeframe: str,
                                     limit: int,
                                     start_time: int | None = None
                                     ) -> tuple[NDArray, float] | None:
        # Validate timeframe is supported by exchange
        try:
            exchange_timeframes = self.exchange.timeframes
            if exchange_timeframes and timeframe not in exchange_timeframes:
                self.logger.error("Timeframe %s not supported by %s. Supported: %s", timeframe, self.exchange.id, ', '.join(exchange_timeframes.keys()))
                return None
        except AttributeError:
            if not TimeframeValidator.is_ccxt_compatible(timeframe):
                self.logger.warning("Timeframe %s may not be supported by %s. Attempting fetch anyway...", timeframe, self.exchange.id)

        if limit > 1000:
            self.logger.warning("Requested limit %s exceeds exchange standard limits, may be truncated", limit)

        ohlcv = await self.exchange.fetch_ohlcv(pair, timeframe, since=start_time, limit=limit + 1)

        if ohlcv is None or len(ohlcv) == 0:
            self.logger.warning("No data returned for %s on %s", pair, self.exchange.id)
            return None

        # Sanitize data: Replace None with np.nan for float64 conversion
        ohlcv_sanitized = [
            [x if x is not None else np.nan for x in candle]
            for candle in ohlcv
        ]

        ohlcv_array = np.array(ohlcv_sanitized, dtype=np.float64)
        # Use only COMPLETED candles
        if len(ohlcv_array) < 2:
             self.logger.warning("Not enough candles to exclude incomplete one. Received: %s", len(ohlcv_array))
             return None

        closed_candles = ohlcv_array[:-1]  # Exclude last incomplete candle
        actual_current_price = float(ohlcv_array[-1, 4])  # Real-time price from the unclosed candle
        # Verify we have enough data
        expected_candles = self._expected_closed_candles(timeframe, limit)
        if len(closed_candles) < expected_candles:
            coverage_days = self._coverage_days(timeframe, len(closed_candles))
            self.logger.warning(
                "Received fewer closed candles (%s) than expected (%s) for %s target coverage (~%.1f days available)",
                len(closed_candles),
                expected_candles,
                timeframe,
                coverage_days
            )
            self.logger.debug("First candle timestamp: %s", closed_candles[0][0] if len(closed_candles) > 0 else 'N/A')
            self.logger.debug("Last closed candle timestamp: %s", closed_candles[-1][0] if len(closed_candles) > 0 else 'N/A')

        return closed_candles, actual_current_price

    @staticmethod
    def _expected_closed_candles(timeframe: str, limit: int, target_days: int = 30) -> int:
        try:
            desired_candles = TimeframeValidator.get_candle_limit_for_days(timeframe, target_days)
        except ValueError:
            desired_candles = 720
        return min(desired_candles, max(0, limit - 1))

    @staticmethod
    def _coverage_days(timeframe: str, candle_count: int) -> float:
        try:
            return TimeframeValidator.calculate_coverage_days(timeframe, candle_count)
        except ValueError:
            return 0.0

    @retry_async()
    async def fetch_daily_historical_data(self,
                                         pair: str,
                                         days: int = 365
                                         ) -> dict[str, Any]:
        """
        Fetch historical daily data for a specified number of days.

        Args:
            pair: The trading pair to fetch data for
            days: Number of days of historical data to retrieve (default: 365)

        Returns: dict containing:
                'data': NDArray of OHLCV data if available, or None
                'available_days': Number of days of data actually available
                'is_complete': Boolean indicating if we have full history for requested period
        """

        try:
            result = await self.fetch_candlestick_data(
                pair=pair,
                timeframe="1d",
                limit=days
            )

            if result is None:
                self.logger.warning("No daily historical data available for %s", pair)
                return {
                    'data': None,
                    'available_days': 0,
                    'is_complete': False,
                    'error': "No data returned from exchange"
                }

            ohlcv_data, _ = result
            available_days = len(ohlcv_data)
            is_complete = (available_days >= days - 1)  # Closed candles only (no incomplete)

            if not is_complete:
                self.logger.info("Limited historical data for %s: requested %s days, got %s days", pair, days, available_days)

            return {
                'data': ohlcv_data,
                'available_days': available_days,
                'is_complete': is_complete,
                'error': None
            }

        except Exception as e:
            self.logger.error("Error fetching daily historical data for %s: %s", pair, str(e))
            return {
                'data': None,
                'available_days': 0,
                'is_complete': False,
                'error': str(e)
            }

    @retry_async()
    async def fetch_weekly_historical_data(self, pair: str, target_weeks: int = 300) -> dict[str, Any]:
        """
        Fetch weekly data for macro analysis. Wraps fetch_candlestick_data with weekly metadata.

        Args:
            pair: The trading pair to fetch data for
            target_weeks: Number of weeks of historical data to retrieve (default: 300)

        Returns: dict containing:
                'data': NDArray of OHLCV data if available, or None
                'error': Error message if fetch failed, None otherwise
        """
        try:
            # REUSE existing method - already supports '1w'
            result = await self.fetch_candlestick_data(pair=pair, timeframe="1w", limit=target_weeks)

            if result is None:
                return {
                    'data': None,
                    'error': "No data returned"
                }

            ohlcv_data, _ = result

            return {
                'data': ohlcv_data,
                'error': None
            }
        except Exception as e:
            self.logger.error("Error fetching weekly data: %s", e)
            return {
                'data': None,
                'error': str(e)
            }

    @retry_async()
    async def fetch_multiple_tickers(self, symbols: list[str] = None) -> dict[str, Any]:
        """
        Fetch price data for multiple trading pairs at once using CCXT with caching

        Args:
            symbols: list of trading pair symbols (e.g., ["BTC/USDT", "ETH/USDT"])
                    If None, fetches all available tickers

        Returns:
            Dictionary with processed ticker data in a RAW/DISPLAY-compatible shape
        """

        try:
            # Validate exchange capabilities
            if not self._validate_exchange_support():
                return {}

            # Fetch and process ticker data
            tickers = await self.exchange.fetch_tickers(symbols)
            if not tickers:
                self.logger.warning("No ticker data returned from exchange")
                return {}

            return self._process_ticker_data(tickers)

        except Exception as e:
            self.logger.error("Error fetching multiple tickers: %s", e)
            return {}

    def _validate_exchange_support(self) -> bool:
        """Validate that the exchange supports the required operations."""
        if not self.exchange.has.get('fetchTickers', False):
            self.logger.warning("Exchange %s does not support fetchTickers", self.exchange.id)
            return False
        return True

    def _process_ticker_data(self, tickers: dict[str, Any]) -> dict[str, Any]:
        """Process ticker data into a RAW/DISPLAY-compatible format."""
        result = {"RAW": {}, "DISPLAY": {}}

        for symbol, ticker in tickers.items():
            base_currency, quote_currency = self._extract_currencies(symbol)
            if not base_currency or not quote_currency:
                continue

            if not self._has_required_ticker_data(ticker):
                continue

            self._add_ticker_to_result(result, base_currency, quote_currency, ticker)

        return result

    def _extract_currencies(self, symbol: str) -> tuple[str | None, str | None]:
        """Extract base and quote currencies from symbol."""
        if '/' not in symbol:
            return None, None
        parts = symbol.split('/', 1)
        return parts[0], parts[1]

    def _has_required_ticker_data(self, ticker: dict[str, Any]) -> bool:
        """Check if ticker has required data fields."""
        return 'last' in ticker and ticker['last'] is not None

    def _add_ticker_to_result(self, result: dict[str, Any], base_currency: str,
                            quote_currency: str, ticker: dict[str, Any]) -> None:
        """Add processed ticker data to result structure."""
        # Initialize structure if needed
        if base_currency not in result["RAW"]:
            result["RAW"][base_currency] = {}
        if base_currency not in result["DISPLAY"]:
            result["DISPLAY"][base_currency] = {}

        # Add RAW data
        result["RAW"][base_currency][quote_currency] = self._create_raw_ticker_data(ticker)

        # Add DISPLAY data
        result["DISPLAY"][base_currency][quote_currency] = self._create_display_ticker_data(
            ticker, quote_currency
        )

    def _create_raw_ticker_data(self, ticker: dict[str, Any]) -> dict[str, Any]:
        """Create raw ticker data structure with comprehensive 24h statistics."""
        return {
            # Core price data
            "PRICE": ticker.get('last', 0),
            "OPEN24HOUR": ticker.get('open', 0),
            "HIGH24HOUR": ticker.get('high', 0),
            "LOW24HOUR": ticker.get('low', 0),
            "PREVCLOSE": ticker.get('previousClose', 0),

            # Price changes
            "CHANGE24HOUR": ticker.get('change', 0),
            "CHANGEPCT24HOUR": ticker.get('percentage', 0),

            # Volume data
            "VOLUME24HOUR": ticker.get('baseVolume', 0),  # Volume in base currency (e.g., BTC)
            "QUOTEVOLUME24HOUR": ticker.get('quoteVolume', 0),  # Volume in quote currency (e.g., USDT)

            # Price metrics
            "VWAP": ticker.get('vwap', 0),  # Volume-weighted average price
            "AVERAGE": ticker.get('average', 0),  # Simple average price

            # Order book top-of-book
            "BID": ticker.get('bid', 0),  # Best bid price
            "ASK": ticker.get('ask', 0),  # Best ask price
            "BIDVOLUME": ticker.get('bidVolume', 0),  # Size at best bid
            "ASKVOLUME": ticker.get('askVolume', 0),  # Size at best ask

            # Metadata
            "LASTUPDATE": ticker.get('timestamp', 0),
            "MKTCAP": None,  # Market cap not typically available in CCXT ticker

            # Additional useful fields from CCXT
            "INFO": ticker.get('info', {}),  # Raw exchange data (for advanced analysis)
        }

    def _create_display_ticker_data(self, ticker: dict[str, Any], quote_currency: str) -> dict[str, Any]:
        """Create display ticker data structure with formatted values."""
        is_usd_quote = quote_currency in ("USD", "USDT")

        def format_price(value: float) -> str:
            if is_usd_quote:
                return f"$ {value:,.2f}"
            return f"{value:,.8f}"

        return {
            "PRICE": format_price(ticker.get('last', 0)),
            "CHANGEPCT24HOUR": f"{ticker.get('percentage', 0):,.2f}",
            "VOLUME24HOUR": f"{ticker.get('baseVolume', 0):,.2f}",
            "HIGH24HOUR": format_price(ticker.get('high', 0)),
            "LOW24HOUR": format_price(ticker.get('low', 0)),
            "VWAP": format_price(ticker.get('vwap', 0)),
            "BID": format_price(ticker.get('bid', 0)),
            "ASK": format_price(ticker.get('ask', 0)),
        }

    def _calculate_depth_imbalance(self, bid_depth: float, ask_depth: float) -> float:
        """Calculate normalized bid/ask imbalance for comparable depth windows."""
        total_depth = bid_depth + ask_depth
        return (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0

    def _calculate_depth_bucket(self, bids: NDArray, asks: NDArray, level_count: int) -> dict[str, float]:
        """Summarize bid and ask depth for a fixed number of visible levels."""
        bid_subset = bids[:level_count]
        ask_subset = asks[:level_count]
        bid_depth = float(np.sum(bid_subset[:, 1])) if len(bid_subset) else 0.0
        ask_depth = float(np.sum(ask_subset[:, 1])) if len(ask_subset) else 0.0
        bid_notional = float(np.sum(bid_subset[:, 0] * bid_subset[:, 1])) if len(bid_subset) else 0.0
        ask_notional = float(np.sum(ask_subset[:, 0] * ask_subset[:, 1])) if len(ask_subset) else 0.0

        return {
            'levels_used': int(min(len(bid_subset), len(ask_subset))),
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'bid_notional': bid_notional,
            'ask_notional': ask_notional,
            'imbalance': self._calculate_depth_imbalance(bid_depth, ask_depth)
        }

    def _calculate_near_mid_liquidity(
        self,
        bids: NDArray,
        asks: NDArray,
        mid_price: float,
        basis_points: int
    ) -> dict[str, float]:
        """Summarize liquidity close to the mid price within a basis-point band."""
        if mid_price <= 0:
            return {
                'basis_points': basis_points,
                'bid_depth': 0.0,
                'ask_depth': 0.0,
                'imbalance': 0.0
            }

        bid_threshold = mid_price * (1 - basis_points / 10000)
        ask_threshold = mid_price * (1 + basis_points / 10000)
        bid_subset = bids[bids[:, 0] >= bid_threshold]
        ask_subset = asks[asks[:, 0] <= ask_threshold]
        bid_depth = float(np.sum(bid_subset[:, 1])) if len(bid_subset) else 0.0
        ask_depth = float(np.sum(ask_subset[:, 1])) if len(ask_subset) else 0.0

        return {
            'basis_points': basis_points,
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'imbalance': self._calculate_depth_imbalance(bid_depth, ask_depth)
        }

    def _calculate_largest_wall(self, levels: NDArray, mid_price: float) -> dict[str, float]:
        """Find the largest visible resting order on one side of the book."""
        if len(levels) == 0:
            return {
                'price': 0.0,
                'amount': 0.0,
                'notional': 0.0,
                'distance_bps': 0.0
            }

        wall_index = int(np.argmax(levels[:, 1]))
        wall_price = float(levels[wall_index][0])
        wall_amount = float(levels[wall_index][1])
        distance_bps = abs(wall_price - mid_price) / mid_price * 10000 if mid_price > 0 else 0.0

        return {
            'price': wall_price,
            'amount': wall_amount,
            'notional': wall_price * wall_amount,
            'distance_bps': distance_bps
        }

    @retry_async()
    async def fetch_order_book_depth(self, pair: str, limit: int = 100) -> dict[str, Any] | None:
        """
        Fetch order book depth for liquidity and support/resistance analysis.

        Provides real-time market depth showing bid/ask walls and liquidity zones.
        Useful for identifying:
        - Strong support/resistance levels (large order concentrations)
        - Market depth and liquidity
        - Buy/sell pressure imbalance
        - Potential manipulation (spoofing via large fake walls)

        Args:
            pair: Trading pair symbol (e.g., "BTC/USDT")
            limit: Maximum number of order book levels to fetch (default: 100)

        Returns: dict containing:
                'bids': list of [price, amount] bid orders (buy side)
                'asks': list of [price, amount] ask orders (sell side)
                'timestamp': Unix timestamp in milliseconds
                'spread': Absolute spread between best bid and ask
                'spread_percent': Spread as percentage of best bid price
                'bid_depth': Total volume on bid side (base currency)
                'ask_depth': Total volume on ask side (base currency)
                'imbalance': Buy/sell pressure ratio (-1 to +1, positive = more bids)
                'mid_price': Midpoint between best bid and ask
            Returns None if fetch fails or exchange doesn't support order books
        """
        try:
            # Check exchange support
            if not self.exchange.has.get('fetchOrderBook', False):
                self.logger.debug("Exchange %s does not support fetchOrderBook", self.exchange.id)
                return None

            order_book = await self.exchange.fetch_order_book(pair, limit=limit)

            if not order_book or not order_book.get('bids') or not order_book.get('asks'):
                self.logger.warning("Empty order book returned for %s", pair)
                return None

            bids = np.array(order_book['bids'], dtype=np.float64)
            asks = np.array(order_book['asks'], dtype=np.float64)

            if len(bids) == 0 or len(asks) == 0:
                self.logger.warning("Order book has empty bids or asks for %s", pair)
                return None

            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            best_bid_size = float(bids[0][1])
            best_ask_size = float(asks[0][1])
            spread = best_ask - best_bid
            spread_percent = (spread / best_bid * 100) if best_bid > 0 else 0
            mid_price = (best_bid + best_ask) / 2

            bid_depth = float(np.sum(bids[:, 1]))
            ask_depth = float(np.sum(asks[:, 1]))
            imbalance = self._calculate_depth_imbalance(bid_depth, ask_depth)

            depth_by_level = {
                str(level): self._calculate_depth_bucket(bids, asks, min(level, len(bids), len(asks)))
                for level in (5, 10, 20)
            }
            liquidity_near_mid = {
                '10bps': self._calculate_near_mid_liquidity(bids, asks, mid_price, 10),
                '25bps': self._calculate_near_mid_liquidity(bids, asks, mid_price, 25)
            }

            result = {
                'bids': order_book['bids'],
                'asks': order_book['asks'],
                'timestamp': order_book.get('timestamp'),
                'levels_requested': limit,
                'levels_analyzed': int(min(len(bids), len(asks))),
                'spread': spread,
                'spread_percent': spread_percent,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'imbalance': imbalance,
                'mid_price': mid_price,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'best_bid_size': best_bid_size,
                'best_ask_size': best_ask_size,
                'bid_notional': float(np.sum(bids[:, 0] * bids[:, 1])),
                'ask_notional': float(np.sum(asks[:, 0] * asks[:, 1])),
                'depth_by_level': depth_by_level,
                'liquidity_near_mid': liquidity_near_mid,
                'largest_bid_wall': self._calculate_largest_wall(bids, mid_price),
                'largest_ask_wall': self._calculate_largest_wall(asks, mid_price)
            }


            return result

        except Exception as e:
            self.logger.error("Error fetching order book for %s: %s", pair, e)
            return None

    @retry_async()
    async def fetch_recent_trades(self, pair: str, limit: int = 1000) -> dict[str, Any] | None:
        """
        Fetch recent trades for order flow and momentum analysis.

        Provides actual executed trades showing real market activity.
        Useful for identifying:
        - Order flow direction (aggressive buyers vs sellers)
        - Trade velocity and market activity
        - Average trade sizes (retail vs institutional flow)
        - Buy/sell pressure trends

        Args:
            pair: Trading pair symbol (e.g., "BTC/USDT")
            limit: Maximum number of recent trades to fetch (default: 1000)

        Returns: dict containing:
                'trades': list of trade dicts with timestamp, price, amount, side
                'buy_volume': Total volume from buy-side trades (base currency)
                'sell_volume': Total volume from sell-side trades (base currency)
                'buy_sell_ratio': Ratio of buy to sell volume (>1 = more buying)
                'buy_pressure_percent': Percentage of volume from buys (50 = balanced)
                'avg_trade_size': Average trade size across all trades
                'trade_velocity': Trades per minute
                'total_trades': Total number of trades returned
                'time_span_minutes': Time span covered by trades
            Returns None if fetch fails or exchange doesn't support trades
        """
        try:
            # Check exchange support
            if not self.exchange.has.get('fetchTrades', False):
                self.logger.debug("Exchange %s does not support fetchTrades", self.exchange.id)
                return None

            trades = await self.exchange.fetch_trades(pair, limit=limit)

            if not trades or len(trades) == 0:
                self.logger.warning("No trades returned for %s", pair)
                return None

            # Calculate buy/sell volumes
            buy_volume = sum(float(t['amount']) for t in trades if t.get('side') == 'buy')
            sell_volume = sum(float(t['amount']) for t in trades if t.get('side') == 'sell')
            total_volume = buy_volume + sell_volume

            # Calculate time span and velocity
            time_span_ms = trades[-1]['timestamp'] - trades[0]['timestamp']
            time_span_minutes = time_span_ms / (1000 * 60) if time_span_ms > 0 else 1
            trade_velocity = len(trades) / time_span_minutes if time_span_minutes > 0 else 0

            # Calculate buy/sell ratio
            buy_sell_ratio = buy_volume / sell_volume if sell_volume > 0 else float('inf')
            buy_pressure_percent = (buy_volume / total_volume * 100) if total_volume > 0 else 50

            result = {
                'trades': trades,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'buy_sell_ratio': buy_sell_ratio,
                'buy_pressure_percent': buy_pressure_percent,
                'avg_trade_size': total_volume / len(trades) if len(trades) > 0 else 0,
                'trade_velocity': trade_velocity,
                'total_trades': len(trades),
                'time_span_minutes': time_span_minutes
            }


            return result

        except Exception as e:
            self.logger.error("Error fetching trades for %s: %s", pair, e)
            return None

    @retry_async()
    async def fetch_funding_rate(self, pair: str) -> dict[str, Any] | None:
        """
        Fetch current funding rate for perpetual futures contracts.

        Funding rates represent periodic payments between long and short positions.
        Useful for identifying:
        - Market sentiment (positive rate = longs pay shorts = bullish sentiment)
        - Overleveraged positions (extreme rates signal potential reversals)
        - Cost of holding leveraged positions

        Args:
            pair: Trading pair symbol for perpetual futures (e.g., "BTC/USDT:USDT")

        Returns: dict containing:
                'funding_rate': Current funding rate (decimal, e.g., 0.0001 = 0.01%)
                'funding_rate_percent': Funding rate as percentage
                'funding_timestamp': Next funding time (Unix timestamp in ms)
                'annualized_rate': Annualized funding rate percentage (assumes 3x daily)
                'sentiment': Sentiment interpretation ('Bullish'/'Bearish'/'Neutral')
            Returns None if not a perpetual contract or exchange doesn't support funding rates

        Note:
            - Only applicable to perpetual futures contracts (not spot markets)
            - Positive rate: Longs pay shorts (bullish sentiment)
            - Negative rate: Shorts pay longs (bearish sentiment)
            - Typical funding occurs every 8 hours (3 times per day)
        """
        try:
            # Check exchange support
            if not self.exchange.has.get('fetchFundingRate', False):
                self.logger.debug("Exchange %s does not support fetchFundingRate", self.exchange.id)
                return None

            funding = await self.exchange.fetch_funding_rate(pair)

            if not funding or 'fundingRate' not in funding:
                self.logger.debug("No funding rate available for %s", pair)
                return None

            rate = float(funding.get('fundingRate', 0))
            # Funding typically happens every 8 hours (3x daily), annualize it
            annualized_rate = rate * 3 * 365 * 100

            # Interpret sentiment based on rate
            if rate > 0.01:  # > 1% per funding
                sentiment = 'Strong Bullish'
            elif rate > 0.0001:  # > 0.01% per funding
                sentiment = 'Bullish'
            elif rate < -0.01:
                sentiment = 'Strong Bearish'
            elif rate < -0.0001:
                sentiment = 'Bearish'
            else:
                sentiment = 'Neutral'

            result = {
                'funding_rate': rate,
                'funding_rate_percent': rate * 100,
                'funding_timestamp': funding.get('fundingTime'),
                'annualized_rate': annualized_rate,
                'sentiment': sentiment
            }


            return result

        except Exception as e:
            self.logger.debug("Funding rate not available for %s: %s", pair, e)
            return None

    async def fetch_market_microstructure(self, pair: str, cached_ticker: dict | None = None) -> dict[str, Any]:
        """
        Fetch comprehensive market microstructure data for a trading pair.

        Combines ticker statistics, order book depth, recent trades, and funding rate
        (for futures) into a single comprehensive market snapshot. This provides AI
        models with rich context about current market conditions, liquidity, and
        order flow dynamics.

        Args:
            pair: Trading pair symbol (e.g., "BTC/USDT")
            cached_ticker: Optional pre-fetched ticker data to avoid redundant API calls

        Returns: dict containing:
                'ticker': 24h ticker statistics (price, volume, changes)
                'order_book': Order book depth analysis (liquidity, spread, imbalance)
                'recent_trades': Recent trade flow analysis (buy/sell pressure, velocity)
                'funding_rate': Funding rate data (futures only, None for spot)
                'available_data': list of successfully fetched data types
                'timestamp': Collection timestamp

        Note:
            - Not all data may be available depending on exchange capabilities
            - Spot markets will have None for funding_rate
            - Failures in individual fetches are logged but don't fail the entire call
        """
        result = {
            'ticker': None,
            'order_book': None,
            'recent_trades': None,
            'funding_rate': None,
            'snapshot_context': {
                'is_live_snapshot': True,
                'comparison_basis': 'previous_analysis_cycle_snapshot',
                'comparison_available': False
            },
            'available_data': [],
            'timestamp': int(time.time() * 1000)
        }

        # Fetch ticker data (or use cached)
        try:
            if cached_ticker:
                result['ticker'] = cached_ticker
                result['available_data'].append('ticker')
            else:
                ticker_data = await self.fetch_multiple_tickers([pair])
                if ticker_data and 'RAW' in ticker_data:
                    # Extract the ticker for this pair
                    base, quote = self._extract_currencies(pair)
                    if base and quote and base in ticker_data['RAW'] and quote in ticker_data['RAW'][base]:
                        result['ticker'] = ticker_data['RAW'][base][quote]
                        result['available_data'].append('ticker')
        except Exception as e:
            self.logger.warning("Could not fetch ticker for %s: %s", pair, e)

        # Fetch order book
        try:
            order_book = await self.fetch_order_book_depth(pair, limit=50)
            if order_book:
                result['order_book'] = order_book
                result['available_data'].append('order_book')
        except Exception as e:
            self.logger.warning("Could not fetch order book for %s: %s", pair, e)

        # Fetch recent trades
        try:
            trades = await self.fetch_recent_trades(pair, limit=500)
            if trades:
                result['recent_trades'] = trades
                result['available_data'].append('recent_trades')
        except Exception as e:
            self.logger.warning("Could not fetch recent trades for %s: %s", pair, e)

        # Fetch funding rate (for futures only)
        try:
            funding = await self.fetch_funding_rate(pair)
            if funding:
                result['funding_rate'] = funding
                result['available_data'].append('funding_rate')
        except Exception as e:
            self.logger.debug("Funding rate not available for %s: %s", pair, e)


        return result
