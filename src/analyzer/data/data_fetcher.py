from typing import Optional, Tuple, Dict, Any, List
import time

import numpy as np
from numpy.typing import NDArray

from src.logger.logger import Logger
from src.utils.decorators import retry_async
from src.utils.timeframe_validator import TimeframeValidator


class DataFetcher:
    def __init__(self, exchange, logger: Logger):
        self.exchange = exchange
        self.logger: Logger = logger

    @retry_async()
    async def fetch_candlestick_data(self,
                                     pair: str,
                                     timeframe: str,
                                     limit: int,
                                     start_time: Optional[int] = None
                                     ) -> Optional[Tuple[NDArray, float]]:
        self.logger.debug(f"Fetching {pair} OHLCV data on {timeframe} timeframe with limit {limit}")
        
        # Validate timeframe is supported by exchange
        if hasattr(self.exchange, 'timeframes') and self.exchange.timeframes:
            if timeframe not in self.exchange.timeframes:
                self.logger.error(
                    f"Timeframe {timeframe} not supported by {self.exchange.id}. "
                    f"Supported: {', '.join(self.exchange.timeframes.keys())}"
                )
                return None
        elif not TimeframeValidator.is_ccxt_compatible(timeframe):
            self.logger.warning(
                f"Timeframe {timeframe} may not be supported by {self.exchange.id}. "
                f"Attempting fetch anyway..."
            )
        
        if limit > 1000:
            self.logger.warning(f"Requested limit {limit} exceeds exchange standard limits, may be truncated")
            
        ohlcv = await self.exchange.fetch_ohlcv(pair, timeframe, since=start_time, limit=limit + 1)

        if ohlcv is None or len(ohlcv) == 0:
            self.logger.warning(f"No data returned for {pair} on {self.exchange.id}")
            return None
        
        self.logger.debug(f"Received {len(ohlcv)} raw candles from exchange for {pair}")
        
        ohlcv_array = np.array(ohlcv)
        # Include all candles (including the incomplete current candle)
        all_candles = ohlcv_array
        latest_close = float(ohlcv_array[-1, 4])
        
        self.logger.debug(f"Processed {len(all_candles)} candles (including current incomplete candle), latest close: {latest_close}")
        
        # Verify we have enough data
        if len(all_candles) < min(720, limit):
            self.logger.warning(f"Received fewer candles ({len(all_candles)}) than expected ({min(720, limit)})")
            self.logger.debug(f"First candle timestamp: {all_candles[0][0] if len(all_candles) > 0 else 'N/A'}")
            self.logger.debug(f"Last candle timestamp: {all_candles[-1][0] if len(all_candles) > 0 else 'N/A'}")

        return all_candles, latest_close

    @retry_async()
    async def fetch_daily_historical_data(self,
                                         pair: str,
                                         days: int = 365
                                         ) -> Dict[str, Any]:
        """
        Fetch historical daily data for a specified number of days.
        
        Args:
            pair: The trading pair to fetch data for
            days: Number of days of historical data to retrieve (default: 365)
            
        Returns:
            Dict containing:
                'data': NDArray of OHLCV data if available, or None
                'latest_close': Latest closing price
                'available_days': Number of days of data actually available
                'is_complete': Boolean indicating if we have full history for requested period
        """
        self.logger.debug(f"Fetching historical daily data for {pair}, {days} days")
        
        try:
            result = await self.fetch_candlestick_data(
                pair=pair,
                timeframe="1d",
                limit=days
            )
            
            if result is None:
                self.logger.warning(f"No daily historical data available for {pair}")
                return {
                    'data': None,
                    'latest_close': None,
                    'available_days': 0,
                    'is_complete': False,
                    'error': "No data returned from exchange"
                }
                
            ohlcv_data, latest_close = result
            available_days = len(ohlcv_data)
            is_complete = (available_days >= days)  # All candles now included
            
            if not is_complete:
                self.logger.info(f"Limited historical data for {pair}: requested {days} days, got {available_days} days")
            else:
                self.logger.debug(f"Successfully fetched {available_days} days of historical data for {pair}")
                
            return {
                'data': ohlcv_data,
                'latest_close': latest_close,
                'available_days': available_days,
                'is_complete': is_complete,
                'error': None
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching daily historical data for {pair}: {str(e)}")
            return {
                'data': None,
                'latest_close': None,
                'available_days': 0,
                'is_complete': False,
                'error': str(e)
            }

    @retry_async()
    async def fetch_weekly_historical_data(self, pair: str, target_weeks: int = 300) -> Dict[str, Any]:
        """
        Fetch weekly data for macro analysis. Wraps fetch_candlestick_data with weekly metadata.
        
        Args:
            pair: The trading pair to fetch data for
            target_weeks: Number of weeks of historical data to retrieve (default: 300)
            
        Returns:
            Dict containing:
                'data': NDArray of OHLCV data if available, or None
                'latest_close': Latest closing price
                'available_weeks': Number of weeks of data actually available
                'meets_200w_threshold': Boolean indicating if we have 200+ weeks for full 200W SMA analysis
                'error': Error message if fetch failed, None otherwise
        """
        self.logger.debug(f"Fetching weekly data for {pair}: {target_weeks} weeks")
        
        try:
            # REUSE existing method - already supports '1w'
            result = await self.fetch_candlestick_data(pair=pair, timeframe="1w", limit=target_weeks)
            
            if result is None:
                return {
                    'data': None,
                    'latest_close': None,
                    'available_weeks': 0,
                    'meets_200w_threshold': False,
                    'error': "No data returned"
                }
                
            ohlcv_data, latest_close = result
            available_weeks = len(ohlcv_data)
            
            return {
                'data': ohlcv_data,
                'latest_close': latest_close,
                'available_weeks': available_weeks,
                'meets_200w_threshold': available_weeks >= 200,
                'error': None
            }
        except Exception as e:
            self.logger.error(f"Error fetching weekly data: {e}")
            return {
                'data': None,
                'latest_close': None,
                'available_weeks': 0,
                'meets_200w_threshold': False,
                'error': str(e)
            }

    @retry_async()
    async def fetch_multiple_tickers(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Fetch price data for multiple trading pairs at once using CCXT with caching
        
        Args:
            symbols: List of trading pair symbols (e.g., ["BTC/USDT", "ETH/USDT"])
                    If None, fetches all available tickers
                    
        Returns:
            Dictionary with processed ticker data in a format similar to CryptoCompare API
        """
        self.logger.debug(f"Fetching multiple tickers: {symbols if symbols else 'all'}")
        
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
            self.logger.error(f"Error fetching multiple tickers: {e}")
            return {}

    def _validate_exchange_support(self) -> bool:
        """Validate that the exchange supports the required operations."""
        if not self.exchange.has.get('fetchTickers', False):
            self.logger.warning(f"Exchange {self.exchange.id} does not support fetchTickers")
            return False
        return True

    def _process_ticker_data(self, tickers: Dict[str, Any]) -> Dict[str, Any]:
        """Process ticker data into CryptoCompare-like format."""
        result = {"RAW": {}, "DISPLAY": {}}
        
        for symbol, ticker in tickers.items():
            base_currency, quote_currency = self._extract_currencies(symbol)
            if not base_currency or not quote_currency:
                continue
                
            if not self._has_required_ticker_data(ticker):
                continue
                
            self._add_ticker_to_result(result, base_currency, quote_currency, ticker)
        
        return result

    def _extract_currencies(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract base and quote currencies from symbol."""
        if '/' not in symbol:
            return None, None
        parts = symbol.split('/', 1)
        return parts[0], parts[1]

    def _has_required_ticker_data(self, ticker: Dict[str, Any]) -> bool:
        """Check if ticker has required data fields."""
        return 'last' in ticker and ticker['last'] is not None

    def _add_ticker_to_result(self, result: Dict[str, Any], base_currency: str, 
                            quote_currency: str, ticker: Dict[str, Any]) -> None:
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

    def _create_raw_ticker_data(self, ticker: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create raw ticker data structure with comprehensive 24h statistics.
        
        Includes price, volume, volatility, and bid/ask data useful for:
        - Momentum analysis (percentage changes, volume trends)
        - Volatility assessment (high/low range, VWAP deviation)
        - Liquidity evaluation (bid/ask spread, volumes)
        - Market microstructure (bid/ask sizes, quote volume)
        """
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

    def _create_display_ticker_data(self, ticker: Dict[str, Any], quote_currency: str) -> Dict[str, Any]:
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

    @retry_async()
    async def fetch_order_book_depth(self, pair: str, limit: int = 100) -> Optional[Dict[str, Any]]:
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
            
        Returns:
            Dict containing:
                'bids': List of [price, amount] bid orders (buy side)
                'asks': List of [price, amount] ask orders (sell side)
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
                self.logger.debug(f"Exchange {self.exchange.id} does not support fetchOrderBook")
                return None
            
            order_book = await self.exchange.fetch_order_book(pair, limit=limit)
            
            if not order_book or not order_book.get('bids') or not order_book.get('asks'):
                self.logger.warning(f"Empty order book returned for {pair}")
                return None
            
            bids = np.array(order_book['bids'])
            asks = np.array(order_book['asks'])
            
            if len(bids) == 0 or len(asks) == 0:
                self.logger.warning(f"Order book has empty bids or asks for {pair}")
                return None
            
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = best_ask - best_bid
            spread_percent = (spread / best_bid * 100) if best_bid > 0 else 0
            
            bid_depth = float(np.sum(bids[:, 1]))
            ask_depth = float(np.sum(asks[:, 1]))
            total_depth = bid_depth + ask_depth
            imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
            
            result = {
                'bids': order_book['bids'],
                'asks': order_book['asks'],
                'timestamp': order_book.get('timestamp'),
                'spread': spread,
                'spread_percent': spread_percent,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'imbalance': imbalance,
                'mid_price': (best_bid + best_ask) / 2,
                'best_bid': best_bid,
                'best_ask': best_ask
            }
            
            self.logger.debug(
                f"Order book for {pair}: spread={spread_percent:.3f}%, "
                f"imbalance={imbalance:+.3f}, depth={bid_depth:.2f}/{ask_depth:.2f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching order book for {pair}: {e}")
            return None

    @retry_async()
    async def fetch_recent_trades(self, pair: str, limit: int = 1000) -> Optional[Dict[str, Any]]:
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
            
        Returns:
            Dict containing:
                'trades': List of trade dicts with timestamp, price, amount, side
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
                self.logger.debug(f"Exchange {self.exchange.id} does not support fetchTrades")
                return None
            
            trades = await self.exchange.fetch_trades(pair, limit=limit)
            
            if not trades or len(trades) == 0:
                self.logger.warning(f"No trades returned for {pair}")
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
            
            self.logger.debug(
                f"Recent trades for {pair}: {len(trades)} trades, "
                f"buy_pressure={buy_pressure_percent:.1f}%, "
                f"velocity={trade_velocity:.1f}/min"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching trades for {pair}: {e}")
            return None

    @retry_async()
    async def fetch_funding_rate(self, pair: str) -> Optional[Dict[str, Any]]:
        """
        Fetch current funding rate for perpetual futures contracts.
        
        Funding rates represent periodic payments between long and short positions.
        Useful for identifying:
        - Market sentiment (positive rate = longs pay shorts = bullish sentiment)
        - Overleveraged positions (extreme rates signal potential reversals)
        - Cost of holding leveraged positions
        
        Args:
            pair: Trading pair symbol for perpetual futures (e.g., "BTC/USDT:USDT")
            
        Returns:
            Dict containing:
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
                self.logger.debug(f"Exchange {self.exchange.id} does not support fetchFundingRate")
                return None
            
            funding = await self.exchange.fetch_funding_rate(pair)
            
            if not funding or 'fundingRate' not in funding:
                self.logger.debug(f"No funding rate available for {pair}")
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
            
            self.logger.debug(
                f"Funding rate for {pair}: {rate*100:.4f}% ({sentiment}), "
                f"annualized: {annualized_rate:.2f}%"
            )
            
            return result
            
        except Exception as e:
            self.logger.debug(f"Funding rate not available for {pair}: {e}")
            return None

    async def fetch_market_microstructure(self, pair: str) -> Dict[str, Any]:
        """
        Fetch comprehensive market microstructure data for a trading pair.
        
        Combines ticker statistics, order book depth, recent trades, and funding rate
        (for futures) into a single comprehensive market snapshot. This provides AI
        models with rich context about current market conditions, liquidity, and
        order flow dynamics.
        
        Args:
            pair: Trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            Dict containing:
                'ticker': 24h ticker statistics (price, volume, changes)
                'order_book': Order book depth analysis (liquidity, spread, imbalance)
                'recent_trades': Recent trade flow analysis (buy/sell pressure, velocity)
                'funding_rate': Funding rate data (futures only, None for spot)
                'available_data': List of successfully fetched data types
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
            'available_data': [],
            'timestamp': int(time.time() * 1000)
        }
        
        # Fetch ticker data
        try:
            ticker_data = await self.fetch_multiple_tickers([pair])
            if ticker_data and 'RAW' in ticker_data:
                # Extract the ticker for this pair
                base, quote = self._extract_currencies(pair)
                if base and quote and base in ticker_data['RAW'] and quote in ticker_data['RAW'][base]:
                    result['ticker'] = ticker_data['RAW'][base][quote]
                    result['available_data'].append('ticker')
        except Exception as e:
            self.logger.warning(f"Could not fetch ticker for {pair}: {e}")
        
        # Fetch order book
        try:
            order_book = await self.fetch_order_book_depth(pair, limit=50)
            if order_book:
                result['order_book'] = order_book
                result['available_data'].append('order_book')
        except Exception as e:
            self.logger.warning(f"Could not fetch order book for {pair}: {e}")
        
        # Fetch recent trades
        try:
            trades = await self.fetch_recent_trades(pair, limit=500)
            if trades:
                result['recent_trades'] = trades
                result['available_data'].append('recent_trades')
        except Exception as e:
            self.logger.warning(f"Could not fetch recent trades for {pair}: {e}")
        
        # Fetch funding rate (for futures only)
        try:
            funding = await self.fetch_funding_rate(pair)
            if funding:
                result['funding_rate'] = funding
                result['available_data'].append('funding_rate')
        except Exception as e:
            self.logger.debug(f"Funding rate not available for {pair}: {e}")
        
        self.logger.info(
            f"Market microstructure for {pair}: "
            f"available_data={result['available_data']}"
        )
        
        return result

