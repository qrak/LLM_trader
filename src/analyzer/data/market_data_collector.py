from datetime import datetime
from typing import Dict, List, Any, Optional

import aiohttp

from .data_fetcher import DataFetcher
from src.logger.logger import Logger
from src.utils.timeframe_validator import TimeframeValidator
from src.platforms.alternative_me import AlternativeMeAPI
from src.rag import RagEngine


class MarketDataCollector:
    """Handles market data collection from various sources"""
    
    def __init__(self, 
                logger: Logger, 
                rag_engine: RagEngine,
                alternative_me_api: Optional[AlternativeMeAPI] = None):
        self.logger = logger
        self.rag_engine = rag_engine
        self.alternative_me_api = alternative_me_api
        
        # Will be set by initialize method
        self.data_fetcher = None
        self.symbol = None
        self.exchange = None
        self.timeframe = "1h"
        self.limit = None  # Will be calculated dynamically
        
        # Storage for collected data
        self.article_urls = {}
    
    def initialize(self, 
                  data_fetcher: DataFetcher, 
                  symbol: str, 
                  exchange, 
                  timeframe: str = "1h", 
                  limit: int = None) -> None:
        """
        Initialize the collector with required parameters.
        
        Args:
            data_fetcher: DataFetcher instance for fetching market data
            symbol: Trading symbol (e.g., "BTC/USDT")
            exchange: Exchange instance
            timeframe: Timeframe for candles (e.g., "1h", "4h", "1d")
            limit: Optional candle limit (auto-calculated if None)
        """
        self.data_fetcher = data_fetcher
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        
        # Calculate limit dynamically if not provided
        if limit is None:
            target_days = 30
            self.limit = TimeframeValidator.get_candle_limit_for_days(timeframe, target_days)
            self.logger.debug(
                f"Calculated candle limit: {self.limit} for {timeframe} timeframe "
                f"(~{target_days} days of data)"
            )
        else:
            self.limit = limit
        
        self.article_urls = {}
    
    async def collect_data(self, context) -> Dict[str, Any]:
        """Collect all required data for market analysis"""
        result = {
            "success": True,
            "errors": []
        }
        
        try:
            if not self.symbol or not self.exchange or not self.data_fetcher:
                error_msg = "Cannot collect data: collector not properly initialized"
                self.logger.error(error_msg)
                return {"success": False, "errors": [error_msg]}
            
            # Fetch OHLCV data
            ohlcv_success = await self.fetch_ohlcv(context)
            if not ohlcv_success:
                result["errors"].append("Failed to fetch OHLCV data")
                result["success"] = False
            
            # Fetch news context via RAG engine
            market_context = await self.rag_engine.retrieve_context(
                "current market news analysis trends", 
                self.symbol, 
                k=5
            )
            result["market_context"] = market_context
            
            # Store article URLs from RAG engine
            try:
                self.article_urls = self.rag_engine.context_builder.get_latest_article_urls()
                self.logger.debug(f"Retrieved {len(self.article_urls)} article URLs from RAG engine")
            except Exception as e:
                self.logger.warning(f"Could not retrieve article URLs from RAG engine: {e}")
                self.article_urls = {}
                
            result["article_urls"] = self.article_urls
                
        except Exception as e:
            self.logger.exception(f"Error collecting market data: {e}")
            result["success"] = False
            result["errors"].append(str(e))
            
        return result
        
    async def fetch_ohlcv(self, context) -> bool:
        """Fetch OHLCV candle data and update context"""
        try:
            if not self.symbol or not self.exchange or not self.data_fetcher:
                self.logger.error("Cannot fetch OHLCV: symbol, exchange, or data_fetcher not initialized")
                return False
            
            result = await self.data_fetcher.fetch_candlestick_data(
                pair=self.symbol,
                timeframe=self.timeframe,
                limit=self.limit
            )

            if result is None:
                self.logger.error("No data returned from exchange")
                return False

            context.ohlcv_candles, context.current_price = result
            
            # Convert timestamps once here for reuse across all components
            try:
                context.timestamps = [datetime.fromtimestamp(ts / 1000) for ts in context.ohlcv_candles[:, 0]]
            except Exception as e:
                self.logger.warning(f"Could not extract timestamps from OHLCV data: {e}")
                context.timestamps = None

            if len(context.ohlcv_candles) < 720:
                hours_available = len(context.ohlcv_candles)
                days_available = hours_available / 24
                self.logger.warning(f"Insufficient data for full 30-day analysis. Have {hours_available} hours (~{days_available:.1f} days)")
                
            # Fetch long-term historical data for additional context
            await self.fetch_long_term_historical_data(context)
            
            # Fetch weekly macro data for 200W SMA analysis
            await self.fetch_weekly_macro_data(context, target_weeks=300)

            # Fetch and process market sentiment data
            await self.fetch_and_process_sentiment_data(context)
            
            return True

        except Exception as e:
            self.logger.exception(f"OHLCV fetch failed: {str(e)}")
            return False
        
    async def fetch_long_term_historical_data(self, context, days: int = 365) -> bool:
        """Fetch long-term (365 days by default) historical data"""
        try:
            if not self.symbol or not self.exchange or not self.data_fetcher:
                self.logger.error("Cannot fetch long-term data: symbol, exchange, or data_fetcher not initialized")
                return False
                
            result = await self.data_fetcher.fetch_daily_historical_data(
                pair=self.symbol,
                days=days
            )
            
            if result['error'] is not None:
                self.logger.error(f"Error fetching long-term data: {result['error']}")
                return False
                
            if result['data'] is None or len(result['data']) == 0:
                self.logger.warning(f"No long-term historical data available for {self.symbol}")
                context.long_term_data = {
                    'is_new_token': True,
                    'available_days': 0,
                    'sma_values': {},
                    'volume_sma_values': {},
                    'price_change': None,
                    'volume_change': None,
                    'volatility': None
                }
                return True
                
            ohlcv_data = result['data']
            available_days = result['available_days']
            is_complete = result['is_complete']
            
            # We'll calculate the indicators in MarketAnalyzer,
            # we just store the raw data properties here
            context.long_term_data = {
                'data': ohlcv_data,
                'is_new_token': not is_complete and available_days < 100,
                'available_days': available_days,
                'is_complete': is_complete
            }
            
            if not is_complete:
                self.logger.info(f"Note: {self.symbol} has limited historical data ({available_days}/{days} days)")
                if available_days < 30:
                    self.logger.warning(f"Very limited historical data for {self.symbol} - this may be a new token")
            
            return True
            
        except Exception as e:
            self.logger.exception(f"Long-term data fetch failed: {str(e)}")
            context.long_term_data = None
            return False

    async def fetch_weekly_macro_data(self, context, target_weeks: int = 300) -> bool:
        """Fetch weekly data for macro trend analysis (200W SMA methodology)"""
        try:
            if not self.symbol or not self.exchange or not self.data_fetcher:
                self.logger.error("Cannot fetch weekly data: symbol, exchange, or data_fetcher not initialized")
                return False
            
            self.logger.info(f"Fetching weekly macro data for {self.symbol}")
            
            result = await self.data_fetcher.fetch_weekly_historical_data(self.symbol, target_weeks)
            
            if result['data'] is None:
                self.logger.warning(f"Weekly data unavailable: {result.get('error', 'Unknown')}")
                context.weekly_ohlcv = None
                context.available_weeks = 0
                context.meets_200w_threshold = False
                return False
            
            context.weekly_ohlcv = result['data']
            context.available_weeks = result['available_weeks']
            context.meets_200w_threshold = result['meets_200w_threshold']
            
            self.logger.info(
                f"Weekly data: {result['available_weeks']} weeks, "
                f"200W SMA: {'Available' if result['meets_200w_threshold'] else 'Insufficient'}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Error fetching weekly macro data: {e}")
            context.weekly_ohlcv = None
            context.available_weeks = 0
            context.meets_200w_threshold = False
            return False

        
    async def fetch_and_process_sentiment_data(self, context) -> bool:
        """Fetch and process sentiment data (fear & greed index)"""
        try:
            fear_greed_data = await self._fetch_fear_greed_index(limit=7)
            
            # If no fear & greed data available, return early
            if not fear_greed_data or len(fear_greed_data) == 0:
                self.logger.warning("No Fear & Greed data available")
                return False
                
            # Get the latest sentiment data
            latest_fg = fear_greed_data[0]
            
            # Process the sentiment history
            historical = []
            for fg in fear_greed_data:
                historical.append({
                    'timestamp': datetime.fromtimestamp(int(fg["timestamp"])),
                    'value': int(fg["value"]),
                    'value_classification': fg["value_classification"]
                })
            
            # Set context sentiment with properly formatted data
            context.sentiment = {
                'timestamp': datetime.fromtimestamp(int(latest_fg["timestamp"])),
                'fear_greed_index': int(latest_fg["value"]),
                'value_classification': latest_fg["value_classification"],
                'historical': historical
            }
            
            self.logger.debug(f"Set sentiment data: {context.sentiment}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error fetching sentiment data: {e}")
            return False
        
    async def _fetch_fear_greed_index(self, limit: int = 0) -> List[Dict[str, Any]]:
        """Fetch Fear & Greed Index data using AlternativeMeAPI client"""
        try:
            # Use the API client if available
            if self.alternative_me_api:
                if limit > 0:
                    # Get historical data if limit is specified
                    history = await self.alternative_me_api.get_historical_fear_greed(days=limit)
                    # Convert to the format expected by the rest of the code
                    result = []
                    for item in history:
                        result.append({
                            "value": str(item["value"]),
                            "value_classification": item["value_classification"],
                            "timestamp": str(item["timestamp"])
                        })
                    return result
                else:
                    # Get current data only
                    current = await self.alternative_me_api.get_fear_greed_index()
                    return [{
                        "value": str(current["value"]),
                        "value_classification": current["value_classification"],
                        "timestamp": str(current["timestamp"])
                    }]
            else:
                # Fallback to direct API call if client not available
                self.logger.warning("AlternativeMeAPI client not available, falling back to direct API call")
                params = {"limit": limit, "format": "json"} if limit > 0 else {}
                
                # Use a context manager to ensure the session is properly closed
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                            "https://api.alternative.me/fng/",
                            params=params,
                            timeout=10
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        if data["metadata"]["error"]:
                            raise ValueError(f"API Error: {data['metadata']['error']}")
                        
                        return data["data"]
        except Exception as e:
            self.logger.error(f"Fear & Greed index fetch failed: {e}")
            return []
    
    def extract_ohlcv_data(self, context) -> list:
        """Extract OHLCV data from context for metrics calculation
        
        Args:
            context: Analysis context containing OHLCV candles
            
        Returns:
            list: List of processed market data points
        """
        data = []
        
        if not hasattr(context, 'ohlcv_candles') or context.ohlcv_candles is None:
            self.logger.warning("No OHLCV data available for metrics calculation")
            return data
        
        # Use pre-computed timestamps if available
        timestamps = context.timestamps if hasattr(context, 'timestamps') and context.timestamps else None
        
        for idx in range(len(context.ohlcv_candles)):
            # Use pre-computed timestamp or fallback to conversion
            if timestamps and idx < len(timestamps):
                timestamp = timestamps[idx]
            else:
                timestamp = datetime.fromtimestamp(float(context.ohlcv_candles[idx, 0]) / 1000.0)
            
            market_data = dict(
                timestamp=timestamp,
                open=float(context.ohlcv_candles[idx, 1]),
                high=float(context.ohlcv_candles[idx, 2]),
                low=float(context.ohlcv_candles[idx, 3]),
                close=float(context.ohlcv_candles[idx, 4]),
                volume=float(context.ohlcv_candles[idx, 5]),
                sentiment=getattr(context, 'sentiment', None)
            )
            data.append(market_data)
        
        return data
