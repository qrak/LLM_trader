"""
Context building for prompt building system.
Handles building context sections like trading context, sentiment, market data, etc.
"""

import numpy as np
from datetime import datetime
from typing import Any, Dict, Optional

from src.logger.logger import Logger
from src.utils.timeframe_validator import TimeframeValidator
from ..formatting.indicator_formatter import IndicatorFormatter


class ContextBuilder:
    """Builds context sections for prompts including trading context, sentiment, and market data."""
    
    def __init__(self, timeframe: str = "1h", logger: Optional[Logger] = None, format_utils=None, data_processor=None):
        """Initialize the context builder.
        
        Args:
            timeframe: Primary timeframe for analysis
            logger: Optional logger instance for debugging
        """
        self.timeframe = timeframe
        self.logger = logger
        self.format_utils = format_utils
        self.formatter = IndicatorFormatter(logger, format_utils, data_processor)
    
    def build_trading_context(self, context) -> str:
        """Build trading context section with current market information.
        
        Args:
            context: Analysis context containing symbol and current price
            
        Returns:
            str: Formatted trading context section
        """
        # Get the current time to understand candle formation
        current_time = datetime.now()
        
        # Create candle status message dynamically based on timeframe
        candle_status = ""
        timeframe_minutes = TimeframeValidator.to_minutes(self.timeframe)
        
        # Calculate current position within the candle (only for intraday timeframes)
        if timeframe_minutes < 1440:  # Less than 1 day
            total_minutes = current_time.hour * 60 + current_time.minute
            minutes_into_candle = total_minutes % timeframe_minutes
            
            candle_progress = (minutes_into_candle / timeframe_minutes) * 100
            candle_status = (
                f"\n- Current Candle: {minutes_into_candle}/{timeframe_minutes} minutes "
                f"({candle_progress:.1f}% complete)"
            )
            candle_status += f"\n- Analysis Note: Technical indicators calculated including the current incomplete candle"
        
        # Get analysis timeframes description
        analysis_timeframes = f"{self.timeframe.upper()}, 1D, 7D, 30D, 365D, and WEEKLY timeframes"
        
        trading_context = f"""
        TRADING CONTEXT:
        - Symbol: {context.symbol if hasattr(context, 'symbol') else 'BTC/USDT'}
        - Current Day: {self.format_utils.format_current_time("%A")}
        - Current Price: {context.current_price}
        - Analysis Time: {self.format_utils.format_current_time('%Y-%m-%d %H:%M:%S')}{candle_status}
        - Primary Timeframe: {self.timeframe}
        - Analysis Includes: {analysis_timeframes}"""
        
        return trading_context
    
    def build_sentiment_section(self, sentiment_data: Optional[Dict[str, Any]]) -> str:
        """Build sentiment analysis section.
        
        Args:
            sentiment_data: Sentiment data including fear & greed index
            
        Returns:
            str: Formatted sentiment section
        """
        if not sentiment_data:
            return ""
        
        historical_data = sentiment_data.get('historical', [])
        
        sentiment_section = f"""
        MARKET SENTIMENT:
        - Current Fear & Greed Index: {sentiment_data.get('fear_greed_index', 'N/A')}
        - Classification: {sentiment_data.get('value_classification', 'N/A')}"""
        
        if historical_data:
            sentiment_section += "\n\n    Historical Fear & Greed (Last 7 days):"
            for day in historical_data:
                # Use centralized timestamp formatting
                if isinstance(day['timestamp'], datetime):
                    date_str = day['timestamp'].strftime('%Y-%m-%d')
                elif isinstance(day['timestamp'], (int, float)):
                    date_str = self.format_utils.format_date_from_timestamp(day['timestamp'])
                else:
                    date_str = str(day['timestamp'])
                sentiment_section += f"\n    - {date_str}: {day['value']} ({day['value_classification']})"
        
        return sentiment_section
    
    def _calculate_period_candles(self) -> Dict[str, int]:
        """Calculate candle counts for standard periods based on current timeframe.
        
        Returns:
            Dict mapping period names to candle counts needed (filters out periods smaller than timeframe)
        """
        base_minutes = TimeframeValidator.to_minutes(self.timeframe)
        
        period_targets = {
            "4h": 4 * 60,      # 240 minutes
            "12h": 12 * 60,    # 720 minutes
            "24h": 24 * 60,    # 1440 minutes
            "3d": 72 * 60,     # 4320 minutes
            "7d": 168 * 60     # 10080 minutes
        }
        
        # Calculate candles needed and filter out periods smaller than base timeframe
        result = {}
        for name, target_mins in period_targets.items():
            candles_needed = target_mins // base_minutes
            # Only include periods that need at least 1 candle (i.e., period >= timeframe)
            if candles_needed >= 1:
                result[name] = candles_needed
        
        return result
    
    def build_market_data_section(self, ohlcv_candles: np.ndarray) -> str:
        """Build market data section with multi-timeframe price summary.
        
        Args:
            ohlcv_candles: OHLCV candle data array
            
        Returns:
            str: Formatted market data section
        """
        if ohlcv_candles is None or ohlcv_candles.size == 0:
            return "MARKET DATA:\nNo OHLCV data available"

        if ohlcv_candles.shape[0] < 24:
            return "MARKET DATA:\nInsufficient historical data (less than 25 candles)"

        available_candles = ohlcv_candles.shape[0]
        data = "MARKET DATA:\n"

        # Keep multi-timeframe price summary if desired
        if available_candles >= 100:
            last_close = float(ohlcv_candles[-1, 4])
            periods = self._calculate_period_candles()

            data += f"\nMulti-Timeframe Price Summary (Based on {self.timeframe} candles):\n"
            for period_name, candle_count in periods.items():
                if (candle_count + 1) <= available_candles:
                    period_start = float(ohlcv_candles[-(candle_count + 1), 4])
                    change_pct = ((last_close / period_start) - 1) * 100
                    high = max([float(candle[2]) for candle in ohlcv_candles[-candle_count:]])
                    low = min([float(candle[3]) for candle in ohlcv_candles[-candle_count:]])
                    
                    # Format very small numbers using the imported fmt function
                    high_formatted = self.format_utils.fmt(high)
                    low_formatted = self.format_utils.fmt(low)
                    
                    data += f"{period_name}: {change_pct:.2f}% change | High: {high_formatted} | Low: {low_formatted}\n"

        return data if data != "MARKET DATA:\n" else ""
    
    def build_market_period_metrics_section(self, market_metrics: Optional[Dict[str, Any]]) -> str:
        """Build market period metrics section.
        
        Args:
            market_metrics: Market metrics data
            
        Returns:
            str: Formatted market period metrics section
        """
        if not market_metrics:
            return ""
        
        return self.formatter.format_market_period_metrics(market_metrics)
    
    def build_long_term_analysis_section(self, long_term_data: Optional[Dict[str, Any]], 
                                        current_price: Optional[float],
                                        weekly_macro_indicators: Optional[Dict[str, Any]] = None) -> str:
        """Build long-term analysis section (daily only - weekly handled by PromptBuilder).
        
        Args:
            long_term_data: Long-term historical data (daily)
            current_price: Current asset price
            weekly_macro_indicators: Weekly macro trend data (passed through, not used here)
            
        Returns:
            str: Formatted long-term analysis section (daily only)
        """
        if not long_term_data:
            return ""
        
        return self.formatter.format_long_term_analysis(long_term_data, current_price)
    
    def build_coin_details_section(self, coin_details: Optional[Dict[str, Any]]) -> str:
        """Build cryptocurrency details section.
        
        Args:
            coin_details: Coin details data including description, taxonomy, and ratings
            
        Returns:
            str: Formatted coin details section
        """
        if not coin_details:
            return ""
        
        return self.formatter.format_coin_details_section(coin_details)
