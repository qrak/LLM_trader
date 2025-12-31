"""
Context building for prompt building system.
Handles building context sections like trading context, sentiment, market data, etc.
"""

import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.logger.logger import Logger
from src.utils.timeframe_validator import TimeframeValidator
from ..formatters import MarketFormatter, MarketPeriodFormatter, LongTermFormatter


class ContextBuilder:
    """Builds context sections for prompts including trading context, sentiment, and market data."""
    
    def __init__(
        self,
        timeframe: str = "1h",
        logger: Optional[Logger] = None,
        format_utils=None,
        data_processor=None,
        market_formatter: Optional[MarketFormatter] = None,
        period_formatter: Optional[MarketPeriodFormatter] = None,
        long_term_formatter: Optional[LongTermFormatter] = None
    ):
        """Initialize the context builder.
        
        Args:
            timeframe: Primary timeframe for analysis
            logger: Optional logger instance for debugging
            format_utils: Format utilities
            data_processor: Data processing utilities
            market_formatter: MarketFormatter instance (for coin details, ticker, etc.)
            period_formatter: MarketPeriodFormatter instance (for period metrics)
            long_term_formatter: LongTermFormatter instance (for long-term analysis)
        """
        self.timeframe = timeframe
        self.logger = logger
        self.format_utils = format_utils
        self.market_formatter = market_formatter
        self.period_formatter = period_formatter
        self.long_term_formatter = long_term_formatter
    
    def build_trading_context(self, context) -> str:
        """Build trading context section with current market information.
        
        Args:
            context: Analysis context containing symbol and current price
            
        Returns:
            str: Formatted trading context section
        """
        # Get the current time to understand candle formation
        current_time = datetime.now(timezone.utc)  # Use UTC to match exchange candle boundaries
        
        # Create candle status message dynamically based on timeframe
        candle_status = ""
        timeframe_minutes = TimeframeValidator.to_minutes(self.timeframe)
        
        # Calculate time until next candle closes (for intraday timeframes)
        # Exchange candles align to UTC boundaries (00:00, 04:00, 08:00 UTC etc.)
        if timeframe_minutes < 1440:  # Less than 1 day
            total_minutes = current_time.hour * 60 + current_time.minute
            minutes_into_candle = total_minutes % timeframe_minutes
            minutes_until_close = timeframe_minutes - minutes_into_candle
            
            candle_status = (
                f"\n- Next Candle Close: in {minutes_until_close} minutes"
            )
            candle_status += f"\n- Data Quality: All indicators based on CLOSED CANDLES ONLY (professional trading standard)"
        
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
                    high = max(float(candle[2]) for candle in ohlcv_candles[-candle_count:])
                    low = min(float(candle[3]) for candle in ohlcv_candles[-candle_count:])
                    
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
        
        return self.period_formatter.format_market_period_metrics(market_metrics)
    
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
        
        return self.long_term_formatter.format_long_term_analysis(long_term_data, current_price)
    
    def build_coin_details_section(self, coin_details: Optional[Dict[str, Any]]) -> str:
        """Build cryptocurrency details section.
        
        Args:
            coin_details: Coin details data including description, taxonomy, and ratings
            
        Returns:
            str: Formatted coin details section
        """
        if not coin_details:
            return ""
        
        return self.market_formatter.format_coin_details_section(coin_details)
    
    def build_previous_indicators_section(self, previous_indicators: Dict[str, Any], current_indicators: Dict[str, Any]) -> str:
        """Build comparison section showing how key indicators changed since last analysis.
        
        Args:
            previous_indicators: Previous technical indicator values
            current_indicators: Current technical indicator values
            
        Returns:
            str: Formatted previous indicators comparison section
        """
        if not previous_indicators or not current_indicators:
            return ""
        
        lines = [
            "INDICATOR CHANGES (Previous → Current):",
            ""
        ]
        
        # Key indicators to compare (in priority order)
        key_indicators = [
            ('rsi', 'RSI'),
            ('macd_line', 'MACD Line'),
            ('macd_signal', 'MACD Signal'),
            ('macd_hist', 'MACD Histogram'),
            ('adx', 'ADX'),
            ('plus_di', '+DI'),
            ('minus_di', '-DI'),
            ('stoch_k', 'Stochastic %K'),
            ('stoch_d', 'Stochastic %D'),
            ('mfi', 'MFI'),
            ('obv', 'OBV'),
            ('atr', 'ATR'),
            ('sma_20', '20 SMA'),
            ('sma_50', '50 SMA'),
            ('sma_200', '200 SMA'),
        ]
        

        changes = []
        for key, label in key_indicators:
            prev_val = previous_indicators.get(key)
            curr_val = current_indicators.get(key)
            
            # Skip if either value is missing or invalid
            if prev_val is None or curr_val is None:
                continue
            
            # Handle array values (take last element)
            if isinstance(prev_val, (list, tuple)):
                prev_val = prev_val[-1] if len(prev_val) > 0 else None
            if isinstance(curr_val, (list, tuple)):
                curr_val = curr_val[-1] if len(curr_val) > 0 else None
            
            if prev_val is None or curr_val is None:
                continue
            
            try:
                prev_val = float(prev_val)
                curr_val = float(curr_val)
                
                # Calculate change
                if abs(prev_val) > 0.0001:  # Avoid division by tiny numbers
                    change_pct = ((curr_val - prev_val) / abs(prev_val)) * 100
                    
                    if abs(change_pct) >= 1.0:
                        arrow = "↑" if change_pct > 0 else "↓"
                        sign = "+" if change_pct > 0 else ""
                        
                        # Format values appropriately
                        line = ""
                        if abs(curr_val) >= 1:
                            line = f"- {label}: {prev_val:.2f} → {curr_val:.2f} ({arrow} {sign}{change_pct:.1f}%)"
                        else:
                            line = f"- {label}: {prev_val:.4f} → {curr_val:.4f} ({arrow} {sign}{change_pct:.1f}%)"
                        
                        changes.append(line)

            except (ValueError, TypeError):
                continue
        
        # If no significant changes were found, but we did process valid indicators
        if not changes:
            lines.append("No significant indicator changes (< 1.0%) observed since last analysis.")
        else:
            lines.extend(changes)
            lines.append("")
            lines.append("(Note: Indicators with < 1.0% change are omitted)")
            
        lines.append("")
        lines.append("INTERPRETATION: Look for trend continuation (momentum building) vs reversal (divergence, exhaustion).")
        
        return "\n".join(lines)
