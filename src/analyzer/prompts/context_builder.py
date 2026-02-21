"""
Context building for prompt building system.
Handles building context sections like trading context, sentiment, market data, etc.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

from src.logger.logger import Logger
from ..formatters import MarketFormatter, MarketPeriodFormatter, LongTermFormatter

if TYPE_CHECKING:
    from src.utils.format_utils import FormatUtils


class ContextBuilder:
    """Builds context sections for prompts including trading context, sentiment, and market data."""

    def __init__(
        self,
        timeframe: str = "1h",
        logger: Optional[Logger] = None,
        format_utils: "FormatUtils" = None,
        market_formatter: MarketFormatter = None,
        period_formatter: MarketPeriodFormatter = None,
        long_term_formatter: LongTermFormatter = None,
        timeframe_validator: Any = None
    ):
        # pylint: disable=too-many-arguments
        """Initialize the context builder.

        Args:
            timeframe: Primary timeframe for analysis
            logger: Optional logger instance for debugging
            format_utils: Format utilities (required)
            market_formatter: MarketFormatter instance (required)
            period_formatter: MarketPeriodFormatter instance (required)
            long_term_formatter: LongTermFormatter instance (required)
            timeframe_validator: TimeframeValidator instance (injected)
        """
        self.timeframe = timeframe
        self.logger = logger
        if format_utils is None:
            raise ValueError("format_utils is required for ContextBuilder")
        self.format_utils = format_utils
        if market_formatter is None:
            raise ValueError("market_formatter is required for ContextBuilder")
        self.market_formatter = market_formatter
        if period_formatter is None:
            raise ValueError("period_formatter is required for ContextBuilder")
        self.period_formatter = period_formatter
        if long_term_formatter is None:
            raise ValueError("long_term_formatter is required for ContextBuilder")
        self.long_term_formatter = long_term_formatter
        self.timeframe_validator = timeframe_validator

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

        # Calculate timeframe minutes using injected validator if available, else fallback
        timeframe_minutes = 60
        if self.timeframe_validator:
            timeframe_minutes = self.timeframe_validator.to_minutes(self.timeframe)

        # Calculate time until next candle closes (for intraday timeframes)
        # Exchange candles align to UTC boundaries (00:00, 04:00, 08:00 UTC etc.)
        if timeframe_minutes < 1440:  # Less than 1 day
            total_minutes = current_time.hour * 60 + current_time.minute
            minutes_into_candle = total_minutes % timeframe_minutes
            minutes_until_close = timeframe_minutes - minutes_into_candle

            candle_status = (
                f"\n- Next Candle Close: in {minutes_until_close} minutes"
            )
            candle_status += "\n- Data Quality: All indicators based on CLOSED CANDLES ONLY (professional trading standard)"

        # Get analysis timeframes description
        analysis_timeframes = f"{self.timeframe.upper()}, 1D, 7D, 30D, 365D, and WEEKLY timeframes"

        # Determine day of week and add weekend warning if applicable
        day_of_week = current_time.strftime("%A")
        weekend_note = ""
        if day_of_week in ["Saturday", "Sunday"]:
            weekend_note = (
                "\n        - WEEKEND MODE: Trading volume/liquidity is typically lower. "
                "Be cautious of fakeouts/manipulation."
            )

        # Get market milestones countdown
        milestones = self._get_market_milestones(current_time)

        trading_context = f"""
        ## Trading Context
        - Symbol: {context.symbol}
        - Current Day: {day_of_week} (UTC)
        - Current Price: {context.current_price}
        - Analysis Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC{candle_status}{weekend_note}
        - Primary Timeframe: {self.timeframe} ({timeframe_minutes} min/candle)
        - Analysis Includes: {analysis_timeframes}

        ### Market Milestones (UTC)
{milestones}"""

        return trading_context

    def _get_market_milestones(self, current_time: datetime) -> str:
        """Calculate and format countdowns to major market milestones.

        Args:
            current_time: Current time in UTC

        Returns:
            str: Formatted milestones list
        """
        milestones = []

        def format_delta(delta: timedelta) -> str:
            total_seconds = int(delta.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            return f"{hours}h {minutes}m"

        # 1. Daily Close (00:00 UTC)
        next_daily = (current_time + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        milestones.append(f"        - Daily Close: in {format_delta(next_daily - current_time)} (00:00 UTC)")

        # 2. CME/Service Open (Sunday 22:00 UTC) - Show only on Sunday
        if current_time.weekday() == 6: # Sunday
            cme_open = current_time.replace(hour=22, minute=0, second=0, microsecond=0)
            if current_time < cme_open:
                milestones.append(
                    f"        - CME/Service Open: in {format_delta(cme_open - current_time)} (Today 22:00 UTC)"
                )
            else:
                milestones.append("        - CME/Service Open: Currently Active (since 22:00 UTC)")

        # 3. Weekly Open (Monday 00:00 UTC) - Show on weekend
        if current_time.weekday() >= 5: # Sat or Sun
            days_until_monday = (7 - current_time.weekday()) % 7 or 7
            next_weekly = (current_time + timedelta(days=days_until_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
            milestones.append(
                f"        - Weekly Open: in {format_delta(next_weekly - current_time)} (Mon 00:00 UTC)"
            )

        return "\n".join(milestones) if milestones else "        - No major milestones within 24h"

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
        ## Market Sentiment
        - Current Fear & Greed Index: {sentiment_data.get('fear_greed_index', 'N/A')}
        - Classification: {sentiment_data.get('value_classification', 'N/A')}"""

        if historical_data:
            sentiment_section += "\n\n        ### Historical Fear & Greed (Last 7 days):"
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
        # Default fallback
        base_minutes = 60
        if self.timeframe_validator:
            base_minutes = self.timeframe_validator.to_minutes(self.timeframe)

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
        data = "## Market Data\n"

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
                                        _weekly_macro_indicators: Optional[Dict[str, Any]] = None) -> str:
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
            "### Indicator Changes (Previous → Current):",
            ""
        ]

        # Key indicators to compare (organized by category)
        key_indicators = [
            # Momentum Indicators
            ('rsi', 'RSI'),
            ('macd_line', 'MACD Line'),
            ('macd_signal', 'MACD Signal'),
            ('macd_hist', 'MACD Histogram'),
            ('ppo', 'PPO'),
            ('stoch_k', 'Stochastic %K'),
            ('stoch_d', 'Stochastic %D'),
            ('williams_r', 'Williams %R'),
            ('tsi', 'TSI'),
            ('roc_14', 'ROC'),
            ('mfi', 'MFI'),

            # Trend Indicators
            ('adx', 'ADX'),
            ('plus_di', '+DI'),
            ('minus_di', '-DI'),
            ('trix', 'TRIX'),
            ('vortex_plus', 'Vortex VI+'),
            ('vortex_minus', 'Vortex VI-'),

            # Volume Indicators
            ('obv', 'OBV'),
            ('obv_slope', 'OBV Slope'),
            ('cci', 'CCI'),
            ('cmf', 'Chaikin MF'),

            # Volatility Indicators
            ('atr', 'ATR'),
            ('atr_percent', 'ATR%'),
            ('bb_percent_b', 'BB %B'),
            ('choppiness', 'Choppiness'),

            # Moving Averages
            ('sma_20', '20 SMA'),
            ('sma_50', '50 SMA'),
            ('sma_200', '200 SMA'),

            # Statistical/Trend Quality
            ('linreg_slope', 'LinReg Slope'),
            ('linreg_r2', 'LinReg R²'),
        ]


        # Indicators that oscillate around zero (signs matter more than %, % can be misleading on zero-cross)
        zero_cross_indicators = {
            'macd_line', 'macd_hist', 'macd_signal', 'roc_14', 'cci', 'cmf',
            'trix', 'tsi', 'ppo', 'linreg_slope', 'coppock', 'kst'
        }

        changes = []
        for key, label in key_indicators:
            prev_val = previous_indicators.get(key)
            curr_val = current_indicators.get(key)

            # Skip if either value is missing or invalid
            if prev_val is None or curr_val is None:
                continue

            # Handle array values (take last element)
            if isinstance(prev_val, (list, tuple, np.ndarray)):
                prev_val = prev_val[-1] if len(prev_val) > 0 else None
            if isinstance(curr_val, (list, tuple, np.ndarray)):
                curr_val = curr_val[-1] if len(curr_val) > 0 else None

            if prev_val is None or curr_val is None:
                continue

            try:
                prev_val = float(prev_val)
                curr_val = float(curr_val)

                is_zero_cross_type = key in zero_cross_indicators
                
                line = self._format_indicator_change(label, prev_val, curr_val, is_zero_cross_type)

                if line:
                    changes.append(line)
            except (ValueError, TypeError):
                continue

        # If no significant changes were found, but we did process valid indicators
        if not changes:
            lines.append("No significant indicator changes observed since last analysis.")
        else:
            lines.extend(changes)
            lines.append("")
            lines.append("(Note: Indicators with < 1.0% change or specific zero-cross logic are filtered)")

        lines.append("")
        lines.append("INTERPRETATION: Look for trend continuation (momentum building) vs reversal (divergence, exhaustion).")

        return "\n".join(lines)

    def _format_indicator_change(self, label: str, prev_val: float, curr_val: float,
                                 is_zero_cross_type: bool) -> Optional[str]:
        """Format a single indicator change."""
        # Calculate change
        diff = curr_val - prev_val
        abs_prev = abs(prev_val)

        # Skip if no meaningful change
        if abs(diff) < 0.000001:
            return None

        # Handle zero-crossing oscillators or small basis values
        crossed_zero = prev_val * curr_val < 0

        arrow = "↑" if diff > 0 else "↓"
        sign = "+" if diff > 0 else ""

        # Logic for display:
        if (is_zero_cross_type and crossed_zero) or (abs_prev < 0.1 and is_zero_cross_type):
            change_desc = f"({arrow} zero-cross)" if crossed_zero else f"({arrow} Δ{diff:+.4f})"
            if abs(curr_val) >= 1:
                return f"- {label}: {prev_val:.2f} → {curr_val:.2f} {change_desc}"
            return f"- {label}: {prev_val:.4f} → {curr_val:.4f} {change_desc}"

        if abs_prev > 0.0001:
            change_pct = (diff / abs_prev) * 100

            # Only show if change is significant (at least 1% or it's a zero-cross type with small basis)
            if abs(change_pct) >= 1.0:
                if abs(curr_val) >= 1:
                    return f"- {label}: {prev_val:.2f} → {curr_val:.2f} ({arrow} {sign}{change_pct:.1f}%)"
                return f"- {label}: {prev_val:.4f} → {curr_val:.4f} ({arrow} {sign}{change_pct:.1f}%)"
        else:
            # Tiny basis, just show delta
            return f"- {label}: {prev_val:.4f} → {curr_val:.4f} ({arrow} Δ{diff:+.4f})"
        
        return None
