from datetime import datetime
from typing import Optional

from logger.logger import Logger
from utils.dataclass import SentimentData, Position, PromptContext


class TradingPromptBuilder:
    def __init__(self, logger: Logger):
        self.logger = logger

    def build_prompt(self, context: PromptContext) -> str:
        sections = [
            self._build_header(context.symbol),
            self._build_market_data(context),
            self._build_trading_context(context),
            self._build_technical_analysis(context),
            self._build_market_period_metrics(context),
            self._build_position_management(context),
            self._build_analysis_steps(),
            self._build_decision_template(context.current_position),
            self._build_sentiment_section(context.sentiment)
        ]

        return "\n\n".join(filter(None, sections))

    def _build_header(self, symbol: str) -> str:
        return f"""You are a professional futures crypto trader analyzing {symbol} on 1 hour timeframe. 
                Provide clear, structured analysis with concrete numbers from the data. Consider both 
                long and short opportunities."""

    def _build_market_data(self, context: PromptContext) -> str:
        if context.ohlcv_candles is None or context.ohlcv_candles.size == 0:
            return "MARKET DATA:\nNo OHLCV data available"

        if context.ohlcv_candles.shape[0] < 24:  # Check number of rows
            return "MARKET DATA:\nInsufficient historical data (less than 25 candles)"

        data = "MARKET DATA:\nRaw OHLCV Data (Last 24 Candles):\n"
        data += "Timestamp,Open,High,Low,Close,Volume\n"
        for candle in context.ohlcv_candles[-24:]:
            ts = datetime.fromtimestamp(candle[0] / 1000).strftime('%Y-%m-%d %H:%M')
            data += (
                f"{ts},"
                f"{candle[1]:.4f},"  # Open
                f"{candle[2]:.4f},"  # High
                f"{candle[3]:.4f},"  # Low
                f"{candle[4]:.4f},"  # Close
                f"{candle[5]:.2f}\n"  # Volume
            )
        return data

    def _build_market_period_metrics(self, context: PromptContext) -> str:

        if not context.market_metrics:
            return ""

        lines = ["MARKET PERIOD METRICS:"]
        for period_name, market_period in context.market_metrics.items():
            lines.append(f"\nPeriod Name: {period_name}")
            metrics = market_period.metrics
            for key, value in metrics.items():
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _build_trading_context(self, context: PromptContext) -> str:
        trading_context = f"""
        Trading Context:
        - Current Day: {datetime.now().strftime("%A")}
        - Current Price: ${context.current_price:,.2f}
        - Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        - Timeframe: 1 hour
        - Market Type: Futures"""

        if context.current_position:
            position = context.current_position
            if position.direction == 'LONG':
                price_diff = context.current_price - position.entry_price
            else:
                price_diff = position.entry_price - context.current_price

            position_pl = (price_diff / position.entry_price) * 100
            pl_status = "Profit" if price_diff > 0 else "Loss"

            position_duration = datetime.now() - position.entry_time
            trading_context += f"""
            Active Position:
            - Direction: {position.direction}
            - Entry Price: ${position.entry_price:,.2f}
            - Current Price: ${context.current_price:,.2f}
            - P/L: {position_pl:+.2f}% ({pl_status})
            - Distance to SL: {abs(context.current_price - position.stop_loss):.2f} $
            - Distance to TP: {abs(context.current_price - position.take_profit):.2f} $
            - Duration: {position_duration.total_seconds() / 3600:.1f} hours
            - Stop Loss: ${position.stop_loss:,.2f}
            - Take Profit: ${position.take_profit:,.2f}"""

        return trading_context

    def _build_technical_analysis(self, context: PromptContext) -> str:
        td = context.technical_data

        return f"""
        TECHNICAL ANALYSIS (Default Lengths):
        1. Price Action (1H):
        - Current Price: ${context.current_price:.2f}
        - Rolling VWAP (14): ${td.vwap_5m:.2f}
        - TWAP (14): ${td.twap:.2f}

        2. Momentum Indicators:
        - RSI(14): {td.rsi_5m_14:.1f}
        " RSI Interpretation Guide: <30=Oversold, 30-70=Neutral, >70=Overbought"
        - MACD Line (12,26,9): {td.macd_line:.4f}
        - MACD Signal (12,26,9): {td.macd_signal:.4f}
        - MACD Hist (12,26,9): {td.macd_hist:.4f}
        - Stochastic %K(5,3,3): {td.stoch_k:.1f}
        - Stochastic %D(5,3,3): {td.stoch_d:.1f}
        - Williams %R(14): {td.williams_r:.1f}

        3. Trend Indicators:
        - ADX(14): {td.adx:.1f}
        - +DI(14): {td.plus_di:.1f}
        - -DI(14): {td.minus_di:.1f}
        - Supertrend(7,3.0): {td.supertrend:.2f}
        - Parabolic SAR(0.02,0.2): {td.psar:.2f}

        4. Volatility Analysis:
        - ATR(14): {td.atr_5m_14:.2f}
        - Bollinger Bands(20,2.0): {td.bb_upper:.2f} / {td.bb_middle:.2f} / {td.bb_lower:.2f}

        5. Volume Analysis:
        - MFI(14): {td.mfi_14:.1f}
        - On Balance Volume(14): {td.obv:.0f}
        - Chaikin MF(20): {td.cmf:.4f}
        - Force Index(13): {td.force_index:.0f}

        6. Key Levels:
        - Resistance (3D): ${context.market_metrics['3D'].metrics['highest_price']:,.2f}
        - Support (3D): ${context.market_metrics['3D'].metrics['lowest_price']:,.2f}

        7. Statistical Metrics:
        - Hurst Exponent(20): {td.hurst:.2f}
        - Z-Score(30): {td.zscore:.2f}
        - Kurtosis(30): {td.kurtosis:.2f}
        8. Multi-Timeframe Performance:
           48H: {context.market_metrics['1D'].metrics['price_change']:+.2f}%
           72H: {context.market_metrics['2D'].metrics['price_change']:+.2f}%
           730H (1 month): {context.market_metrics['3D'].metrics['price_change']:+.2f}%"""

    def _build_position_management(self, context: PromptContext) -> str:
        trade_history_str = "\n".join(
            f"- {decision}" for decision in context.trade_history
        ) if context.trade_history else "No recent trades"

        return f"""
        POSITION MANAGEMENT:
        - Active Position Status: {context.current_position if context.current_position else 'None'}
        - Previous Analysis: {context.previous_response}

        Recent Trade History:
        {trade_history_str}"""

    def _build_analysis_steps(self) -> str:
        return """
        ANALYSIS STEPS:
        1. Analyze price action and volume patterns
        2. Evaluate position context and trade history
        3. Check technical signals and divergences
        4. Validate support/resistance levels
        5. Calculate risk/reward scenarios
        6. Target for at least 1% profits
        7. Prioritize trades with 3+ confirming indicators
        8. Validate RSI against Bollinger Band position
        9. When you respond, take into account that you will perform analysis after 1 hour and your current response will be saved for further analysis."""

    def _build_decision_template(self, current_position: Optional[Position]) -> str:
        if current_position:
            return """
            TRADING_DECISION:
            Signal: [CLOSE/HOLD]
            Confidence: [HIGH/MEDIUM/LOW]
            Stop Loss: [Price]
            Take Profit: [Price]

            Note: Signal must be exactly as shown in brackets."""
        else:
            return """
            TRADING_DECISION:
            Signal: [BUY/SELL/HOLD]
            Confidence: [HIGH/MEDIUM/LOW]
            Entry Price: [Current]
            Stop Loss: [Price]
            Take Profit: [Price]
            Position Size: [0.5-5%]

            Note: 
            - BUY = Open long position
            - SELL = Open short position
            - CLOSE = Close current position
            - Signal must be exactly as shown in brackets."""

    def _build_sentiment_section(self, sentiment: Optional[SentimentData]) -> str:
        if not sentiment:
            return ""
        return f"""
        MARKET SENTIMENT:
        - Fear & Greed Index: {sentiment.fear_greed_index}
        - Classification: {sentiment.value_classification}"""