"""
Template management for prompt building system.
Handles system prompts, response templates, and analysis steps for TRADING DECISIONS.
"""

from typing import Optional, Any

from src.logger.logger import Logger


class TemplateManager:
    """Manages prompt templates, system prompts, and analysis steps for trading decisions."""
    
    def __init__(self, config: Any, logger: Optional[Logger] = None):
        """Initialize the template manager.
        
        Args:
            config: Configuration module providing prompt defaults
            logger: Optional logger instance for debugging
        """
        self.logger = logger
        self.config = config
    
    def build_system_prompt(self, symbol: str, timeframe: str = "1h", language: Optional[str] = None, has_chart_image: bool = False, previous_response: Optional[str] = None) -> str:
        """Build the system prompt for trading decision AI.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            timeframe: Timeframe for analysis (e.g., "1h", "4h", "1d")
            language: Optional language (unused - always English for trading)
            has_chart_image: Whether a chart image is being provided for visual analysis
            previous_response: Previous AI response for context continuity
            
        Returns:
            str: Formatted system prompt
        """
        header_lines = [
            f"You are a professional automated trading system for {symbol} on {timeframe} timeframe.",
            "",
            "CORE PRINCIPLES:",
            "- All data is based on CLOSED CANDLES ONLY (no incomplete candle data)",
            "- Trading decisions must be based on confirmed signals, not speculation",
            "- Risk management is paramount: every trade requires proper stop loss and take profit",
            "- Confidence must match signal strength: only high-confidence trades in strong setups",
            "",
            "YOUR TASK:",
            "Analyze technical indicators, price action, volume, patterns, market sentiment, and news.",
            "Provide a clear trading decision: BUY (long), SELL (short), HOLD (no action), or CLOSE (exit position).",
            "Include specific entry, stop loss, and take profit levels with your reasoning.",
        ]

        if has_chart_image:
            cfg_limit = int(self.config.AI_CHART_CANDLE_LIMIT)
            header_lines.extend([
                "",
                f"CHART ANALYSIS:",
                f"A chart image (~{cfg_limit} candlesticks) is provided for visual pattern recognition.",
                "Integrate chart patterns with numerical indicators. Only report clear, well-formed patterns.",
            ])
        
        # Add previous response context if available
        if previous_response:
            header_lines.extend([
                "",
                "PREVIOUS ANALYSIS CONTEXT:",
                "Your last analysis reasoning (for continuity):",
                previous_response,
                "",
                "Use this context to maintain consistency in your analysis approach.",
            ])

        return "\n".join(header_lines)
    
    def build_response_template(self, has_chart_analysis: bool = False) -> str:
        """Build the response template for trading decision output.
        
        Args:
            has_chart_analysis: Whether chart image analysis is available
            
        Returns:
            str: Formatted response template
        """
        response_template = '''RESPONSE FORMAT:

Provide analysis reasoning (2-4 paragraphs: technical indicators, patterns, market context), then output JSON:

```json
{
    "analysis": {
        "signal": "BUY|SELL|HOLD|CLOSE",
        "confidence": 0-100,
        "entry_price": number,
        "stop_loss": number,
        "take_profit": number,
        "position_size": 0.0-1.0,
        "reasoning": "1-2 sentence summary",
        "key_levels": {"support": [level1, level2], "resistance": [level1, level2]},
        "trend": {"direction": "BULLISH|BEARISH|NEUTRAL", "strength": 0-100, "timeframe_alignment": "ALIGNED|MIXED|DIVERGENT"},
        "risk_reward_ratio": number
    }
}
```

TRADING SIGNALS & CONFIDENCE:
- BUY (70-100 confidence): Strong bullish trend (ADX >25, aligned SMAs) + 3+ indicator confluence + volume confirmation + clear SL below support, TP above resistance + minimum 2:1 R/R
- SELL (70-100 confidence): Strong bearish trend + 3+ indicator confluence + volume confirmation + clear SL above resistance, TP below support + minimum 2:1 R/R  
- HOLD (any confidence <70): Mixed signals, weak trend, conflicting indicators, low volume, or insufficient setup quality. Patience over forced trades.
- CLOSE: Exit position when SL/TP hit, signal reversal, or thesis invalidated

RISK MANAGEMENT (Stop Loss & Take Profit):
LONG trades:
- SL: Below swing low + 1x ATR buffer (max 2-3% from entry) | Example: Entry $100, Swing Low $97, ATR $1 → SL $96
- TP: Key resistance levels, Fibonacci (0.618/0.786/1.0), previous highs | Multiple targets: TP1=1.5R, TP2=2.5R, TP3=3.5R

SHORT trades:  
- SL: Above swing high + 1x ATR buffer (max 2-3% from entry) | Example: Entry $100, Swing High $103, ATR $1 → SL $104
- TP: Key support levels, Fibonacci (0.382/0.236/0.0), previous lows | Multiple targets: TP1=1.5R, TP2=2.5R, TP3=3.5R

Mandatory: All trades require stops based on technical levels (not arbitrary %), accounting for ATR volatility, positioned to invalidate thesis if hit. Minimum 1.5:1 R/R (prefer 2:1+). Scale out at multiple targets (50%/30%/20%).'''
        
        return response_template
    
    def build_analysis_steps(self, symbol: str, has_advanced_support_resistance: bool = False, has_chart_analysis: bool = False, available_periods: dict = None) -> str:
        """Build analysis steps instructions for the AI model.
        
        Args:
            symbol: Trading symbol being analyzed
            has_advanced_support_resistance: Whether advanced S/R indicators are detected
            has_chart_analysis: Whether chart image analysis is available (Google AI only)
            available_periods: Dict of period names to candle counts (e.g., {"12h": 2, "24h": 4, "3d": 12, "7d": 28})
            
        Returns:
            str: Formatted analysis steps
        """
        # Get the base asset for market comparisons
        analyzed_base = symbol.split('/')[0] if '/' in symbol else symbol
        
        # Build dynamic timeframe description based on available periods
        if available_periods:
            period_names = list(available_periods.keys())
            timeframe_desc = f"Analyze the provided Multi-Timeframe Price Summary periods: {', '.join(period_names)}"
        else:
            timeframe_desc = "Analyze the provided Multi-Timeframe Price Summary periods (dynamically calculated based on your analysis timeframe)"
        
        analysis_steps = f"""
ANALYSIS STEPS (use findings to determine trading signal):

1. MULTI-TIMEFRAME ASSESSMENT:
   {timeframe_desc} | Compare short vs multi-day vs long-term (30d+, 365d) | Review weekly macro (200-week SMA, institutions) | Identify alignment (strong signal) vs divergence (caution)

2. TECHNICAL INDICATORS:
   Momentum: RSI (<30/>70), MACD (crosses, histogram) | Trend: ADX (>25), DI+/DI- | Volatility: ATR, Bollinger Bands | Volume: MFI, OBV, Force Index | SMAs: 20/50/200 crosses | Advanced: TSI, Vortex, PFE, RMI, Ultimate, Supertrend | Assess confluence (strong) vs divergence (weak)

3. PATTERN RECOGNITION:
   Chart patterns (wedges, triangles, H&S, double tops/bottoms) | Divergences (price vs RSI/MACD) | Candlesticks (engulfing, doji, hammer, shooting star) | Fibonacci levels (50-period, pullback/extension zones) | Overbought/oversold extremes | Prioritize RECENT patterns

4. SUPPORT/RESISTANCE:
   Map key levels across timeframes | Historical reaction zones (multiple touches) | Technical confluences (S/R + Fib + SMA) | Volume profile (high nodes) | Calculate risk/reward for SL/TP placement

5. MARKET CONTEXT:
   Market Overview (global cap, dominance)"""
        
        if "BTC" not in analyzed_base:
            analysis_steps += "\n   - Compare performance relative to BTC (correlation/divergence)"
        
        if "ETH" not in analyzed_base:
            analysis_steps += "\n   - Compare performance relative to ETH if relevant"
        
        analysis_steps += """
 | Fear & Greed Index (extremes) | Asset alignment with market | Relevant events and impact | Assess if context supports or contradicts technicals

6. NEWS & SENTIMENT:
   Recent asset news | Market-moving events/announcements | Sentiment evaluation | News-to-price action correlation | Institutional/corporate developments | Regulatory impacts | Identify news that could override technical signals

7. STATISTICAL ANALYSIS:
   Z-Score (extremes may revert) | Kurtosis (fat tails = extreme move risk) | Hurst Exponent (>0.5 trending, <0.5 mean-reverting) | Distribution anomalies | Volatility cycles | Assess continuation vs reversal probability"""
        
        # Add chart analysis steps only if chart images are available
        step_number = 8
        if has_chart_analysis:
            cfg_limit = int(self.config.AI_CHART_CANDLE_LIMIT)

            analysis_steps += f"""

{step_number}. CHART ANALYSIS (~{cfg_limit} candlesticks):
   Scan for patterns: H&S, Double Tops/Bottoms, Wedges, Triangles, Flags/Pennants (3-5% min range, state "None detected" if unclear) | Validate each: name, bias, range, confidence, key levels | Candlestick patterns (doji, hammer, engulfing) & momentum shifts | Cross-validate visual vs numerical indicators (prioritize numerical if ambiguous) | Confirm or contradict?"""
            step_number += 1
        
        analysis_steps += f"""

{step_number}. SYNTHESIS:
   Trend direction & strength? | Multiple indicator confluence? | Key SL/TP levels? | Risk/reward ratio? | Confidence level? | Trade invalidation triggers?

IMPORTANT: ALL data uses CLOSED CANDLES ONLY (no incomplete data). Decisions based on confirmed price action, preventing premature entries on unconfirmed signals."""
        
        if has_advanced_support_resistance:
            analysis_steps += """

ADVANCED S/R: Volume-weighted pivots [Pivot=(H+L+C)/3, S1=2P-H, R1=2P-L] with consecutive touches, above-average volume filters. Only strong levels provided."""

        return analysis_steps
