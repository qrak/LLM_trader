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
    
    def build_system_prompt(self, symbol: str, timeframe: str = "1h", language: Optional[str] = None, has_chart_image: bool = False) -> str:
        """Build the system prompt for trading decision AI.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            timeframe: Timeframe for analysis (e.g., "1h", "4h", "1d")
            language: Optional language (unused - always English for trading)
            has_chart_image: Whether a chart image is being provided for visual analysis
            
        Returns:
            str: Formatted system prompt
        """
        header_lines = [
            f"You are an automated trading system analyzing {symbol} on {timeframe} timeframe.",
            "Your task is to analyze all provided technical indicators, market data, patterns, and news to make a trading decision.",
            "",
            "ANALYSIS APPROACH:",
            "- Thoroughly analyze ALL provided data: technical indicators, price action, volume, sentiment, news",
            "- Identify trend direction and strength from multiple indicators (ADX, MACD, RSI, SMAs)",
            "- Detect chart patterns, divergences, and key price levels",
            "- Consider market context (Fear & Greed, BTC correlation, broader market)",
            "- Evaluate risk/reward based on support/resistance levels",
            "",
            "OUTPUT REQUIREMENT:",
            "- After your analysis, output a JSON trading decision",
            "- Be decisive: BUY, SELL, HOLD, or CLOSE",
            "- Provide specific price levels for stop_loss and take_profit",
        ]

        if has_chart_image:
            cfg_limit = int(self.config.AI_CHART_CANDLE_LIMIT)
            header_lines.extend([
                "",
                f"A chart image (~{cfg_limit} candlesticks) is provided. Integrate visual pattern analysis with numerical indicators.",
                "Stay conservative - only report patterns that are clear and well-formed.",
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

First, provide your analysis reasoning (2-4 paragraphs covering key findings from technical indicators, patterns, and market context).

Then output your trading decision in this exact JSON format:
```json
{
    "analysis": {
        "signal": "BUY|SELL|HOLD|CLOSE",
        "confidence": 0-100,
        "entry_price": number,
        "stop_loss": number,
        "take_profit": number,
        "position_size": 0.0-1.0,
        "reasoning": "1-2 sentence summary of why this decision",
        "key_levels": {
            "support": [level1, level2],
            "resistance": [level1, level2]
        },
        "trend": {
            "direction": "BULLISH|BEARISH|NEUTRAL",
            "strength": 0-100,
            "timeframe_alignment": "ALIGNED|MIXED|DIVERGENT"
        },
        "risk_reward_ratio": number
    }
}
```

SIGNAL DEFINITIONS:
- BUY: Open LONG position (expect price to rise)
- SELL: Open SHORT position (expect price to fall)  
- HOLD: No action - wait for clearer setup or let existing position run
- CLOSE: Close existing position (hit target, stop, or reversal signal)

DECISION CRITERIA:
- High confidence (70+): Strong trend + indicator confluence + volume confirmation
- Medium confidence (50-70): Good setup but some mixed signals
- Low confidence (<50): Unclear - prefer HOLD

STOP LOSS PLACEMENT:
- For BUY: Below recent swing low or key support
- For SELL: Above recent swing high or key resistance
- Consider ATR for volatility-adjusted stops

TAKE PROFIT PLACEMENT:
- Target key resistance (for BUY) or support (for SELL)
- Aim for minimum 1.5:1 risk/reward ratio
- Consider Fibonacci extensions and previous highs/lows'''
        
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
ANALYSIS STEPS:
Follow these steps to analyze the market. Use findings to determine your trading signal.

1. MULTI-TIMEFRAME ASSESSMENT:
   - {timeframe_desc}
   - Compare shorter periods vs multi-day periods vs long-term (30d+, 365d) price action
   - Review weekly macro trend indicators if provided (200-week SMA, institutional positioning)
   - Identify alignment or divergence across different timeframes
   - KEY QUESTION: Are timeframes aligned (strong signal) or divergent (caution)?

2. TECHNICAL INDICATOR ANALYSIS:
   - Evaluate core momentum: RSI (<30 oversold, >70 overbought), MACD (signal crosses, histogram)
   - Assess trend strength: ADX (>25 = trending), DI+/DI- (direction)
   - Check volatility: ATR (range expansion/contraction), Bollinger Bands (squeeze/breakout)
   - Analyze volume: MFI (money flow), OBV (accumulation/distribution), Force Index
   - SMA relationships: Price vs 20/50/200 SMA, golden/death crosses
   - Advanced indicators: TSI, Vortex, PFE, RMI, Ultimate Oscillator, Supertrend
   - KEY QUESTION: Do indicators show confluence (strong) or divergence (weak)?

3. PATTERN RECOGNITION:
   - Identify chart patterns: wedges, triangles, H&S, double tops/bottoms
   - Detect divergences: Price vs RSI, Price vs MACD (bullish/bearish divergence)
   - Look for candlestick patterns: engulfing, doji, hammer, shooting star
   - Note Fibonacci levels and harmonic patterns if visible
   - Identify overbought/oversold extremes across indicators
   - Prioritize RECENT patterns over older ones
   - KEY QUESTION: What patterns suggest about next likely move?

4. SUPPORT/RESISTANCE & KEY LEVELS:
   - Map key price levels from all timeframes
   - Identify historical price reaction zones (multiple touches)
   - Determine areas with technical confluences (S/R + Fib + SMA)
   - Note volume profile levels (high volume nodes)
   - Calculate risk/reward from current price to key levels
   - KEY QUESTION: Where are the best stop loss and take profit levels?

5. MARKET CONTEXT:
   - Reference Market Overview data (global market cap, dominance)"""
        
        if "BTC" not in analyzed_base:
            analysis_steps += "\n   - Compare performance relative to BTC (correlation/divergence)"
        
        if "ETH" not in analyzed_base:
            analysis_steps += "\n   - Compare performance relative to ETH if relevant"
        
        analysis_steps += """
   - Consider Fear & Greed Index (extreme fear = potential buy, extreme greed = caution)
   - Analyze if asset is aligned with or diverging from general market
   - Note relevant market events and their historical impact
   - KEY QUESTION: Does market context support or contradict your technical view?

6. NEWS & SENTIMENT ANALYSIS:
   - Summarize relevant recent news about the asset
   - Identify potential market-moving events or announcements
   - Evaluate overall sentiment from news coverage
   - Connect news events to recent price action
   - Note institutional actions or corporate developments
   - Identify regulatory news that might impact the asset
   - KEY QUESTION: Any news that could override technical signals?

7. STATISTICAL ANALYSIS:
   - Evaluate Z-Score (deviation from mean - extreme values may revert)
   - Consider Kurtosis (fat tails = higher risk of extreme moves)
   - Hurst Exponent (>0.5 = trending, <0.5 = mean-reverting)
   - Note abnormal distribution patterns in price/volume
   - Assess volatility cycles (expansion/contraction phases)
   - KEY QUESTION: What do statistics say about probability of continuation vs reversal?"""
        
        # Add chart analysis steps only if chart images are available
        step_number = 8
        if has_chart_analysis:
            cfg_limit = int(self.config.AI_CHART_CANDLE_LIMIT)

            analysis_steps += f"""

{step_number}. CHART PATTERN ANALYSIS & VISUAL INTEGRATION:
   CHART CONTEXT:
   - Review the provided chart image (~{cfg_limit} candlesticks)
   
   VISUAL PATTERN DETECTION:
   - Scan for classic patterns: Head and Shoulders, Double Tops/Bottoms, Wedges, Triangles, Flags/Pennants
   - Only report patterns that are clear and well-formed (3-5% minimum price range)
   - If no clear patterns visible, state "No clear classic patterns detected"
   
   PATTERN VALIDATION:
   - For each pattern found, note:
     * Pattern name and directional bias
     * Price range and approximate location
     * Confidence level (high/medium/low)
     * Key levels (neckline, breakout points)
   
   CANDLESTICK ANALYSIS:
   - Identify meaningful candle patterns (doji, hammer, engulfing)
   - Note momentum shifts through candle size/color changes
   
   VISUAL-NUMERICAL INTEGRATION:
   - Validate visual patterns against numerical indicators
   - If chart contradicts indicators, note the discrepancy
   - Prioritize numerical data over ambiguous visual signals
   
   KEY QUESTION: Do visual patterns confirm or contradict indicator analysis?"""
            step_number += 1
        
        analysis_steps += f"""

{step_number}. TRADING DECISION SYNTHESIS:
   Synthesize all analysis into your trading decision:
   - What is the predominant trend direction and strength?
   - Are multiple indicators confirming the same signal?
   - What are the key support/resistance levels for stop loss and take profit?
   - What is the risk/reward ratio?
   - What confidence level does the analysis support?
   - What could invalidate this trade setup?

TECHNICAL INDICATORS NOTE:
- Current incomplete candle IS INCLUDED in all technical indicator calculations
- This provides real-time assessment as the candle progresses
- Indicator values will update as price action continues"""
        
        if has_advanced_support_resistance:
            analysis_steps += """

ADVANCED SUPPORT/RESISTANCE REFERENCE:
- Volume-weighted pivot points with strength thresholds
- Pivot = (High + Low + Close) / 3
- S1 = (2 * Pivot) - High, R1 = (2 * Pivot) - Low
- Tracks consecutive touches to measure level strength
- Filters for above-average volume at reaction points
- Returns ONLY strong S/R levels meeting all criteria"""

        return analysis_steps
