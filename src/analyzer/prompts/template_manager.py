"""
Template management for prompt building system.
Handles system prompts, response templates, and analysis steps for TRADING DECISIONS.
"""

import re
from datetime import datetime, timezone
from typing import Optional, Any, Dict

from src.logger.logger import Logger


class TemplateManager:
    """Manages prompt templates, system prompts, and analysis steps for trading decisions."""
    
    def __init__(self, config: Any, logger: Optional[Logger] = None, timeframe_validator: Any = None):
        """Initialize the template manager.
        
        Args:
            config: Configuration module providing prompt defaults
            logger: Optional logger instance for debugging
            timeframe_validator: TimeframeValidator instance (injected)
        """
        self.logger = logger
        self.config = config
        self.timeframe_validator = timeframe_validator
    
    def build_system_prompt(self, symbol: str, timeframe: str = "1h", previous_response: Optional[str] = None, performance_context: Optional[str] = None, brain_context: Optional[str] = None, last_analysis_time: Optional[str] = None) -> str:
        """Build the system prompt for trading decision AI.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            timeframe: Timeframe for analysis (e.g., "1h", "4h", "1d")
            previous_response: Previous AI response for context continuity (JSON stripped)
            performance_context: Recent trading history and performance metrics
            brain_context: Distilled trading insights from closed trades
            last_analysis_time: Formatted timestamp of last analysis (e.g., "2025-12-26 14:30:00")
            
        Returns:
            str: Formatted system prompt
        """
        # Validate dependency needed for relevance window calculation
        if self.timeframe_validator is None:
             # Fallback if somehow not injected, though it should be. 
             # Ideally raise error, but to avoid crash in prod if missed, maybe safe default or error?
             # Given strict DI rules, raising error or logging warning is better.
             pass

        header_lines = [
            f"You are an Institutional-Grade Crypto Trading Analyst managing {symbol} on {timeframe} timeframe.",
            "Analyze technical indicators, price action, volume, patterns, provided chart if available, market sentiment, and news.",
            "Provide exactly ONE decision (BUY/SELL/HOLD/CLOSE/UPDATE) with entry, stop loss, and take profit level reasoning.",
            "",
            "## Analytical Framework (Chain of Thought)",
            "Step through: (1) Market Structure phase, (2) Timeframe alignment, (3) Momentum/volatility state,",
            "(4) Microstructure bias, (5) Risk/reward assessment, (6) Historical Evidence (Trading Brain insights).",
            "",
        ]
        
        if last_analysis_time:
            header_lines.extend([
                "## Temporal Context",
                f"Last analysis: {last_analysis_time} UTC",
                "",
            ])
        
        header_lines.extend([
            "## Core Principles",
            "- Indicators calculated on CLOSED CANDLES ONLY (no repaint). Current price is REAL-TIME (incomplete candle).",
            "- Decisions must be based on CONFIRMED signals, not speculation.",
            "- Risk management is paramount: SL and TP required for every trade.",
            "- Confidence must match signal strength: >70 required for trades (strong setups only).",
            "- MAXIMIZE PROFIT: Learn from past trades, avoid repeated mistakes, improve win rate.",
            "",
        ])
        
        # Add performance context if available
        if performance_context:
            header_lines.extend([
                "",
                performance_context.strip(),
                "",
                "",
                "## Profit Maximization Strategy",
                "- LEARN from closed trades: Why did stops get hit? Were entries premature? Was trend strength misjudged?",
                "- IMPROVE win rate: Only trade when multiple factors align strongly (3+ confluences)",
                "- AVOID repeated mistakes: If recent trades failed due to weak setups, demand stronger confirmation",
                "- HOLD discipline: Better to miss a trade than force a weak setup (HOLD is valid when confidence <70)",
                "- UPDATE positions actively: Move SL to breakeven after 1:1 or 1.5:1 gain, trail stops on strong trends, adjust TP if momentum extends",
                "- CLOSE proactively: Don't wait for SL if market structure breaks, trend reverses, or thesis invalidates",
                "- ADAPT to performance: If win rate is low, increase entry standards and risk/reward requirements",
            ])
        
        if brain_context:
            header_lines.extend([
                "",
                brain_context.strip(),
            ])
        
        # Add previous response context if available (strip JSON to save tokens)
        if previous_response:
            # Extract only text reasoning, exclude JSON block
            text_reasoning = re.split(r'```json', previous_response, flags=re.IGNORECASE)[0].strip()
            if text_reasoning:
                # Calculate window duration safely using injected validator
                window_minutes = 120 # Default fallback
                if self.timeframe_validator:
                    try:
                        window_minutes = self.timeframe_validator.to_minutes(timeframe) * 2
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"Failed to calculate relevance window for {timeframe}: {e}")
                
                header_lines.extend([
                    "",
                    "## PREVIOUS ANALYSIS CONTEXT",
                    "Your last analysis reasoning (for continuity):",
                    text_reasoning,
                    "",
                    "### DETERMINISTIC TIME CHECK",
                    f"- **Current Time**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
                    "- **Relevance**: PREVIOUS reasoning MUST be verified against CURRENT data. If previous claims (e.g., 'approaching' events) contradict the current clock or were based on now-outdated milestones, you MUST ignore or correct them.",
                    f"- **Relevance Window**: Only consider an event 'imminent' if it occurs within the next 2 full candles (Window: {window_minutes} minutes).",
                    "",
                    "Use this context to maintain consistency in your analysis approach while prioritizing ground truth temporal data.",
                ])

        return "\n".join(header_lines)
    
    def build_response_template(self, has_chart_analysis: bool = False, dynamic_thresholds: Optional[Dict[str, Any]] = None) -> str:
        """Build the response template for trading decision output.
        
        Args:
            has_chart_analysis: Whether chart image analysis is available
            dynamic_thresholds: Brain-learned thresholds for dynamic values
            
        Returns:
            str: Formatted response template
        """
        thresholds = dynamic_thresholds or {}
        # Core thresholds
        adx_strong = thresholds.get("adx_strong_threshold", 25)
        avg_sl = thresholds.get("avg_sl_pct", 2.5)
        min_rr = thresholds.get("min_rr_recommended", 2.0)
        conf_threshold = thresholds.get("confidence_threshold", 70)
        # Extended thresholds
        adx_weak = thresholds.get("adx_weak_threshold", 20)
        conf_weak = thresholds.get("min_confluences_weak", 4)
        conf_std = thresholds.get("min_confluences_standard", 3)
        pos_reduce_mixed = thresholds.get("position_reduce_mixed", 0.20)
        pos_reduce_div = thresholds.get("position_reduce_divergent", 0.35)
        min_pos_size = thresholds.get("min_position_size", 0.10)
        rr_borderline = thresholds.get("rr_borderline_min", 1.5)
        rr_strong = thresholds.get("rr_strong_setup", 2.5)
        # Safe MAE line
        safe_mae_pct = thresholds.get("safe_mae_pct", 0)
        safe_mae_line = ""
        if safe_mae_pct > 0:
            safe_mae_line = f"\n- **Safe Drawdown**: Historical winning trades survived up to {safe_mae_pct*100:.2f}% drawdown. Ensure stop isn't too tight."
            
        response_template = f'''## Response Format

Structure your analysis before JSON:
1. **MARKET STRUCTURE**: Current phase and trend state
2. **INDICATOR ASSESSMENT**: Key technical signals and confluence
3. **CONTEXT & CATALYST**: Macro alignment, news, microstructure (if relevant)
4. **RISK/REWARD**: Invalidation point, targets, R/R ratio
5. **DECISION**: Signal with confidence justification

Then output JSON:

```json
{{
    "analysis": {{
        "signal": "BUY|SELL|HOLD|CLOSE|UPDATE",
        "confidence": 0-100,
        "confluence_factors": {{
            "trend_alignment": 0-100,
            "momentum_strength": 0-100,
            "volume_support": 0-100,
            "pattern_quality": 0-100,
            "support_resistance_strength": 0-100
        }},
        "entry_price": number,
        "stop_loss": number,
        "take_profit": number,
        "position_size": 0.0-1.0,
        "reasoning": "1-2 sentence summary",
        "key_levels": {{"support": [level1, level2], "resistance": [level1, level2]}},
        "trend": {{"direction": "BULLISH|BEARISH|NEUTRAL", "strength": 0-100, "timeframe_alignment": "ALIGNED|MIXED|DIVERGENT"}},
        "risk_reward_ratio": number
    }}
}}
```

CONFLUENCE SCORING:
Before finalizing your signal, rate each factor (0-100) based on how strongly it SUPPORTS your chosen signal:
- 0 = Factor opposes the signal or is irrelevant
- 50 = Neutral / Mixed signals
- 100 = Factor strongly confirms the signal

Factors to score:
1. **trend_alignment**: Multi-timeframe trend confluence (short/medium/long align with signal direction)
2. **momentum_strength**: RSI, MACD, momentum oscillators supporting the signal
3. **volume_support**: Volume profile confirms the move (buying volume for BUY, selling for SELL)
4. **pattern_quality**: Chart patterns, candlestick formations, swing structure quality
5. **support_resistance_strength**: Proximity and strength of S/R levels supporting the trade setup

CRITICAL: Provide EXACTLY ONE signal. Never say "CLOSE then HOLD" or "BUY followed by SELL". Make only the immediate action decision.

=== Trend Strength Rules (Advisory) ===
ADX + CHOPPINESS ASSESSMENT:
- ADX < {adx_weak} AND Choppiness > 50: ⚠️ Weak trend + choppy. Needs {conf_weak}+ confluences.
- ADX < {adx_weak} but Choppiness < 50: Potential trend emerging. Trade with strong confirmation.
- ADX {adx_weak}-{adx_strong}: Developing trend. Standard {conf_std}+ confluences.
- ADX >= {adx_strong}: Strong trend environment.

CHOPPINESS INDEX CONTEXT:
- > 61.8: Ranging | < 38.2: Trending | 38-62: Transitional

NOTE: You may OVERRIDE these guidelines if you have exceptional conviction ( catalyst, {conf_weak + 1}+ confluences). State reasoning.

POSITION SIZING FORMULA (calculate before finalizing - SHOW YOUR WORK in RISK/REWARD section):
- Base size = confidence / 100 (e.g., 75 confidence = 0.75 base)
- If timeframe_alignment = "MIXED": reduce by {pos_reduce_mixed:.2f} (e.g., 0.75 - {pos_reduce_mixed:.2f} = {0.75 - pos_reduce_mixed:.2f})
- If timeframe_alignment = "DIVERGENT": reduce by {pos_reduce_div:.2f} (e.g., 0.75 - {pos_reduce_div:.2f} = {0.75 - pos_reduce_div:.2f})
- In weak trend environments (ADX < {adx_weak}): consider smaller sizes
- Final position_size = max({min_pos_size:.2f}, calculated_value)
- FORMAT: "Position: base [X] - alignment [Y] = [Z]"

MACRO TIMEFRAME CONFLICT (CRITICAL):
If the 365D macro trend is BEARISH and you are going LONG (or vice versa):
- Explicitly state: "⚠️ 365D MACRO CONFLICT: [direction]" in your analysis
- Require 4+ strong confluences (standard is 3) or a clear REVERSAL structure (e.g. divergence, pattern break)
- Differentiate between "Structural Bearishness" (don't buy) and "Overextended Bullishness" (valid short opportunity)
If both 365D and Weekly macro conflict with your trade: Exercise EXTREME CAUTION. Only proceed if you identify a clear "Cycle Top/Bottom" or "Major Reversal" setup with 5+ confluences. Otherwise, HOLD is preferred.

SHORT TRADE OPPORTUNITIES:
These are GUIDELINES, not hard rules. Use your judgment based on overall confluence, just as with LONG trades.
When Weekly Macro is BULLISH but short-term conditions suggest exhaustion, SHORT may be valid if you observe:
- Statistical overextension (elevated Z-score, overbought oscillators at extremes)
- Momentum divergence (price making new highs, but indicators failing to confirm)
- Volume climax with rejection (exceptionally high volume at resistance with reversal candle pattern)
- One-sided order book pressure (heavy sell-side absorption visible in microstructure data)
🧠 THINK STEP BY STEP: Before dismissing SHORT, ask: "What would a professional mean-reversion trader see here?"
SHORT trades require stricter confluence than LONG in a bull macro - but they are NOT forbidden. If your analysis shows a clear exhaustion setup, state your reasoning and proceed with appropriate position sizing.

TRADING SIGNALS & CONFIDENCE:
- BUY ({conf_threshold}-100 confidence): Strong multi-indicator confluence + volume confirmation + clear SL/TP + minimum {min_rr:.1f}:1 R/R preferred
- SELL ({conf_threshold}-100 confidence): Strong multi-indicator confluence + volume confirmation + clear SL/TP + minimum {min_rr:.1f}:1 R/R preferred
- HOLD (any confidence <{conf_threshold}): Mixed signals, weak trend, conflicting indicators, low volume, or insufficient setup quality
- CLOSE: Exit position when SL/TP hit, signal reversal, or thesis invalidated
- UPDATE: Adjust existing position SL/TP when market structure improves

RISK/REWARD GUIDELINES:
- R/R < {rr_borderline:.1f}: Very unfavorable - strongly consider HOLD
- R/R {rr_borderline:.1f}-{min_rr:.1f}: Borderline - only trade with exceptional confluence ({conf_weak}+)
- R/R >= {min_rr:.1f}: Standard acceptable quality
- R/R >= {rr_strong:.1f}: Strong setup - preferred for counter-trend trades

RISK MANAGEMENT (Stop Loss & Take Profit):{safe_mae_line}
LONG trades:
- SL: Below swing low + 1x ATR buffer (max {avg_sl:.1f}% from entry) | Example: Entry $100, Swing Low $97, ATR $1 → SL $96
- TP: Key resistance levels, Fibonacci (0.618/0.786/1.0), previous highs | Multiple targets: TP1=1.5R, TP2=2.5R, TP3=3.5R

SHORT trades:  
- SL: Above swing high + 1x ATR buffer (max {avg_sl:.1f}% from entry) | Example: Entry $100, Swing High $103, ATR $1 → SL $104
- TP: Key support levels, Fibonacci (0.382/0.236/0.0), previous lows | Multiple targets: TP1=1.5R, TP2=2.5R, TP3=3.5R


Mandatory: All trades require stops based on technical levels (not arbitrary %), accounting for ATR volatility, positioned to invalidate thesis if hit.'''
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
## Analysis Steps (use findings to determine trading signal):

1. MULTI-TIMEFRAME ASSESSMENT:
   {timeframe_desc} | Compare short vs multi-day vs long-term (30d+, 365d) | Weekly macro (200-week SMA)
   🧠 Are timeframes aligned or divergent? Which dominates?

2. TECHNICAL INDICATORS:
   Analyze all provided Momentum, Trend, Volatility, and Volume indicators (RSI, MACD, ADX, ATR, ROC, MFI, etc.)
   🧠 Do indicators confirm each other or show divergence?

3. PATTERN RECOGNITION (Conservative Approach):
   **Swing Structure:** HH/HL = uptrend, LH/LL = downtrend | **Classic:** H&S, double tops/bottoms, wedges, flags | **Candlesticks:** doji, hammer, engulfing at key S/R | **Divergences:** Price vs RSI/MACD/OBV | **IMPORTANT:** If unclear, state "No clear pattern" - do NOT force conclusions
   🧠 Is pattern complete or forming? Could this be a fakeout?

4. SUPPORT/RESISTANCE:
   Key levels across timeframes | Historical reaction zones (3+ touches) | Confluences (S/R + Fib + SMA) | Volume nodes | Risk/reward for SL/TP
   🧠 How did price react last time at this level?

5. MARKET CONTEXT:
   Market Overview (global cap, dominance)"""
        
        if "BTC" not in analyzed_base:
            analysis_steps += "\n   - Compare performance relative to BTC (correlation/divergence)"
        
        if "ETH" not in analyzed_base:
            analysis_steps += "\n   - Compare performance relative to ETH if relevant"
        
        analysis_steps += """
 | Fear & Greed Index (extremes) | Asset alignment with market | Relevant events

5.5. BULL vs BEAR CASE (Forced Dialectical Analysis):
   🐂 BULL CASE: What confluence supports LONG? What would need to happen for price to rise?
   🐻 BEAR CASE: What confluence supports SHORT? What would need to happen for price to fall?
   🧠 WHICH PERSPECTIVE WINS? Justify with data. If brain has relevant semantic rules for either direction, weight those appropriately.

6. NEWS & SENTIMENT:
   Asset news | Market events | Sentiment | Institutional activity | Regulatory impacts | News that could override technicals

7. STATISTICAL ANALYSIS:
   Z-Score (extremes revert) | Kurtosis (tail risk) | Hurst (>0.5 trending, <0.5 reverting) | Volatility cycles"""
        
        # Add chart analysis steps only if chart images are available
        step_number = 8
        if has_chart_analysis:
            cfg_limit = int(self.config.AI_CHART_CANDLE_LIMIT)

            analysis_steps += f"""

{step_number}. CHART IMAGE ANALYSIS (~{cfg_limit} candles, 4 panels):
   **P1-PRICE:** SMA50 (orange), SMA200 (purple) - Golden/Death Cross? | Read MIN/MAX labels | Apply patterns from Step 3 to visual data
   **P2-RSI:** Read value from annotation | Zone (>70 overbought, <30 oversold) | Check divergence vs price
   **P3-VOLUME:** Trend direction | Spikes align with price moves? | Green/red bar ratio
   **P4-CMF/OBV:** CMF (cyan, left axis): >0 buying, <0 selling | OBV (magenta, right): rising=accumulation
   **VALIDATE:** Cross-check visuals with numerical indicators - flag discrepancies"""
            step_number += 1
        
        analysis_steps += f"""

{step_number}. SYNTHESIS:
   Trend direction/strength | Indicator confluence | SL/TP levels | R/R ratio | Confidence | Invalidation triggers

NOTE: Indicators calculated from CLOSED CANDLES ONLY. No pattern = state "No clear pattern detected"."""
        
        if has_advanced_support_resistance:
            analysis_steps += """
ADVANCED S/R: Volume-weighted pivots with 3+ touches, above-average volume. Only strong levels provided."""

        return analysis_steps
