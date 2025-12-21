"""
Template management for prompt building system.
Handles system prompts, response templates, and analysis steps.
"""

from typing import Optional, Any

from src.logger.logger import Logger


class TemplateManager:
    """Manages prompt templates, system prompts, and analysis steps."""
    
    def __init__(self, config: Any, logger: Optional[Logger] = None):
        """Initialize the template manager.
        
        Args:
            config: Configuration module providing prompt defaults
            logger: Optional logger instance for debugging
        """
        self.logger = logger
        self.config = config
    
    def build_system_prompt(self, symbol: str, timeframe: str = "1h", language: Optional[str] = None, has_chart_image: bool = False) -> str:
        """Build the system prompt for the AI model.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            timeframe: Timeframe for analysis (e.g., "1h", "4h", "1d")
            language: Optional language for response (defaults to English)
            has_chart_image: Whether a chart image is being provided for visual analysis
            
        Returns:
            str: Formatted system prompt
        """
        language = language or self.config.DEFAULT_LANGUAGE

        header_lines = [
            f"You are providing educational crypto market analysis of {symbol} on {timeframe} timeframe along with multi-timeframe technical metrics and recent market data.",
            "Focus on objective technical indicator readings and historical pattern recognition (e.g., identify potential chart patterns like triangles, head and shoulders, flags based on OHLCV data) for educational purposes only.",
            "Present clear, data-driven observations with specific numeric values from the provided metrics. Prioritize recent price action and technical indicators over older news unless the news is highly significant.",
            "After presenting quantitative data in each section, provide brief interpretive commentary explaining what the numbers mean for traders. Balance technical precision with accessible explanations.",
            "Identify key price levels based solely on technical analysis concepts (Support, Resistance, Pivot Points, Fibonacci levels if applicable).",
        ]

        if has_chart_image:
            cfg_limit = int(self.config.AI_CHART_CANDLE_LIMIT)
            header_lines.extend([
                "",
                f"A chart image containing approximately {cfg_limit} candlesticks is provided for visual review. Integrate chart observations with numerical indicators as described in the analysis steps, stay conservative when qualifying patterns, and explicitly state when no clear classic patterns are visible instead of forcing a conclusion.",
            ])

        header_lines.extend([
            "",
            "THIS IS EDUCATIONAL CONTENT ONLY. All analysis is for informational and educational purposes - NOT financial advice.",
            "Always include disclaimers that this is not investment advice and users must do their own research.",
        ])

        header_base = "\n".join(header_lines)

        if language == self.config.DEFAULT_LANGUAGE or language == "English":
            header = header_base
        else:
            header = (
                f"{header_base}\n"
                f"Write your entire response in {language} language. Only the JSON structure should remain in English, but all text content must be in {language}.\n"
                f"Use appropriate {language} terminology for technical analysis concepts."
            )

        return header
    
    def build_response_template(self, has_chart_analysis: bool = False) -> str:
        """Build the response template for structured output.
        
        Args:
            has_chart_analysis: Whether chart image analysis is available
            
        Returns:
            str: Formatted response template
        """
        response_template = '''RESPONSE FORMAT:
        Please structure your response in JSON format as follows:
        ```json
        {
            "analysis": {
                "summary": "Concise overview of the current market situation",
                "observed_trend": "BULLISH|BEARISH|NEUTRAL", // Justify this based on indicators/patterns
                "trend_strength": 0-100, // Based on ADX or similar
                "timeframes": {
                    "short_term": "BULLISH|BEARISH|NEUTRAL",
                    "medium_term": "BULLISH|BEARISH|NEUTRAL",
                    "long_term": "BULLISH|BEARISH|NEUTRAL"
                },
                "key_levels": {
                    "support": [level1, level2],
                    "resistance": [level1, level2]
                },
                "price_scenarios": {
                    "bullish_scenario": number, // Potential target/resistance if trend turns bullish
                    "bearish_scenario": number // Potential target/support if trend continues bearish
                },
                "confidence_score": 0-100, // Overall confidence in the analysis
                "technical_bias": "BULLISH|BEARISH|NEUTRAL", // Justify this based on indicator confluence
                "risk_ratio": number, // Estimated risk/reward based on scenarios/levels
                "market_structure": "BULLISH|BEARISH|NEUTRAL", // Based on price action patterns (higher highs/lows etc.)
                "news_summary": "Brief summary of relevant recent news and their potential market impact"
            }
        }
        ```        
        After the JSON block, include a detailed human-readable analysis. 
        **IMPORTANT: Format this detailed analysis using Markdown syntax.** Use headings (`##`), bold (`**text**`), italics (`*text*`), bullet points (`-` or `*`), numbered lists for enhanced readability. 
        **Quantify observations where possible (e.g., "Price is X% below the 50-period MA", "RSI dropped Y points").**
        
        **IMPORTANT: Begin your analysis with a clear disclaimer that this is educational content only and not financial advice.**
        
        **IMPORTANT: Include a note that technical indicators are calculated including the current incomplete candle for real-time analysis.**
        
        **INTERPRETIVE GUIDANCE: After presenting raw numerical data in each section, provide a brief 4-8 sentence interpretation explaining what these numbers mean for traders. Use phrases like "What this means:", "Interpretation:", or "Key takeaway:" to clearly separate data from analysis.**
        
        Organize the Markdown analysis into these sections:
        
        - Disclaimer (emphasize this is for educational purposes only, not financial advice)
        - Technical Analysis Overview (objective description of what the indicators show, quantified) - INCLUDE NOTE: "All technical indicators in this analysis are calculated including the current incomplete {timeframe} candle for real-time market assessment. Values will update as the candle progresses."
          * After the quantified metrics, add: "**What this means:** [brief interpretation of the overall technical picture]"
        - Multi-Timeframe Assessment (describe short, medium, long-term patterns with quantified changes)
          * After each timeframe subsection, add: "**Interpretation:** [what this timeframe suggests for traders]"
        - Technical Indicators Summary (describe indicators in organized paragraphs grouped by category)
          * After each indicator category (Momentum, Trend, Volatility, Volume), add: "**Key takeaway:** [what these indicators collectively suggest]"
        - Key Technical Levels (describe support and resistance levels in text format with specific prices and distances)
          * After presenting levels, add: "**Trading implication:** [how these levels should inform decision-making]"
        - Market Context (describe asset performance vs broader market)
          * After the data, add: "**What this tells us:** [interpretation of relative performance and market positioning]"
        - News Summary (summarize relevant recent news and their potential impact on the asset)
          * After each major news item, briefly explain: "**Impact assessment:** [potential effect on price action]"'''
        
        # Add chart analysis sections only if chart images are available
        if has_chart_analysis:
            response_template += '''
        - Chart Pattern Analysis & Visual Integration:
          * **Identified Patterns**: List each pattern found with:
            - Pattern name and type (e.g., "Bearish Head and Shoulders Top")
            - Location (price range and approximate candle indices)
            - Bias (bullish/bearish) and confidence level (high/medium/low)
            - Key structural levels (neckline, shoulders, peaks, breakout points)
            - Current status (forming, completed, breached)
          * **Visual Observations**: Describe what you see in the chart image
          * **Pattern-Indicator Alignment**: How visual patterns confirm or contradict technical indicators
          * After pattern analysis, add: "**Visual interpretation:** [what the chart patterns suggest for near-term price action]"'''
        
        response_template += '''
        - Potential Catalysts (Summarize factors like news, events, strong technical signals that could drive future price movement)
          * Add: "**Catalyst assessment:** [which catalysts are most likely to materialize and their potential impact]"
        - Educational Context (explain technical concepts related to the current market conditions)
        - Historical Patterns (similar technical setups in the past and what they typically indicate)
          * Add: "**Historical context takeaway:** [what past patterns suggest about current probabilities]"
        - Risk Considerations (discuss technical factors that may invalidate the analysis)
          * Add: "**Risk summary:** [key levels and scenarios that would invalidate the current thesis]"
        
      
        **FINAL SECTION - Add this at the very end after all educational content:**
        
        - **Final Thoughts & Synthesis**:
          * Provide a concise 3-5 paragraph synthesis that:
            - Summarizes the current market state in plain language
            - Highlights the 2-3 most critical factors driving the analysis
            - Explains the highest-probability scenarios (bullish/bearish/neutral) with percentage likelihood estimates if appropriate
            - Identifies the key levels or events that would confirm or invalidate each scenario
            - Offers perspective on the risk/reward profile at current levels
            - Concludes with what traders should be watching most closely in the coming days/weeks
          * This should read as a cohesive "bottom line" summary that synthesizes all the technical, fundamental, and sentiment analysis into actionable insights while maintaining the educational disclaimer framework
        '''
        
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
        Follow these steps to generate the analysis. In the final JSON response, briefly justify the 'observed_trend' and 'technical_bias' fields by referencing specific indicators or patterns from the provided data (e.g., "Bearish due to MACD crossover and price below Supertrend").
        
        IMPORTANT: For each analysis step below, first present the quantitative data/observations, then provide a brief interpretation of what those findings mean for traders. This interpretive commentary should explain the practical implications and trading context.

        1. Multi-Timeframe Assessment:
            - {timeframe_desc}
            - Compare shorter periods vs multi-day periods vs long-term (30d+, 365d) price action
            - Review weekly macro trend indicators if provided (200-week SMA, institutional positioning)
            - Identify alignment or divergence across different timeframes
        
        2. Technical Indicator Analysis:
            - Evaluate core momentum indicators (RSI, MACD, Stochastic)
            - Observe trend strength using ADX, DI readings
            - Check volatility levels with ATR and Bollinger Bands
            - Analyze volume indicators (MFI, OBV, Force Index) for context
            - Consider SMA relationships (e.g., 50 vs 200) for trend context
            - Assess advanced indicators (TSI, Vortex, PFE, RMI, Ultimate Oscillator)
           
        3. Key Pattern Recognition:
            - Identify chart patterns (wedges, triangles, H&S, double tops/bottoms)
            - Detect divergences between price and momentum indicators
            - Look for candlestick reversal patterns
            - Note potential harmonic patterns and Fibonacci relationships
            - Identify overbought/oversold conditions across indicators
            - Prioritize recent patterns over older ones; older patterns may be less relevant given the dynamic timeframe analysis (e.g., 1D).
        
        4. Support/Resistance Validation:
            - Map key price levels from all timeframes
            - Identify historical price reaction zones
            - Determine areas with multiple technical confluences
            - Compare current price with historical significant levels
            - Basic Support/Resistance Indicator: Rolling min/max of high/low over specified period
            - Volume profile analysis: Note price levels with high historical volume
        
        5. Market Context Integration:
            - Reference the provided Market Overview data in your analysis
            - Compare the asset's performance with the broader market (market cap %, dominance trends)"""
        
        if "BTC" not in analyzed_base:
            analysis_steps += "\n            - Compare the asset's performance relative to BTC"
        
        if "ETH" not in analyzed_base:
            analysis_steps += "\n            - Compare the asset's performance relative to ETH if relevant"
        
        analysis_steps += """
            - Consider market sentiment metrics including Fear & Greed Index
            - Analyze if the asset is aligned with or diverging from general market trends
            - Note relevant market events and their historical impact
            - Consider market structures observed in similar historical contexts
        
        6. News Analysis:
            - Summarize relevant recent news articles about the asset
            - Identify potential market-moving events or announcements
            - Evaluate sentiment from news coverage
            - Connect news events to recent price action when applicable
            - Note institutional actions or corporate developments mentioned in news
            - Identify any regulatory news that might impact the asset
        
        7. Statistical Analysis:
            - Evaluate statistical indicators like Z-Score and Kurtosis
            - Consider Hurst Exponent for trending vs mean-reverting behavior
            - Note abnormal distribution patterns in price/volume
            - Assess volatility cycles and potential expansion/contraction phases"""
        
        # Add chart analysis steps only if chart images are available
        step_number = 8
        if has_chart_analysis:
            cfg_limit = int(self.config.AI_CHART_CANDLE_LIMIT)

            analysis_steps += f"""
        {step_number}. Chart Pattern Analysis & Visual Integration:
           CHART CONTEXT:
           - Review the provided chart image (~{cfg_limit} candlesticks) optimized with high contrast and thin wicks for pattern clarity.
           VISUAL PATTERN DETECTION (Priority):
           - Systematically scan the chart image for classic patterns:
             * Head and Shoulders (and inverse variants)
             * Double Tops/Bottoms (compare peak/trough heights and spacing)
             * Wedges (rising/falling with converging trend lines)
             * Triangles (ascending/descending/symmetrical consolidations)
             * Flags and Pennants (continuation patterns after strong moves)
             * Support and resistance breakouts/breakdowns
           PATTERN QUALITY & HONESTY:
           - Only report patterns that are clear, well-formed, and span adequate range (target at least 3-5% move and 20-30+ candles for H&S/double tops).
           - Reject noisy or ambiguous formations; if none qualify, explicitly state "No clear classic patterns detected."
           - Stay conservative when uncertain rather than forcing a conclusion.
           PATTERN STRUCTURE & REPORTING:
           - For each accepted pattern, capture:
             * Pattern name/type, directional bias (bullish/bearish), and status (forming, completed, breached)
             * Exact price range and approximate candle indices
             * Confidence level (high/medium/low) with justification
             * Key structural components (neckline, shoulders, peaks, troughs) and breakout/breakdown levels
             * Brief rationale for validity (structure quality, symmetry, price range, volume alignment)
           CANDLESTICK FORMATIONS:
           - Identify meaningful single/multi-candle patterns (doji, hammer, shooting star, engulfing) and note how they support or contradict broader setups.
           - Observe momentum shifts through candle size and color changes.
           SUPPORT & RESISTANCE VISUALIZATION:
           - Mark horizontal levels where price repeatedly bounced/rejected.
           - Identify trend lines connecting swing highs or swing lows and highlight channels where applicable.
           VISUAL-NUMERICAL INTEGRATION:
           - Validate each pattern against numerical indicators: ADX (>25) for trend strength, volume spikes on breakouts, momentum alignment (RSI, MACD), and divergences.
           - Prioritize numerical data if it contradicts what the chart suggests and document any mismatches.
           VISUAL SYNTHESIS:
           - Fuse visual observations with calculated metrics, explaining how chart evidence reinforces or challenges the technical narrative.
        """
            step_number += 1
        
        analysis_steps += f"""
        
        {step_number}. Educational Information:
            - Explain possible scenarios based on technical analysis concepts
            - Describe what traders typically watch for in similar situations
            - Present educational information about risk management concepts
            - Focus on explaining the "what" and "why" of technical patterns
            - Offer context about typical behavior of similar assets in comparable market conditions
        
        TECHNICAL INDICATORS NOTE:
            - Current incomplete candle is included in all technical indicator calculations
            - This provides real-time market assessment as the candle progresses
            - Indicator values will update as price action continues within the current timeframe"""
        
        if has_advanced_support_resistance:
            analysis_steps += """
        
        CUSTOM INDICATORS REFERENCE:
        
        Advanced Support/Resistance:
            - Volume-weighted pivot points with strength thresholds
            - Creates pivot points using (H+L+C)/3 formula
            - Calculates S1 = (2*PP)-H and R1 = (2*PP)-L levels
            - Tracks consecutive touches to measure level strength
            - Filters for above-average volume at reaction points
            - Returns ONLY strong support and resistance levels that meet all criteria 
            - Uses price momentum and volume confirmations"""

        return analysis_steps
