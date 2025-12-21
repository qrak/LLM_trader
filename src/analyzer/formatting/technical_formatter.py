"""
Consolidated Technical Analysis Formatter.
Handles all technical analysis formatting in a single comprehensive class.
"""
from typing import Optional
from src.logger.logger import Logger


class TechnicalFormatter:
    """Consolidated formatter for all technical analysis sections."""
    
    def __init__(self, technical_calculator, logger: Optional[Logger] = None, format_utils=None):
        """Initialize the technical analysis formatter.
        
        Args:
            technical_calculator: TechnicalCalculator instance for thresholds and calculations
            logger: Optional logger instance for debugging
        """
        self.technical_calculator = technical_calculator
        self.logger = logger
        self.INDICATOR_THRESHOLDS = technical_calculator.INDICATOR_THRESHOLDS
        self.format_utils = format_utils
    
    def format_technical_analysis(self, context, timeframe: str) -> str:
        """Format complete technical analysis section.
        
        Args:
            context: Analysis context containing technical data
            timeframe: Primary timeframe for analysis
            
        Returns:
            str: Formatted technical analysis section
        """
        if not context.technical_data:
            return "TECHNICAL ANALYSIS:\nNo technical data available."

        td = context.technical_data
        crypto_data = {'current_price': context.current_price}
        
        # Build all sections
        patterns_section = self._format_patterns_section(context)
        momentum_section = self.format_momentum_section(td)
        trend_section = self.format_trend_section(td)
        volume_section = self.format_volume_section(td)
        volatility_section = self.format_volatility_section(td, crypto_data)
        advanced_section = self.format_advanced_indicators_section(td, crypto_data)
        key_levels_section = self.format_key_levels_section(td)

        # Build main technical analysis content
        technical_analysis = f"""\nTECHNICAL ANALYSIS ({timeframe}):\n\n## Price Action:\n- Current Price: {self.format_utils.fmt(context.current_price)}\n- Rolling VWAP (20): {self.format_utils.fmt_ta(self.technical_calculator, td, 'vwap', 8)}\n- TWAP (20): {self.format_utils.fmt_ta(self.technical_calculator, td, 'twap', 8)}\n\n{momentum_section}\n\n{trend_section}\n\n{volatility_section}\n\n{volume_section}\n\n## Statistical Metrics:\n- Hurst Exponent(20): {self.format_utils.fmt_ta(self.technical_calculator, td, 'hurst', 2)} [~0.5: Random Walk, >0.5: Trending, <0.5: Mean Reverting]\n- Z-Score(20): {self.format_utils.fmt_ta(self.technical_calculator, td, 'zscore', 2)} [Distance from mean in std deviations]\n- Kurtosis(20): {self.format_utils.fmt_ta(self.technical_calculator, td, 'kurtosis', 2)} [Tail risk indicator; >3 suggests fatter tails]\n\n{key_levels_section}\n\n{advanced_section}\n\n{patterns_section}"""

        return technical_analysis
    
    def format_momentum_section(self, td: dict) -> str:
        """Format the momentum indicators section."""
        return f"""## Momentum Indicators:
- RSI(14): {self.format_utils.fmt_ta(self.technical_calculator, td, 'rsi', 1)} [<{self.INDICATOR_THRESHOLDS['rsi']['oversold']}=Oversold, {self.INDICATOR_THRESHOLDS['rsi']['oversold']}-{self.INDICATOR_THRESHOLDS['rsi']['overbought']}=Neutral, >{self.INDICATOR_THRESHOLDS['rsi']['overbought']}=Overbought]
- MACD (12,26,9): [Pattern detector provides crossover analysis]
  * Line: {self.format_utils.fmt_ta(self.technical_calculator, td, 'macd_line', 8)}
  * Signal: {self.format_utils.fmt_ta(self.technical_calculator, td, 'macd_signal', 8)}
  * Histogram: {self.format_utils.fmt_ta(self.technical_calculator, td, 'macd_hist', 8)}
- Stochastic %K(14,3,3): {self.format_utils.fmt_ta(self.technical_calculator, td, 'stoch_k', 1)} [<{self.INDICATOR_THRESHOLDS['stoch_k']['oversold']}=Oversold, >{self.INDICATOR_THRESHOLDS['stoch_k']['overbought']}=Overbought]
- Stochastic %D(14,3,3): {self.format_utils.fmt_ta(self.technical_calculator, td, 'stoch_d', 1)} [<{self.INDICATOR_THRESHOLDS['stoch_d']['oversold']}=Oversold, >{self.INDICATOR_THRESHOLDS['stoch_d']['overbought']}=Overbought]
- Williams %R(14): {self.format_utils.fmt_ta(self.technical_calculator, td, 'williams_r', 1)} [<{self.INDICATOR_THRESHOLDS['williams_r']['oversold']}=Oversold, >{self.INDICATOR_THRESHOLDS['williams_r']['overbought']}=Overbought]
- TSI(20,10): {self.format_utils.fmt_ta(self.technical_calculator, td, 'tsi', 2)} [True Strength Index - momentum oscillator with signal line crossovers]
- RMI(20,5): {self.format_utils.fmt_ta(self.technical_calculator, td, 'rmi', 1)} [Relative Momentum Index - similar to RSI but uses momentum instead of price changes]
- PPO(12,26): {self.format_utils.fmt_ta(self.technical_calculator, td, 'ppo', 2)} [Percentage Price Oscillator - MACD in percentage terms]"""

    def format_trend_section(self, td: dict) -> str:
        """Format the trend indicators section."""
        supertrend_direction = self.format_utils.get_supertrend_direction_string(td.get('supertrend_direction', 0))

        return (
            "## Trend Indicators:\n"
            f"- ADX(14): {self.format_utils.fmt_ta(self.technical_calculator, td, 'adx', 1)} [0-{self.INDICATOR_THRESHOLDS['adx']['weak']}: Weak/No Trend, {self.INDICATOR_THRESHOLDS['adx']['weak']}-{self.INDICATOR_THRESHOLDS['adx']['strong']}: Strong, {self.INDICATOR_THRESHOLDS['adx']['strong']}-{self.INDICATOR_THRESHOLDS['adx']['very_strong']}: Very Strong, >{self.INDICATOR_THRESHOLDS['adx']['very_strong']}: Extremely Strong]\n"
            f"- +DI(14): {self.format_utils.fmt_ta(self.technical_calculator, td, 'plus_di', 1)} [Pattern detector analyzes DI crossovers]\n"
            f"- -DI(14): {self.format_utils.fmt_ta(self.technical_calculator, td, 'minus_di', 1)}\n"
            f"- Supertrend(20,3.0) Direction: {supertrend_direction}\n"
            f"- TRIX(20): {self.format_utils.fmt_ta(self.technical_calculator, td, 'trix', 4)} [Triple exponential average momentum oscillator]\n"
            f"- PFE(20,5): {self.format_utils.fmt_ta(self.technical_calculator, td, 'pfe', 2)} [Polarized Fractal Efficiency - trend strength indicator]\n"
            f"- Vortex VI+(20): {self.format_utils.fmt_ta(self.technical_calculator, td, 'vortex_plus', 2)} [Vortex Indicator positive]\n"
            f"- Vortex VI-(20): {self.format_utils.fmt_ta(self.technical_calculator, td, 'vortex_minus', 2)} [Vortex Indicator negative]"
        )

    def format_volume_section(self, td: dict) -> str:
        """Format the volume indicators section."""
        cmf_interpretation = self.format_utils.format_cmf_interpretation(self.technical_calculator, td)
        
        return (
            "## Volume Indicators:\n"
            f"- MFI(14): {self.format_utils.fmt_ta(self.technical_calculator, td, 'mfi', 1)} [<{self.INDICATOR_THRESHOLDS['mfi']['oversold']}=Oversold, >{self.INDICATOR_THRESHOLDS['mfi']['overbought']}=Overbought]\n"
            f"- On Balance Volume (OBV): {self.format_utils.fmt_ta(self.technical_calculator, td, 'obv', 0)}\n"
            f"- Chaikin MF(20): {self.format_utils.fmt_ta(self.technical_calculator, td, 'cmf', 4)}{cmf_interpretation}\n"
            f"- Force Index(20): {self.format_utils.fmt_ta(self.technical_calculator, td, 'force_index', 0)}"
        )

    def format_volatility_section(self, td: dict, crypto_data: dict) -> str:
        """Format the volatility indicators section."""
        bb_interpretation = self.format_utils.format_bollinger_interpretation(self.technical_calculator, td)
        
        return (
            "## Volatility Indicators:\n"
            f"- Bollinger Bands(20,2): {self.format_utils.fmt_ta(self.technical_calculator, td, 'bb_upper', 8)} | {self.format_utils.fmt_ta(self.technical_calculator, td, 'bb_middle', 8)} | {self.format_utils.fmt_ta(self.technical_calculator, td, 'bb_lower', 8)}{bb_interpretation}\n"
            f"- BB %B: {self.format_utils.fmt_ta(self.technical_calculator, td, 'bb_percent_b', 2)} [0-1 range, >0.8=near upper, <0.2=near lower]\n"
            f"- ATR(14): {self.format_utils.fmt_ta(self.technical_calculator, td, 'atr', 8)}\n"
            f"- Keltner Channels(20,2): {self.format_utils.fmt_ta(self.technical_calculator, td, 'kc_upper', 8)} | {self.format_utils.fmt_ta(self.technical_calculator, td, 'kc_middle', 8)} | {self.format_utils.fmt_ta(self.technical_calculator, td, 'kc_lower', 8)}"
        )

    def format_key_levels_section(self, td: dict) -> str:
        """Format key levels section."""
        return (
            "## Key Levels:\n"
            f"- Basic Support: {self.format_utils.fmt_ta(self.technical_calculator, td, 'basic_support', 8)}\n"
            f"- Basic Resistance: {self.format_utils.fmt_ta(self.technical_calculator, td, 'basic_resistance', 8)}\n"
            f"- Pivot Point: {self.format_utils.fmt_ta(self.technical_calculator, td, 'pivot_point', 8)}\n"
            f"- Pivot S1: {self.format_utils.fmt_ta(self.technical_calculator, td, 'pivot_s1', 8)} | S2: {self.format_utils.fmt_ta(self.technical_calculator, td, 'pivot_s2', 8)} | S3: {self.format_utils.fmt_ta(self.technical_calculator, td, 'pivot_s3', 8)} | S4: {self.format_utils.fmt_ta(self.technical_calculator, td, 'pivot_s4', 8)}\n"
            f"- Pivot R1: {self.format_utils.fmt_ta(self.technical_calculator, td, 'pivot_r1', 8)} | R2: {self.format_utils.fmt_ta(self.technical_calculator, td, 'pivot_r2', 8)} | R3: {self.format_utils.fmt_ta(self.technical_calculator, td, 'pivot_r3', 8)} | R4: {self.format_utils.fmt_ta(self.technical_calculator, td, 'pivot_r4', 8)}"
        )

    def format_advanced_indicators_section(self, td: dict, crypto_data: dict) -> str:
        """Format advanced indicators section."""
        return (
            "## Advanced Indicators:\n"
            f"- Advanced Support: {self.format_utils.fmt_ta(self.technical_calculator, td, 'advanced_support', 8)}\n"
            f"- Advanced Resistance: {self.format_utils.fmt_ta(self.technical_calculator, td, 'advanced_resistance', 8)}\n"
            f"- Commodity Channel Index CCI(14): {self.format_utils.fmt_ta(self.technical_calculator, td, 'cci', 1)} [>100=Overbought, <-100=Oversold]\n"
            f"- Average True Range %: {self.format_utils.fmt_ta(self.technical_calculator, td, 'atr_percent', 2)}%\n"
            f"- Parabolic SAR: {self.format_utils.fmt_ta(self.technical_calculator, td, 'sar', 8)} [Price above SAR=Bullish, below=Bearish]\n"
            f"- Donchian Channels(20): {self.format_utils.fmt_ta(self.technical_calculator, td, 'donchian_upper', 8)} | {self.format_utils.fmt_ta(self.technical_calculator, td, 'donchian_lower', 8)}\n"
            f"- Ultimate Oscillator: {self.format_utils.fmt_ta(self.technical_calculator, td, 'uo', 1)} [>70=Overbought, <30=Oversold]\n"
            f"- Coppock Curve: {self.format_utils.fmt_ta(self.technical_calculator, td, 'coppock', 2)} [Momentum oscillator, values above 0 suggest bullish momentum]\n"
            f"- KST (Know Sure Thing): {self.format_utils.fmt_ta(self.technical_calculator, td, 'kst', 2)} [Momentum oscillator, crossovers signal trend changes]\n"
            f"- Chandelier Exit Long: {self.format_utils.fmt_ta(self.technical_calculator, td, 'chandelier_long', 8)} [Trailing stop level for long positions]\n"
            f"- Chandelier Exit Short: {self.format_utils.fmt_ta(self.technical_calculator, td, 'chandelier_short', 8)} [Trailing stop level for short positions]"
        )
    
    def _format_patterns_section(self, context) -> str:
        """Format patterns section using detected patterns from context.
        
        Args:
            context: Analysis context containing technical data
            
        Returns:
            str: Formatted patterns section
        """
        # Use stored technical_patterns from analysis engine
        if context.technical_patterns:
            try:
                pattern_summaries = []
                last_candle_index = len(context.ohlcv_candles) - 1 if context.ohlcv_candles is not None else None
                
                for category, patterns_list in context.technical_patterns.items():
                    if patterns_list:  # Only process non-empty pattern lists
                        for pattern_dict in patterns_list:
                            # Filter patterns based on recency relative to total candles analyzed
                            # Strategy: Show patterns from most recent 20% of data (e.g., last 200 of 999 candles)
                            # This makes filtering adaptive to different timeframes and candle counts
                            pattern_index = pattern_dict.get('index', None)
                            periods_ago = pattern_dict.get('details', {}).get('periods_ago', 0)
                            
                            # Determine recency threshold based on pattern type and data size
                            if last_candle_index is not None and pattern_index is not None:
                                total_candles = last_candle_index + 1
                                
                                # For long-term signals (MA crossovers), use wider window (50% of data)
                                if category == 'ma_crossover':
                                    recency_threshold = int(total_candles * 0.5)  # Last 50% of candles
                                # For persistent patterns (volatility, volume), use narrow window (5% of data)
                                elif category in ['volatility', 'volume']:
                                    recency_threshold = max(10, int(total_candles * 0.05))  # Last 5% or min 10 candles
                                # For other patterns (divergences, crossovers, etc.), use moderate window (20% of data)
                                else:
                                    recency_threshold = max(20, int(total_candles * 0.2))  # Last 20% or min 20 candles
                                
                                is_recent = pattern_index >= (last_candle_index - recency_threshold)
                            else:
                                # If no index info, include the pattern
                                is_recent = True
                            
                            if is_recent:
                                description = pattern_dict.get('description', f'Unknown {category} pattern')
                                pattern_summaries.append(f"- {description}")
                
                if pattern_summaries:
                    if self.logger:
                        self.logger.debug(f"Including {len(pattern_summaries)} recent patterns in technical analysis (adaptive recency filter)")
                    return "\n\n## Detected Patterns:\n" + "\n".join(pattern_summaries[-50:])  # Show last 50 recent patterns
            except Exception as e:
                if self.logger:
                    self.logger.debug(f"Error using stored technical_patterns: {e}")
        
        # Fallback: try direct pattern detection
        try:
            ohlcv_data = context.ohlcv_candles
            technical_history = context.technical_data.get('history', {})
            
            patterns = self.technical_calculator.get_all_patterns(ohlcv_data, technical_history)
            
            if self.logger:
                self.logger.debug(f"Using fallback pattern detection, found {len(patterns)} patterns")
            
            if patterns:
                pattern_summaries = []
                for pattern in patterns[-5:]:  # Show last 5 patterns
                    description = pattern.get('description', 'Unknown pattern')
                    pattern_summaries.append(f"- {description}")
                
                if pattern_summaries:
                    return "\n\n## Detected Patterns:\n" + "\n".join(pattern_summaries)
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Could not use fallback pattern detection: {e}")
        
        return ""
