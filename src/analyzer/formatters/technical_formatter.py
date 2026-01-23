"""
Consolidated Technical Analysis Formatter.
Handles all technical analysis formatting in a single comprehensive class.
"""
from typing import Optional
import re
import numpy as np
from src.logger.logger import Logger
from src.utils.data_utils import get_last_valid_value, get_last_n_valid
from src.utils.timeframe_validator import TimeframeValidator


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
        
        patterns_section = self._format_patterns_section(context, timeframe)
        price_action_section = self.format_price_action_section(context, td)
        momentum_section = self.format_momentum_section(td)
        trend_section = self.format_trend_section(td)
        volume_section = self.format_volume_section(td)
        volatility_section = self.format_volatility_section(td, crypto_data)
        advanced_section = self.format_advanced_indicators_section(td, crypto_data)
        key_levels_section = self.format_key_levels_section(td)

        technical_analysis = f"""\n## Technical Analysis ({timeframe})\n\n{price_action_section}\n\n{momentum_section}\n\n{trend_section}\n\n{volatility_section}\n\n{volume_section}\n\n## Statistical Metrics:\n- Hurst:{self.format_utils.fmt_ta(td, 'hurst', 2)} Z:{self.format_utils.fmt_ta(td, 'zscore', 2)} Kurt:{self.format_utils.fmt_ta(td, 'kurtosis', 2)}\n- Entropy:{self.format_utils.fmt_ta(td, 'entropy', 3)} Skew:{self.format_utils.fmt_ta(td, 'skewness', 2)} Var:{self.format_utils.fmt_ta(td, 'variance', 8)}\n- LinReg: Slope:{self.format_utils.fmt_ta(td, 'linreg_slope', 8)} R²:{self.format_utils.fmt_ta(td, 'linreg_r2', 3)}\n\n{key_levels_section}\n\n{advanced_section}\n\n{patterns_section}"""

        return technical_analysis
    
    def format_price_action_section(self, context, td: dict) -> str:
        """Format price action section with OHLCV temporal context (last 24 candles).
        
        Shows text-based trend descriptions for:
        - Price trend direction (Close)
        - Volatility expansion/contraction (High/Low range)
        - Volume confirmation (Volume trend)
        """
        try:
            
            # Get OHLCV data from context
            ohlcv_data = context.ohlcv_candles
            if ohlcv_data is None or len(ohlcv_data) < 2:
                # Fallback to simple format
                return f"## Price Action:\n- Price:{self.format_utils.fmt(context.current_price)} | VWAP:{self.format_utils.fmt_ta(td, 'vwap', 8)} TWAP:{self.format_utils.fmt_ta(td, 'twap', 8)}"
            
            # Extract last 24 candles (or all available)
            lookback = 24
            ohlcv_slice = ohlcv_data[-lookback:] if len(ohlcv_data) >= lookback else ohlcv_data
            
            # Extract OHLCV columns (timestamp, open, high, low, close, volume)
            opens = ohlcv_slice[:, 1]
            highs = ohlcv_slice[:, 2]
            lows = ohlcv_slice[:, 3]
            closes = ohlcv_slice[:, 4]
            volumes = ohlcv_slice[:, 5]
            
            # Calculate close trend
            close_delta = float(closes[-1] - closes[0])
            close_delta_pct = (close_delta / closes[0] * 100) if closes[0] != 0 else 0
            
            # Count green/red candles
            green_candles = sum(1 for i in range(len(closes)) if closes[i] >= opens[i])
            red_candles = len(closes) - green_candles
            
            # Determine close trend text
            if abs(close_delta_pct) < 1.0:
                close_trend = f"→FLAT ({green_candles}G/{red_candles}R, {close_delta_pct:+.1f}%)"
            elif close_delta_pct > 0:
                close_trend = f"↑RISING ({green_candles}G/{red_candles}R, {close_delta_pct:+.1f}%)"
            else:
                close_trend = f"↓FALLING ({green_candles}G/{red_candles}R, {close_delta_pct:+.1f}%)"
            
            # Calculate volume trend
            volume_delta = float(volumes[-1] - volumes[0])
            volume_delta_pct = (volume_delta / volumes[0] * 100) if volumes[0] != 0 else 0
            
            if abs(volume_delta_pct) < 10.0:
                volume_trend = f"→STABLE ({volume_delta_pct:+.0f}%)"
            elif volume_delta_pct > 0:
                volume_trend = f"↑INCREASING ({volume_delta_pct:+.0f}%)"
            else:
                volume_trend = f"↓DECLINING ({volume_delta_pct:+.0f}%)"
            
            # Calculate High-Low range evolution (volatility indicator)
            hl_ranges = highs - lows
            hl_range_current = float(hl_ranges[-1])
            hl_range_avg = float(np.mean(hl_ranges))
            range_expansion = ((hl_range_current - hl_range_avg) / hl_range_avg * 100) if hl_range_avg != 0 else 0
            
            # Volatility assessment
            if abs(range_expansion) < 15.0:
                volatility_text = "NORMAL"
            elif range_expansion > 0:
                volatility_text = f"EXPANDING (+{range_expansion:.0f}%)"
            else:
                volatility_text = f"CONTRACTING ({range_expansion:.0f}%)"
            
            # Build OHLCV section with text-based descriptions
            price_action = (
                "## Price Action:\n"
                f"- Price:{self.format_utils.fmt(context.current_price)} | Close Trend: {close_trend}\n"
                f"- Range: {self.format_utils.fmt(hl_range_current)} ({volatility_text}) | H:{self.format_utils.fmt(float(highs[-1]))} L:{self.format_utils.fmt(float(lows[-1]))}\n"
                f"- Volume: {volume_trend} | VWAP:{self.format_utils.fmt_ta(td, 'vwap', 8)} TWAP:{self.format_utils.fmt_ta(td, 'twap', 8)}"
            )
            
            return price_action
            
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Error formatting price action with OHLCV: {e}")
            # Fallback to simple format
            return f"## Price Action:\n- Price:{self.format_utils.fmt(context.current_price)} | VWAP:{self.format_utils.fmt_ta(td, 'vwap', 8)} TWAP:{self.format_utils.fmt_ta(td, 'twap', 8)}"
    
    def format_momentum_section(self, td: dict) -> str:
        """Format the momentum indicators section with temporal context (last 12 candles)."""
        # Get temporal arrays for critical momentum indicators
        rsi_temporal = self._format_temporal_array(td, 'rsi', 12, 1)
        macd_hist_temporal = self._format_temporal_array(td, 'macd_hist', 12, 8)
        stoch_k_temporal = self._format_temporal_array(td, 'stoch_k', 12, 1)
        
        return f"""## Momentum Indicators:
- RSI:{self.format_utils.fmt_ta(td, 'rsi', 1)}{rsi_temporal} | MACD:{self.format_utils.fmt_ta(td, 'macd_line', 8)}/{self.format_utils.fmt_ta(td, 'macd_signal', 8)} Hist:{self.format_utils.fmt_ta(td, 'macd_hist', 8)}{macd_hist_temporal}
- Stoch %K:{self.format_utils.fmt_ta(td, 'stoch_k', 1)}{stoch_k_temporal} %D:{self.format_utils.fmt_ta(td, 'stoch_d', 1)} | Williams %R:{self.format_utils.fmt_ta(td, 'williams_r', 1)}
- TSI:{self.format_utils.fmt_ta(td, 'tsi', 2)} | RMI:{self.format_utils.fmt_ta(td, 'rmi', 1)} | PPO:{self.format_utils.fmt_ta(td, 'ppo', 2)} | ROC:{self.format_utils.fmt_ta(td, 'roc_14', 2)}"""

    def format_trend_section(self, td: dict) -> str:
        """Format the trend indicators section with temporal context for trend strength evolution."""
        supertrend_direction = self.format_utils.get_supertrend_direction_string(td.get('supertrend_direction', 0))
        
        # TD Sequential (trend exhaustion indicator)
        td_seq_str = self._format_td_sequential(td)
        
        # SMA structure and crossovers
        sma_str = self._format_sma_structure(td)
        
        # Ichimoku cloud position
        ichimoku_str = self._format_ichimoku_signal(td)
        
        # ADX temporal array to show trend strength evolution
        adx_temporal = self._format_temporal_array(td, 'adx', 12, 1)

        return (
            "## Trend Indicators:\n"
            f"- ADX:{self.format_utils.fmt_ta(td, 'adx', 1)}{adx_temporal} +DI:{self.format_utils.fmt_ta(td, 'plus_di', 1)} -DI:{self.format_utils.fmt_ta(td, 'minus_di', 1)} | Supertrend:{supertrend_direction}{td_seq_str}\n"
            f"- TRIX:{self.format_utils.fmt_ta(td, 'trix', 4)} | PFE:{self.format_utils.fmt_ta(td, 'pfe', 2)}{ichimoku_str}\n"
            f"- Vortex+ VI+:{self.format_utils.fmt_ta(td, 'vortex_plus', 2)} VI-:{self.format_utils.fmt_ta(td, 'vortex_minus', 2)}\n"
            f"{sma_str}"
        )

    def format_volume_section(self, td: dict) -> str:
        """Format the volume indicators section with temporal context for volume trends."""
        cmf_interpretation = self.format_utils.format_cmf_interpretation(td)
        
        mfi_temporal = self._format_temporal_array(td, 'mfi', 12, 1)
        
        return (
            "## Volume Indicators:\n"
            f"- MFI:{self.format_utils.fmt_ta(td, 'mfi', 1)}{mfi_temporal} | OBV:{self.format_utils.fmt_ta(td, 'obv', 0)}\n"
            f"- Chaikin MF:{self.format_utils.fmt_ta(td, 'cmf', 4)}{cmf_interpretation} | Force Index:{self.format_utils.fmt_ta(td, 'force_index', 0)}\n"
            f"- PVT:{self.format_utils.fmt_ta(td, 'pvt', 0)} | AD Line:{self.format_utils.fmt_ta(td, 'ad_line', 0)}"
        )

    def format_volatility_section(self, td: dict, crypto_data: dict) -> str:
        """Format the volatility indicators section with temporal context for volatility evolution."""
        bb_interpretation = self.format_utils.format_bollinger_interpretation(td)
        
        # ATR temporal array to show volatility expansion/contraction
        atr_temporal = self._format_temporal_array(td, 'atr', 12, 8)
        
        # BB %B temporal array to show price position in bands over time
        bb_percent_b_temporal = self._format_temporal_array(td, 'bb_percent_b', 12, 2)
        
        # Choppiness Index interpretation
        chop_str = self._format_choppiness(td)
        
        return (
            "## Volatility Indicators:\n"
            f"- BB: U:{self.format_utils.fmt_ta(td, 'bb_upper', 8)} M:{self.format_utils.fmt_ta(td, 'bb_middle', 8)} L:{self.format_utils.fmt_ta(td, 'bb_lower', 8)}{bb_interpretation} %B:{self.format_utils.fmt_ta(td, 'bb_percent_b', 2)}{bb_percent_b_temporal}\n"
            f"- ATR:{self.format_utils.fmt_ta(td, 'atr', 8)}{atr_temporal} | KC: U:{self.format_utils.fmt_ta(td, 'kc_upper', 8)} M:{self.format_utils.fmt_ta(td, 'kc_middle', 8)} L:{self.format_utils.fmt_ta(td, 'kc_lower', 8)}\n"
            f"{chop_str}"
        )

    def format_key_levels_section(self, td: dict) -> str:
        """Format key levels section (compressed format)."""
        return (
            "## Key Levels:\n"
            f"- S/R: Support:{self.format_utils.fmt_ta(td, 'basic_support', 8)} Resistance:{self.format_utils.fmt_ta(td, 'basic_resistance', 8)}\n"
            f"- Pivot:{self.format_utils.fmt_ta(td, 'pivot_point', 8)} S[{self.format_utils.fmt_ta(td, 'pivot_s1', 8)},{self.format_utils.fmt_ta(td, 'pivot_s2', 8)},{self.format_utils.fmt_ta(td, 'pivot_s3', 8)}] R[{self.format_utils.fmt_ta(td, 'pivot_r1', 8)},{self.format_utils.fmt_ta(td, 'pivot_r2', 8)},{self.format_utils.fmt_ta(td, 'pivot_r3', 8)}]\n"
            f"- FibPivot:{self.format_utils.fmt_ta(td, 'fib_pivot_point', 8)} S[{self.format_utils.fmt_ta(td, 'fib_pivot_s1', 8)},{self.format_utils.fmt_ta(td, 'fib_pivot_s2', 8)}] R[{self.format_utils.fmt_ta(td, 'fib_pivot_r1', 8)},{self.format_utils.fmt_ta(td, 'fib_pivot_r2', 8)}]"
        )

    def format_advanced_indicators_section(self, td: dict, crypto_data: dict) -> str:
        """Format advanced indicators section with temporal context for advanced signals."""
        # CCI temporal array to show commodity channel momentum
        cci_temporal = self._format_temporal_array(td, 'cci', 12, 1)
        
        # Coppock temporal array to show long-term momentum shifts
        coppock_temporal = self._format_temporal_array(td, 'coppock', 12, 2)
        
        # KST temporal array to show know sure thing momentum
        kst_temporal = self._format_temporal_array(td, 'kst', 12, 2)
        
        return (
            "## Advanced Indicators:\n"
            f"- Adv S/R: {self.format_utils.fmt_ta(td, 'advanced_support', 8)}/{self.format_utils.fmt_ta(td, 'advanced_resistance', 8)}\n"
            f"- CCI:{self.format_utils.fmt_ta(td, 'cci', 1)}{cci_temporal} | ATR%:{self.format_utils.fmt_ta(td, 'atr_percent', 2)}% | SAR:{self.format_utils.fmt_ta(td, 'sar', 8)}\n"
            f"- Donchian: U:{self.format_utils.fmt_ta(td, 'donchian_upper', 8)} L:{self.format_utils.fmt_ta(td, 'donchian_lower', 8)}\n"
            f"- UltOsc:{self.format_utils.fmt_ta(td, 'uo', 1)} | Coppock:{self.format_utils.fmt_ta(td, 'coppock', 2)}{coppock_temporal} | KST:{self.format_utils.fmt_ta(td, 'kst', 2)}{kst_temporal}\n"
            f"- Chandelier: Long:{self.format_utils.fmt_ta(td, 'chandelier_long', 8)} Short:{self.format_utils.fmt_ta(td, 'chandelier_short', 8)}"
        )
    
    def _format_patterns_section(self, context, timeframe: str = '4h') -> str:
        """Format patterns section using detected patterns from context.
        
        Args:
            context: Analysis context containing technical data
            timeframe: Timeframe string for dynamic threshold calculation
            
        Returns:
            str: Formatted patterns section
        """
        if context.technical_patterns:
            try:
                pattern_summaries = []
                last_candle_index = len(context.ohlcv_candles) - 1 if context.ohlcv_candles is not None else None
                # Deduplication: track most recent pattern per (category, base_type)
                # This prevents showing redundant signals like:
                #   - Stochastic bull × 2 bars ago + Stochastic bear × 3 bars ago
                # Instead, only the most recent crossover per indicator is shown.
                dedup_tracker: dict[tuple[str, str], dict] = {}
                for category, patterns_list in context.technical_patterns.items():
                    if patterns_list:  # Only process non-empty pattern lists
                        for pattern_dict in patterns_list:
                            # Filter patterns based on recency relative to total candles analyzed
                            # Strategy: Show patterns from most recent 20% of data (e.g., last 200 of 999 candles)
                            # This makes filtering adaptive to different timeframes and candle counts
                            pattern_index = pattern_dict.get('index', None)
                            # Determine recency threshold based on pattern type and data size
                            if last_candle_index is not None and pattern_index is not None:
                                total_candles = last_candle_index + 1
                                periods_ago = last_candle_index - pattern_index
                                
                                # Category-specific ABSOLUTE thresholds (in bars) based on timeframe
                                # These ensure stale patterns are filtered regardless of data window size
                                abs_threshold = self._calculate_staleness_threshold(category, timeframe)
                                
                                # Calculate percentage-based threshold (existing logic)
                                if category == 'ma_crossover':
                                    pct_threshold = int(total_candles * 0.3)
                                elif category in ['volatility', 'volume']:
                                    pct_threshold = max(10, int(total_candles * 0.05))
                                elif category == 'divergence':
                                    pct_threshold = max(20, int(total_candles * 0.10))
                                else:
                                    pct_threshold = max(20, int(total_candles * 0.15))
                                
                                # Use the MORE RESTRICTIVE of the two thresholds
                                recency_threshold = min(abs_threshold, pct_threshold)
                                
                                is_recent = periods_ago <= recency_threshold
                            else:
                                # If no index info, include the pattern
                                is_recent = True
                            
                            if is_recent:
                                # Extract base type for deduplication
                                # e.g., 'stoch_bullish_crossover' -> 'stoch_crossover'
                                # This groups bullish/bearish variants together
                                pattern_type = pattern_dict.get('type', '')
                                base_type = self._get_dedup_key(category, pattern_type)
                                dedup_key = (category, base_type)
                                periods_ago_val = pattern_dict.get('details', {}).get('periods_ago', 999)
                                # Keep only the most recent pattern per dedup key
                                if dedup_key not in dedup_tracker or periods_ago_val < dedup_tracker[dedup_key]['periods']:
                                    dedup_tracker[dedup_key] = {
                                        'pattern': pattern_dict,
                                        'periods': periods_ago_val,
                                        'category': category
                                    }
                # Convert dedup_tracker to pattern_summaries
                for dedup_key, entry in dedup_tracker.items():
                    pattern_dict = entry['pattern']
                    category = entry['category']
                    description = pattern_dict.get('description', f'Unknown {category} pattern')
                    compressed_desc = self._compress_pattern_description(description)
                    pattern_summaries.append(f"- {compressed_desc}")
                if pattern_summaries:
                    if self.logger:
                        self.logger.debug(f"Including {len(pattern_summaries)} recent patterns in technical analysis (dedup + recency filter)")
                    return "\n\n## Detected Patterns:\n" + "\n".join(pattern_summaries[-25:])
            except Exception as e:
                if self.logger:
                    self.logger.debug(f"Error using stored technical_patterns: {e}")
        
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
                    compressed_desc = self._compress_pattern_description(description)
                    pattern_summaries.append(f"- {compressed_desc}")
                
                if pattern_summaries:
                    return "\n\n## Detected Patterns:\n" + "\n".join(pattern_summaries)
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Could not use fallback pattern detection: {e}")
        
        return ""
    
    def _compress_pattern_description(self, description: str) -> str:
        """Compress pattern descriptions to save tokens (e.g., 'MACD bearish×now' instead of 'MACD bearish crossover now at 2025-12-23 18:00:00').
        
        Args:
            description: Original pattern description
            
        Returns:
            str: Compressed pattern description
        """

        
        # Remove full timestamps (keep relative time if present)
        description = re.sub(r' at \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?: UTC)?', '', description)
        
        # Remove index numbers
        description = re.sub(r' \(index \d+\)', '', description)
        
        # Compress common words
        replacements = {
            'crossover': '×',
            'bullish': 'bull',
            'bearish': 'bear',
            'histogram': 'hist',
            'periods ago': 'bars ago',
            'period ago': 'bar ago',
            'detected': '',
            'squeeze': 'sqz',
            'breakout imminent': 'breakout',
            'zero-line cross': 'zero×',
        }
        
        for old, new in replacements.items():
            description = description.replace(old, new)
        
        # Compress multiple spaces
        description = re.sub(r'\s+', ' ', description).strip()
        
        return description
    def _get_dedup_key(self, category: str, pattern_type: str) -> str:
        """Extract base pattern type for deduplication.
        Groups bullish/bearish variants together, e.g.:
        - 'stoch_bullish_crossover' -> 'stoch_crossover'
        - 'macd_bullish_zero_cross' -> 'macd_zero_cross'
        - 'rsi_oversold' -> 'rsi_level' (group with overbought)
        - 'rsi_overbought' -> 'rsi_level'
        """
        # Remove directional qualifiers to group similar patterns
        base = pattern_type.replace('bullish_', '').replace('bearish_', '')
        # Group oversold/overbought together
        if 'oversold' in base or 'overbought' in base:
            return f"{category}_level"
        # Group divergences by indicator
        if 'divergence' in base:
            return base.split('_')[0] + '_divergence'
        return base
    
    def _calculate_staleness_threshold(self, category: str, timeframe: str) -> int:
        """Calculate staleness threshold dynamically based on timeframe."""
        try:
            # Get minutes per candle, defaulting to 4h (240m) on error
            minutes_per_candle = TimeframeValidator.to_minutes(timeframe)
        except (ValueError, TypeError):
            minutes_per_candle = 240 
        
        # Target hours relative to pattern significance
        target_hours = {
            'rsi': 40,           # Momentum: ~40h
            'macd': 40,          # Momentum: ~40h
            'stochastic': 40,    # Momentum: ~40h
            'ma_crossover': 200, # Trend: ~8 days
            'divergence': 80,    # Divergence: ~3 days
            'volatility': 20,    # Volatility: ~20h
            'volume': 40,        # Volume: ~40h
        }
        
        target_minutes = target_hours.get(category, 40) * 60
        # Ensure at least 1 bar
        return max(1, target_minutes // minutes_per_candle)

    def _format_td_sequential(self, td: dict) -> str:
        """Format TD Sequential indicator (trend exhaustion detector).
        
        TD Sequential counts consecutive candles (up to 9) where close > close[4] (bullish)
        or close < close[4] (bearish). Count of 8-9 signals potential trend exhaustion.
        """
        td_seq = td.get('td_sequential')
        if td_seq is None:
            return ""
        
        td_val = get_last_valid_value(td_seq)
        if td_val is None:
            return ""
        if td_val > 0:
            count = int(abs(td_val))
            return f" | TD:{count}↑⚠️" if count >= 8 else f" | TD:{count}↑" if count >= 1 else ""
        elif td_val < 0:
            count = int(abs(td_val))
            return f" | TD:{count}↓⚠️" if count >= 8 else f" | TD:{count}↓" if count >= 1 else ""
        return ""
    
    def _format_sma_structure(self, td: dict) -> str:
        """Format SMA structure with crossover detection and dynamic S/R levels.
        
        Shows SMA 20/50/200 values and detects:
        - Golden Cross (50 > 200, bullish)
        - Death Cross (50 < 200, bearish)
        - Price position relative to SMAs
        """
        try:
            sma_20 = td.get('sma_20')
            sma_50 = td.get('sma_50')
            sma_200 = td.get('sma_200')
            
            # Extract last values from arrays using shared utility
            sma_20_val = get_last_valid_value(sma_20)
            sma_50_val = get_last_valid_value(sma_50)
            sma_200_val = get_last_valid_value(sma_200)
            
            # Build SMA line
            sma_parts = []
            if sma_20_val is not None:
                sma_parts.append(f"20:{self.format_utils.fmt(sma_20_val, 8)}")
            if sma_50_val is not None:
                sma_parts.append(f"50:{self.format_utils.fmt(sma_50_val, 8)}")
            if sma_200_val is not None:
                sma_parts.append(f"200:{self.format_utils.fmt(sma_200_val, 8)}")
            
            if not sma_parts:
                return ""
            
            # Detect crossovers
            cross_signal = ""
            if sma_50_val is not None and sma_200_val is not None:
                if sma_50_val > sma_200_val:
                    cross_signal = " | Golden×"
                elif sma_50_val < sma_200_val:
                    cross_signal = " | Death×"
            
            return f"- SMAs: {' '.join(sma_parts)}{cross_signal}"
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Error formatting SMA structure: {e}")
            return ""
    
    def _format_ichimoku_signal(self, td: dict) -> str:
        """Format Ichimoku cloud position signal.
        
        Shows whether price is above cloud (bullish), below cloud (bearish), or in cloud (neutral).
        Calculates signal on-demand from raw span data.
        """
        try:
            # Get ichimoku spans
            span_a = td.get('ichimoku_span_a')
            span_b = td.get('ichimoku_span_b')
            
            if span_a is None or span_b is None:
                return ""
            
            # Extract last values
            span_a_val = get_last_valid_value(span_a)
            span_b_val = get_last_valid_value(span_b)
            
            if span_a_val is None or span_b_val is None:
                return ""
            
            # Get current price - try multiple sources
            current_price = None
            
            # Method 1: Try from close prices in technical data
            close_data = td.get('close')
            if close_data is not None:
                from src.utils.data_utils import safe_array_to_scalar
                current_price = safe_array_to_scalar(close_data, -1)
            
            if current_price is None:
                return ""
            
            # Calculate cloud boundaries
            cloud_top = max(span_a_val, span_b_val)
            cloud_bottom = min(span_a_val, span_b_val)
            
            # Determine signal based on price position relative to cloud
            if current_price > cloud_top:
                return " | Ichi:☁️↑"
            elif current_price < cloud_bottom:
                return " | Ichi:☁️↓"
            else:
                return " | Ichi:☁️="
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Error calculating ichimoku signal: {e}")
            return ""
    
    def _format_temporal_array(self, td: dict, key: str, lookback: int, decimals: int) -> str:
        """Format temporal array with text-based trend description.

        Args:
            td: Technical data dictionary
            key: Indicator key name
            lookback: Number of historical candles to analyze
            decimals: Decimal places for formatting

        Returns:
            Formatted string with trend direction and delta, e.g., " (↑UP Δ+5.2)"
        """
        indicator_data = td.get(key)
        if indicator_data is None:
            return ""
        if not isinstance(indicator_data, np.ndarray):
            return ""
        last_n = get_last_n_valid(indicator_data, lookback)
        if len(last_n) < 2:
            return ""
        delta = float(last_n[-1] - last_n[0])
        delta_sign = "+" if delta >= 0 else ""
        if abs(last_n[0]) > 0.0001:
            delta_pct = abs(delta / last_n[0]) * 100
        else:
            delta_pct = abs(delta) * 100
        threshold_pct = 3.0 if key in ['rsi', 'stoch_k', 'stoch_d', 'mfi', 'cci'] else 5.0
        if delta_pct >= threshold_pct:
            trend_text = "↑UP" if delta > 0 else "↓DOWN"
        else:
            trend_text = "→FLAT"
        return f" ({trend_text} Δ{delta_sign}{delta:.{decimals}f})"
    
    def _format_choppiness(self, td: dict) -> str:
        """Format Choppiness Index with market state interpretation.
        
        Thresholds: >61.8 Choppy, <38.2 Trending, else Transition.
        """
        chop = td.get('choppiness')
        if chop is None:
            return ""
        chop_val = get_last_valid_value(chop)
        if chop_val is None:
            return ""
        if chop_val > 61.8:
            state = "Chop"
        elif chop_val < 38.2:
            state = "Trend"
        else:
            state = "Transition"
        return f"- Choppiness:{chop_val:.1f} ({state})"

