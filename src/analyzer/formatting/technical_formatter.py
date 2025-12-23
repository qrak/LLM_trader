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
        price_action_section = self.format_price_action_section(context, td)
        momentum_section = self.format_momentum_section(td)
        trend_section = self.format_trend_section(td)
        volume_section = self.format_volume_section(td)
        volatility_section = self.format_volatility_section(td, crypto_data)
        advanced_section = self.format_advanced_indicators_section(td, crypto_data)
        key_levels_section = self.format_key_levels_section(td)

        # Build main technical analysis content
        technical_analysis = f"""\nTECHNICAL ANALYSIS ({timeframe}):\n\n{price_action_section}\n\n{momentum_section}\n\n{trend_section}\n\n{volatility_section}\n\n{volume_section}\n\n## Statistical Metrics:\n- Hurst:{self.format_utils.fmt_ta(self.technical_calculator, td, 'hurst', 2)} | Z-Score:{self.format_utils.fmt_ta(self.technical_calculator, td, 'zscore', 2)} | Kurtosis:{self.format_utils.fmt_ta(self.technical_calculator, td, 'kurtosis', 2)}\n\n{key_levels_section}\n\n{advanced_section}\n\n{patterns_section}"""

        return technical_analysis
    
    def format_price_action_section(self, context, td: dict) -> str:
        """Format price action section with OHLCV temporal context (last 24 candles).
        
        Shows individual sparklines for Open, High, Low, Close, and Volume to reveal:
        - Price trend direction (Close)
        - Volatility expansion/contraction (High/Low range)
        - Gap patterns (Open vs previous Close)
        - Volume confirmation (Volume trend)
        """
        try:
            import numpy as np
            
            # Get OHLCV data from context
            ohlcv_data = context.ohlcv_candles
            if ohlcv_data is None or len(ohlcv_data) < 2:
                # Fallback to simple format
                return f"## Price Action:\n- Price:{self.format_utils.fmt(context.current_price)} | VWAP:{self.format_utils.fmt_ta(self.technical_calculator, td, 'vwap', 8)} TWAP:{self.format_utils.fmt_ta(self.technical_calculator, td, 'twap', 8)}"
            
            # Extract last 24 candles (or all available)
            lookback = 24
            ohlcv_slice = ohlcv_data[-lookback:] if len(ohlcv_data) >= lookback else ohlcv_data
            
            # Extract OHLCV columns (timestamp, open, high, low, close, volume)
            opens = ohlcv_slice[:, 1]
            highs = ohlcv_slice[:, 2]
            lows = ohlcv_slice[:, 3]
            closes = ohlcv_slice[:, 4]
            volumes = ohlcv_slice[:, 5]
            
            # Generate sparklines and deltas
            close_sparkline = self._generate_sparkline(closes)
            close_delta = float(closes[-1] - closes[0])
            close_delta_pct = (close_delta / closes[0] * 100) if closes[0] != 0 else 0
            
            open_sparkline = self._generate_sparkline(opens)
            open_delta = float(opens[-1] - opens[0])
            
            high_sparkline = self._generate_sparkline(highs)
            high_delta = float(highs[-1] - highs[0])
            
            low_sparkline = self._generate_sparkline(lows)
            low_delta = float(lows[-1] - lows[0])
            
            volume_sparkline = self._generate_sparkline(volumes)
            volume_delta = float(volumes[-1] - volumes[0])
            volume_delta_pct = (volume_delta / volumes[0] * 100) if volumes[0] != 0 else 0
            
            # Calculate High-Low range evolution (volatility indicator)
            hl_ranges = highs - lows
            hl_range_current = float(hl_ranges[-1])
            hl_range_avg = float(np.mean(hl_ranges))
            range_expansion = ((hl_range_current - hl_range_avg) / hl_range_avg * 100) if hl_range_avg != 0 else 0
            
            # Format with signs
            close_sign = "+" if close_delta >= 0 else ""
            close_pct_sign = "+" if close_delta_pct >= 0 else ""
            volume_sign = "+" if volume_delta_pct >= 0 else ""
            range_sign = "+" if range_expansion >= 0 else ""
            
            # Build OHLCV section
            price_action = (
                "## Price Action:\n"
                f"- Price:{self.format_utils.fmt(context.current_price)} | Close[{close_sparkline}](Δ{close_sign}{close_delta:.2f} {close_pct_sign}{close_delta_pct:.1f}%)\n"
                f"- OHLC: O[{open_sparkline}] H[{high_sparkline}] L[{low_sparkline}] | Range:{self.format_utils.fmt(hl_range_current)}({range_sign}{range_expansion:.0f}%)\n"
                f"- Volume[{volume_sparkline}](Δ{volume_sign}{volume_delta_pct:.0f}%) | VWAP:{self.format_utils.fmt_ta(self.technical_calculator, td, 'vwap', 8)} TWAP:{self.format_utils.fmt_ta(self.technical_calculator, td, 'twap', 8)}"
            )
            
            return price_action
            
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Error formatting price action with OHLCV: {e}")
            # Fallback to simple format
            return f"## Price Action:\n- Price:{self.format_utils.fmt(context.current_price)} | VWAP:{self.format_utils.fmt_ta(self.technical_calculator, td, 'vwap', 8)} TWAP:{self.format_utils.fmt_ta(self.technical_calculator, td, 'twap', 8)}"
    
    def format_momentum_section(self, td: dict) -> str:
        """Format the momentum indicators section with temporal context (last 12 candles)."""
        # Get temporal arrays for critical momentum indicators
        rsi_temporal = self._format_temporal_array(td, 'rsi', 12, 1)
        macd_hist_temporal = self._format_temporal_array(td, 'macd_hist', 12, 8)
        stoch_k_temporal = self._format_temporal_array(td, 'stoch_k', 12, 1)
        
        return f"""## Momentum Indicators:
- RSI:{self.format_utils.fmt_ta(self.technical_calculator, td, 'rsi', 1)}{rsi_temporal} | MACD:{self.format_utils.fmt_ta(self.technical_calculator, td, 'macd_line', 8)}/{self.format_utils.fmt_ta(self.technical_calculator, td, 'macd_signal', 8)} Hist:{self.format_utils.fmt_ta(self.technical_calculator, td, 'macd_hist', 8)}{macd_hist_temporal}
- Stoch %K:{self.format_utils.fmt_ta(self.technical_calculator, td, 'stoch_k', 1)}{stoch_k_temporal} %D:{self.format_utils.fmt_ta(self.technical_calculator, td, 'stoch_d', 1)} | Williams %R:{self.format_utils.fmt_ta(self.technical_calculator, td, 'williams_r', 1)}
- TSI:{self.format_utils.fmt_ta(self.technical_calculator, td, 'tsi', 2)} | RMI:{self.format_utils.fmt_ta(self.technical_calculator, td, 'rmi', 1)} | PPO:{self.format_utils.fmt_ta(self.technical_calculator, td, 'ppo', 2)}"""

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
            f"- ADX:{self.format_utils.fmt_ta(self.technical_calculator, td, 'adx', 1)}{adx_temporal} +DI:{self.format_utils.fmt_ta(self.technical_calculator, td, 'plus_di', 1)} -DI:{self.format_utils.fmt_ta(self.technical_calculator, td, 'minus_di', 1)} | Supertrend:{supertrend_direction}{td_seq_str}\n"
            f"- TRIX:{self.format_utils.fmt_ta(self.technical_calculator, td, 'trix', 4)} | PFE:{self.format_utils.fmt_ta(self.technical_calculator, td, 'pfe', 2)}{ichimoku_str}\n"
            f"- Vortex+ VI+:{self.format_utils.fmt_ta(self.technical_calculator, td, 'vortex_plus', 2)} VI-:{self.format_utils.fmt_ta(self.technical_calculator, td, 'vortex_minus', 2)}\n"
            f"{sma_str}"
        )

    def format_volume_section(self, td: dict) -> str:
        """Format the volume indicators section with temporal context for volume trends."""
        cmf_interpretation = self.format_utils.format_cmf_interpretation(self.technical_calculator, td)
        
        # MFI temporal array to show buying/selling pressure evolution
        mfi_temporal = self._format_temporal_array(td, 'mfi', 12, 1)
        
        return (
            "## Volume Indicators:\n"
            f"- MFI:{self.format_utils.fmt_ta(self.technical_calculator, td, 'mfi', 1)}{mfi_temporal} | OBV:{self.format_utils.fmt_ta(self.technical_calculator, td, 'obv', 0)}\n"
            f"- Chaikin MF:{self.format_utils.fmt_ta(self.technical_calculator, td, 'cmf', 4)}{cmf_interpretation} | Force Index:{self.format_utils.fmt_ta(self.technical_calculator, td, 'force_index', 0)}"
        )

    def format_volatility_section(self, td: dict, crypto_data: dict) -> str:
        """Format the volatility indicators section with temporal context for volatility evolution."""
        bb_interpretation = self.format_utils.format_bollinger_interpretation(self.technical_calculator, td)
        
        # ATR temporal array to show volatility expansion/contraction
        atr_temporal = self._format_temporal_array(td, 'atr', 12, 8)
        
        # BB %B temporal array to show price position in bands over time
        bb_percent_b_temporal = self._format_temporal_array(td, 'bb_percent_b', 12, 2)
        
        return (
            "## Volatility Indicators:\n"
            f"- BB: U:{self.format_utils.fmt_ta(self.technical_calculator, td, 'bb_upper', 8)} M:{self.format_utils.fmt_ta(self.technical_calculator, td, 'bb_middle', 8)} L:{self.format_utils.fmt_ta(self.technical_calculator, td, 'bb_lower', 8)}{bb_interpretation} %B:{self.format_utils.fmt_ta(self.technical_calculator, td, 'bb_percent_b', 2)}{bb_percent_b_temporal}\n"
            f"- ATR:{self.format_utils.fmt_ta(self.technical_calculator, td, 'atr', 8)}{atr_temporal} | KC: U:{self.format_utils.fmt_ta(self.technical_calculator, td, 'kc_upper', 8)} M:{self.format_utils.fmt_ta(self.technical_calculator, td, 'kc_middle', 8)} L:{self.format_utils.fmt_ta(self.technical_calculator, td, 'kc_lower', 8)}"
        )

    def format_key_levels_section(self, td: dict) -> str:
        """Format key levels section (compressed format)."""
        # Format Fibonacci retracement levels if available (compressed)
        fib_section = ""
        if 'fibonacci_retracement' in td:
            fib_levels = td['fibonacci_retracement']
            if isinstance(fib_levels, list) and len(fib_levels) == 7:
                fib_section = (
                    f" | Fib50: 0.0={self.format_utils.format_number(fib_levels[0], 8)} "
                    f"0.382={self.format_utils.format_number(fib_levels[2], 8)} "
                    f"0.618={self.format_utils.format_number(fib_levels[4], 8)} "
                    f"1.0={self.format_utils.format_number(fib_levels[6], 8)}"
                )
        
        return (
            "## Key Levels:\n"
            f"- S/R: Support:{self.format_utils.fmt_ta(self.technical_calculator, td, 'basic_support', 8)} Resistance:{self.format_utils.fmt_ta(self.technical_calculator, td, 'basic_resistance', 8)}\n"
            f"- Pivot:{self.format_utils.fmt_ta(self.technical_calculator, td, 'pivot_point', 8)} S[{self.format_utils.fmt_ta(self.technical_calculator, td, 'pivot_s1', 8)},{self.format_utils.fmt_ta(self.technical_calculator, td, 'pivot_s2', 8)},{self.format_utils.fmt_ta(self.technical_calculator, td, 'pivot_s3', 8)}] R[{self.format_utils.fmt_ta(self.technical_calculator, td, 'pivot_r1', 8)},{self.format_utils.fmt_ta(self.technical_calculator, td, 'pivot_r2', 8)},{self.format_utils.fmt_ta(self.technical_calculator, td, 'pivot_r3', 8)}]\n"
            f"- FibPivot:{self.format_utils.fmt_ta(self.technical_calculator, td, 'fib_pivot_point', 8)} S[{self.format_utils.fmt_ta(self.technical_calculator, td, 'fib_pivot_s1', 8)},{self.format_utils.fmt_ta(self.technical_calculator, td, 'fib_pivot_s2', 8)}] R[{self.format_utils.fmt_ta(self.technical_calculator, td, 'fib_pivot_r1', 8)},{self.format_utils.fmt_ta(self.technical_calculator, td, 'fib_pivot_r2', 8)}]{fib_section}"
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
            f"- Adv S/R: {self.format_utils.fmt_ta(self.technical_calculator, td, 'advanced_support', 8)}/{self.format_utils.fmt_ta(self.technical_calculator, td, 'advanced_resistance', 8)}\n"
            f"- CCI:{self.format_utils.fmt_ta(self.technical_calculator, td, 'cci', 1)}{cci_temporal} | ATR%:{self.format_utils.fmt_ta(self.technical_calculator, td, 'atr_percent', 2)}% | SAR:{self.format_utils.fmt_ta(self.technical_calculator, td, 'sar', 8)}\n"
            f"- Donchian: U:{self.format_utils.fmt_ta(self.technical_calculator, td, 'donchian_upper', 8)} L:{self.format_utils.fmt_ta(self.technical_calculator, td, 'donchian_lower', 8)}\n"
            f"- UltOsc:{self.format_utils.fmt_ta(self.technical_calculator, td, 'uo', 1)} | Coppock:{self.format_utils.fmt_ta(self.technical_calculator, td, 'coppock', 2)}{coppock_temporal} | KST:{self.format_utils.fmt_ta(self.technical_calculator, td, 'kst', 2)}{kst_temporal}\n"
            f"- Chandelier: Long:{self.format_utils.fmt_ta(self.technical_calculator, td, 'chandelier_long', 8)} Short:{self.format_utils.fmt_ta(self.technical_calculator, td, 'chandelier_short', 8)}"
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
                                
                                # For long-term signals (MA crossovers), use wider window (30% of data)
                                if category == 'ma_crossover':
                                    recency_threshold = int(total_candles * 0.3)  # Last 30% of candles
                                # For persistent patterns (volatility, volume), use narrow window (5% of data)
                                elif category in ['volatility', 'volume']:
                                    recency_threshold = max(10, int(total_candles * 0.05))  # Last 5% or min 10 candles
                                # For divergences, use narrow window (10% of data) - they're time-sensitive
                                elif category == 'divergence':
                                    recency_threshold = max(20, int(total_candles * 0.10))  # Last 10% or min 20 candles
                                # For other patterns (crossovers, etc.), use moderate window (15% of data)
                                else:
                                    recency_threshold = max(20, int(total_candles * 0.15))  # Last 15% or min 20 candles
                                
                                is_recent = pattern_index >= (last_candle_index - recency_threshold)
                            else:
                                # If no index info, include the pattern
                                is_recent = True
                            
                            if is_recent:
                                description = pattern_dict.get('description', f'Unknown {category} pattern')
                                # Compress pattern descriptions for token efficiency
                                compressed_desc = self._compress_pattern_description(description)
                                pattern_summaries.append(f"- {compressed_desc}")
                
                if pattern_summaries:
                    if self.logger:
                        self.logger.debug(f"Including {len(pattern_summaries)} recent patterns in technical analysis (adaptive recency filter)")
                    return "\n\n## Detected Patterns:\n" + "\n".join(pattern_summaries[-25:])  # Show last 25 recent patterns
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
        import re
        
        # Remove full timestamps (keep relative time if present)
        description = re.sub(r' at \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', description)
        
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
    
    def _format_td_sequential(self, td: dict) -> str:
        """Format TD Sequential indicator (trend exhaustion detector).
        
        Returns empty string if no valid TD Sequential data.
        TD Sequential counts consecutive candles (up to 9) where close > close[4] (bullish) or close < close[4] (bearish).
        Count of 8-9 signals potential trend exhaustion and reversal zone.
        """
        try:
            td_seq = td.get('td_sequential')
            if td_seq is None:
                return ""
            
            # Get the last value
            if hasattr(td_seq, '__iter__') and not isinstance(td_seq, str):
                import numpy as np
                # Find last non-NaN value
                valid_indices = np.where(~np.isnan(td_seq))[0]
                if len(valid_indices) > 0:
                    td_val = float(td_seq[valid_indices[-1]])
                else:
                    return ""
            else:
                td_val = float(td_seq)
            
            # Format: positive = bullish count, negative = bearish count
            if td_val > 0:
                count = int(abs(td_val))
                if count >= 8:  # Exhaustion warning
                    return f" | TD:{count}↑⚠️"
                elif count >= 1:
                    return f" | TD:{count}↑"
            elif td_val < 0:
                count = int(abs(td_val))
                if count >= 8:  # Exhaustion warning
                    return f" | TD:{count}↓⚠️"
                elif count >= 1:
                    return f" | TD:{count}↓"
            
            return ""
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Error formatting TD Sequential: {e}")
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
            
            # Extract last values from arrays
            import numpy as np
            
            def get_last_valid(arr):
                if arr is None:
                    return None
                if hasattr(arr, '__iter__') and not isinstance(arr, str):
                    valid_idx = np.where(~np.isnan(arr))[0]
                    if len(valid_idx) > 0:
                        return float(arr[valid_idx[-1]])
                return float(arr) if not np.isnan(arr) else None
            
            sma_20_val = get_last_valid(sma_20)
            sma_50_val = get_last_valid(sma_50)
            sma_200_val = get_last_valid(sma_200)
            
            # Build SMA line
            sma_parts = []
            if sma_20_val is not None:
                sma_parts.append(f"20:{self.format_utils.format_number(sma_20_val, 8)}")
            if sma_50_val is not None:
                sma_parts.append(f"50:{self.format_utils.format_number(sma_50_val, 8)}")
            if sma_200_val is not None:
                sma_parts.append(f"200:{self.format_utils.format_number(sma_200_val, 8)}")
            
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
        """Format Ichimoku cloud position signal for regular timeframe.
        
        Shows whether price is above cloud (bullish), below cloud (bearish), or in cloud (neutral).
        """
        try:
            ichimoku_signal = td.get('ichimoku_signal')
            if ichimoku_signal is None:
                return ""
            
            # Get last value if array
            if hasattr(ichimoku_signal, '__iter__') and not isinstance(ichimoku_signal, str):
                import numpy as np
                valid_idx = np.where(~np.isnan(ichimoku_signal))[0]
                if len(valid_idx) > 0:
                    signal_val = int(ichimoku_signal[valid_idx[-1]])
                else:
                    return ""
            else:
                signal_val = int(ichimoku_signal)
            
            # Format signal
            if signal_val == 1:
                return " | Ichi:☁️↑"
            elif signal_val == -1:
                return " | Ichi:☁️↓"
            elif signal_val == 0:
                return " | Ichi:☁️="
            
            return ""
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Error formatting Ichimoku signal: {e}")
            return ""
    
    def _format_temporal_array(self, td: dict, key: str, lookback: int, decimals: int) -> str:
        """Format temporal array of indicator values with sparkline and momentum indicator.
        
        Args:
            td: Technical data dictionary
            key: Indicator key name
            lookback: Number of historical candles to show
            decimals: Decimal places for formatting
            
        Returns:
            Formatted string with sparkline and delta, e.g., " [▁▂▃▅▆▇█](Δ+5.2)"
            Empty string if data not available
        """
        try:
            import numpy as np
            
            indicator_data = td.get(key)
            if indicator_data is None:
                return ""
            
            # Extract array values
            if not hasattr(indicator_data, '__iter__') or isinstance(indicator_data, str):
                return ""  # Not an array
            
            # Get last N valid values
            valid_mask = ~np.isnan(indicator_data)
            if not np.any(valid_mask):
                return ""
            
            valid_data = indicator_data[valid_mask]
            if len(valid_data) < 2:
                return ""  # Need at least 2 points for trend
            
            # Get last N values
            last_n = valid_data[-lookback:] if len(valid_data) >= lookback else valid_data
            
            if len(last_n) < 2:
                return ""
            
            # Calculate delta (current - oldest in window)
            delta = float(last_n[-1] - last_n[0])
            delta_sign = "+" if delta >= 0 else ""
            
            # Generate sparkline (8 levels: ▁▂▃▄▅▆▇█)
            sparkline = self._generate_sparkline(last_n)
            
            # Format: [sparkline](Δdelta)
            return f" [{sparkline}](Δ{delta_sign}{delta:.{decimals}f})"
            
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Error formatting temporal array for {key}: {e}")
            return ""
    
    def _generate_sparkline(self, values) -> str:
        """Generate ASCII sparkline from array of values.
        
        Args:
            values: Array of numeric values (numpy array or list)
            
        Returns:
            Sparkline string using ▁▂▃▄▅▆▇█ characters
        """
        import numpy as np
        
        if len(values) == 0:
            return ""
        
        # Normalize to 0-7 range for 8 sparkline levels
        min_val = np.min(values)
        max_val = np.max(values)
        
        if max_val == min_val:
            # Flat line
            return "▄" * len(values)
        
        # Normalize and map to sparkline characters
        normalized = (values - min_val) / (max_val - min_val) * 7
        sparkline_chars = "▁▂▃▄▅▆▇█"
        
        sparkline = "".join([sparkline_chars[min(7, int(round(v)))] for v in normalized])
        
        return sparkline

