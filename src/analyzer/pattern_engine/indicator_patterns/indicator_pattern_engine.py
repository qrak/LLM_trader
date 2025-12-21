"""
Indicator Pattern Engine - Orchestrator

Coordinates all indicator pattern detection and returns unified results.
Complements chart pattern engine by providing momentum and confirmation signals.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

from .rsi_patterns import (
    detect_rsi_oversold_numba,
    detect_rsi_overbought_numba,
    detect_rsi_w_bottom_numba,
    detect_rsi_m_top_numba
)
from .macd_patterns import (
    detect_macd_crossover_numba,
    detect_macd_zero_cross_numba,
    get_macd_histogram_trend_numba
)
from .divergence_patterns import (
    detect_bullish_divergence_numba,
    detect_bearish_divergence_numba
)
from .volatility_patterns import (
    detect_atr_spike_numba,
    detect_bb_squeeze_numba,
    detect_volatility_trend_numba,
    detect_keltner_squeeze_numba
)
from .ma_crossover_patterns import (
    detect_golden_cross_numba,
    detect_death_cross_numba,
    detect_short_term_crossover_numba,
    check_ma_alignment_numba
)
from .stochastic_patterns import (
    detect_stoch_oversold_numba,
    detect_stoch_overbought_numba,
    detect_stoch_bullish_crossover_numba,
    detect_stoch_bearish_crossover_numba
)
from .volume_patterns import (
    detect_volume_spike_numba,
    detect_volume_dryup_numba,
    detect_volume_price_divergence_numba,
    detect_accumulation_distribution_numba,
    detect_climax_volume_numba
)


class IndicatorPatternEngine:
    """
    Orchestrates indicator pattern detection across RSI, MACD, divergences, and volatility.
    
    Pure NumPy/Numba implementation - no heavy classes, fast execution.
    """
    
    def __init__(self, logger=None, format_utils=None):
        """Initialize indicator pattern engine"""
        self.logger = logger
        self.format_utils = format_utils
    
    def _format_pattern_time(self, periods_ago: int, index: int, timestamps: Optional[List]) -> str:
        """
        Format pattern timing with timestamp.
        
        Args:
            periods_ago: Number of periods ago
            index: Index where pattern occurred
            timestamps: Optional list of datetime objects
            
        Returns:
            Formatted string like "4 periods ago at 2025-10-30 12:00:00"
        """
        if timestamps and 0 <= index < len(timestamps):
            timestamp = timestamps[index]
            if hasattr(timestamp, 'strftime'):
                timestamp_str = f" at {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            else:
                timestamp_str = ""
        else:
            timestamp_str = ""
        
        if periods_ago == 0:
            return "now" + timestamp_str
        elif periods_ago == 1:
            return "1 period ago" + timestamp_str
        else:
            return f"{periods_ago} periods ago" + timestamp_str
    
    def detect_patterns(
        self,
        technical_history: Dict[str, np.ndarray],
        ohlcv_data: Optional[np.ndarray] = None,
        long_term_sma_values: Optional[Dict[int, float]] = None,
        timestamps: Optional[List] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect all indicator patterns from technical history.
        
        Args:
            technical_history: Dict of indicator name -> numpy array
                Expected keys: rsi, macd_line, macd_signal, macd_hist, stoch_k,
                              atr, bb_upper, bb_lower, kc_upper, kc_lower
            ohlcv_data: Optional OHLCV array for price data (for divergences)
            long_term_sma_values: Optional dict of SMA period -> value for MA crossovers
            timestamps: Optional list of datetime objects for timestamp formatting
            
        Returns:
            Dict with pattern categories:
            {
                'rsi': [...],
                'macd': [...],
                'divergence': [...],
                'volatility': [...],
                'stochastic': [...],
                'ma_crossover': [...],
                'volume': [...]
            }
        """
        patterns = {
            'rsi': [],
            'macd': [],
            'divergence': [],
            'volatility': [],
            'stochastic': [],
            'ma_crossover': [],
            'volume': []
        }
        
        # Extract price and volume data if available
        prices = None
        volume = None
        if ohlcv_data is not None and len(ohlcv_data) > 0:
            prices = ohlcv_data[:, 4]  # Close prices (column 4)
            volume = ohlcv_data[:, 5]  # Volume (column 5)
        
        # RSI Patterns
        if 'rsi' in technical_history:
            rsi_patterns = self._detect_rsi_patterns(
                technical_history['rsi'],
                prices,
                timestamps
            )
            patterns['rsi'].extend(rsi_patterns)
        
        # MACD Patterns
        if 'macd_line' in technical_history and 'macd_signal' in technical_history:
            macd_patterns = self._detect_macd_patterns(
                technical_history['macd_line'],
                technical_history['macd_signal'],
                technical_history.get('macd_hist'),
                timestamps
            )
            patterns['macd'].extend(macd_patterns)
        
        # Divergence Patterns
        if prices is not None:
            divergence_patterns = self._detect_divergence_patterns(
                prices,
                technical_history,
                timestamps
            )
            patterns['divergence'].extend(divergence_patterns)
        
        # Volatility Patterns
        volatility_patterns = self._detect_volatility_patterns(
            technical_history,
            timestamps
        )
        patterns['volatility'].extend(volatility_patterns)
        
        # Stochastic Patterns
        if 'stoch_k' in technical_history and 'stoch_d' in technical_history:
            stoch_patterns = self._detect_stochastic_patterns(
                technical_history['stoch_k'],
                technical_history['stoch_d'],
                prices,
                timestamps
            )
            patterns['stochastic'].extend(stoch_patterns)
        
        # MA Crossover Patterns (uses SMA arrays from technical_history if available)
        if long_term_sma_values is not None or any(k in technical_history for k in ['sma_20', 'sma_50', 'sma_200']):
            ma_patterns = self._detect_ma_crossover_patterns(
                long_term_sma_values if long_term_sma_values is not None else {},
                technical_history,
                timestamps
            )
            patterns['ma_crossover'].extend(ma_patterns)
        
        # Volume Patterns
        if volume is not None and prices is not None:
            volume_patterns = self._detect_volume_patterns(
                volume,
                prices,
                timestamps
            )
            patterns['volume'].extend(volume_patterns)
        
        return patterns
    
    def _detect_rsi_patterns(
        self,
        rsi: np.ndarray,
        prices: Optional[np.ndarray],
        timestamps: Optional[List]
    ) -> List[Dict[str, Any]]:
        """Detect RSI-based patterns"""
        patterns = []
        
        # Oversold
        is_oversold, periods_ago, rsi_value = detect_rsi_oversold_numba(rsi)
        if is_oversold:
            pattern_index = len(rsi) - 1 - periods_ago
            timestamp_str = self._format_pattern_time(periods_ago, pattern_index, timestamps)
            patterns.append({
                'type': 'rsi_oversold',
                'description': f'RSI oversold at {rsi_value:.2f} {timestamp_str}',
                'index': pattern_index,
                'details': {
                    'rsi_value': float(rsi_value),
                    'threshold': 30.0,
                    'periods_ago': int(periods_ago)
                }
            })
        
        # Overbought
        is_overbought, periods_ago, rsi_value = detect_rsi_overbought_numba(rsi)
        if is_overbought:
            pattern_index = len(rsi) - 1 - periods_ago
            timestamp_str = self._format_pattern_time(periods_ago, pattern_index, timestamps)
            patterns.append({
                'type': 'rsi_overbought',
                'description': f'RSI overbought at {rsi_value:.2f} {timestamp_str}',
                'index': pattern_index,
                'details': {
                    'rsi_value': float(rsi_value),
                    'threshold': 70.0,
                    'periods_ago': int(periods_ago)
                }
            })
        
        # W-Bottom (requires price data)
        if prices is not None and len(prices) == len(rsi):
            found, first_idx, second_idx, first_rsi, second_rsi = detect_rsi_w_bottom_numba(
                rsi, prices
            )
            if found:
                patterns.append({
                    'type': 'rsi_w_bottom',
                    'description': f'RSI W-Bottom detected (bullish reversal confirmation)',
                    'index': len(rsi) - 1,
                    'details': {
                        'first_rsi': float(first_rsi),
                        'second_rsi': float(second_rsi),
                        'first_idx': int(first_idx),
                        'second_idx': int(second_idx)
                    }
                })
        
        # M-Top (requires price data)
        if prices is not None and len(prices) == len(rsi):
            found, first_idx, second_idx, first_rsi, second_rsi = detect_rsi_m_top_numba(
                rsi, prices
            )
            if found:
                patterns.append({
                    'type': 'rsi_m_top',
                    'description': f'RSI M-Top detected (bearish reversal confirmation)',
                    'index': len(rsi) - 1,
                    'details': {
                        'first_rsi': float(first_rsi),
                        'second_rsi': float(second_rsi),
                        'first_idx': int(first_idx),
                        'second_idx': int(second_idx)
                    }
                })
        
        return patterns
    
    def _detect_macd_patterns(
        self,
        macd_line: np.ndarray,
        signal_line: np.ndarray,
        macd_hist: Optional[np.ndarray],
        timestamps: Optional[List]
    ) -> List[Dict[str, Any]]:
        """Detect MACD-based patterns"""
        patterns = []
        
        # Crossover
        found, is_bullish, periods_ago, macd_val, signal_val = detect_macd_crossover_numba(
            macd_line, signal_line
        )
        if found:
            crossover_type = 'bullish' if is_bullish else 'bearish'
            pattern_index = len(macd_line) - 1 - periods_ago
            timestamp_str = self._format_pattern_time(periods_ago, pattern_index, timestamps)
            patterns.append({
                'type': f'macd_{crossover_type}_crossover',
                'description': f'MACD {crossover_type} crossover {timestamp_str}',
                'index': pattern_index,
                'details': {
                    'is_bullish': is_bullish,
                    'macd_value': float(macd_val),
                    'signal_value': float(signal_val),
                    'periods_ago': int(periods_ago)
                }
            })
        
        # Zero-line cross
        found, is_bullish, periods_ago, macd_val = detect_macd_zero_cross_numba(macd_line)
        if found:
            cross_type = 'bullish' if is_bullish else 'bearish'
            pattern_index = len(macd_line) - 1 - periods_ago
            timestamp_str = self._format_pattern_time(periods_ago, pattern_index, timestamps)
            patterns.append({
                'type': f'macd_{cross_type}_zero_cross',
                'description': f'MACD {cross_type} zero-line cross {timestamp_str}',
                'index': pattern_index,
                'details': {
                    'is_bullish': is_bullish,
                    'macd_value': float(macd_val),
                    'periods_ago': int(periods_ago)
                }
            })
        
        # Histogram trend
        if macd_hist is not None:
            hist_trend = get_macd_histogram_trend_numba(macd_hist)
            if hist_trend != 0:
                trend_name = 'increasing' if hist_trend > 0 else 'decreasing'
                pattern_index = len(macd_hist) - 1
                timestamp_str = self._format_pattern_time(0, pattern_index, timestamps)
                patterns.append({
                    'type': f'macd_histogram_{trend_name}',
                    'description': f'MACD histogram {trend_name} (momentum shift) {timestamp_str}',
                    'index': pattern_index,
                    'details': {
                        'trend': int(hist_trend),
                        'current_hist': float(macd_hist[-1]),
                        'periods_ago': 0
                    }
                })
        
        return patterns
    
    def _create_divergence_pattern(
        self,
        pattern_type: str,
        indicator_name: str,
        is_bullish: bool,
        first_idx: int,
        second_idx: int,
        first_p: float,
        second_p: float,
        first_i: float,
        second_i: float,
        timestamps: Optional[List]
    ) -> Dict[str, Any]:
        """
        Create a divergence pattern dictionary (helper method).
        
        Args:
            pattern_type: Pattern type string (e.g., 'rsi_bullish_divergence')
            indicator_name: Indicator name (e.g., 'rsi', 'macd')
            is_bullish: True for bullish divergence, False for bearish
            first_idx: Index of first extreme
            second_idx: Index of second extreme
            first_p: First price value
            second_p: Second price value
            first_i: First indicator value
            second_i: Second indicator value
            timestamps: Optional timestamp list
            
        Returns:
            Pattern dictionary
        """
        timestamp_str = self._format_pattern_time(0, second_idx, timestamps)
        
        if is_bullish:
            if indicator_name == 'rsi':
                description = f'RSI Bullish Divergence: Price lower low (${second_p:.2f}), RSI higher low ({second_i:.2f}) {timestamp_str}'
            else:
                description = f'MACD Bullish Divergence: Price lower low, MACD higher low {timestamp_str}'
        else:
            if indicator_name == 'rsi':
                description = f'RSI Bearish Divergence: Price higher high (${second_p:.2f}), RSI lower high ({second_i:.2f}) {timestamp_str}'
            else:
                description = f'MACD Bearish Divergence: Price higher high, MACD lower high {timestamp_str}'
        
        return {
            'type': pattern_type,
            'description': description,
            'index': second_idx,
            'details': {
                'indicator': indicator_name,
                'first_price': float(first_p),
                'second_price': float(second_p),
                'first_indicator': float(first_i),
                'second_indicator': float(second_i),
                'first_idx': int(first_idx),
                'second_idx': int(second_idx),
                'periods_ago': 0
            }
        }
    
    def _detect_divergence_patterns(
        self,
        prices: np.ndarray,
        technical_history: Dict[str, np.ndarray],
        timestamps: Optional[List]
    ) -> List[Dict[str, Any]]:
        """Detect divergence patterns across multiple indicators"""
        patterns = []
        
        # RSI Divergences
        if 'rsi' in technical_history:
            rsi = technical_history['rsi']
            
            # Bullish divergence
            found, first_idx, second_idx, first_p, second_p, first_i, second_i = \
                detect_bullish_divergence_numba(prices, rsi)
            if found:
                patterns.append(self._create_divergence_pattern(
                    'rsi_bullish_divergence', 'rsi', True,
                    first_idx, second_idx, first_p, second_p, first_i, second_i, timestamps
                ))
            
            # Bearish divergence
            found, first_idx, second_idx, first_p, second_p, first_i, second_i = \
                detect_bearish_divergence_numba(prices, rsi)
            if found:
                patterns.append(self._create_divergence_pattern(
                    'rsi_bearish_divergence', 'rsi', False,
                    first_idx, second_idx, first_p, second_p, first_i, second_i, timestamps
                ))
        
        # MACD Divergences
        if 'macd_line' in technical_history:
            macd = technical_history['macd_line']
            
            # Bullish divergence
            found, first_idx, second_idx, first_p, second_p, first_i, second_i = \
                detect_bullish_divergence_numba(prices, macd)
            if found:
                patterns.append(self._create_divergence_pattern(
                    'macd_bullish_divergence', 'macd', True,
                    first_idx, second_idx, first_p, second_p, first_i, second_i, timestamps
                ))
            
            # Bearish divergence
            found, first_idx, second_idx, first_p, second_p, first_i, second_i = \
                detect_bearish_divergence_numba(prices, macd)
            if found:
                patterns.append(self._create_divergence_pattern(
                    'macd_bearish_divergence', 'macd', False,
                    first_idx, second_idx, first_p, second_p, first_i, second_i, timestamps
                ))
        
        return patterns
    
    def _detect_volatility_patterns(
        self,
        technical_history: Dict[str, np.ndarray],
        timestamps: Optional[List]
    ) -> List[Dict[str, Any]]:
        """Detect volatility-based patterns"""
        patterns = []
        
        # ATR Spike
        if 'atr' in technical_history:
            atr = technical_history['atr']
            found, periods_ago, current_atr, avg_atr = detect_atr_spike_numba(atr)
            if found:
                pattern_index = len(atr) - 1
                timestamp_str = self._format_pattern_time(0, pattern_index, timestamps)
                patterns.append({
                    'type': 'atr_spike',
                    'description': f'ATR spike detected: {current_atr:.4f} vs avg {avg_atr:.4f} {timestamp_str}',
                    'index': pattern_index,
                    'details': {
                        'current_atr': float(current_atr),
                        'average_atr': float(avg_atr),
                        'spike_ratio': float(current_atr / avg_atr),
                        'periods_ago': 0
                    }
                })
            
            # Volatility trend
            vol_trend = detect_volatility_trend_numba(atr)
            if vol_trend != 0:
                trend_name = 'increasing' if vol_trend > 0 else 'decreasing'
                pattern_index = len(atr) - 1
                timestamp_str = self._format_pattern_time(0, pattern_index, timestamps)
                patterns.append({
                    'type': f'volatility_{trend_name}',
                    'description': f'Volatility {trend_name} {timestamp_str}',
                    'index': pattern_index,
                    'details': {
                        'trend': int(vol_trend),
                        'current_atr': float(atr[-1]),
                        'periods_ago': 0
                    }
                })
        
        # Bollinger Band Squeeze
        if 'bb_upper' in technical_history and 'bb_lower' in technical_history:
            bb_upper = technical_history['bb_upper']
            bb_lower = technical_history['bb_lower']
            found, current_width, percentile_width = detect_bb_squeeze_numba(
                bb_upper, bb_lower
            )
            if found:
                pattern_index = len(bb_upper) - 1
                timestamp_str = self._format_pattern_time(0, pattern_index, timestamps)
                patterns.append({
                    'type': 'bb_squeeze',
                    'description': f'Bollinger Band squeeze detected (low volatility, breakout imminent) {timestamp_str}',
                    'index': pattern_index,
                    'details': {
                        'current_width': float(current_width),
                        'percentile_width': float(percentile_width),
                        'periods_ago': 0
                    }
                })
        
        # TTM Squeeze (Bollinger Bands inside Keltner Channels)
        if all(k in technical_history for k in ['bb_upper', 'bb_lower', 'kc_upper', 'kc_lower']):
            found = detect_keltner_squeeze_numba(
                technical_history['kc_upper'],
                technical_history['kc_lower'],
                technical_history['bb_upper'],
                technical_history['bb_lower']
            )
            if found:
                pattern_index = len(technical_history['bb_upper']) - 1
                timestamp_str = self._format_pattern_time(0, pattern_index, timestamps)
                patterns.append({
                    'type': 'ttm_squeeze',
                    'description': f'TTM Squeeze detected (extreme low volatility) {timestamp_str}',
                    'index': pattern_index,
                    'details': {
                        'squeeze_type': 'ttm',
                        'periods_ago': 0
                    }
                })
        
        return patterns
    
    def _detect_stochastic_patterns(
        self,
        stoch_k: np.ndarray,
        stoch_d: np.ndarray,
        prices: Optional[np.ndarray],
        timestamps: Optional[List]
    ) -> List[Dict[str, Any]]:
        """Detect Stochastic oscillator patterns"""
        patterns = []
        
        # Oversold
        is_oversold, periods_ago, stoch_value = detect_stoch_oversold_numba(stoch_k)
        if is_oversold:
            pattern_index = len(stoch_k) - 1 - periods_ago
            timestamp_str = self._format_pattern_time(periods_ago, pattern_index, timestamps)
            patterns.append({
                'type': 'stoch_oversold',
                'description': f'Stochastic oversold at {stoch_value:.2f} {timestamp_str}',
                'index': pattern_index,
                'details': {
                    'stoch_k_value': float(stoch_value),
                    'threshold': 20.0,
                    'periods_ago': int(periods_ago)
                }
            })
        
        # Overbought
        is_overbought, periods_ago, stoch_value = detect_stoch_overbought_numba(stoch_k)
        if is_overbought:
            pattern_index = len(stoch_k) - 1 - periods_ago
            timestamp_str = self._format_pattern_time(periods_ago, pattern_index, timestamps)
            patterns.append({
                'type': 'stoch_overbought',
                'description': f'Stochastic overbought at {stoch_value:.2f} {timestamp_str}',
                'index': pattern_index,
                'details': {
                    'stoch_k_value': float(stoch_value),
                    'threshold': 80.0,
                    'periods_ago': int(periods_ago)
                }
            })
        
        # Bullish crossover
        found, periods_ago, k_val, d_val, in_oversold = detect_stoch_bullish_crossover_numba(stoch_k, stoch_d)
        if found:
            pattern_index = len(stoch_k) - 1 - periods_ago
            timestamp_str = self._format_pattern_time(periods_ago, pattern_index, timestamps)
            desc = f'Stochastic bullish crossover {timestamp_str}'
            if in_oversold:
                desc += ' in oversold territory (strong signal)'
            patterns.append({
                'type': 'stoch_bullish_crossover',
                'description': desc,
                'index': pattern_index,
                'details': {
                    'stoch_k': float(k_val),
                    'stoch_d': float(d_val),
                    'in_oversold': in_oversold,
                    'periods_ago': int(periods_ago)
                }
            })
        
        # Bearish crossover
        found, periods_ago, k_val, d_val, in_overbought = detect_stoch_bearish_crossover_numba(stoch_k, stoch_d)
        if found:
            pattern_index = len(stoch_k) - 1 - periods_ago
            timestamp_str = self._format_pattern_time(periods_ago, pattern_index, timestamps)
            desc = f'Stochastic bearish crossover {timestamp_str}'
            if in_overbought:
                desc += ' in overbought territory (strong signal)'
            patterns.append({
                'type': 'stoch_bearish_crossover',
                'description': desc,
                'index': pattern_index,
                'details': {
                    'stoch_k': float(k_val),
                    'stoch_d': float(d_val),
                    'in_overbought': in_overbought,
                    'periods_ago': int(periods_ago)
                }
            })
        
        return patterns
    
    def _detect_ma_crossover_patterns(
        self,
        sma_values: Dict[int, float],
        technical_history: Dict[str, np.ndarray],
        timestamps: Optional[List]
    ) -> List[Dict[str, Any]]:
        """Detect Moving Average crossover patterns using arrays from technical_history"""
        patterns = []
        
        # Try to use SMA arrays from technical_history first (preferred for crossover detection)
        sma_20_array = technical_history.get('sma_20')
        sma_50_array = technical_history.get('sma_50')
        sma_200_array = technical_history.get('sma_200')
        
        # If sma_values not provided but we have arrays, populate from current array values
        if (sma_values is None or len(sma_values) == 0) and (sma_50_array is not None or sma_200_array is not None):
            sma_values = {}
            if sma_20_array is not None and len(sma_20_array) > 0:
                sma_values[20] = float(sma_20_array[-1])
            if sma_50_array is not None and len(sma_50_array) > 0:
                sma_values[50] = float(sma_50_array[-1])
            if sma_200_array is not None and len(sma_200_array) > 0:
                sma_values[200] = float(sma_200_array[-1])
        
        # Detect actual crossovers if we have arrays
        if sma_50_array is not None and sma_200_array is not None:
            # Golden Cross (50 SMA crosses above 200 SMA)
            found, periods_ago, sma_50_val, sma_200_val = detect_golden_cross_numba(sma_50_array, sma_200_array)
            if found:
                pattern_index = len(sma_50_array) - 1 - periods_ago
                timestamp_str = self._format_pattern_time(periods_ago, pattern_index, timestamps)
                patterns.append({
                    'type': 'golden_cross',
                    'description': f'Golden Cross: 50 SMA crossed above 200 SMA {timestamp_str} (bullish long-term signal)',
                    'index': pattern_index,
                    'details': {
                        'sma_50': float(sma_50_val),
                        'sma_200': float(sma_200_val),
                        'periods_ago': int(periods_ago)
                    }
                })
            
            # Death Cross (50 SMA crosses below 200 SMA)
            found, periods_ago, sma_50_val, sma_200_val = detect_death_cross_numba(sma_50_array, sma_200_array)
            if found:
                pattern_index = len(sma_50_array) - 1 - periods_ago
                timestamp_str = self._format_pattern_time(periods_ago, pattern_index, timestamps)
                patterns.append({
                    'type': 'death_cross',
                    'description': f'Death Cross: 50 SMA crossed below 200 SMA {timestamp_str} (bearish long-term signal)',
                    'index': pattern_index,
                    'details': {
                        'sma_50': float(sma_50_val),
                        'sma_200': float(sma_200_val),
                        'periods_ago': int(periods_ago)
                    }
                })
        
        # Short-term crossover (20 SMA vs 50 SMA)
        if sma_20_array is not None and sma_50_array is not None:
            found, is_bullish, periods_ago, sma_20_val, sma_50_val = detect_short_term_crossover_numba(sma_20_array, sma_50_array)
            if found:
                cross_type = 'bullish' if is_bullish else 'bearish'
                pattern_index = len(sma_20_array) - 1 - periods_ago
                timestamp_str = self._format_pattern_time(periods_ago, pattern_index, timestamps)
                patterns.append({
                    'type': f'ma_short_term_{cross_type}_crossover',
                    'description': f'20 SMA crossed {"above" if is_bullish else "below"} 50 SMA {timestamp_str} ({cross_type} short-term signal)',
                    'index': pattern_index,
                    'details': {
                        'is_bullish': is_bullish,
                        'sma_20': float(sma_20_val),
                        'sma_50': float(sma_50_val),
                        'periods_ago': int(periods_ago)
                    }
                })
        
        # MA alignment detection (uses current values or falls back to arrays)
        if sma_values is not None and 20 in sma_values and 50 in sma_values and 200 in sma_values:
            sma_20 = sma_values[20]
            sma_50 = sma_values[50]
            sma_200 = sma_values[200]
            
            # Bullish alignment: 20 > 50 > 200
            if sma_20 > sma_50 and sma_50 > sma_200:
                patterns.append({
                    'type': 'ma_bullish_alignment',
                    'description': f'Bullish MA alignment (20>50>200 SMA) - current configuration',
                    'index': 0,
                    'details': {
                        'sma_20': float(sma_20),
                        'sma_50': float(sma_50),
                        'sma_200': float(sma_200),
                        'periods_ago': 0
                    }
                })
            
            # Bearish alignment: 20 < 50 < 200
            elif sma_20 < sma_50 and sma_50 < sma_200:
                patterns.append({
                    'type': 'ma_bearish_alignment',
                    'description': f'Bearish MA alignment (20<50<200 SMA) - current configuration',
                    'index': 0,
                    'details': {
                        'sma_20': float(sma_20),
                        'sma_50': float(sma_50),
                        'sma_200': float(sma_200),
                        'periods_ago': 0
                    }
                })
        
        # Detect Golden/Death Cross potential (50 vs 200 relationship)
        if sma_values is not None and 50 in sma_values and 200 in sma_values:
            sma_50 = sma_values[50]
            sma_200 = sma_values[200]
            
            # Calculate percentage distance
            pct_distance = abs((sma_50 / sma_200 - 1.0) * 100)
            
            # If very close (<2%), potential crossover imminent
            if pct_distance < 2.0:
                if sma_50 > sma_200:
                    patterns.append({
                        'type': 'golden_cross_active',
                        'description': f'Golden Cross active: 50 SMA above 200 SMA (bullish) - current configuration',
                        'index': 0,
                        'details': {
                            'sma_50': float(sma_50),
                            'sma_200': float(sma_200),
                            'pct_distance': float(pct_distance),
                            'periods_ago': 0
                        }
                    })
                else:
                    patterns.append({
                        'type': 'death_cross_active',
                        'description': f'Death Cross active: 50 SMA below 200 SMA (bearish) - current configuration',
                        'index': 0,
                        'details': {
                            'sma_50': float(sma_50),
                            'sma_200': float(sma_200),
                            'pct_distance': float(pct_distance),
                            'periods_ago': 0
                        }
                    })
        
        return patterns
    
    def _detect_volume_patterns(
        self,
        volume: np.ndarray,
        prices: np.ndarray,
        timestamps: Optional[List]
    ) -> List[Dict[str, Any]]:
        """Detect volume-based patterns"""
        patterns = []
        
        # Volume spike
        is_spike, current_vol, avg_vol, spike_ratio = detect_volume_spike_numba(volume)
        if is_spike:
            pattern_index = len(volume) - 1
            timestamp_str = self._format_pattern_time(0, pattern_index, timestamps)
            patterns.append({
                'type': 'volume_spike',
                'description': f'Volume spike: {spike_ratio:.2f}x average ({current_vol:.0f} vs {avg_vol:.0f}) {timestamp_str}',
                'index': pattern_index,
                'details': {
                    'current_volume': float(current_vol),
                    'average_volume': float(avg_vol),
                    'spike_ratio': float(spike_ratio),
                    'periods_ago': 0
                }
            })
        
        # Volume dry-up
        is_dryup, current_vol, avg_vol, dryup_ratio = detect_volume_dryup_numba(volume)
        if is_dryup:
            pattern_index = len(volume) - 1
            timestamp_str = self._format_pattern_time(0, pattern_index, timestamps)
            patterns.append({
                'type': 'volume_dryup',
                'description': f'Volume dry-up: {dryup_ratio:.2f}x average (potential breakout setup) {timestamp_str}',
                'index': pattern_index,
                'details': {
                    'current_volume': float(current_vol),
                    'average_volume': float(avg_vol),
                    'dryup_ratio': float(dryup_ratio),
                    'periods_ago': 0
                }
            })
        
        # Climax volume
        is_climax, current_vol, avg_vol, climax_ratio = detect_climax_volume_numba(volume)
        if is_climax:
            pattern_index = len(volume) - 1
            timestamp_str = self._format_pattern_time(0, pattern_index, timestamps)
            patterns.append({
                'type': 'climax_volume',
                'description': f'Climax volume: {climax_ratio:.2f}x average (potential exhaustion) {timestamp_str}',
                'index': pattern_index,
                'details': {
                    'current_volume': float(current_vol),
                    'average_volume': float(avg_vol),
                    'climax_ratio': float(climax_ratio),
                    'periods_ago': 0
                }
            })
        
        # Volume-price divergence
        found, is_bearish, price_chg, vol_chg = detect_volume_price_divergence_numba(volume, prices)
        if found:
            div_type = 'bearish' if is_bearish else 'bullish'
            pattern_index = len(volume) - 1
            timestamp_str = self._format_pattern_time(0, pattern_index, timestamps)
            desc = f'{div_type.capitalize()} volume-price divergence'
            if is_bearish:
                desc += f' (price rising, volume falling - weak rally) {timestamp_str}'
            else:
                desc += f' (price falling, volume falling - weak selloff) {timestamp_str}'
            
            patterns.append({
                'type': f'volume_price_divergence_{div_type}',
                'description': desc,
                'index': pattern_index,
                'details': {
                    'is_bearish': is_bearish,
                    'price_change_pct': float(price_chg),
                    'volume_change_pct': float(vol_chg),
                    'periods_ago': 0
                }
            })
        
        # Accumulation/Distribution
        found, is_accumulation, strength, up_vol_ratio = detect_accumulation_distribution_numba(volume, prices)
        if found:
            phase = 'Accumulation' if is_accumulation else 'Distribution'
            pattern_index = len(volume) - 1
            timestamp_str = self._format_pattern_time(0, pattern_index, timestamps)
            desc = f'{phase} detected (strength: {strength:.2f}) over last 10 periods {timestamp_str}'
            
            patterns.append({
                'type': f'volume_{phase.lower()}',
                'description': desc,
                'index': pattern_index,
                'details': {
                    'is_accumulation': is_accumulation,
                    'strength': float(strength),
                    'up_volume_ratio': float(up_vol_ratio),
                    'lookback_periods': 10,
                    'periods_ago': 0
                }
            })
        
        return patterns

