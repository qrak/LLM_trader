"""
Indicator Pattern Detection Module

Pure NumPy/Numba implementations for detecting patterns in technical indicators.
Complements chart patterns by providing momentum, overbought/oversold, and divergence signals.
"""

from .rsi_patterns import (
    detect_rsi_oversold_numba,
    detect_rsi_overbought_numba,
    detect_rsi_w_bottom_numba,
    detect_rsi_m_top_numba
)

from .macd_patterns import (
    detect_macd_crossover_numba,
    detect_macd_zero_cross_numba
)

from .divergence_patterns import (
    detect_bullish_divergence_numba,
    detect_bearish_divergence_numba
)

from .volatility_patterns import (
    detect_atr_spike_numba,
    detect_bb_squeeze_numba
)

from .indicator_pattern_engine import IndicatorPatternEngine

__all__ = [
    # RSI patterns
    'detect_rsi_oversold_numba',
    'detect_rsi_overbought_numba',
    'detect_rsi_w_bottom_numba',
    'detect_rsi_m_top_numba',
    # MACD patterns
    'detect_macd_crossover_numba',
    'detect_macd_zero_cross_numba',
    # Divergence patterns
    'detect_bullish_divergence_numba',
    'detect_bearish_divergence_numba',
    # Volatility patterns
    'detect_atr_spike_numba',
    'detect_bb_squeeze_numba',
    # Engine
    'IndicatorPatternEngine',
]
