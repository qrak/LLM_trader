import numpy as np
from typing import Dict, List, Any
from datetime import datetime

# Note: Chart pattern detection (Head & Shoulders, Double Top/Bottom, Triangle, Wedge, Channel)
# has been removed. The AI model detects these patterns visually from the chart image more accurately.
# Only swing detection is kept for use by indicator pattern engine.

from src.analyzer.pattern_engine.swing_detection import (
    detect_swing_highs_numba,
    detect_swing_lows_numba
)


class PatternEngine:
    
    def __init__(self, lookback: int = 7, lookahead: int = 7, format_utils=None):
        self.lookback = lookback
        self.lookahead = lookahead
        self.format_utils = format_utils
    
    def detect_patterns(self, ohlcv: np.ndarray, timestamps: List[datetime] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect patterns from OHLCV data.
        
        Note: Chart patterns (H&S, Double Top/Bottom, Triangle, Wedge, Channel) removed.
        AI model detects these visually from chart image. Only returns empty dict now
        as indicator patterns are handled by IndicatorPatternEngine.
        """
        # Chart pattern detection removed - AI detects from chart image visually
        # Swing detection still available for other components via get_swing_points()
        return {}
    
    def get_swing_points(self, ohlcv: np.ndarray) -> tuple:
        """Get swing highs and lows for use by other components."""
        if len(ohlcv) < self.lookback + self.lookahead:
            return np.array([]), np.array([])
        
        high = ohlcv[:, 1].astype(np.float64)
        low = ohlcv[:, 2].astype(np.float64)
        
        swing_highs = detect_swing_highs_numba(high, self.lookback, self.lookahead)
        swing_lows = detect_swing_lows_numba(low, self.lookback, self.lookahead)
        
        return swing_highs, swing_lows
