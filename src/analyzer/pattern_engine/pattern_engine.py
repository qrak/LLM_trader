from datetime import datetime
from typing import Any

import numpy as np

from src.analyzer.pattern_engine.swing_detection import (
    detect_swing_highs_numba,
    detect_swing_lows_numba
)


class PatternEngine:

    def __init__(self, lookback: int = 7, lookahead: int = 7, format_utils=None):
        self.lookback = lookback
        self.lookahead = lookahead
        self.format_utils = format_utils

    def detect_patterns(self, _ohlcv: np.ndarray, _timestamps: list[datetime] = None) -> dict[str, list[dict[str, Any]]]:
        """
        Detect patterns from OHLCV data.

        Indicator patterns are handled by IndicatorPatternEngine.
        """
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
