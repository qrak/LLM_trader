from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime

from src.analyzer.pattern_engine import PatternEngine
from src.analyzer.pattern_engine.indicator_patterns import IndicatorPatternEngine
from src.logger.logger import Logger


class PatternAnalyzer:
    
    def __init__(self, logger: Optional[Logger] = None, format_utils=None):
        self.logger = logger
        self.format_utils = format_utils
        self.pattern_engine = PatternEngine(lookback=5, lookahead=5, format_utils=format_utils)
        self.indicator_pattern_engine = IndicatorPatternEngine(format_utils=format_utils)
        self._warmed_up = False
    
    def detect_patterns(
        self,
        ohlcv_data: np.ndarray,
        technical_history: Dict[str, np.ndarray],
        long_term_data: Optional[Dict] = None,
        timestamps: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Detect all chart and indicator patterns from current market data.
        
        Note: No caching - always runs fresh detection for real-time analysis.
        """
        if self.logger:
            self.logger.debug(f"Running pattern detection on {len(ohlcv_data)} candles")
        
        # Use provided timestamps or extract from OHLCV data as fallback
        if timestamps is None and ohlcv_data is not None and len(ohlcv_data) > 0:
            try:
                timestamps = [datetime.fromtimestamp(ts / 1000) for ts in ohlcv_data[:, 0]]
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Could not extract timestamps from OHLCV data: {e}")
        
        # Detect chart patterns
        chart_patterns = self.pattern_engine.detect_patterns(ohlcv_data, timestamps)
        
        # Extract SMA values for MA crossover detection
        sma_values = None
        if long_term_data is not None and 'sma_values' in long_term_data:
            sma_values = long_term_data['sma_values']
        
        # Detect indicator patterns
        indicator_patterns = {}
        try:
            indicator_patterns = self.indicator_pattern_engine.detect_patterns(
                technical_history, ohlcv_data, sma_values, timestamps
            )
            if self.logger:
                ind_count = sum(len(p) for p in indicator_patterns.values())
                self.logger.debug(f"Detected {ind_count} indicator patterns")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error detecting indicator patterns: {e}")
        
        # Combine both types of patterns
        patterns = {
            **chart_patterns,  # candlestick, trend, reversal patterns
            **indicator_patterns  # RSI, MACD, divergence, volatility, stochastic, MA, volume patterns
        }
        
        if self.logger:
            total_patterns = sum(len(p) for p in patterns.values())
            chart_count = sum(len(p) for p in chart_patterns.values())
            ind_count = sum(len(p) for p in indicator_patterns.values())
            self.logger.debug(f"Detected {total_patterns} patterns: {chart_count} chart + {ind_count} indicator")
        
        return patterns
    
    def warmup(self) -> None:
        """Run a lightweight detection pass to prime numba caches."""
        if self._warmed_up:
            return

        sample_count = max(self.pattern_engine.lookback + self.pattern_engine.lookahead + 5, 64)
        try:
            dummy_ohlcv = self._build_dummy_ohlcv(sample_count)
            dummy_history = self._build_dummy_history(sample_count, dummy_ohlcv[:, 4])
            # Run both engines to trigger numba compilation
            self.pattern_engine.detect_patterns(dummy_ohlcv, None)
            self.indicator_pattern_engine.detect_patterns(dummy_history, dummy_ohlcv, None, None)
            self._warmed_up = True
            if self.logger:
                self.logger.debug("PatternAnalyzer warm-up completed (Numba cache primed)")
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"PatternAnalyzer warm-up skipped: {exc}")

    def _build_dummy_ohlcv(self, sample_count: int) -> np.ndarray:
        """Create deterministic OHLCV data for warm-up."""
        timestamps = np.arange(sample_count) * 60_000
        base = np.linspace(100.0, 110.0, sample_count)
        noise = np.sin(np.linspace(0, np.pi * 3, sample_count)) * 0.5
        close = base + noise
        open_prices = close - 0.1
        high = close + 0.5
        low = close - 0.5
        volume = np.linspace(1_000.0, 1_500.0, sample_count)
        return np.column_stack((timestamps, open_prices, high, low, close, volume))

    def _build_dummy_history(self, sample_count: int, close_series: np.ndarray) -> Dict[str, np.ndarray]:
        """Construct the minimal indicator history needed for pattern warm-up."""
        ramp = np.linspace(-1.0, 1.0, sample_count)
        rsi = np.clip(50 + 20 * np.sin(ramp * np.pi), 0, 100)
        macd_line = ramp
        macd_signal = ramp * 0.8
        macd_hist = macd_line - macd_signal
        stoch_k = np.clip(50 + 30 * np.sin(ramp * np.pi), 0, 100)
        stoch_d = np.clip(50 + 20 * np.cos(ramp * np.pi), 0, 100)
        atr = np.linspace(0.5, 1.5, sample_count)
        bb_upper = close_series + 1.0
        bb_lower = close_series - 1.0
        kc_upper = close_series + 0.8
        kc_lower = close_series - 0.8

        return {
            'rsi': rsi.astype(np.float64),
            'macd_line': macd_line.astype(np.float64),
            'macd_signal': macd_signal.astype(np.float64),
            'macd_hist': macd_hist.astype(np.float64),
            'stoch_k': stoch_k.astype(np.float64),
            'stoch_d': stoch_d.astype(np.float64),
            'atr': atr.astype(np.float64),
            'bb_upper': bb_upper.astype(np.float64),
            'bb_lower': bb_lower.astype(np.float64),
            'kc_upper': kc_upper.astype(np.float64),
            'kc_lower': kc_lower.astype(np.float64),
            'sma_20': close_series.astype(np.float64),
            'sma_50': close_series.astype(np.float64),
            'sma_200': close_series.astype(np.float64)
        }
        
    def get_all_patterns(
        self,
        ohlcv_data: np.ndarray,
        technical_history: Dict[str, np.ndarray],
        long_term_data: Optional[Dict] = None
    ) -> List[Dict]:
        try:
            patterns_dict = self.detect_patterns(ohlcv_data, technical_history, long_term_data)
            
            all_patterns = []
            for _, patterns_list in patterns_dict.items():
                all_patterns.extend(patterns_list)
            
            if self.logger:
                self.logger.debug(f"Detected {len(all_patterns)} patterns")
                
            return all_patterns
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error in pattern detection: {e}")
            return []