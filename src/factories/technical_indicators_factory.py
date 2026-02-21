"""Factory for creating TechnicalIndicators instances."""
import numpy as np
from src.indicators.base.technical_indicators import TechnicalIndicators


class TechnicalIndicatorsFactory:
    """
    Factory for creating fresh TechnicalIndicators instances.

    Creates separate instances for different timeframe calculations to avoid
    state interference between main, long-term, and weekly analyses.
    """

    @staticmethod
    def create_for_current_timeframe(ohlcv_data: np.ndarray) -> TechnicalIndicators:
        """Create TechnicalIndicators instance for current timeframe analysis."""
        ti = TechnicalIndicators()
        ti.get_data(ohlcv_data)
        return ti

    @staticmethod
    def create_for_long_term(ohlcv_data: np.ndarray) -> TechnicalIndicators:
        """Create TechnicalIndicators instance for long-term (365-day) analysis."""
        ti = TechnicalIndicators()
        ti.get_data(ohlcv_data)
        return ti

    @staticmethod
    def create_for_weekly(ohlcv_data: np.ndarray) -> TechnicalIndicators:
        """Create TechnicalIndicators instance for weekly macro analysis."""
        ti = TechnicalIndicators()
        ti.get_data(ohlcv_data)
        return ti
