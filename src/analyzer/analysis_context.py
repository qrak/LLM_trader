"""Data container for market analysis data.

Provides typed storage for OHLCV data, technical indicators, market metrics,
and other analysis context using dataclass pattern.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from src.utils.data_utils import SerializableMixin


@dataclass
class AnalysisContext(SerializableMixin):
    """Data container for market analysis data.

    All fields use proper typing. Validation is trusted via type hints
    rather than redundant runtime shape checks.
    """
    symbol: str
    exchange: str | None = None
    timeframe: str | None = None
    ohlcv_candles: np.ndarray | None = None
    current_price: float | None = None
    timestamps: list[datetime] | None = None
    technical_data: dict[str, Any] = field(default_factory=dict)
    technical_history: dict[str, Any] = field(default_factory=dict)
    technical_patterns: dict[str, Any] = field(default_factory=dict)
    market_metrics: dict[str, Any] = field(default_factory=dict)
    sentiment: dict[str, Any] | None = None
    long_term_data: dict[str, Any] | None = None
    weekly_macro_indicators: dict[str, Any] | None = None
    market_overview: dict[str, Any] = field(default_factory=dict)
    market_microstructure: dict[str, Any] = field(default_factory=dict)
    weekly_ohlcv: np.ndarray | None = None
    coin_details: dict[str, Any] = field(default_factory=dict)
