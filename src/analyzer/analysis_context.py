"""Data container for market analysis data.

Provides typed storage for OHLCV data, technical indicators, market metrics,
and other analysis context using dataclass pattern.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np

from src.utils.data_utils import SerializableMixin


@dataclass
class AnalysisContext(SerializableMixin):
    """Data container for market analysis data.
    
    All fields use proper typing. Validation is trusted via type hints
    rather than redundant isinstance() checks.
    """
    symbol: str
    exchange: Optional[str] = None
    timeframe: Optional[str] = None
    ohlcv_candles: Optional[np.ndarray] = None
    current_price: Optional[float] = None
    timestamps: Optional[List[datetime]] = None
    technical_data: Dict[str, Any] = field(default_factory=dict)
    technical_history: Dict[str, Any] = field(default_factory=dict)
    technical_patterns: Dict[str, Any] = field(default_factory=dict)
    market_metrics: Dict[str, Any] = field(default_factory=dict)
    sentiment: Optional[Dict[str, Any]] = None
    long_term_data: Optional[Dict[str, Any]] = None
    weekly_macro_indicators: Optional[Dict[str, Any]] = None
    market_overview: Dict[str, Any] = field(default_factory=dict)
    market_microstructure: Dict[str, Any] = field(default_factory=dict)
    weekly_ohlcv: Optional[np.ndarray] = None
    available_weeks: int = 0
    meets_200w_threshold: bool = False
    news_articles: List[Dict[str, Any]] = field(default_factory=list)
    coin_details: Dict[str, Any] = field(default_factory=dict)
