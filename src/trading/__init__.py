"""Trading module for automated trading decisions and position management."""

from .dataclasses import Position, TradeDecision, TradingMemory
from .brain import TradingBrainService
from .memory import TradingMemoryService
from .statistics import TradingStatisticsService
from .position_extractor import PositionExtractor
from .trading_strategy import TradingStrategy
from .statistics_calculator import TradingStatistics, StatisticsCalculator
from .vector_memory import VectorMemoryService

__all__ = [
    'Position',
    'TradeDecision',
    'TradingMemory',
    'TradingStatistics',
    'StatisticsCalculator',
    'TradingBrainService',
    'TradingMemoryService',
    'TradingStatisticsService',
    'PositionExtractor',
    'TradingStrategy',
    'VectorMemoryService',
]

