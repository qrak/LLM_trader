"""Trading module for automated trading decisions and position management."""

from .dataclasses import Position, TradeDecision, TradingMemory, TradingBrain, TradingInsight, ConfidenceStats, FactorStats
from .persistence import TradingPersistence
from .brain import TradingBrainService
from .memory import TradingMemoryService
from .statistics import TradingStatisticsService
from .position_extractor import PositionExtractor
from .trading_strategy import TradingStrategy
from .statistics_calculator import TradingStatistics, StatisticsCalculator

__all__ = [
    'Position',
    'TradeDecision',
    'TradingMemory',
    'TradingBrain',
    'TradingInsight',
    'ConfidenceStats',
    'FactorStats',
    'TradingStatistics',
    'StatisticsCalculator',
    'TradingPersistence',
    'TradingBrainService',
    'TradingMemoryService',
    'TradingStatisticsService',
    'PositionExtractor',
    'TradingStrategy',
]

