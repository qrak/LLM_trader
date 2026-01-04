"""Trading module for automated trading decisions and position management."""

from .dataclasses import Position, TradeDecision, TradingMemory, TradingBrain, TradingInsight, ConfidenceStats
from .data_persistence import DataPersistence
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
    'TradingStatistics',
    'StatisticsCalculator',
    'DataPersistence',
    'PositionExtractor',
    'TradingStrategy',
]

