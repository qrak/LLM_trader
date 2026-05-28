"""Analyzer module for market analysis logic."""

# Core analysis components
from .analysis_engine import AnalysisEngine
from .analysis_context import AnalysisContext
from .analysis_result_processor import AnalysisResultProcessor

# Data components
from .market_data_collector import MarketDataCollector
from .data_fetcher import DataFetcher

# Calculation components
from .market_metrics_calculator import MarketMetricsCalculator
from .technical_calculator import TechnicalCalculator
from .pattern_analyzer import PatternAnalyzer

# Formatting components
from .formatters.technical_formatter import TechnicalFormatter
from .formatters.market_formatter import MarketFormatter

# Prompt components
from .prompts import PromptBuilder, TemplateManager

__all__ = [
    # Core
    'AnalysisEngine',
    'AnalysisContext',
    'AnalysisResultProcessor',

    # Data
    'MarketDataCollector',
    'DataFetcher',

    # Calculations
    'MarketMetricsCalculator',
    'TechnicalCalculator',
    'PatternAnalyzer',

    # Formatting
    'TechnicalFormatter',
    'MarketFormatter',

    # Prompts
    'PromptBuilder',
    'TemplateManager'
]
