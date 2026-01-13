"""
Analyzer module with improved organization and structure.

This module provides a clean, logical organization of analysis components:

- core/: Main analysis engine, context management, and result processing
- data/: Data collection, fetching, and processing components  
- calculations/: All calculation logic including indicators, metrics, and patterns
- formatting/: Output formatting for AI prompts and console display
- prompts/: Prompt building, context construction, and template management

Key Components:
- AnalysisEngine: Main analysis orchestrator
- TechnicalCalculator: Technical indicator calculations
- PromptBuilder: AI prompt construction
"""

# Core analysis components
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
from .prompts import PromptBuilder, ContextBuilder, TemplateManager

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
    'ContextBuilder',
    'TemplateManager'
]