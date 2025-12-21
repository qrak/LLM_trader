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
from .core import AnalysisEngine, AnalysisContext, AnalysisResultProcessor

# Data components  
from .data import MarketDataCollector, DataFetcher, DataProcessor

# Calculation components
from .calculations import MarketMetricsCalculator, TechnicalCalculator, PatternAnalyzer

# Formatting components  
from .formatting import TechnicalFormatter, MarketFormatter, IndicatorFormatter

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
    'DataProcessor',
    
    # Calculations
    'MarketMetricsCalculator', 
    'TechnicalCalculator',
    'PatternAnalyzer',
    
    # Formatting
    'TechnicalFormatter',
    'MarketFormatter', 
    'IndicatorFormatter',
    
    # Prompts
    'PromptBuilder',
    'ContextBuilder',
    'TemplateManager'
]