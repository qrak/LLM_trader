"""
Analyzer module with improved organization and structure.

This module provides a clean, logical organization of analysis components:

- core/: Main analysis engine, context management, and result processing
- data/: Data collection, fetching, and processing components  
- calculations/: All calculation logic including indicators, metrics, and patterns
- formatting/: Output formatting separated by concern (technical vs market analysis)
- prompts/: Prompt building, context construction, and template management
- publishing/: Result publishing and output management

Key Components:
- AnalysisEngine: Main analysis orchestrator (formerly MarketAnalyzer)
- TechnicalCalculator: Technical indicator calculations
- TechnicalAnalysisFormatter: Technical analysis output formatting
- PromptBuilder: AI prompt construction (formerly prompt.py)
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
# Publishing components
from .publishing import AnalysisPublisher

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
    'TemplateManager',

    
    # Publishing
    'AnalysisPublisher'
]