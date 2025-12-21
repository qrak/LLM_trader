"""
Indicator Constants and Thresholds

Single source of truth for all indicator threshold values used across the application.
These values define the standard interpretation levels for technical indicators.
"""

# Technical Indicator Thresholds
# These thresholds are used for formatting, analysis, and pattern detection
INDICATOR_THRESHOLDS = {
    # Momentum Indicators
    'rsi': {
        'oversold': 30,
        'overbought': 70
    },
    'stoch_k': {
        'oversold': 20,
        'overbought': 80
    },
    'stoch_d': {
        'oversold': 20,
        'overbought': 80
    },
    'williams_r': {
        'oversold': -80,
        'overbought': -20
    },
    'mfi': {
        'oversold': 20,
        'overbought': 80
    },
    
    # Trend Indicators
    'adx': {
        'weak': 25,
        'strong': 50,
        'very_strong': 75
    },
    
    # Volatility Indicators
    'bb_width': {
        'tight': 2,
        'wide': 10
    },
    'bb_percent_b': {
        'oversold': 0.2,
        'overbought': 0.8
    }
}
