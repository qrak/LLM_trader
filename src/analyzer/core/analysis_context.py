from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np


class AnalysisContext:
    """
    Enhanced data container for market analysis data
    Provides proper storage, validation and serialization
    """
    
    def __init__(self, symbol: str):
        # Basic info
        self._symbol = symbol
        self._exchange = None
        self._timeframe = None
        
        # OHLCV data
        self._ohlcv_candles = None
        self._current_price = None
        self._timestamps = None
        
        # Indicator data
        self._technical_data = {}
        self._technical_history = {}
        self._technical_patterns = {}
        
        # Market analysis data
        self._market_metrics = {}
        self._sentiment = None
        self._long_term_data = None
        self._weekly_macro_indicators = None
        self._market_overview = {}  # Initialize as an empty dictionary
        self._market_microstructure = {}  # Order book, trades, funding rate data
        
        # News and articles
        self._news_articles = []
        
        # Cryptocurrency details
        self._coin_details = {}
        
    @property
    def symbol(self) -> str:
        """Get the trading symbol"""
        return self._symbol
        
    @property
    def exchange(self) -> Optional[str]:
        """Get the exchange name"""
        return self._exchange
        
    @exchange.setter
    def exchange(self, value: str):
        """Set the exchange name"""
        self._exchange = value
        
    @property
    def timeframe(self) -> Optional[str]:
        """Get the analysis timeframe"""
        return self._timeframe
        
    @timeframe.setter
    def timeframe(self, value: str):
        """Set the analysis timeframe"""
        self._timeframe = value
        
    @property
    def ohlcv_candles(self) -> Optional[np.ndarray]:
        """Get the OHLCV candle data"""
        return self._ohlcv_candles
        
    @ohlcv_candles.setter
    def ohlcv_candles(self, value: np.ndarray):
        """Set the OHLCV candle data with validation"""
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError("OHLCV data must be a numpy array")
        self._ohlcv_candles = value
        
    @property
    def current_price(self) -> Optional[float]:
        """Get the current price"""
        return self._current_price
        
    @current_price.setter
    def current_price(self, value: float):
        """Set the current price with validation"""
        if value is not None:
            try:
                self._current_price = float(value)
            except (ValueError, TypeError):
                raise ValueError("Current price must be a valid number")
        else:
            self._current_price = None
    
    @property
    def timestamps(self) -> Optional[List[datetime]]:
        """Get datetime timestamps extracted from OHLCV data"""
        return self._timestamps
    
    @timestamps.setter
    def timestamps(self, value: Optional[List[datetime]]):
        """Set timestamps"""
        self._timestamps = value
            
    @property
    def technical_data(self) -> Dict[str, Any]:
        """Get technical indicator point data"""
        return self._technical_data
        
    @technical_data.setter
    def technical_data(self, value: Dict[str, Any]):
        """Set technical indicator point data"""
        if not isinstance(value, dict):
            raise TypeError("Technical data must be a dictionary")
        self._technical_data = value
        
    @property
    def technical_history(self) -> Dict[str, Any]:
        """Get historical technical indicator data"""
        return self._technical_history
        
    @technical_history.setter
    def technical_history(self, value: Dict[str, Any]):
        """Set historical technical indicator data"""
        if not isinstance(value, dict):
            raise TypeError("Technical history must be a dictionary")
        self._technical_history = value
        
    @property
    def technical_patterns(self) -> Dict[str, Any]:
        """Get detected technical patterns"""
        return self._technical_patterns
        
    @technical_patterns.setter
    def technical_patterns(self, value: Dict[str, Any]):
        """Set detected technical patterns"""
        if value is not None and not isinstance(value, dict):
            raise TypeError("Technical patterns must be a dictionary")
        self._technical_patterns = value or {}
        
    @property
    def market_metrics(self) -> Dict[str, Any]:
        """Get market metrics by period"""
        return self._market_metrics
        
    @market_metrics.setter
    def market_metrics(self, value: Dict[str, Any]):
        """Set market metrics by period"""
        if not isinstance(value, dict):
            raise TypeError("Market metrics must be a dictionary")
        self._market_metrics = value
        
    @property
    def sentiment(self) -> Optional[Dict[str, Any]]:
        """Get market sentiment data"""
        return self._sentiment
        
    @sentiment.setter
    def sentiment(self, value: Dict[str, Any]):
        """Set market sentiment data"""
        if value is not None and not isinstance(value, dict):
            raise TypeError("Sentiment data must be a dictionary")
        self._sentiment = value
        
    @property
    def long_term_data(self) -> Optional[Dict[str, Any]]:
        """Get long-term historical data"""
        return self._long_term_data
        
    @long_term_data.setter
    def long_term_data(self, value: Dict[str, Any]):
        """Set long-term historical data"""
        if value is not None and not isinstance(value, dict):
            raise TypeError("Long-term data must be a dictionary")
        self._long_term_data = value
    
    @property
    def weekly_macro_indicators(self) -> Optional[Dict[str, Any]]:
        """Get weekly macro indicators (200W SMA analysis)"""
        return self._weekly_macro_indicators

    @weekly_macro_indicators.setter
    def weekly_macro_indicators(self, value: Dict[str, Any]):
        """Set weekly macro indicators"""
        if value is not None and not isinstance(value, dict):
            raise TypeError("Weekly macro indicators must be a dictionary")
        self._weekly_macro_indicators = value
        
    @property
    def market_overview(self) -> Optional[Dict[str, Any]]:
        """Get market overview data"""
        return self._market_overview

    @market_overview.setter
    def market_overview(self, value: Dict[str, Any]):
        """Set market overview data"""
        if value is not None and not isinstance(value, dict):
            raise TypeError("Market overview data must be a dictionary")
        self._market_overview = value or {}
    
    @property
    def market_microstructure(self) -> Dict[str, Any]:
        """Get market microstructure data (order book, trades, funding rate)"""
        return self._market_microstructure
    
    @market_microstructure.setter
    def market_microstructure(self, value: Dict[str, Any]):
        """Set market microstructure data"""
        if value is not None and not isinstance(value, dict):
            raise TypeError("Market microstructure data must be a dictionary")
        self._market_microstructure = value or {}
        
    @property
    def news_articles(self) -> List[Dict[str, Any]]:
        """Get related news articles"""
        return self._news_articles
        
    @news_articles.setter
    def news_articles(self, value: List[Dict[str, Any]]):
        """Set related news articles"""
        if not isinstance(value, list):
            raise TypeError("News articles must be a list")
        self._news_articles = value
        
    @property
    def coin_details(self) -> Dict[str, Any]:
        """Get cryptocurrency details including description, taxonomy, and ratings"""
        return self._coin_details
        
    @coin_details.setter
    def coin_details(self, value: Dict[str, Any]):
        """Set cryptocurrency details"""
        if not isinstance(value, dict):
            raise TypeError("Coin details must be a dictionary")
        self._coin_details = value
