"""Verify MarketMetricsCalculator refactor works correctly."""
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.timeframe_validator import TimeframeValidator

@dataclass
class MockContext:
    """Mock AnalysisContext for testing."""
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    ohlcv_candles: Optional[np.ndarray] = None
    timestamps: Optional[List[datetime]] = None
    technical_data: Dict[str, Any] = field(default_factory=dict)
    technical_history: Dict[str, Any] = field(default_factory=dict)
    market_metrics: Dict[str, Any] = field(default_factory=dict)


def generate_mock_ohlcv(num_candles: int = 100, interval_hours: int = 1) -> np.ndarray:
    """Generate realistic mock OHLCV data.
    
    Args:
        num_candles: Number of candles to generate
        interval_hours: Interval in hours between candles (e.g., 1 for 1h, 4 for 4h)
        
    Returns:
        NumPy array of shape (num_candles, 6) with [timestamp, open, high, low, close, volume]
    """
    base_ts = datetime.now().timestamp() * 1000
    data = []
    price = 50000.0
    
    for i in range(num_candles):
        # Time interval calculation
        ts = base_ts - (num_candles - i) * 3600000 * interval_hours
        
        change = np.random.uniform(-0.02, 0.02)
        open_p = price
        close_p = price * (1 + change)
        high_p = max(open_p, close_p) * (1 + np.random.uniform(0, 0.01))
        low_p = min(open_p, close_p) * (1 - np.random.uniform(0, 0.01))
        volume = np.random.uniform(100, 1000)
        data.append([ts, open_p, high_p, low_p, close_p, volume])
        price = close_p
        
    return np.array(data, dtype=np.float64)


def test_metrics_calculation():
    """Test that metrics calculation works with NumPy arrays."""
    from src.logger.logger import Logger
    from src.analyzer.market_metrics_calculator import MarketMetricsCalculator
    
    logger = Logger("test")
    calc = MarketMetricsCalculator(logger)
    
    print("\n" + "="*50)
    print("TEST 1: Standard 1h Timeframe")
    print("="*50)
    
    # Test with 100 candles (1h)
    ctx = MockContext()
    ctx.timeframe = "1h"
    ctx.ohlcv_candles = generate_mock_ohlcv(100, interval_hours=1)
    
    # Mock technical history
    ctx.technical_history = {
        "rsi": np.random.uniform(30, 70, 100),
        "macd_line": np.random.uniform(-100, 100, 100),
        "adx": np.random.uniform(10, 40, 100)
    }
    
    calc.update_period_metrics(ctx)
    
    # Verify 1h output
    # 1D metrics (24h) needs 24 candles. We gave 100.
    assert "1D" in ctx.market_metrics, "Missing 1D metrics for 1h timeframe"
    metrics = ctx.market_metrics["1D"]["metrics"]
    print(f"✅ 1D Metrics calculated with {metrics['data_points']} candles (Expected: 24)")
    assert metrics['data_points'] == 24, "Should use exactly 24 candles for 1D"


    print("\n" + "="*50)
    print("TEST 2: 4h Timeframe (Dynamic Calculation Check)")
    print("="*50)
    
    # Test with 20 candles (4h)
    # Total time = 20 * 4h = 80 hours (enough for 3 days)
    # 1D (24h) should need only 6 candles (24/4 = 6)
    ctx_4h = MockContext()
    ctx_4h.timeframe = "4h"
    ctx_4h.ohlcv_candles = generate_mock_ohlcv(20, interval_hours=4)
    ctx_4h.technical_history = {
        "rsi": np.random.uniform(30, 70, 20),
        "macd_line": np.random.uniform(-100, 100, 20)
    }
    
    calc.update_period_metrics(ctx_4h)
    
    # Verify 4h output
    assert "1D" in ctx_4h.market_metrics, "Missing 1D metrics for 4h timeframe"
    metrics_4h = ctx_4h.market_metrics["1D"]["metrics"]
    
    print(f"✅ 1D Metrics for 4h timeframe calculated with {metrics_4h['data_points']} candles")
    
    # Critical assertion: Did it use 6 candles?
    expected_candles = 6  # 24h / 4h = 6
    if metrics_4h['data_points'] == expected_candles:
        print(f"✅ CORRECT: Used {expected_candles} candles for 1D (24h) period")
    else:
        print(f"❌ INCORRECT: Used {metrics_4h['data_points']} candles instead of {expected_candles}")
        raise AssertionError(f"Dynamic period calculation failed. Expected {expected_candles}, got {metrics_4h['data_points']}")

    print("\n" + "="*50)
    print("ALL TESTS PASSED ✅")
    print("="*50)


if __name__ == "__main__":
    test_metrics_calculation()
