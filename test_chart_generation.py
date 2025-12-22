"""
Test script for chart generation - generates a sample chart image for visual inspection.
This script creates a chart using real market data and saves it to disk.
"""
import asyncio
import sys
import os
from pathlib import Path
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.analyzer.pattern_engine.chart_generator import ChartGenerator
import ccxt.async_support as ccxt


async def test_chart_generation():
    """Generate a test chart using real market data."""
    
    print("=" * 60)
    print("Chart Generation Test")
    print("=" * 60)
    
    # Simple config class for test
    class TestConfig:
        AI_CHART_CANDLE_LIMIT = 100
        DEBUG_CHART_SAVE_PATH = "test_images"
    
    config = TestConfig()
    
    # Initialize chart generator with minimal dependencies
    chart_generator = ChartGenerator(
        logger=None,  # We'll print to console
        config=config,
        format_utils=None  # Will use default formatter
    )
    
    # Fetch sample data from Binance
    print("\n📊 Fetching market data from Binance...")
    exchange = ccxt.binance()
    
    try:
        # Fetch OHLCV data (last 300 candles for 1h timeframe)
        symbol = "BTC/USDT"
        timeframe = "1h"
        limit = 100  # Fetch more than AI limit to test truncation
        
        print(f"   Symbol: {symbol}")
        print(f"   Timeframe: {timeframe}")
        print(f"   Candles: {limit}")
        print(f"   AI Chart Limit: {config.AI_CHART_CANDLE_LIMIT}")
        
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        await exchange.close()
        
        # Convert to numpy array
        ohlcv_array = np.array(ohlcv, dtype=float)
        
        print(f"\n✅ Fetched {len(ohlcv_array)} candles successfully")
        print(f"   Price range: ${ohlcv_array[:, 4].min():.2f} - ${ohlcv_array[:, 4].max():.2f}")
        
        # Generate chart with timestamps
        print(f"\n🎨 Generating chart image...")
        print(f"   Chart will be limited to last {config.AI_CHART_CANDLE_LIMIT} candles")
        print(f"   Save location: {config.DEBUG_CHART_SAVE_PATH}/")
        
        # Create output directory if it doesn't exist
        os.makedirs(config.DEBUG_CHART_SAVE_PATH, exist_ok=True)
        
        chart_path = chart_generator.create_chart_image(
            ohlcv=ohlcv_array,
            pair_symbol=symbol,
            timeframe=timeframe,
            save_to_disk=True,
            timestamps=None  # Let chart generator convert timestamps
        )
        
        print(f"\n✅ Chart saved successfully!")
        print(f"   Path: {chart_path}")
        print(f"\n💡 Open the image to inspect the chart quality for AI analysis.")
        print(f"   Look for:")
        print(f"   - High contrast (black background, bright colors)")
        print(f"   - Clear candlestick patterns")
        print(f"   - Dense grid lines for pattern analysis")
        print(f"   - Highest/Lowest point annotations")
        print(f"   - Current price reference line")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        if exchange:
            await exchange.close()
        raise
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_chart_generation())
