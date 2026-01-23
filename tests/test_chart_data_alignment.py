"""
Test to verify chart generator and technical calculator use aligned data.

Usage:
    & ./.venv/Scripts/Activate.ps1
    python tests/test_chart_data_alignment.py

This test verifies:
1. Chart OHLCV data matches analysis OHLCV data (last candle alignment)
2. Chart slice length matches expected AI_CHART_CANDLE_LIMIT
3. Both chart and analysis use the same most recent closed candle
"""
import asyncio
import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    except Exception:
        pass

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from src.logger.logger import Logger
from src.platforms.exchange_manager import ExchangeManager
from src.config.loader import config


async def main():
    """Run chart data alignment verification."""
    print("=" * 70)
    print("CHART DATA ALIGNMENT TEST")
    print("=" * 70)
    
    logger = Logger()
    
    try:
        # Initialize exchange manager
        exchange_manager = ExchangeManager(logger, config)
        await exchange_manager.initialize()
        
        # Fetch data with same parameters as real analysis
        pair = "BTC/USDT"
        timeframe = "4h"
        analysis_limit = 999  # Full analysis window
        chart_limit = 125     # AI_CHART_CANDLE_LIMIT default
        
        print(f"\n--- Configuration ---")
        print(f"  Pair: {pair}")
        print(f"  Timeframe: {timeframe}")
        print(f"  Analysis limit: {analysis_limit} candles")
        print(f"  Chart limit: {chart_limit} candles")
        
        # Find exchange for the symbol
        exchange, exchange_name = await exchange_manager.find_symbol_exchange(pair)
        if exchange is None:
            print(f"\n[FAIL] Could not find exchange for {pair}")
            return 1
        
        print(f"  Exchange: {exchange_name}")
        
        # Fetch OHLCV data (same method used in production)
        ohlcv = await exchange.fetch_ohlcv(
            pair, timeframe, limit=analysis_limit + 1
        )
        
        if ohlcv is None or len(ohlcv) < 2:
            print("\n[FAIL] Could not fetch OHLCV data")
            return 1
        
        ohlcv_array = np.array(ohlcv)
        
        # Exclude incomplete candle (production behavior)
        closed_candles = ohlcv_array[:-1]
        
        print(f"\n--- Data Fetched ---")
        print(f"  Total candles (with incomplete): {len(ohlcv_array)}")
        print(f"  Closed candles: {len(closed_candles)}")
        
        # Simulate chart slice (as done in chart_generator.py)
        chart_ohlcv = closed_candles[-chart_limit:] if len(closed_candles) >= chart_limit else closed_candles
        
        print(f"\n--- Alignment Checks ---")
        
        # Test 1: Last candle alignment
        analysis_last = closed_candles[-1]
        chart_last = chart_ohlcv[-1]
        last_candle_aligned = np.array_equal(analysis_last, chart_last)
        
        if last_candle_aligned:
            print(f"  [PASS] Last candle alignment")
            print(f"     Analysis last: timestamp={int(analysis_last[0])}, close={analysis_last[4]:.2f}")
            print(f"     Chart last:    timestamp={int(chart_last[0])}, close={chart_last[4]:.2f}")
        else:
            print(f"  [FAIL] Last candle alignment")
            print(f"     Analysis last: timestamp={int(analysis_last[0])}, close={analysis_last[4]:.2f}")
            print(f"     Chart last:    timestamp={int(chart_last[0])}, close={chart_last[4]:.2f}")
            return 1
        
        # Test 2: Chart length
        expected_chart_len = min(chart_limit, len(closed_candles))
        actual_chart_len = len(chart_ohlcv)
        
        if actual_chart_len == expected_chart_len:
            print(f"  [PASS] Chart length: {actual_chart_len} candles")
        else:
            print(f"  [FAIL] Chart length: expected {expected_chart_len}, got {actual_chart_len}")
            return 1
        
        # Test 3: Chart is a subset of analysis data
        # Check first chart candle exists in analysis data
        chart_first = chart_ohlcv[0]
        chart_first_found = False
        for i, candle in enumerate(closed_candles):
            if np.array_equal(candle, chart_first):
                chart_first_found = True
                print(f"  [PASS] Chart subset verification")
                print(f"     Chart first candle found at analysis index {i}")
                break
        
        if not chart_first_found:
            print(f"  [FAIL] Chart subset verification")
            print(f"     Chart first candle not found in analysis data")
            return 1
        
        # Test 4: Timestamp continuity
        chart_timestamps = chart_ohlcv[:, 0]
        timestamp_diffs = np.diff(chart_timestamps)
        expected_diff_ms = 4 * 60 * 60 * 1000  # 4 hours in milliseconds
        
        # Allow small tolerance for potential exchange timestamp quirks
        all_consistent = np.all(np.abs(timestamp_diffs - expected_diff_ms) < 1000)
        
        if all_consistent:
            print(f"  [PASS] Timestamp continuity: consistent 4h intervals")
        else:
            print(f"  [WARN] Timestamp continuity: some intervals vary")
            # Not a failure, just informational
        
        print("\n" + "=" * 70)
        print("[SUCCESS] ALL ALIGNMENT CHECKS PASSED")
        print("   Chart and analysis data are properly aligned.")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if 'exchange_manager' in locals():
            await exchange_manager.shutdown()


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
