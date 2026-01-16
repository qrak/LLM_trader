"""Standalone script to test Google AI chart analysis with real CCXT data and mocked API calls.

Usage:
    python tests/run_googleai_chart_test.py [--pair=BTC/USDC] [--timeframe=4h] [--limit=999]

This will:
- Connect to real exchanges via CCXT and fetch actual OHLCV data
- Generate real chart images saved to chart_images/ directory
- Mock the Google AI API calls to avoid costs
- Run the full prompt flow with chart analysis

The chart images can be visually inspected to verify how they look
before integrating indicators.
"""
import sys
import os
import argparse
from pathlib import Path

# Fix Windows console encoding issues with Unicode/emoji characters
if sys.platform == 'win32':
    try:
        # Try to set UTF-8 encoding for stdout/stderr
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
        # Also set environment variable for subprocess
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    except Exception:
        pass  # If reconfigure fails, continue anyway

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import asyncio

from src.logger.logger import Logger
from src.config.loader import config
from src.analyzer.prompts.prompt_builder import PromptBuilder
from src.analyzer.analysis_context import AnalysisContext
from src.analyzer.data_fetcher import DataFetcher
from src.utils.format_utils import FormatUtils
from src.platforms.ai_providers.mock import MockClient
from src.managers.model_manager import ModelManager
from src.parsing.unified_parser import UnifiedParser
from src.analyzer.technical_calculator import TechnicalCalculator
from src.factories.technical_indicators_factory import TechnicalIndicatorsFactory
from src.analyzer.pattern_engine.chart_generator import ChartGenerator
from src.platforms.exchange_manager import ExchangeManager
from src.analyzer.formatters import (
    MarketOverviewFormatter,
    LongTermFormatter,
    MarketFormatter,
    MarketPeriodFormatter,
    TechnicalFormatter
)


async def main():
    parser = argparse.ArgumentParser(description='Test Google AI chart analysis with real CCXT data')
    parser.add_argument('--pair', type=str, default=None,
                       help=f'Trading pair to analyze (default: {config.CRYPTO_PAIR})')
    parser.add_argument('--timeframe', type=str, default=None,
                       help=f'Chart timeframe (default: {config.TIMEFRAME})')
    parser.add_argument('--limit', type=int, default=999,
                       help='Number of candles to fetch (default: 999, then ai_chart_candle_limit limits display)')
    args = parser.parse_args()

    pair = args.pair or config.CRYPTO_PAIR
    timeframe = args.timeframe or config.TIMEFRAME

    logger = Logger("googleai_chart_test", logger_debug=True)
    
    print(f"\n{'='*60}")
    print(f"GOOGLE AI CHART TEST - REAL CCXT DATA, MOCKED API")
    print(f"{'='*60}")
    print(f"Pair: {pair}")
    print(f"Timeframe: {timeframe}")
    print(f"Candle limit: {args.limit} (chart displays {config.AI_CHART_CANDLE_LIMIT})")
    print(f"Chart save path: {config.DEBUG_CHART_SAVE_PATH}")
    print(f"{'='*60}\n")

    # Initialize core utilities
    format_utils = FormatUtils()
    unified_parser = UnifiedParser(logger=logger, format_utils=format_utils)
    ti_factory = TechnicalIndicatorsFactory()
    technical_calculator = TechnicalCalculator(logger=logger, format_utils=format_utils, ti_factory=ti_factory)

    # Initialize exchange manager and connect to real exchanges
    print("Connecting to exchange...")
    exchange_manager = ExchangeManager(logger, config)
    await exchange_manager.initialize()
    
    # Find exchange that supports the pair
    exchange, exchange_name = await exchange_manager.find_symbol_exchange(pair)
    if exchange is None:
        print(f"ERROR: Could not find exchange supporting {pair}")
        await exchange_manager.shutdown()
        return
    
    print(f"Using exchange: {exchange_name}")

    # Fetch real OHLCV data
    print(f"Fetching {args.limit} candles of {pair} ({timeframe})...")
    data_fetcher = DataFetcher(exchange, logger)
    result = await data_fetcher.fetch_candlestick_data(pair, timeframe, args.limit)
    
    if result is None:
        print(f"ERROR: Failed to fetch OHLCV data for {pair}")
        await exchange_manager.shutdown()
        return
    
    # fetch_candlestick_data returns tuple (ohlcv_array, latest_close)
    ohlcv, latest_close = result
    
    # Ensure we have a numpy array
    import numpy as np
    if not isinstance(ohlcv, np.ndarray):
        ohlcv = np.array(ohlcv)
    
    if len(ohlcv) < 50:
        print(f"WARNING: Only {len(ohlcv)} candles available, chart may look sparse")
    
    print(f"Fetched {len(ohlcv)} candles, latest close: {latest_close}")
    import datetime
    if len(ohlcv) > 0:
        last_ts = ohlcv[-1][0]
        last_time_str = datetime.datetime.fromtimestamp(last_ts / 1000).strftime('%Y-%m-%d %H:%M:%S')
        print(f"DEBUG: Last candle timestamp: {last_ts} ({last_time_str})")
    
    current_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"DEBUG: Current System Time: {current_time_str}")

    # Build analysis context with real data
    context = AnalysisContext(pair)
    context.timeframe = timeframe
    context.ohlcv_candles = ohlcv
    context.current_price = float(latest_close)
    context.market_overview = {'coin_data': {pair.split('/')[0]: {'price': context.current_price}}}

    # Calculate technical indicators for chart visualization
    print(f"\nCalculating technical indicators...")
    indicators = technical_calculator.get_indicators(ohlcv)
    print(f"Indicators calculated: {list(indicators.keys())[:10]}... ({len(indicators)} total)")

    # Generate chart image with real data and indicators
    print(f"\nGenerating chart image with indicators (RSI, SMA 50/200, Volume, CMF, OBV)...")
    chart_generator = ChartGenerator(
        logger=logger, 
        config=config, 
        formatter=format_utils.fmt, 
        format_utils=format_utils
    )
    
    chart_result = chart_generator.create_chart_image(
        ohlcv=ohlcv,
        technical_history=indicators,  # Pass indicators for visualization
        pair_symbol=pair,
        timeframe=timeframe,
        save_to_disk=True  # Save to chart_images/ directory
    )
    
    # chart_result is a file path when save_to_disk=True
    if isinstance(chart_result, str):
        print(f"Chart saved to: {chart_result}")
        # Read back into BytesIO for API call
        with open(chart_result, 'rb') as f:
            import io
            chart_buffer = io.BytesIO(f.read())
            chart_buffer.seek(0)
    else:
        chart_buffer = chart_result
        print("Chart generated in memory")

    # Initialize formatters for dependency injection
    overview_formatter = MarketOverviewFormatter(logger, format_utils)
    long_term_formatter = LongTermFormatter(logger, format_utils)
    market_formatter = MarketFormatter(logger, format_utils)
    period_formatter = MarketPeriodFormatter(logger, format_utils)
    technical_formatter = TechnicalFormatter(technical_calculator, logger, format_utils)

    # Build prompt
    pb = PromptBuilder(
        timeframe=timeframe,
        logger=logger,
        technical_calculator=technical_calculator,
        format_utils=format_utils,
        config=config,
        overview_formatter=overview_formatter,
        long_term_formatter=long_term_formatter,
        market_formatter=market_formatter,
        period_formatter=period_formatter,
        technical_formatter=technical_formatter
    )
    
    prompt = pb.build_prompt(context)
    system_prompt = pb.build_system_prompt(pair)

    # Initialize ModelManager with mocked Google AI
    print("\nInitializing ModelManager with mocked Google AI...")
    manager = ModelManager(logger=logger, config=config, unified_parser=unified_parser)
    
    # Replace Google client with mock
    manager.google_client = MockClient(logger=logger)
    if 'googleai' in manager.PROVIDER_METADATA:
        manager.PROVIDER_METADATA['googleai']['client'] = manager.google_client
    
    # Force provider to googleai
    manager.provider = 'googleai'

    # Prepare messages
    prepared_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt + f"\n\nTEST_HINT: last_close={context.current_price}"}
    ]

    print("\n--- System Prompt (truncated) ---")
    print(system_prompt[:500])
    print("\n--- Main Prompt (truncated) ---")
    print(prompt[:1000])

    # Run chart analysis with mocked API
    print("\n--- Sending to Mocked Google AI with Chart ---")
    try:
        response = await manager.send_prompt_with_chart_analysis(
            prompt + f"\n\nTEST_HINT: last_close={context.current_price}",
            chart_buffer,
            system_message=system_prompt,
            provider='googleai'
        )
        print("\n--- Mock Google AI Response ---")
        print(response)
    except Exception as e:
        print(f"ERROR during chart analysis: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup
    await exchange_manager.shutdown()

    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print(f"{'='*60}")
    print(f"\nCheck the chart image in: {config.DEBUG_CHART_SAVE_PATH}/")


if __name__ == '__main__':
    asyncio.run(main())
