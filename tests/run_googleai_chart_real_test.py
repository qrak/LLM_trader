"""Standalone script to test Google AI chart analysis with real CCXT data and REAL API calls.

Usage:
    python tests/run_googleai_chart_real_test.py [--pair=BTC/USDC] [--timeframe=4h] [--limit=999]

This will:
- Connect to real exchanges via CCXT and fetch actual OHLCV data
- Generate real chart images saved to chart_images/ directory
- Make REAL Google AI API calls (uses gemini-3-flash-preview)
- Verify if the model correctly interprets the chart

WARNING: This test uses real API calls and may consume API quota.
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
import numpy as np
import io

from src.logger.logger import Logger
from src.config.loader import config
from src.analyzer.prompts.prompt_builder import PromptBuilder
from src.analyzer.analysis_context import AnalysisContext
from src.analyzer.data_processor import DataProcessor
from src.analyzer.data_fetcher import DataFetcher
from src.utils.format_utils import FormatUtils
from src.contracts.manager import ModelManager
from src.parsing.unified_parser import UnifiedParser
from src.analyzer.technical_calculator import TechnicalCalculator
from src.factories.technical_indicators_factory import TechnicalIndicatorsFactory
from src.analyzer.pattern_engine.chart_generator import ChartGenerator
from src.platforms.exchange_manager import ExchangeManager


def create_chart_analysis_prompt(pair: str, timeframe: str, current_price: float) -> str:
    """Create a comprehensive chart analysis prompt to test AI interpretation of all indicators."""
    return f"""Analyze this multi-panel candlestick chart for {pair} on the {timeframe} timeframe.
Current price: ${current_price:.2f}

The chart has 4 panels. Please extract and describe what you see in EACH panel:

**PANEL 1 - PRICE CHART (Top, largest panel):**
1. Overall trend direction (bullish/bearish/sideways) based on candlestick sequence
2. SMA 50 (orange line) position relative to price - is price above or below?
3. SMA 200 (purple line) position relative to price and SMA 50
4. Is there a Golden Cross (SMA50 above SMA200) or Death Cross (SMA50 below SMA200)?
5. Key price levels visible: MIN annotation value, MAX annotation value
6. Any notable candlestick patterns (doji, hammer, engulfing, etc.)
7. Support/resistance levels where price has reacted multiple times

**PANEL 2 - RSI (Second panel, yellow line):**
1. Current RSI value (read from the annotation or estimate from line position)
2. Is RSI in overbought zone (>70), oversold zone (<30), or neutral?
3. Any divergence between RSI and price? (price making new highs while RSI making lower highs = bearish divergence)

**PANEL 3 - VOLUME (Third panel, green/red bars):**
1. Recent volume trend (increasing/decreasing)
2. Are volume spikes aligned with price movements?
3. Volume bar colors: green = bullish candle, red = bearish candle

**PANEL 4 - CMF & OBV (Bottom panel):**
1. CMF value (cyan area chart, left y-axis): positive = buying pressure, negative = selling pressure
2. Is CMF above or below the zero line?
3. OBV trend (magenta line, right y-axis): rising = accumulation, falling = distribution
4. Do CMF and OBV confirm or contradict the price trend?

**SYNTHESIS:**
Based on ALL indicators visible in the chart:
- What is the overall market bias? (Bullish/Bearish/Neutral)
- Are indicators aligned (confirming) or divergent (conflicting)?
- What would be a reasonable short-term outlook?

Be specific with numbers you can read from the chart (prices, RSI value, etc.)."""


async def main():
    parser = argparse.ArgumentParser(description='Test Google AI chart analysis with real API calls')
    parser.add_argument('--pair', type=str, default=None,
                       help=f'Trading pair to analyze (default: {config.CRYPTO_PAIR})')
    parser.add_argument('--timeframe', type=str, default=None,
                       help=f'Chart timeframe (default: {config.TIMEFRAME})')
    parser.add_argument('--limit', type=int, default=999,
                       help='Number of candles to fetch (default: 999, then ai_chart_candle_limit limits display)')
    args = parser.parse_args()

    pair = args.pair or config.CRYPTO_PAIR
    timeframe = args.timeframe or config.TIMEFRAME

    logger = Logger("googleai_chart_real_test", logger_debug=True)
    
    print(f"\n{'='*60}")
    print(f"GOOGLE AI CHART TEST - REAL CCXT DATA, REAL API CALLS")
    print(f"{'='*60}")
    print(f"Pair: {pair}")
    print(f"Timeframe: {timeframe}")
    print(f"Model: gemini-3-flash-preview")
    print(f"Candle limit: {args.limit} (chart displays {config.AI_CHART_CANDLE_LIMIT})")
    print(f"Chart save path: {config.DEBUG_CHART_SAVE_PATH}")
    print(f"{'='*60}")
    print(f"\n⚠️  WARNING: This test makes REAL API calls to Google AI!")
    print(f"{'='*60}\n")

    # Initialize core utilities
    data_processor = DataProcessor()
    format_utils = FormatUtils(data_processor)
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
    if not isinstance(ohlcv, np.ndarray):
        ohlcv = np.array(ohlcv)
    
    if len(ohlcv) < 50:
        print(f"WARNING: Only {len(ohlcv)} candles available, chart may look sparse")
    
    print(f"Fetched {len(ohlcv)} candles, latest close: {latest_close}")

    # Build analysis context with real data
    context = AnalysisContext(pair)
    context.timeframe = timeframe
    context.ohlcv_candles = ohlcv
    context.current_price = float(ohlcv[-1, 4])
    context.market_overview = {'coin_data': {pair.split('/')[0]: {'price': context.current_price}}}

    # Calculate technical indicators for chart visualization
    print(f"\nCalculating technical indicators...")
    indicators = technical_calculator.get_indicators(ohlcv)
    print(f"Indicators calculated: {len(indicators)} total")
    
    # Print key indicator values for comparison with AI response
    print("\n--- GROUND TRUTH VALUES (for comparison) ---")
    if 'rsi' in indicators and len(indicators['rsi']) > 0:
        rsi_val = indicators['rsi'][-1]
        print(f"RSI (14): {rsi_val:.1f}")
    if 'sma_50' in indicators and len(indicators['sma_50']) > 0:
        sma50_val = indicators['sma_50'][-1]
        print(f"SMA 50: {sma50_val:.2f}")
    if 'sma_200' in indicators and len(indicators['sma_200']) > 0:
        sma200_val = indicators['sma_200'][-1]
        print(f"SMA 200: {sma200_val:.2f}")
    if 'cmf' in indicators and len(indicators['cmf']) > 0:
        cmf_val = indicators['cmf'][-1]
        print(f"CMF (20): {cmf_val:.3f} ({'Buying Pressure' if cmf_val > 0 else 'Selling Pressure'})")
    print(f"Current Price: {context.current_price:.2f}")
    print(f"Price vs SMA50: {'Above' if context.current_price > sma50_val else 'Below'}")
    print(f"Price vs SMA200: {'Above' if context.current_price > sma200_val else 'Below'}")
    print(f"SMA50 vs SMA200: {'Golden Cross (Bullish)' if sma50_val > sma200_val else 'Death Cross (Bearish)'}")
    print("-------------------------------------------\n")

    # Generate chart image with real data and indicators
    print(f"Generating chart image with indicators...")
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
            chart_buffer = io.BytesIO(f.read())
            chart_buffer.seek(0)
    else:
        chart_buffer = chart_result
        print("Chart generated in memory")

    # Create simplified chart-focused prompt for testing
    prompt = create_chart_analysis_prompt(pair, timeframe, context.current_price)
    system_prompt = f"You are a professional crypto chart analyst. Analyze the provided chart image for {pair}."

    # Initialize ModelManager with REAL Google AI client
    print("\nInitializing ModelManager with REAL Google AI client...")
    manager = ModelManager(logger=logger, config=config, unified_parser=unified_parser)
    
    # Verify Google client is available
    if manager.google_client is None:
        print("ERROR: Google AI client not initialized!")
        print("Make sure GOOGLE_AI_API_KEY environment variable is set.")
        await exchange_manager.shutdown()
        return
    
    # Force provider to googleai
    manager.provider = 'googleai'
    
    print("\n--- System Prompt ---")
    print(system_prompt)
    print("\n--- Analysis Prompt ---")
    print(prompt)

    # Run chart analysis with REAL API
    print(f"\n{'='*60}")
    print("SENDING REQUEST TO GOOGLE AI (gemini-3-flash-preview)...")
    print(f"{'='*60}\n")
    
    try:
        response = await manager.send_prompt_with_chart_analysis(
            prompt,
            chart_buffer,
            system_message=system_prompt,
            provider='googleai'
        )
        
        print(f"\n{'='*60}")
        print("GOOGLE AI RESPONSE")
        print(f"{'='*60}\n")
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
    print(f"\nChart image saved in: {config.DEBUG_CHART_SAVE_PATH}/")
    print("Review the AI response above to verify chart interpretation accuracy.")


if __name__ == '__main__':
    asyncio.run(main())
