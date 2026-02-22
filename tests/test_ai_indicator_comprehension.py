"""
Test AI comprehension of technical indicators with real API calls.

Usage:
    python tests/test_ai_indicator_comprehension.py [--no-chart]

This test:
1. Fetches real 4h OHLCV data from Binance
2. Calculates all technical indicators + detects patterns
3. Sends the EXACT same prompt format used by the trading bot
4. Asks AI to echo back specific indicator values
5. Validates AI correctly extracted and interpreted values

WARNING: Uses real Google AI API calls.
"""
import sys
import os
import argparse
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import asyncio  # noqa: E402
import numpy as np  # noqa: E402
import io  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Optional  # noqa: E402

from src.logger.logger import Logger  # noqa: E402
from src.config.loader import config  # noqa: E402
from src.analyzer.data_fetcher import DataFetcher  # noqa: E402
from src.utils.format_utils import FormatUtils  # noqa: E402
from src.managers.model_manager import ModelManager  # noqa: E402
from src.parsing.unified_parser import UnifiedParser  # noqa: E402
from src.analyzer.technical_calculator import TechnicalCalculator  # noqa: E402
from src.factories.technical_indicators_factory import TechnicalIndicatorsFactory  # noqa: E402
from src.analyzer.pattern_analyzer import PatternAnalyzer  # noqa: E402
from src.analyzer.pattern_engine.chart_generator import ChartGenerator  # noqa: E402
from src.platforms.exchange_manager import ExchangeManager  # noqa: E402
from src.utils.format_utils import timestamps_from_ms_array  # noqa: E402


@dataclass
class ValidationCase:
    """Represents a single test case for AI comprehension validation."""
    name: str
    description: str
    expected_keywords: list[str]
    ground_truth: str
    passed: Optional[bool] = None
    ai_excerpt: Optional[str] = None


def create_indicator_comprehension_prompt(
    pair: str,
    timeframe: str,
    current_price: float,
    indicators: dict,
    patterns: dict,
    format_utils: FormatUtils
) -> str:
    """Create a prompt that tests AI's ability to extract and interpret indicator values."""
    
    # Extract key indicator values for validation
    rsi = indicators.get('rsi', np.array([]))[-1] if len(indicators.get('rsi', [])) > 0 else None
    macd_line = indicators.get('macd_line', np.array([]))[-1] if len(indicators.get('macd_line', [])) > 0 else None
    macd_signal = indicators.get('macd_signal', np.array([]))[-1] if len(indicators.get('macd_signal', [])) > 0 else None
    macd_hist = indicators.get('macd_hist', np.array([]))[-1] if len(indicators.get('macd_hist', [])) > 0 else None
    adx = indicators.get('adx', np.array([]))[-1] if len(indicators.get('adx', [])) > 0 else None
    plus_di = indicators.get('plus_di', np.array([]))[-1] if len(indicators.get('plus_di', [])) > 0 else None
    minus_di = indicators.get('minus_di', np.array([]))[-1] if len(indicators.get('minus_di', [])) > 0 else None
    stoch_k = indicators.get('stoch_k', np.array([]))[-1] if len(indicators.get('stoch_k', [])) > 0 else None
    stoch_d = indicators.get('stoch_d', np.array([]))[-1] if len(indicators.get('stoch_d', [])) > 0 else None
    sma_50 = indicators.get('sma_50', np.array([]))[-1] if len(indicators.get('sma_50', [])) > 0 else None
    sma_200 = indicators.get('sma_200', np.array([]))[-1] if len(indicators.get('sma_200', [])) > 0 else None
    
    # Format patterns
    pattern_lines = []
    for category, pattern_list in patterns.items():
        for p in pattern_list:
            desc = p.get('description', 'Unknown pattern')
            pattern_lines.append(f"- {desc}")
    patterns_str = "\n".join(pattern_lines[:10]) if pattern_lines else "No patterns detected"
    
    
    # Use FormatUtils for consistent formatting
    def fmt(val):
        return format_utils.fmt(val, precision=2)

    prompt = f"""INDICATOR COMPREHENSION TEST

You are analyzing {pair} on the {timeframe} timeframe.
Current Price: ${fmt(current_price)}

I will provide you with technical indicator values. Your task is to:
1. EXTRACT the exact values I provide
2. INTERPRET what each indicator suggests
3. SYNTHESIZE an overall market assessment

## TECHNICAL INDICATORS:

**Momentum:**
- RSI (14): {fmt(rsi)}
- MACD Line: {fmt(macd_line)}
- MACD Signal: {fmt(macd_signal)}
- MACD Histogram: {fmt(macd_hist)}
- Stochastic %K: {fmt(stoch_k)}
- Stochastic %D: {fmt(stoch_d)}

**Trend:**
- ADX: {fmt(adx)}
- +DI: {fmt(plus_di)}
- -DI: {fmt(minus_di)}

**Moving Averages:**
- SMA 50: {fmt(sma_50)}
- SMA 200: {fmt(sma_200)}
- Price vs SMA 50: {'Above' if sma_50 and current_price > sma_50 else 'Below'} (${fmt(abs(current_price - sma_50))} difference) if sma_50 else 'N/A'
- Price vs SMA 200: {'Above' if sma_200 and current_price > sma_200 else 'Below'} (${fmt(abs(current_price - sma_200))} difference) if sma_200 else 'N/A'

## DETECTED PATTERNS:
{patterns_str}

## YOUR RESPONSE MUST INCLUDE:

1. **RSI INTERPRETATION**: State the RSI value and whether it indicates overbought (>70), oversold (<30), or neutral (30-70)

2. **MACD INTERPRETATION**: State whether MACD is bullish (line > signal, histogram positive) or bearish (line < signal, histogram negative)

3. **TREND STRENGTH**: State the ADX value and classify as weak (<20), developing (20-25), or strong (>25) trend. State whether +DI > -DI (bullish) or -DI > +DI (bearish)

4. **MA STRUCTURE**: State whether there is a Golden Cross (SMA50 > SMA200) or Death Cross (SMA50 < SMA200)

5. **PATTERN RECOGNITION**: Acknowledge at least ONE pattern from the detected patterns list

6. **OVERALL ASSESSMENT**: Provide a one-sentence summary of the market condition

Begin your analysis:"""
    
    return prompt


def validate_response(response: str, test_cases: list[ValidationCase]) -> tuple[int, int]:
    """Validate AI response against test cases."""
    passed = 0
    failed = 0
    
    for tc in test_cases:
        # Check if any expected keyword is in the response
        found_keywords = [kw for kw in tc.expected_keywords if kw.lower() in response.lower()]
        
        if found_keywords:
            tc.passed = True
            tc.ai_excerpt = found_keywords[0]
            passed += 1
            print(f"  ✅ {tc.name}: PASS")
            print(f"     Ground Truth: {tc.ground_truth}")
            print(f"     Found: {', '.join(found_keywords)}")
        else:
            tc.passed = False
            failed += 1
            print(f"  ❌ {tc.name}: FAIL")
            print(f"     Ground Truth: {tc.ground_truth}")
            print(f"     Expected keywords: {tc.expected_keywords}")
            print("     (Keywords not found in response)")
    
    return passed, failed


async def main():
    parser = argparse.ArgumentParser(description='Test AI indicator comprehension')
    parser.add_argument('--pair', type=str, default=None,
                       help=f'Trading pair (default: {config.CRYPTO_PAIR})')
    parser.add_argument('--timeframe', type=str, default=None,
                       help=f'Timeframe (default: {config.TIMEFRAME})')
    parser.add_argument('--no-chart', action='store_true',
                       help='Skip chart generation (text-only analysis)')
    args = parser.parse_args()

    pair = args.pair or config.CRYPTO_PAIR
    timeframe = args.timeframe or config.TIMEFRAME

    logger = Logger("ai_indicator_test", logger_debug=True)
    
    print(f"\n{'='*70}")
    print("AI INDICATOR COMPREHENSION TEST")
    print(f"{'='*70}")
    print(f"Pair: {pair}")
    print(f"Timeframe: {timeframe}")
    print("Model: gemini-3-flash-preview")
    print(f"Chart Analysis: {'Disabled' if args.no_chart else 'Enabled'}")
    print(f"{'='*70}")
    print("\n⚠️  WARNING: This test makes REAL API calls to Google AI!")
    print(f"{'='*70}\n")

    # Initialize components
    format_utils = FormatUtils()
    unified_parser = UnifiedParser(logger=logger, format_utils=format_utils)
    ti_factory = TechnicalIndicatorsFactory()
    technical_calculator = TechnicalCalculator(
        logger=logger, format_utils=format_utils, ti_factory=ti_factory
    )
    pattern_analyzer = PatternAnalyzer(logger=logger)

    # Connect to exchange
    print("Connecting to exchange...")
    exchange_manager = ExchangeManager(logger, config)
    await exchange_manager.initialize()
    
    exchange, exchange_name = await exchange_manager.find_symbol_exchange(pair)
    if exchange is None:
        print(f"ERROR: Could not find exchange supporting {pair}")
        await exchange_manager.shutdown()
        return 1
    
    print(f"Using exchange: {exchange_name}")

    # Fetch real data
    print("Fetching OHLCV data...")
    data_fetcher = DataFetcher(exchange, logger)
    result = await data_fetcher.fetch_candlestick_data(pair, timeframe, 999)
    
    if result is None:
        print(f"ERROR: Failed to fetch data for {pair}")
        await exchange_manager.shutdown()
        return 1
    
    ohlcv, current_price = result
    if not isinstance(ohlcv, np.ndarray):
        ohlcv = np.array(ohlcv)
    
    print(f"Fetched {len(ohlcv)} closed candles, current price: ${current_price:.2f}")

    # Calculate indicators
    print("Calculating technical indicators...")
    indicators = technical_calculator.get_indicators(ohlcv)
    print(f"Calculated {len(indicators)} indicators")
    
    # Detect patterns
    print("Detecting patterns...")
    timestamps = timestamps_from_ms_array(ohlcv[:, 0])
    technical_history = {k: v for k, v in indicators.items() if isinstance(v, np.ndarray)}
    patterns = pattern_analyzer.detect_patterns(
        ohlcv_data=ohlcv,
        technical_history=technical_history,
        timestamps=timestamps
    )
    total_patterns = sum(len(p) for p in patterns.values())
    print(f"Detected {total_patterns} patterns")

    # Extract ground truth values for test cases
    rsi = indicators.get('rsi', np.array([]))[-1] if len(indicators.get('rsi', [])) > 0 else 50
    adx = indicators.get('adx', np.array([]))[-1] if len(indicators.get('adx', [])) > 0 else 20
    plus_di = indicators.get('plus_di', np.array([]))[-1] if len(indicators.get('plus_di', [])) > 0 else 20
    minus_di = indicators.get('minus_di', np.array([]))[-1] if len(indicators.get('minus_di', [])) > 0 else 20
    macd_line = indicators.get('macd_line', np.array([]))[-1] if len(indicators.get('macd_line', [])) > 0 else 0
    macd_signal = indicators.get('macd_signal', np.array([]))[-1] if len(indicators.get('macd_signal', [])) > 0 else 0
    sma_50 = indicators.get('sma_50', np.array([]))[-1] if len(indicators.get('sma_50', [])) > 0 else current_price
    sma_200 = indicators.get('sma_200', np.array([]))[-1] if len(indicators.get('sma_200', [])) > 0 else current_price

    # Define test cases based on ground truth
    test_cases = [
        ValidationCase(
            name="RSI Zone Recognition",
            description="AI correctly identifies RSI zone",
            expected_keywords=[
                "overbought" if rsi > 70 else ("oversold" if rsi < 30 else "neutral"),
                str(int(round(rsi)))
            ],
            ground_truth=f"RSI={rsi:.2f} ({'overbought' if rsi > 70 else ('oversold' if rsi < 30 else 'neutral')})"
        ),
        ValidationCase(
            name="MACD Direction",
            description="AI correctly identifies MACD direction",
            expected_keywords=[
                "bullish" if macd_line > macd_signal else "bearish",
                "above" if macd_line > macd_signal else "below"
            ],
            ground_truth=f"MACD {'bullish' if macd_line > macd_signal else 'bearish'} (line {'>' if macd_line > macd_signal else '<'} signal)"
        ),
        ValidationCase(
            name="ADX Trend Strength",
            description="AI correctly classifies trend strength",
            expected_keywords=[
                "strong" if adx > 25 else ("developing" if adx > 20 else "weak"),
                str(int(round(adx)))
            ],
            ground_truth=f"ADX={adx:.2f} ({'strong' if adx > 25 else ('developing' if adx > 20 else 'weak')})"
        ),
        ValidationCase(
            name="DI Direction",
            description="AI correctly identifies DI dominance",
            expected_keywords=[
                "bullish" if plus_di > minus_di else "bearish",
                "+DI" if plus_di > minus_di else "-DI"
            ],
            ground_truth=f"{'+DI' if plus_di > minus_di else '-DI'} dominant ({'bullish' if plus_di > minus_di else 'bearish'})"
        ),
        ValidationCase(
            name="MA Cross Recognition",
            description="AI correctly identifies Golden/Death Cross",
            expected_keywords=[
                "golden" if sma_50 > sma_200 else "death",
                "bullish" if sma_50 > sma_200 else "bearish"
            ],
            ground_truth=f"{'Golden Cross' if sma_50 > sma_200 else 'Death Cross'} (SMA50 {'>' if sma_50 > sma_200 else '<'} SMA200)"
        ),
        ValidationCase(
            name="Pattern Acknowledgment",
            description="AI acknowledges detected patterns",
            expected_keywords=["MACD", "crossover", "stochastic", "cross", "Golden", "Death", "SMA"],
            ground_truth=f"{total_patterns} patterns detected"
        )
    ]

    print("\n--- GROUND TRUTH VALUES ---")
    for tc in test_cases:
        print(f"  {tc.name}: {tc.ground_truth}")
    print("---------------------------\n")

    # Create prompt
    prompt = create_indicator_comprehension_prompt(
        pair, timeframe, current_price, indicators, patterns, format_utils
    )
    system_prompt = f"""You are a professional crypto analyst. 
Analyze the provided technical indicators for {pair}.
Be specific with indicator values and their interpretations.
Follow the response format exactly as requested."""

    # Generate chart if enabled
    chart_buffer = None
    if not args.no_chart:
        print("Generating chart image...")
        chart_generator = ChartGenerator(
            logger=logger, 
            config=config, 
            formatter=format_utils.fmt, 
            format_utils=format_utils
        )
        chart_result = chart_generator.create_chart_image(
            ohlcv=ohlcv,
            technical_history=indicators,
            pair_symbol=pair,
            timeframe=timeframe,
            save_to_disk=True
        )
        if isinstance(chart_result, str):
            print(f"Chart saved to: {chart_result}")
            with open(chart_result, 'rb') as f:
                chart_buffer = io.BytesIO(f.read())
                chart_buffer.seek(0)

    # Initialize model manager
    print("\nInitializing ModelManager...")
    manager = ModelManager(logger=logger, config=config, unified_parser=unified_parser)
    
    if manager._clients.google is None:
        print("ERROR: Google AI client not initialized!")
        await exchange_manager.shutdown()
        return 1
    
    manager.provider = 'googleai'

    print("\n--- SENDING REQUEST TO GOOGLE AI ---\n")
    
    try:
        if chart_buffer and not args.no_chart:
            response = await manager.send_prompt_with_chart_analysis(
                prompt,
                chart_buffer,
                system_message=system_prompt,
                provider='googleai'
            )
        else:
            response = await manager.send_prompt(
                prompt,
                system_message=system_prompt,
                provider='googleai'
            )
        
        print("\n--- AI RESPONSE ---")
        print(response)
        print("\n--- END AI RESPONSE ---\n")
        
        # Validate response
        print(f"{'='*70}")
        print("TEST RESULTS")
        print(f"{'='*70}\n")
        
        passed, failed = validate_response(response, test_cases)
        
        print(f"\n{'='*70}")
        print(f"SUMMARY: {passed}/{len(test_cases)} tests passed")
        if failed == 0:
            print("✅ ALL TESTS PASSED - AI correctly comprehends indicators")
        else:
            print(f"⚠️  {failed} test(s) failed - review AI response for issues")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"ERROR during API call: {e}")
        import traceback
        traceback.print_exc()
        await exchange_manager.shutdown()
        return 1

    # Cleanup
    await exchange_manager.shutdown()
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
