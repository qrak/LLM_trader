import asyncio
import json
import time
import numpy as np
from PIL import Image
import io
import pytest

from src.logger.logger import Logger
from src.utils.loader import config
from src.analyzer.prompts.prompt_builder import PromptBuilder
from src.analyzer.core.analysis_context import AnalysisContext
from src.analyzer.data.data_processor import DataProcessor
from src.utils.format_utils import FormatUtils
from src.platforms.ai_providers.mock import MockClient
from src.contracts.manager import ModelManager


@pytest.mark.asyncio
async def test_prompt_flow_with_mock_provider():
    # Setup logger and utilities
    logger = Logger("test_prompt", logger_debug=True)
    data_processor = DataProcessor()
    format_utils = FormatUtils(data_processor)

    # Initialize prompt builder
    prompt_builder = PromptBuilder(
        timeframe=config.TIMEFRAME,
        logger=logger,
        format_utils=format_utils,
        config=config,
        data_processor=data_processor
    )

    # Generate synthetic but realistic OHLCV data (timestamp, open, high, low, close, volume)
    N = 240  # 10 days of 1h candles
    now = int(time.time())
    base_price = None

    # Try to get a realistic base price from coingecko cache if available
    try:
        import json as _json
        with open('data/market_data/coingecko_global.json', 'r', encoding='utf-8') as fh:
            cg = _json.load(fh)
            top = cg.get('data', {}).get('top_coins', [{}])[0]
            base_price = float(top.get('current_price', 100.0))
    except Exception:
        base_price = 100.0

    rng = np.random.default_rng(seed=42)
    prices = [base_price]
    for _ in range(N - 1):
        change = rng.normal(0, base_price * 0.002)
        prices.append(max(0.0001, prices[-1] + change))

    ohlcv = []
    for i in range(N):
        ts_ms = int((now - (N - 1 - i) * 3600) * 1000)
        close = float(prices[i])
        open_p = float(prices[i] - rng.normal(0, close * 0.0008))
        high = max(open_p, close) + abs(rng.normal(0, close * 0.001))
        low = min(open_p, close) - abs(rng.normal(0, close * 0.001))
        vol = float(abs(rng.normal(1000, 200)))
        ohlcv.append([ts_ms, open_p, high, low, close, vol])

    ohlcv_arr = np.array(ohlcv)

    # Build analysis context with real-looking data
    context = AnalysisContext('BTC/USDT')
    context.timeframe = config.TIMEFRAME
    context.ohlcv_candles = ohlcv_arr
    context.current_price = float(ohlcv_arr[-1, 4])
    # Minimal market_overview info
    context.market_overview = {
        'coin_data': {'BTC': {'price': context.current_price}}
    }

    # Build system prompt & main prompt
    main_prompt = prompt_builder.build_prompt(context, has_chart_analysis=False)
    system_prompt = prompt_builder.build_system_prompt('BTC/USDT', has_chart_image=False)

    # Append a TEST_HINT to help MockClient create deterministic response
    prepared_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": main_prompt + f"\n\nTEST_HINT: last_close={context.current_price}"}
    ]

    # Create ModelManager and inject MockClient as the OpenRouter-like provider
    manager = ModelManager(logger=logger, config=config)
    manager.openrouter_client = MockClient(logger=logger)
    # CRITICAL: Must update PROVIDER_METADATA because it caches the client reference from __init__
    if 'openrouter' in manager.PROVIDER_METADATA:
        manager.PROVIDER_METADATA['openrouter']['client'] = manager.openrouter_client
    
    manager.provider = 'openrouter'

    # Call the prompt flow (text-only)
    response_text = await manager.send_prompt(main_prompt, prepared_messages=prepared_messages, provider='openrouter')

    assert response_text is not None and isinstance(response_text, str)
    assert '```json' in response_text, "Mock response should include JSON block"

    # Extract JSON from the response and validate structure
    start = response_text.find('```json')
    end = response_text.find('```', start + 6)
    json_block = response_text[start + 7:end].strip()
    payload = json.loads(json_block)

    assert 'analysis' in payload
    analysis = payload['analysis']
    assert 'signal' in analysis
    assert 'confidence' in analysis
    assert 'entry_price' in analysis
    
    # Validate new confluence_factors structure
    assert 'confluence_factors' in analysis, "confluence_factors missing from analysis"
    cf = analysis['confluence_factors']
    assert isinstance(cf, dict), "confluence_factors should be a dict"
    required_factors = ['trend_alignment', 'momentum_strength', 'volume_support', 
                       'pattern_quality', 'support_resistance_strength']
    for factor in required_factors:
        assert factor in cf, f"Missing confluence factor: {factor}"
        assert isinstance(cf[factor], (int, float)), f"{factor} should be numeric"
        assert 0 <= cf[factor] <= 100, f"{factor} should be in range 0-100"

    # Test chart analysis flow as well (image provided)
    # Create a small PNG image
    img = Image.new('RGB', (600, 400), color='white')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    response_with_chart = await manager.send_prompt_with_chart_analysis(main_prompt, buf, provider='openrouter')
    assert response_with_chart is not None and 'analysis' in response_with_chart
    # Verify that the mock now returns chart-specific reasoning
    # This confirms that our fix to MockClient is working and satisfying the user request
    if isinstance(response_with_chart, str) and '```json' in response_with_chart:
        # It returned the raw string with JSON block
        assert "Chart Pattern Analysis" in response_with_chart, "Mock response should mention chart analysis"
    elif isinstance(response_with_chart, dict) and "raw_response" in response_with_chart:
        # It returned the processed dict which contains raw_response
        assert "Chart Pattern Analysis" in response_with_chart["raw_response"], "Mock response should mention chart analysis"

    # Done
    logger.info("Mock prompt flow test completed successfully")
