"""Standalone script to build a real prompt using cached data and run it against the Mock AI client.

Usage:
    python scripts/run_prompt_with_mock.py

This will:
- Build an AnalysisContext with synthetic OHLCV seeded from cached Coingecko data
- Use PromptBuilder to create the system + user prompt
- Inject a MockClient into ModelManager and run the full prompt flow
- Print the response (analysis + JSON)
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so we can import the package `src` when running this script directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import asyncio
import json
import time
import numpy as np
from PIL import Image
import io

from src.logger.logger import Logger
from src.utils.loader import config
from src.analyzer.prompts.prompt_builder import PromptBuilder
from src.analyzer.core.analysis_context import AnalysisContext
from src.analyzer.data.data_processor import DataProcessor
from src.utils.format_utils import FormatUtils
from src.platforms.ai_providers.mock import MockClient
from src.contracts.manager import ModelManager


async def main():
    logger = Logger("run_prompt_mock", logger_debug=True)
    data_processor = DataProcessor()
    format_utils = FormatUtils(data_processor)

    pb = PromptBuilder(timeframe=config.TIMEFRAME, logger=logger, format_utils=format_utils, config=config, data_processor=data_processor)

    # Build synthetic OHLCV using cached coingecko base price
    N = 240
    now = int(time.time())
    base_price = 100.0
    try:
        import json as _json
        with open('data/market_data/coingecko_global.json', 'r', encoding='utf-8') as fh:
            cg = _json.load(fh)
            top = cg.get('data', {}).get('top_coins', [{}])[0]
            base_price = float(top.get('current_price', 100.0))
    except Exception:
        pass

    rng = np.random.default_rng(seed=123)
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

    context = AnalysisContext('BTC/USDT')
    context.timeframe = config.TIMEFRAME
    context.ohlcv_candles = ohlcv_arr
    context.current_price = float(ohlcv_arr[-1, 4])
    context.market_overview = {'coin_data': {'BTC': {'price': context.current_price}}}

    prompt = pb.build_prompt(context)
    system_prompt = pb.build_system_prompt('BTC/USDT')

    prepared_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt + f"\n\nTEST_HINT: last_close={context.current_price}"}
    ]

    manager = ModelManager(logger=logger, config=config)
    manager.openrouter_client = MockClient(logger=logger)
    manager.provider = 'openrouter'

    print("--- System Prompt ---")
    print(system_prompt[:1000])
    print("\n--- Main Prompt (truncated) ---")
    print(prompt[:1000])

    response = await manager.send_prompt(prompt, prepared_messages=prepared_messages, provider='openrouter')

    print("\n--- Mock Response ---")
    print(response)

    # Chart analysis example
    img = Image.new('RGB', (600, 400), color='white')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    response_with_chart = await manager.send_prompt_with_chart_analysis(prompt, buf, provider='openrouter')
    print('\n--- Mock Response with Chart ---')
    print(response_with_chart)


if __name__ == '__main__':
    asyncio.run(main())
