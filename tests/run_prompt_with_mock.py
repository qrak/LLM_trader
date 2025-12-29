"""Standalone script to build a real prompt using cached data and run it against the Mock AI client.

Usage:
    python tests/run_prompt_with_mock.py [--scenario=crash|bullish|neutral|real]

This will:
- Build an AnalysisContext with synthetic OHLCV seeded from cached Coingecko data
- Optionally inject synthetic news articles for testing wtpsplit/RAG processing
- Use PromptBuilder to create the system + user prompt
- Inject a MockClient into ModelManager and run the full prompt flow
- Print the response (analysis + JSON)

Scenarios:
- crash: Simulates a major market crash (SEC lawsuit, exchange hack)
- bullish: Simulates bullish news (ETF approval, whale accumulation)
- neutral: Simulates neutral/consolidation news
- real: Uses actual cached news from recent_news.json (default)
"""
import sys
import argparse
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
from src.config.loader import config
from src.analyzer.prompts.prompt_builder import PromptBuilder
from src.analyzer.analysis_context import AnalysisContext
from src.analyzer.data_processor import DataProcessor
from src.utils.format_utils import FormatUtils
from src.platforms.ai_providers.mock import MockClient
from src.contracts.manager import ModelManager
from src.parsing.unified_parser import UnifiedParser
from src.analyzer.technical_calculator import TechnicalCalculator
from src.factories.technical_indicators_factory import TechnicalIndicatorsFactory


# ============================================================================
# SYNTHETIC NEWS SCENARIOS FOR TESTING
# ============================================================================
# These scenarios test how wtpsplit segments text and how the RAG engine
# selects relevant sentences containing key data points.

SYNTHETIC_NEWS_SCENARIOS = {
    "crash": [
        {
            "id": "synthetic_crash_001",
            "title": "BREAKING: SEC Files Lawsuit Against Major Crypto Exchange",
            "body": """The Securities and Exchange Commission has filed an emergency lawsuit against CryptoGlobal Exchange, 
alleging securities violations worth $2.3 billion. Bitcoin dropped 15% within 30 minutes of the announcement, 
falling from $95,000 to $80,750. Trading volumes spiked to $45 billion across major exchanges.

The SEC claims the exchange listed 68 unregistered securities and operated without proper licensing since 2022.
CryptoGlobal's CEO denied all allegations, stating they will "vigorously defend" their position.

Market analysts expect continued volatility, with support levels at $78,000 and $72,500.
The Fear & Greed Index plummeted from 55 to 12 (Extreme Fear) following the news.

Institutional outflows reached $890 million in the past 24 hours according to CoinShares data.
Short positions on Binance jumped to 68% from 52% before the announcement.""",
            "categories": "BTC|REGULATION|MARKET|TRADING|CRYPTOCURRENCY",
            "tags": "SEC|Bitcoin|Crash|Regulation|Exchange",
            "source": "synthetic_test",
            "published_on": int(time.time()) - 3600,  # 1 hour ago
        },
        {
            "id": "synthetic_crash_002",
            "title": "Major Exchange Reports $400M Hot Wallet Hack",
            "body": """CentralCrypto Exchange confirmed a security breach resulting in the theft of approximately 
$400 million in various cryptocurrencies. The stolen funds include 4,500 BTC ($382M) and 50,000 ETH ($148M).

Bitcoin's price fell an additional 8% on the news, testing the $74,200 support level.
The exchange has halted all withdrawals and deposits pending investigation.

The BTC perpetual futures long/short ratio shifted dramatically:
Overall: 32.5% long, 67.5% short
Binance: 31.2% long, 68.8% short
OKX: 33.8% long, 66.2% short

This marks the largest exchange hack since the 2022 incidents.
On-chain analysts traced funds to multiple mixer protocols.""",
            "categories": "BTC|ETH|EXCHANGE|SECURITY|CRYPTOCURRENCY",
            "tags": "Hack|Security|Bitcoin|Ethereum|Exchange",
            "source": "synthetic_test",
            "published_on": int(time.time()) - 1800,  # 30 min ago
        },
    ],
    
    "bullish": [
        {
            "id": "synthetic_bull_001",
            "title": "SEC Approves First Spot Bitcoin ETF with Record $10B Day-One Inflows",
            "body": """In a landmark decision, the SEC has approved the first spot Bitcoin ETF, 
with BlackRock's iShares Bitcoin Trust (IBIT) seeing record $10.2 billion in day-one trading volume.

Bitcoin surged 22% on the news, breaking above $105,000 for the first time.
The approval marks a major milestone for institutional cryptocurrency adoption.

Key metrics following the approval:
- BTC Price: $105,450 (+22.3% in 24h)
- ETF Inflows: $10.2 billion on day one
- Futures Open Interest: Up 45% to $28 billion
- Long/Short Ratio: 72.5% long, 27.5% short

Michael Saylor's Strategy announced an additional $2 billion BTC purchase.
Fidelity and VanEck ETFs are expected to receive approval within 30 days.

The Fear & Greed Index jumped to 85 (Extreme Greed) from 62.""",
            "categories": "BTC|ETF|REGULATION|MARKET|CRYPTOCURRENCY",
            "tags": "ETF|SEC|Bitcoin|Institutional|BlackRock",
            "source": "synthetic_test",
            "published_on": int(time.time()) - 3600,
        },
        {
            "id": "synthetic_bull_002",
            "title": "Whale Alert: Unknown Entity Accumulates 45,000 BTC Worth $4.7 Billion",
            "body": """On-chain data reveals a massive accumulation event over the past 72 hours.
An unknown entity has purchased approximately 45,000 BTC worth $4.7 billion at an average price of $104,500.

The whale's wallet now holds 78,500 BTC total, making it the 4th largest non-exchange wallet.
Analysts speculate this could be a sovereign wealth fund or major tech corporation.

Market impact:
- Exchange reserves dropped to 2.1 million BTC (lowest since 2018)
- Spot buying pressure outpaced selling 3.2:1
- Funding rates remain positive at 0.025% (8h)

Bitcoin is currently trading at $107,200, up 3.1% since the accumulation was first detected.
Resistance levels at $110,000 and $115,000 are being closely watched by traders.""",
            "categories": "BTC|WHALE|MARKET|TRADING|CRYPTOCURRENCY",
            "tags": "Whale|Accumulation|Bitcoin|On-chain|Bullish",
            "source": "synthetic_test",
            "published_on": int(time.time()) - 7200,
        },
    ],
    
    "neutral": [
        {
            "id": "synthetic_neutral_001",
            "title": "Bitcoin Consolidates Near $87,000 as Traders Await Fed Decision",
            "body": """Bitcoin continues to trade in a tight range between $85,500 and $88,200 as 
market participants await the Federal Reserve's interest rate decision scheduled for Wednesday.

Current market metrics show a balanced sentiment:
- BTC Price: $87,150 (unchanged in 24h)
- 24h Trading Volume: $12.4 billion (below 30-day average)
- Long/Short Ratio: 50.2% long, 49.8% short
- Funding Rates: 0.001% (neutral)

Technical analysis suggests a breakout is imminent, though direction remains unclear.
Key resistance sits at $89,500 while support is established at $84,000.

The Fear & Greed Index reads 45 (Neutral), reflecting market indecision.
Options data shows equal call/put interest at the $90,000 strike for month-end expiry.""",
            "categories": "BTC|MARKET|TRADING|MACROECONOMICS|CRYPTOCURRENCY",
            "tags": "Bitcoin|Consolidation|Fed|Trading|Neutral",
            "source": "synthetic_test",
            "published_on": int(time.time()) - 5400,
        },
    ],
}


def get_synthetic_news(scenario: str) -> list:
    """Get synthetic news articles for the specified scenario."""
    if scenario not in SYNTHETIC_NEWS_SCENARIOS:
        print(f"Warning: Unknown scenario '{scenario}', using 'neutral'")
        scenario = "neutral"
    return SYNTHETIC_NEWS_SCENARIOS[scenario]


def load_real_news() -> list:
    """Load actual news from the cache file."""
    news_path = ROOT / "data" / "news_cache" / "recent_news.json"
    try:
        with open(news_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('articles', [])[:10]  # Limit to 10 articles
    except Exception as e:
        print(f"Warning: Could not load real news: {e}")
        return []


def build_news_context(articles: list, logger: Logger) -> str:
    """Build a simple news context string from articles for prompt injection.
    
    This bypasses the full RAG engine to directly test how news appears in prompts.
    For full RAG testing, use the ContextBuilder with wtpsplit.
    """
    if not articles:
        return ""
    
    context_parts = ["## Recent News Context\n"]
    
    for i, article in enumerate(articles[:5], 1):
        title = article.get('title', 'No Title')
        body = article.get('body', '')[:500]  # Truncate for display
        source = article.get('source', 'unknown')
        categories = article.get('categories', '')
        
        context_parts.append(f"### Article {i}: {title}")
        context_parts.append(f"**Source:** {source} | **Topics:** {categories}")
        context_parts.append(f"{body}...\n")
    
    return "\n".join(context_parts)


async def test_wtpsplit_processing(articles: list, logger: Logger):
    """Test wtpsplit sentence splitting on synthetic articles.
    
    This function directly tests the SentenceSplitter to verify
    proper segmentation of news content.
    """
    from src.rag.text_splitting import SentenceSplitter
    from src.rag.context_builder import ContextBuilder
    from src.utils.token_counter import TokenCounter
    from src.rag.article_processor import ArticleProcessor
    from src.parsing.unified_parser import UnifiedParser
    from src.utils.format_utils import FormatUtils
    from src.analyzer.data_processor import DataProcessor
    
    print("\n" + "="*60)
    print("WTPSPLIT SENTENCE SPLITTING TEST")
    print("="*60)
    
    splitter = SentenceSplitter(logger)
    token_counter = TokenCounter()
    data_processor = DataProcessor()
    format_utils = FormatUtils(data_processor)
    unified_parser = UnifiedParser(logger=logger, format_utils=format_utils)
    
    article_processor = ArticleProcessor(
        logger=logger,
        format_utils=format_utils,
        sentence_splitter=splitter,
        unified_parser=unified_parser
    )
    
    context_builder = ContextBuilder(
        logger=logger,
        token_counter=token_counter,
        article_processor=article_processor,
        sentence_splitter=splitter
    )
    
    for article in articles[:2]:  # Test first 2 articles
        title = article.get('title', 'No Title')
        body = article.get('body', '')
        
        print(f"\n--- Article: {title[:50]}... ---")
        
        # Split into sentences
        sentences = splitter.split_text(body.replace('\n', ' ').strip())
        
        print(f"Total sentences found: {len(sentences)}")
        print("\nTop sentences by scoring:")
        
        # Test sentence selection with keywords
        keywords = {'btc', 'bitcoin', 'price', 'long', 'short', 'ratio', '%'}
        selected = context_builder._select_relevant_sentences(sentences, keywords, max_count=5)
        
        for i, sent in enumerate(selected, 1):
            # Highlight key data in sentence
            has_percent = '%' in sent
            has_dollar = '$' in sent
            has_number = any(c.isdigit() for c in sent)
            indicators = []
            if has_percent: indicators.append("%")
            if has_dollar: indicators.append("$")
            if has_number: indicators.append("#")
            
            indicator_str = f"[{','.join(indicators)}]" if indicators else "[text]"
            print(f"  {i}. {indicator_str} {sent[:100]}...")
    
    print("\n" + "="*60)


async def main():
    parser = argparse.ArgumentParser(description='Run prompt with mock AI and optional synthetic news')
    parser.add_argument('--scenario', type=str, default='real',
                       choices=['crash', 'bullish', 'neutral', 'real'],
                       help='News scenario to inject (default: real)')
    parser.add_argument('--test-wtpsplit', action='store_true',
                       help='Run detailed wtpsplit sentence splitting test')
    args = parser.parse_args()
    
    logger = Logger("run_prompt_mock", logger_debug=True)
    data_processor = DataProcessor()
    format_utils = FormatUtils(data_processor)
    unified_parser = UnifiedParser(logger=logger, format_utils=format_utils)

    # TechnicalCalculator is required by PromptBuilder
    ti_factory = TechnicalIndicatorsFactory()
    technical_calculator = TechnicalCalculator(logger=logger, format_utils=format_utils, ti_factory=ti_factory)

    pb = PromptBuilder(
        timeframe=config.TIMEFRAME, 
        logger=logger, 
        technical_calculator=technical_calculator,
        format_utils=format_utils, 
        config=config, 
        data_processor=data_processor
    )

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

    # Load news based on scenario
    print(f"\n{'='*60}")
    print(f"SCENARIO: {args.scenario.upper()}")
    print(f"{'='*60}")
    
    if args.scenario == 'real':
        news_articles = load_real_news()
        print(f"Loaded {len(news_articles)} articles from cache")
    else:
        news_articles = get_synthetic_news(args.scenario)
        print(f"Using {len(news_articles)} synthetic {args.scenario} articles")
    
    # Test wtpsplit if requested
    if args.test_wtpsplit:
        await test_wtpsplit_processing(news_articles, logger)
    
    # Build news context for prompt
    news_context = build_news_context(news_articles, logger)
    
    prompt = pb.build_prompt(context)
    system_prompt = pb.build_system_prompt('BTC/USDT')
    
    # Inject news context into the prompt
    if news_context:
        prompt = news_context + "\n\n" + prompt

    prepared_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt + f"\n\nTEST_HINT: last_close={context.current_price}, scenario={args.scenario}"}
    ]

    manager = ModelManager(logger=logger, config=config, unified_parser=unified_parser)
    manager.openrouter_client = MockClient(logger=logger)
    # CRITICAL: Must update PROVIDER_METADATA because it caches the client reference from __init__
    if 'openrouter' in manager.PROVIDER_METADATA:
        manager.PROVIDER_METADATA['openrouter']['client'] = manager.openrouter_client
    
    manager.provider = 'openrouter'

    print("\n--- System Prompt (truncated) ---")
    print(system_prompt[:500])
    print("\n--- News Context ---")
    print(news_context[:1000] if news_context else "(no news context)")
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
    
    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    asyncio.run(main())
