"""Test semantic search and reasoning flow for correctness."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger.logger import Logger
from src.trading.vector_memory import VectorMemoryService
from src.trading.brain import TradingBrainService
from src.trading.persistence import TradingPersistence

logger = Logger(logger_name="Test", logger_debug=False)
persistence = TradingPersistence(logger, data_dir="data/trading")
brain = TradingBrainService(logger, persistence, symbol="BTC/USDC", timeframe="4h")

print("=" * 80)
print("TEST 1: Semantic Search - Similar Experience Retrieval")
print("=" * 80)

# Test query similar to current market conditions
test_context = "BULLISH + High ADX + MEDIUM Volatility"
experiences = brain.vector_memory.retrieve_similar_experiences(
    test_context, k=3, where={"outcome": {"$ne": "UPDATE"}}
)

if experiences:
    print(f"[OK] Found {len(experiences)} similar experiences")
    for i, exp in enumerate(experiences):
        meta = exp["metadata"]
        print(f"  {i+1}. [{exp['similarity']:.0f}% match] {meta['outcome']} {meta['pnl_pct']:+.2f}%")
        reasoning = meta.get("reasoning", "N/A")
        if reasoning and reasoning != "N/A":
            print(f"     Reasoning: {reasoning[:80]}...")
        else:
            print("     [FAIL] BUG: Reasoning is N/A!")
else:
    print("[FAIL] No experiences found")

print("\n" + "=" * 80)
print("TEST 2: Brain Context Generation")
print("=" * 80)

context = brain.get_context(
    trend_direction="BULLISH",
    adx=30,
    volatility_level="MEDIUM",
    rsi_level="NEUTRAL",
    macd_signal="BULLISH"
)

if context:
    print(f"[OK] Brain context generated ({len(context)} chars)")
    # Check for N/A in the context
    if "N/A" in context and "Key Insight: \"N/A\"" in context:
        print("[FAIL] BUG: Context contains N/A reasoning!")
    else:
        print("[OK] No N/A reasoning in context")
    
    # Check for key insight presence
    if "Key Insight:" in context:
        print("[OK] Key Insights are being injected into prompts")
    else:
        print("[WARN] No Key Insights in context (may be expected if no similar trades)")
    
    print("\n--- Sample Context Output ---")
    print(context[:800] + "..." if len(context) > 800 else context)
else:
    print("[FAIL] No context generated")

print("\n" + "=" * 80)
print("TEST 3: get_context_for_prompt direct check")
print("=" * 80)

prompt_context = brain.vector_memory.get_context_for_prompt("Uptrend + Weak Trend", k=3)
if prompt_context:
    print(f"[OK] Prompt context generated ({len(prompt_context)} chars)")
    if "Key Insight: \"N/A\"" in prompt_context:
        print("[FAIL] BUG: Found N/A reasoning in prompt context!")
    elif "Key Insight:" in prompt_context:
        print("[OK] Key Insights present with real reasoning")
    print("\n--- Prompt Context ---")
    print(prompt_context)
else:
    print("[FAIL] No prompt context generated")

print("\n" + "=" * 80)
print("ALL TESTS COMPLETE")
print("=" * 80)
