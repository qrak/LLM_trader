"""
Vector Brain & Prompt Integration Simulation Test

This script verifies that the Vector RAG Trading Brain correctly:
1. Stores trade experiences in a local vector database.
2. Retrieves relevant experiences based on market context.
3. Injects these experiences into the System Prompt via PromptBuilder.

Usage:
    python tests/run_brain_simulation.py
"""
import sys
import shutil
import time
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import asyncio
from src.logger.logger import Logger
from src.config.loader import config
from src.trading.vector_memory import VectorMemoryService
from src.trading.brain import TradingBrainService
from src.trading.dataclasses import TradingBrain
from src.analyzer.prompts.prompt_builder import PromptBuilder
from src.analyzer.technical_calculator import TechnicalCalculator
from src.factories.technical_indicators_factory import TechnicalIndicatorsFactory
from src.utils.format_utils import FormatUtils
from src.analyzer.data_processor import DataProcessor
from src.analyzer.formatters import (
    MarketOverviewFormatter,
    LongTermFormatter,
    MarketFormatter,
    MarketPeriodFormatter,
    TechnicalFormatter
)


TEST_DB_DIR = f"data/brain_vector_db_test_{int(time.time())}"

def setup_components(logger: Logger):
    """Initialize all required components for the test."""
    # 1. Utilities
    data_processor = DataProcessor()
    format_utils = FormatUtils(data_processor)
    
    # 2. Technical Calculator (Required by PromptBuilder)
    ti_factory = TechnicalIndicatorsFactory()
    technical_calculator = TechnicalCalculator(logger, format_utils, ti_factory)
    
    # 3. Formatters
    overview_formatter = MarketOverviewFormatter(logger, format_utils)
    long_term_formatter = LongTermFormatter(logger, format_utils)
    market_formatter = MarketFormatter(logger, format_utils)
    period_formatter = MarketPeriodFormatter(logger, format_utils)
    technical_formatter = TechnicalFormatter(technical_calculator, logger, format_utils)
    
    # 4. PromptBuilder
    prompt_builder = PromptBuilder(
        timeframe="1h",
        logger=logger,
        technical_calculator=technical_calculator,
        config=config,
        format_utils=format_utils,
        data_processor=data_processor,
        overview_formatter=overview_formatter,
        long_term_formatter=long_term_formatter,
        market_formatter=market_formatter,
        period_formatter=period_formatter,
        technical_formatter=technical_formatter
    )
    
    # 5. Vector Brain
    # Create a fresh test DB
    if Path(TEST_DB_DIR).exists():
        shutil.rmtree(TEST_DB_DIR)
        
    vector_memory = VectorMemoryService(logger, data_dir=TEST_DB_DIR)
    
    # We need a dummy persistence object for BrainService, or mock it
    class MockPersistence:
        data_dir = Path("data")  # Required by BrainService if it tries to init default memory
        def load_brain(self): return TradingBrain()
        def save_brain(self, brain): pass
        
    brain_service = TradingBrainService(
        logger=logger,
        persistence=MockPersistence(),
        vector_memory=vector_memory
    )
    
    return prompt_builder, brain_service, vector_memory

def seed_brain_memory(vector_memory: VectorMemoryService):
    """Seed the vector database with specific 'lessons'."""
    print("\n[SETUP] Seeding Vector Database with 3 experiences...")
    
    # Experience 1: High ADX Win
    vector_memory.store_experience(
        trade_id="trade_sim_001",
        market_context="Strong Uptrend. ADX: 42 (Very Strong). RSI: 75 (Overbought).",
        outcome="WIN",
        pnl_pct=5.5,
        direction="LONG",
        confidence="HIGH",
        reasoning="Trend was too strong to fade. RSI overbought sustained for days. Breakout follow-through."
    )
    
    # Experience 2: Low Volatility Loss
    vector_memory.store_experience(
        trade_id="trade_sim_002",
        market_context="Ranging Market. ADX: 12 (Weak). Bollinger Bands: Tight Squeeze.",
        outcome="LOSS",
        pnl_pct=-1.2,
        direction="LONG",
        confidence="MEDIUM",
        reasoning="Fakeout breakout in low volatility. Choppiness index was high (65). Got stopped out by wick."
    )
    
    # Experience 3: Divergence Win
    vector_memory.store_experience(
        trade_id="trade_sim_003",
        market_context="Downtrend Weakening. RSI Bullish Divergence. MACD Cross.",
        outcome="WIN",
        pnl_pct=3.1,
        direction="LONG",
        confidence="HIGH",
        reasoning="Caught the reversal bottom. RSI divergence confirmed by volume spike."
    )
    print("[SETUP] Seeding complete.")

async def run_simulation():
    logger = Logger("BrainTest", logger_debug=True)
    
    print("="*60)
    print("VECTOR BRAIN INTEGRATION TEST")
    print("="*60)
    
    try:
        # 1. Setup
        pb, brain, vector_mem = setup_components(logger)
        
        # 2. Seed Data
        seed_brain_memory(vector_mem)
        
        # 3. Simulate Run: High ADX Scenario
        # We want to trigger retrieval of Experience #1 (High ADX)
        print("\n[TEST] Simulating 'High ADX Uptrend' Context query...")
        
        # Use get_context to generate the string
        brain_context_str = brain.get_context(
            trend_direction="BULLISH",
            adx=38.5,             # Similar to stored ADX 42
            volatility_level="HIGH",
            rsi_level="OVERBOUGHT",
            macd_signal="BULLISH"
        )
        
        print(f"\n[RESULT] Brain Context String Retrieved ({len(brain_context_str)} chars):")
        print("-" * 40)
        print(brain_context_str)
        print("-" * 40)
        
        # 4. Verify Content
        if "RSI overbought sustained" in brain_context_str:
            print("\n[SUCCESS] Vector Brain successfully retrieved the relevant 'High ADX' experience!")
        else:
            print("\n[FAILURE] Expected experience text not found in retrieval.")
            
        # 5. Build System Prompt (Integration Check)
        print("\n[TEST] Building System Prompt with injected context...")
        
        system_prompt = pb.build_system_prompt(
            symbol="BTC/USDC",
            brain_context=brain_context_str,
            last_analysis_time="2026-01-04 12:00:00"
        )
        
        # 6. Check System Prompt
        if brain_context_str in system_prompt:
             print("\n[SUCCESS] Brain context was correctly injected into the System Prompt.")
        else:
             print("\n[FAILURE] Brain context missing from System Prompt.")
             
        # Optional: Print snippet
        start_idx = system_prompt.find("RELEVANT PAST EXPERIENCES")
        if start_idx != -1:
            print("\n--- System Prompt Snippet ---")
            print(system_prompt[start_idx:start_idx+500] + "...")
            
    finally:
        # Cleanup
        print("\n[CLEANUP] Removing test database...")
        if Path(TEST_DB_DIR).exists():
            # Wait a bit to ensure file handles are released by Chroma
            time.sleep(1)
            try:
                shutil.rmtree(TEST_DB_DIR)
                print("[CLEANUP] Done.")
            except Exception as e:
                print(f"[CLEANUP] Warning: Could not delete test DB: {e}")

if __name__ == "__main__":
    asyncio.run(run_simulation())
