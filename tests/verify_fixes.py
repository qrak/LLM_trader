import unittest
import numpy as np
import sys
import os
from unittest.mock import MagicMock
from datetime import datetime, timezone

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules to test
from src.analyzer.formatters.technical_formatter import TechnicalFormatter
from src.indicators.volatility.volatility_indicators import keltner_channels_numba
from src.rag.rag_engine import RagEngine
from src.trading.data_models import TradingMemory
from src.trading.memory import TradeDecision # Import TradeDecision from memory.py based on recent changes or dataclasses if acceptable. Check imports.

# TradeDecision is in src.trading.data_models based on previous context
from src.trading.data_models import TradeDecision  # noqa: F811

class TestFixes(unittest.TestCase):

    def test_technical_formatter_no_crypto_data(self):
        """Test that TechnicalFormatter runs without the unused crypto_data variable."""
        mock_calculator = MagicMock()
        mock_logger = MagicMock()
        mock_format_utils = MagicMock()
        
        formatter = TechnicalFormatter(mock_calculator, mock_logger, mock_format_utils)
        
        # Mock context
        mock_context = MagicMock()
        mock_context.technical_data = {
            "rsi": 50, "mac": 0, "upper_band": 100, "lower_band": 90, 
            "middle_band": 95, "atr": 2, "adx": 25, "cci": 0, "sar": 98,
            "obv": 1000, "mfi": 50, "stoch_k": 50, "stoch_d": 50,
            "williams_r": -50
        }
        mock_context.current_price = 100.0
        # Ensure regex patterns in formatter don't fail on None
        mock_context.technical_signals = [] 

        # Mock internal formatting methods to isolate the test
        # We only want to ensure the main method runs and doesn't crash on 'crypto_data'
        try:
             # We can't easily mock private methods on the instance we just created if they are called internally
             # But if they are just formatting helpers, they might fail if we don't provide enough data
             # Let's mock the ones that might use crypto_data if it were there, or just run it.
             # The error was "NameError: name 'crypto_data' is not defined" if we removed it but kept usage.
             # If we removed both definition AND usage, it should run fine.
             
             # To make it run smooth, let's mock the sub-formatters
             formatter._format_patterns_section = MagicMock(return_value="Patterns: None")
             formatter.format_price_action_section = MagicMock(return_value="Price Action: Neutral")
             formatter._format_trend_analysis = MagicMock(return_value="Trend: Neutral")
             formatter._format_momentum_oscillators = MagicMock(return_value="Momentum: Neutral")
             formatter._format_volatility_indicators = MagicMock(return_value="Volatility: Normal")
             formatter._format_volume_analysis = MagicMock(return_value="Volume: Normal")
             formatter._format_support_resistance = MagicMock(return_value="S/R: None")

             formatter.format_technical_analysis(mock_context, "1h")
             
        except Exception as e:
            self.fail(f"TechnicalFormatter execution failed: {e}")

    def test_keltner_channels_mamode(self):
        """Test that keltner_channels_numba produces different results for different mamodes."""
        # Create deterministic data
        np.random.seed(42)
        high = np.random.random(100) * 10 + 100
        low = high - np.random.random(100) * 2
        close = (high + low) / 2
        
        # Calculate with EMA (default)
        upper_ema, mid_ema, lower_ema = keltner_channels_numba(high, low, close, length=20, multiplier=2.0, mamode='ema')
        
        # Calculate with SMA
        upper_sma, mid_sma, lower_sma = keltner_channels_numba(high, low, close, length=20, multiplier=2.0, mamode='sma')
        
        # Results should differ in the middle band (EMA vs SMA)
        # We check from index 20 onwards to avoid initial nan/warmup differences
        self.assertFalse(np.allclose(mid_ema[30:], mid_sma[30:]), "Middle band should differ between EMA and SMA")
        
    def test_rag_engine_logs_tokens(self):
        """Test that RagEngine logs the total tokens."""
        import asyncio
        
        mock_logger = MagicMock()
        mock_config = MagicMock()
        mock_config.RAG_UPDATE_INTERVAL_HOURS = 1
        mock_token_counter = MagicMock()
        
        engine = RagEngine(mock_logger, mock_token_counter, mock_config)
        
        # Mock dependencies
        engine.context_builder = MagicMock()
        engine.news_manager = MagicMock()
        engine.news_manager.news_database = []
        engine.index_manager = MagicMock()
        engine.category_processor = MagicMock()
        engine.category_processor.extract_base_coin.return_value = "BTC"
        
        # Mock async methods on engine
        future_false = asyncio.Future()
        future_false.set_result(False)
        engine._ensure_categories_updated = MagicMock(return_value=future_false)
        
        future_none = asyncio.Future()
        future_none.set_result(None)
        engine.update_if_needed = MagicMock(return_value=future_none)
        
        # Mock async method on context_builder
        future_scores = asyncio.Future()
        future_scores.set_result([]) # Return empty list for scores
        engine.context_builder.keyword_search = MagicMock(return_value=future_scores)
        
        # Setup return for context builder synchronous method
        engine.context_builder.add_articles_to_context.return_value = ("Context content", 1234)
        
        # Run async method
        asyncio.run(engine.retrieve_context("query", "BTC/USD"))
        
        # Verify log call
        mock_logger.debug.assert_called()
        # Check if any of the calls contain "1234"
        found = False
        for call in mock_logger.debug.call_args_list:
            if "1234" in str(call):
                found = True
                break
        self.assertTrue(found, "RagEngine did not log the token count (1234)")

    def test_trading_memory_context_summary(self):
        """Test that get_context_summary includes open position P&L using current_price."""
        mock_logger = MagicMock()
        memory = TradingMemory(mock_logger)
        
        # Setup an open position in history
        # Scenario: We bought at 50,000, now price is 51,000.
        # There was a BUY decision but NO matching CLOSE decision in the list.
        trade = TradeDecision(
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USD",
            action="BUY",
            confidence="HIGH",
            price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
            quantity=1.0, 
            reasoning="Test Buy"
        )
        memory.decisions = [trade]
        
        # Execute
        summary = memory.get_context_summary(current_price=51000.0)
        
        # Verify
        self.assertIn("Active History Position (Unrealized)", summary)
        self.assertIn("Current P&L: +2.00%", summary)

if __name__ == '__main__':
    unittest.main()
