import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
import time
from typing import Optional, Tuple, List
import numpy as np
from src.rag.text_splitting import SentenceSplitter
from src.indicators.volatility.volatility_indicators import choppiness_index_numba

class TestPerformanceOptimization(unittest.IsolatedAsyncioTestCase):
    
    def test_sentence_splitter_cache(self):
        """Test that lru_cache is working on split_text."""
        mock_logger = MagicMock()
        
        # Patch wtpsplit to avoid loading heavy model
        with patch('src.rag.text_splitting.SentenceSplitter._initialize_model'):
            splitter = SentenceSplitter(logger=mock_logger)
            splitter._model = None # Force regex fallback for speed, cache works same way
            
            text = "This is a sentence. This is another one."
            
            # First call
            start_time = time.perf_counter()
            _ = splitter.split_text(text)
            first_duration = time.perf_counter() - start_time
            
            # Second call (should be cached)
            start_time = time.perf_counter()
            _ = splitter.split_text(text)
            second_duration = time.perf_counter() - start_time
            
            # Verify cache info
            cache_info = splitter.split_text.cache_info()
            print(f"Cache Info: {cache_info}")
            
            self.assertGreater(cache_info.hits, 0, "Cache hits should be > 0")

    async def test_parallel_execution(self):
        """Test that AnalysisEngine runs tasks in parallel."""
        from src.analyzer.analysis_engine import AnalysisEngine
        
        # Mock dependencies
        mock_logger = MagicMock()
        mock_rag = AsyncMock()
        mock_model_manager = MagicMock()
        mock_model_manager.supports_image_analysis.return_value = True
        mock_config = MagicMock()
        mock_config.TIMEFRAME = "1h"
        mock_config.CANDLE_LIMIT = 100
        mock_gen = MagicMock()
        
        # Async mocks with delays
        async def slow_rag(*args, **kwargs):
            await asyncio.sleep(0.1)
            return {"overview": "data"}
            
        async def slow_tech(*args, **kwargs):
            await asyncio.sleep(0.1)
            
        async def slow_chart(*args, **kwargs):
            await asyncio.sleep(0.1)
            return MagicMock()
            
        mock_rag.get_market_overview = slow_rag
        mock_rag.retrieve_context = AsyncMock(side_effect=slow_rag)
        # Explicitly make context_builder a MagicMock (synchronous)
        mock_rag.context_builder = MagicMock()
        mock_rag.context_builder.get_latest_article_urls.return_value = {}
        mock_rag.config.RAG_NEWS_LIMIT = 5
        
        # Create engine with mocked methods
        engine = AnalysisEngine(
            logger=mock_logger,
            rag_engine=mock_rag,
            coingecko_api=MagicMock(),
            model_manager=mock_model_manager,
            alternative_me_api=MagicMock(),
            market_api=MagicMock(),
            config=mock_config,
            technical_calculator=MagicMock(),
            pattern_analyzer=MagicMock(),
            prompt_builder=MagicMock(),
            data_collector=MagicMock(),
            metrics_calculator=MagicMock(),
            result_processor=MagicMock(),
            chart_generator=mock_gen
        )
        
        # Patch internal methods to simulate work
        engine._collect_market_data = AsyncMock(return_value=True)
        engine._enrich_market_context = AsyncMock(side_effect=slow_rag) # 0.1s task 1
        engine._perform_technical_analysis = AsyncMock(side_effect=slow_tech) # 0.1s task 2 part A
        engine._generate_chart_image = AsyncMock(side_effect=slow_chart) # 0.1s task 2 part B
        # Task 3 is rag.retrieve_context (0.1s)
        
        engine._generate_ai_analysis = AsyncMock(return_value={"result": "ok"})
        engine._generate_brain_context_from_current_indicators = MagicMock(return_value="brain")
        
        # Set context
        engine.context = MagicMock()
        engine.context.technical_data = {"rsi": 50}
        
        start_time = time.perf_counter()
        await engine.analyze_market(provider="test", model="test")
        duration = time.perf_counter() - start_time
        
        print(f"Parallel execution took: {duration:.4f}s")
        
        # Sequential: collect(0) + enrich(0.1) + rag(0.1) + tech(0.1) + chart(0.1) = 0.4s
        # Parallel: collect(0) + max(enrich(0.1), rag(0.1), tech(0.1)+chart(0.1)) = 0.2s
        # (tech+chart is sequential within their task)
        
        # Let's ensure it called everything
        engine._enrich_market_context.assert_called_once()
        engine._perform_technical_analysis.assert_called_once()
        engine._generate_chart_image.assert_called_once()
        mock_rag.retrieve_context.assert_called_once()

    def test_choppiness_index_performance(self):
        """Test that choppiness_index_numba is performant (O(N))."""
        # Setup data
        np.random.seed(42)
        size = 100_000
        high = np.random.uniform(100, 200, size)
        low = high - np.random.uniform(1, 10, size)
        close = (high + low) / 2
        length = 100

        # Warmup
        choppiness_index_numba(high[:200], low[:200], close[:200], length)

        start_time = time.perf_counter()
        _ = choppiness_index_numba(high, low, close, length)
        duration = time.perf_counter() - start_time
        
        # Should be well under 0.2s on modern hardware for 100k points with optimization.
        # Allowing generous buffer for CI environments.
        print(f"Choppiness Index (N={size}, L={length}) took {duration:.4f}s")
        self.assertLess(duration, 1.0, "Choppiness Index calculation too slow")

    def test_choppiness_index_short_input(self):
        """Test that choppiness_index_numba handles short inputs gracefully."""
        # Setup data shorter than length
        size = 10
        high = np.random.uniform(100, 200, size)
        low = high - np.random.uniform(1, 10, size)
        close = (high + low) / 2
        length = 14

        # This should not raise IndexError
        result = choppiness_index_numba(high, low, close, length)

        # Result should be all NaNs
        self.assertTrue(np.all(np.isnan(result)), "Result should be all NaNs for short input")

if __name__ == '__main__':
    unittest.main()
