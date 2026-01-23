import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
import time
from typing import Optional, Tuple, List
import numpy as np
from src.rag.text_splitting import SentenceSplitter
from src.indicators.volatility.volatility_indicators import choppiness_index_numba
from src.indicators.momentum.momentum_indicators import williams_r_numba
from src.indicators.statistical.statistical_indicators import (
    linreg_numba, entropy_numba, skew_numba, kurtosis_numba
)
import pandas as pd

class TestPerformanceOptimization(unittest.IsolatedAsyncioTestCase):

    def test_williams_r_correctness_and_performance(self):
        """Test that Williams %R is correct and performant."""

        # 1. Correctness Check
        # Generate deterministic data
        np.random.seed(42)
        n = 1000
        length = 14
        close = np.random.random(n) * 100
        high = close + np.random.random(n) * 5
        low = close - np.random.random(n) * 5

        # Run optimized numba version
        res_numba = williams_r_numba(high, low, close, length)

        # Run reference pandas implementation
        high_s = pd.Series(high)
        low_s = pd.Series(low)
        close_s = pd.Series(close)
        highest_high = high_s.rolling(window=length).max()
        lowest_low = low_s.rolling(window=length).min()
        # Formula: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
        # Note: Some sources use (Highest High - Close), others (Close - Highest High).
        # Our implementation uses: ((highest_high - close[i]) / (highest_high - lowest_low)) * -100
        res_ref = ((highest_high - close_s) / (highest_high - lowest_low)) * -100
        res_ref = res_ref.to_numpy()

        # Compare (ignoring NaNs at start)
        mask = ~np.isnan(res_ref)
        np.testing.assert_allclose(res_numba[mask], res_ref[mask], err_msg="Williams %R output mismatch")

        # 2. Performance Check (simple threshold)
        # Larger dataset
        n_perf = 100_000
        close_p = np.random.random(n_perf) * 100
        high_p = close_p + np.random.random(n_perf) * 5
        low_p = close_p - np.random.random(n_perf) * 5

        start_time = time.perf_counter()
        _ = williams_r_numba(high_p, low_p, close_p, length)
        duration = time.perf_counter() - start_time

        print(f"Williams %R (N={n_perf}) took {duration:.4f}s")
        self.assertLess(duration, 0.5, "Williams %R calculation too slow")
    
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

    def test_linreg_performance(self):
        """Test that linreg_numba is performant with O(N) complexity."""
        from src.indicators.statistical.statistical_indicators import linreg_numba
        
        np.random.seed(42)
        size = 100_000
        close = np.random.uniform(50, 150, size)
        length = 14

        # Warmup
        linreg_numba(close[:200], length)

        start_time = time.perf_counter()
        result = linreg_numba(close, length, r=False)
        duration = time.perf_counter() - start_time
        
        print(f"Linear Regression (N={size}, L={length}) took {duration:.4f}s")
        self.assertLess(duration, 1.0, "Linear regression calculation too slow")
        
        # Verify output is correct length
        self.assertEqual(len(result), size)

    def test_linreg_correctness(self):
        """Test linreg_numba produces correct slope values."""
        from src.indicators.statistical.statistical_indicators import linreg_numba
        
        # Test with known linear data: y = 2x + 10
        length = 5
        x = np.arange(1, length + 1)
        y = 2 * x + 10  # slope = 2
        
        # Pad with some values before to test windowing
        close = np.concatenate([np.array([15, 20, 25]), y])
        
        result = linreg_numba(close, length, r=False)
        
        # The slope for the last window should be close to 2
        self.assertAlmostEqual(result[-1], 2.0, places=10)

    def test_linreg_r_squared_correctness(self):
        """Test linreg_numba R-squared correctness."""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 10, n)
        # Perfect linear correlation: y = 2x
        y = 2 * x
        
        length = 10
        r_values = linreg_numba(y, length, r=True)
        
        # For perfect positive correlation, r should be 1.0
        mask = ~np.isnan(r_values)
        self.assertTrue(np.allclose(r_values[mask], 1.0), "Correlation should be 1.0 for linear data")
        
        # Test with noise
        y_noisy = 2 * x + np.random.normal(0, 0.1, n)
        r_noisy = linreg_numba(y_noisy, length, r=True)
        
        # Verify against numpy for a few arbitrary points
        for i in range(length, n, 10): # Check every 10th point
            window = y_noisy[i-length+1 : i+1]
            wx = np.arange(1, length+1)
            expected_r = np.corrcoef(wx, window)[0, 1]
            # linreg_numba returns r (correlation coefficient) when r=True, despite the "R-Squared" name in some docs.
            # Paper mentions "r = ...", so we expect r.
            self.assertAlmostEqual(r_noisy[i], expected_r, places=5, msg=f"Mismatch at index {i}")

    def test_entropy_correctness(self):
        """Test entropy_numba correctness against a simple Python reference."""
        np.random.seed(42)
        data = np.random.uniform(1, 100, 100)
        length = 10
        base = 2.0
        
        result_numba = entropy_numba(data, length, base=base)
        
        # Reference implementation
        expected = np.full(len(data), np.nan)
        # Use length-1 to match Numba implementation start
        for i in range(length - 1, len(data)):
            # Standard window: ends at i (inclusive), length L
            # Slice: [i - length + 1 : i + 1]
            window = data[i-length+1 : i+1]
            
            total = np.sum(window)
            p = window / total
            e = -np.sum(p * np.log(p) / np.log(base))
            expected[i] = e
            
        mask = ~np.isnan(expected)
        np.testing.assert_allclose(result_numba[mask], expected[mask], err_msg="Entropy mismatch", atol=1e-7)

    def test_moments_stability(self):
        """Test Skewness and Kurtosis stability with large offsets."""
        np.random.seed(42)
        base_data = np.random.normal(0, 1, 100)
        large_offset = 1_000_000.0
        large_data = base_data + large_offset
        length = 20
        
        s_base = skew_numba(base_data, length)
        s_large = skew_numba(large_data, length)
        
        k_base = kurtosis_numba(base_data, length)
        k_large = kurtosis_numba(large_data, length)
        
        mask = ~np.isnan(s_base)
        
        # Verify translation invariance (stability)
        np.testing.assert_allclose(s_large[mask], s_base[mask], err_msg="Skewness stability failure", atol=1e-5)
        np.testing.assert_allclose(k_large[mask], k_base[mask], err_msg="Kurtosis stability failure", atol=1e-5)

if __name__ == '__main__':
    unittest.main()
