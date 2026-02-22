
import unittest
import numpy as np
from src.indicators.momentum.momentum_indicators import kst_numba

class TestKSTCorrectness(unittest.TestCase):
    def test_kst_values(self):
        """
        Test that KST is calculated correctly.
        """
        np.random.seed(42)
        n = 1000
        close = np.random.random(n) * 100 + 100

        # Calculate KST with default params
        kst = kst_numba(close)

        # Check that result has correct length
        self.assertEqual(len(kst), n)

        # Check NaNs
        expected_start = 28
        self.assertFalse(np.isnan(kst[expected_start]), f"Value at {expected_start} should be valid")
        self.assertTrue(np.isnan(kst[expected_start - 1]), f"Value at {expected_start-1} should be NaN")

        # Check specific value consistency (regression test based on seed=42)
        expected_val = 43.67652374331601
        self.assertTrue(np.isclose(kst[expected_start], expected_val, atol=1e-8),
                        f"Regression check failed. Expected {expected_val}, got {kst[expected_start]}")

    def test_kst_flat_line(self):
        """Test with flat line data (should result in 0 ROC and 0 KST)."""
        n = 100
        # Use full(n, 100.0)
        close = np.full(n, 100.0)
        kst = kst_numba(close)

        # All valid values should be 0
        expected_start = 28
        self.assertTrue(np.all(kst[expected_start:] == 0.0))

    def test_kst_short_data(self):
        """Test with short data."""
        n = 20
        close = np.full(n, 100.0)
        kst = kst_numba(close)

        # Should be all NaNs
        self.assertTrue(np.all(np.isnan(kst)))

if __name__ == '__main__':
    unittest.main()
