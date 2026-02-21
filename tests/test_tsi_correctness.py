import unittest
import numpy as np
from src.indicators.momentum.momentum_indicators import tsi_numba

class TestTSICorrectness(unittest.TestCase):
    def test_tsi_values_and_initialization(self):
        """
        Test that TSI is calculated correctly and the off-by-one initialization error is fixed.
        """
        np.random.seed(42)
        n = 1000
        close = np.random.random(n) * 100
        long_length = 25
        short_length = 13

        # Calculate TSI
        tsi = tsi_numba(close, long_length, short_length)

        # Check that the result has the correct length
        self.assertEqual(len(tsi), n)

        # Check initialization
        # The first valid value should be at index long_length + short_length - 1
        first_valid_idx = long_length + short_length - 1

        # Previous implementation had NaN at this index. New one should not.
        self.assertFalse(np.isnan(tsi[first_valid_idx]),
                         f"Value at index {first_valid_idx} (long+short-1) should not be NaN")

        # Check that values before are NaN
        self.assertTrue(np.isnan(tsi[first_valid_idx - 1]),
                        f"Value at index {first_valid_idx - 1} should be NaN")

        # Check specific value consistency (regression test based on seed=42)
        # Value at index 38 (start of main loop) was -0.7489009092080221
        # Note: Index 38 is long_length + short_length.
        expected_val_38 = -0.7489009092080221
        # Allow small floating point difference due to order of operations
        if n > 38:
             self.assertTrue(np.isclose(tsi[38], expected_val_38, atol=1e-8),
                             f"Regression check failed. Expected {expected_val_38}, got {tsi[38]}")

    def test_tsi_short_data(self):
        """Test with data shorter than required length."""
        close = np.array([10.0] * 10)
        long_length = 25
        short_length = 13

        tsi = tsi_numba(close, long_length, short_length)

        self.assertTrue(np.all(np.isnan(tsi)), "All values should be NaN for short data")

    def test_tsi_exact_length(self):
        """Test with data exactly the required length."""
        # Minimum length to get 1 valid value is long_length + short_length
        # Wait, if init is at long+short-1, then we need long+short values (0 to long+short-1)
        long_length = 5
        short_length = 3
        n = long_length + short_length
        close = np.random.random(n) * 100

        tsi = tsi_numba(close, long_length, short_length)

        # Index long+short-1 is the last element
        self.assertFalse(np.isnan(tsi[-1]), "Last element should be valid")
        self.assertTrue(np.isnan(tsi[-2]), "Second to last element should be NaN")

if __name__ == '__main__':
    unittest.main()
