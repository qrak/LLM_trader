
import unittest
import numpy as np
from src.indicators.statistical.statistical_indicators import kurtosis_numba

class TestKurtosisOptimization(unittest.TestCase):
    def test_kurtosis_correctness(self):
        """Verify kurtosis_numba against naive implementation."""
        np.random.seed(42)
        data_length = 1000
        window_length = 30
        data = np.random.randn(data_length)

        result = kurtosis_numba(data, window_length)

        # Verify a few points
        indices_to_check = [window_length, 100, 500, 999]
        for idx in indices_to_check:
            window = data[idx - window_length + 1 : idx + 1]
            mean = np.mean(window)
            std = np.std(window) # numpy default ddof=0, consistent with implementation

            # Kurtosis calculation
            k_sum = np.sum(((window - mean) / std) ** 4)
            length = window_length
            kurtosis_constant = (length * (length + 1)) / ((length - 1) * (length - 2) * (length - 3))
            expected_k = kurtosis_constant * k_sum - 3 * (length - 1) / ((length - 2) * (length - 3))

            self.assertAlmostEqual(result[idx], expected_k, places=5,
                                   msg=f"Mismatch at index {idx}")

    def test_kurtosis_drift_stability(self):
        """Verify numerical stability with large drift."""
        np.random.seed(42)
        data_length = 2000 # Enough to trigger re-calculation (every 1000)
        window_length = 30
        drift = np.linspace(0, 1_000_000, data_length)
        data = np.random.randn(data_length) + drift

        result = kurtosis_numba(data, window_length)

        # Check index after re-calculation (e.g. 1005)
        idx = 1005
        window = data[idx - window_length + 1 : idx + 1]
        mean = np.mean(window)
        std = np.std(window)

        k_sum = np.sum(((window - mean) / std) ** 4)
        length = window_length
        kurtosis_constant = (length * (length + 1)) / ((length - 1) * (length - 2) * (length - 3))
        expected_k = kurtosis_constant * k_sum - 3 * (length - 1) / ((length - 2) * (length - 3))

        self.assertAlmostEqual(result[idx], expected_k, places=4,
                               msg=f"Mismatch at index {idx} with drift")

if __name__ == '__main__':
    unittest.main()
