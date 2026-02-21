
import unittest
from unittest.mock import MagicMock, patch
import sys
import numpy # Ensure numpy is loaded before patching sys.modules so it is not unloaded/reloaded

# Define a simple mock for VectorSearchResult if needed
class MockVectorSearchResult:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class TestVectorThresholds(unittest.TestCase):
    def setUp(self):
        # Create a mock for src.trading.dataclasses
        mock_dataclasses = MagicMock()
        mock_dataclasses.VectorSearchResult = MockVectorSearchResult

        # Patch sys.modules to mock dependencies before importing the module under test
        # We assume numpy is already in sys.modules and won't be removed by patch.dict cleanup
        self.modules_patcher = patch.dict(sys.modules, {
            'src.logger.logger': MagicMock(),
            'chromadb': MagicMock(),
            'sentence_transformers': MagicMock(),
            'src.trading.dataclasses': mock_dataclasses
        })
        self.modules_patcher.start()

        # Ensure we import a fresh version of the module using the mocks
        # This is necessary because we changed sys.modules
        if 'src.trading.vector_memory' in sys.modules:
            del sys.modules['src.trading.vector_memory']

        from src.trading.vector_memory import VectorMemoryService
        self.VectorMemoryService = VectorMemoryService

        self.logger = MagicMock()
        self.chroma_client = MagicMock()
        self.embedding_model = MagicMock()
        self.service = self.VectorMemoryService(self.logger, self.chroma_client, self.embedding_model)
        # Mock initialization
        self.service._initialized = True
        self.service._collection = MagicMock()
        self.service._semantic_rules_collection = MagicMock()

    def tearDown(self):
        self.modules_patcher.stop()

    def test_compute_optimal_thresholds_rr_borderline(self):
        """Test that rr_borderline_min is correctly identified."""

        metadatas = []
        # Create 10 trades with RR=1.2 (below 1.3)
        # 2 wins, 8 losses -> 20% win rate (< 40%)
        # This should trigger rr_borderline_min = 1.3 and break
        for _ in range(2):
            metadatas.append({"outcome": "WIN", "rr_ratio": 1.2, "sl_distance_pct": 0.01})
        for _ in range(8):
            metadatas.append({"outcome": "LOSS", "rr_ratio": 1.2, "sl_distance_pct": 0.01})

        # Create trades with RR=1.4 (between 1.3 and 1.5)
        # Increase wins here to ensure cumulative win rate at 1.5 is > 40%,
        # avoiding confusion about whether it should trigger 1.5 if 1.3 didn't break.
        # 7 wins, 3 losses at 1.4.
        # Cumulative at 1.5: 2+7=9 wins, 8+3=11 losses. 9/20 = 45% (> 40%).
        for _ in range(7):
            metadatas.append({"outcome": "WIN", "rr_ratio": 1.4, "sl_distance_pct": 0.01})
        for _ in range(3):
            metadatas.append({"outcome": "LOSS", "rr_ratio": 1.4, "sl_distance_pct": 0.01})

        # Create trades with RR=2.0 (high)
        for _ in range(5):
            metadatas.append({"outcome": "WIN", "rr_ratio": 2.0, "sl_distance_pct": 0.01})

        # Mock return value for collection.get()
        mock_response = {
            "ids": ["id" + str(i) for i in range(len(metadatas))],
            "metadatas": metadatas,
            "documents": ["doc"] * len(metadatas),
            "embeddings": [[0.1]*10] * len(metadatas)
        }
        self.service._collection.get.return_value = mock_response

        # Run computation with small sample size to trigger checks
        thresholds = self.service.compute_optimal_thresholds(min_sample_size=2)

        # Verify rr_borderline_min is set to 1.3
        self.assertEqual(thresholds.get("rr_borderline_min"), 1.3)

    def test_compute_optimal_thresholds_no_borderline(self):
        """Test that rr_borderline_min is NOT set if win rate is good."""

        metadatas = []
        # Create 10 trades with RR=1.2 (below 1.3)
        # 6 wins, 4 losses -> 60% win rate (> 40%)
        for _ in range(6):
            metadatas.append({"outcome": "WIN", "rr_ratio": 1.2, "sl_distance_pct": 0.01})
        for _ in range(4):
            metadatas.append({"outcome": "LOSS", "rr_ratio": 1.2, "sl_distance_pct": 0.01})

        mock_response = {
            "ids": ["id" + str(i) for i in range(len(metadatas))],
            "metadatas": metadatas,
            "documents": ["doc"] * len(metadatas),
            "embeddings": [[0.1]*10] * len(metadatas)
        }
        self.service._collection.get.return_value = mock_response

        thresholds = self.service.compute_optimal_thresholds(min_sample_size=2)

        # Should not be set
        self.assertIsNone(thresholds.get("rr_borderline_min"))

if __name__ == '__main__':
    unittest.main()
