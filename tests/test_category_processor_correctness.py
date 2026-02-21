import sys
from unittest.mock import MagicMock
import pytest

# Mock src.config.loader before importing anything that uses it
config_mock = MagicMock()
loader_mock = MagicMock()
loader_mock.config = config_mock
sys.modules['src.config.loader'] = loader_mock

from src.rag.category_processor import CategoryProcessor

class TestCategoryProcessorCorrectness:
    @pytest.fixture
    def processor(self):
        # Mock dependencies
        logger = MagicMock()
        collision_resolver = MagicMock()
        file_handler = MagicMock()

        # CategoryProcessor calls _load_rag_config in __init__
        file_handler.load_rag_priorities.return_value = {}

        return CategoryProcessor(logger, collision_resolver, file_handler)

    def test_extract_base_coin_bug(self, processor):
        """Test the specific bug with overlapping quotes."""
        # Case 1: BTCBUSD -> Should be BTC
        assert processor.extract_base_coin("BTCBUSD") == "BTC"

        # Case 2: ETHUSDT -> Should be ETH
        assert processor.extract_base_coin("ETHUSDT") == "ETH"

        # Case 3: BNBUSD -> Should be BNB
        assert processor.extract_base_coin("BNBUSD") == "BNB"

    def test_extract_base_coin_usdc(self, processor):
        """Test USDC handling (which is in CategoryProcessor but was missing in UnifiedParser)."""
        assert processor.extract_base_coin("ETHUSDC") == "ETH"
