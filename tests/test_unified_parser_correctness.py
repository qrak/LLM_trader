import sys
from unittest.mock import MagicMock
import pytest

# Mock src.config.loader before importing anything that uses it
# This is necessary because UnifiedParser imports Logger which imports Config
# which tries to load keys.env
config_mock = MagicMock()
loader_mock = MagicMock()
loader_mock.config = config_mock
sys.modules['src.config.loader'] = loader_mock

from src.parsing.unified_parser import UnifiedParser  # noqa: E402

class TestUnifiedParserCorrectness:
    @pytest.fixture
    def parser(self):
        return UnifiedParser(logger=MagicMock())

    def test_extract_base_coin_overlapping_quotes(self, parser):
        """Test that overlapping quote currencies are handled correctly (longest match first)."""
        # BUSD vs USD
        assert parser.extract_base_coin("BTCBUSD") == "BTC"
        assert parser.extract_base_coin("ETHBUSD") == "ETH"

        # USD vs USDT
        assert parser.extract_base_coin("ETHUSDT") == "ETH"
        assert parser.extract_base_coin("BTCUSDT") == "BTC"

        # USDC vs USD
        assert parser.extract_base_coin("ETHUSDC") == "ETH" # This will fail until I add USDC

        # Normal USD
        assert parser.extract_base_coin("BNBUSD") == "BNB"

    def test_extract_base_coin_explicit_separators(self, parser):
        """Test symbols with explicit separators."""
        assert parser.extract_base_coin("BTC/USD") == "BTC"
        assert parser.extract_base_coin("ETH-USDT") == "ETH"

    def test_extract_base_coin_no_match(self, parser):
        """Test symbols that don't match any known quote."""
        assert parser.extract_base_coin("UNKNOWN") == "UNKNOWN"
        assert parser.extract_base_coin("") == ""

    def test_extract_base_coin_case_sensitivity(self, parser):
        """Test that input case is handled correctly (output should be upper)."""
        assert parser.extract_base_coin("btcusdt") == "BTC"
        assert parser.extract_base_coin("ethbusd") == "ETH"
