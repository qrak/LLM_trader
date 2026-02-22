import unittest
import sys
from unittest.mock import MagicMock, patch

class TestUnifiedParser(unittest.TestCase):
    def setUp(self):
        # Create a mock for the config loader module
        self.mock_config_loader = MagicMock()
        self.mock_config = MagicMock()
        self.mock_config_loader.config = self.mock_config

        # Patch sys.modules to inject our mock
        self.modules_patcher = patch.dict(sys.modules, {'src.config.loader': self.mock_config_loader})
        self.modules_patcher.start()

        # Ensure we re-import the module under test to pick up the mock
        # We need to clear these from sys.modules to force reload, BUT NOT src.config.loader which we just mocked!
        self._modules_to_restore = {}
        # We only need to reload modules that IMPORT src.config.loader, not src.config.loader itself
        for mod_name in ['src.parsing.unified_parser', 'src.logger.logger']:
            if mod_name in sys.modules:
                self._modules_to_restore[mod_name] = sys.modules[mod_name]
                del sys.modules[mod_name]

        # Import the class under test
        from src.parsing.unified_parser import UnifiedParser
        self.UnifiedParserClass = UnifiedParser

        self.mock_logger = MagicMock()
        self.mock_format_utils = MagicMock()
        self.mock_format_utils.parse_value.side_effect = lambda x, default=None: float(x) if isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).isdigit() else default

        self.parser = self.UnifiedParserClass(self.mock_logger, self.mock_format_utils)

    def tearDown(self):
        self.modules_patcher.stop()
        # Restore original modules if they existed
        for mod_name in ['src.parsing.unified_parser', 'src.logger.logger']:
            if mod_name in self._modules_to_restore:
                sys.modules[mod_name] = self._modules_to_restore[mod_name]
            elif mod_name in sys.modules:
                # If it wasn't there before, remove it to avoid pollution
                del sys.modules[mod_name]

    def test_extract_base_coin(self):
        """Test extraction of base coin, specifically the regression for BUSD."""
        # The bug fix: BUSD should be detected as quote, so DOGEBUSD -> DOGE
        self.assertEqual(self.parser.extract_base_coin("DOGEBUSD"), "DOGE")
        self.assertEqual(self.parser.extract_base_coin("BTC/USDT"), "BTC")
        self.assertEqual(self.parser.extract_base_coin("ETH-USD"), "ETH")
        self.assertEqual(self.parser.extract_base_coin("SOLUSDT"), "SOL")

    def test_extract_json_block_valid(self):
        text = '```json\n{"analysis": {"summary": "Bullish"}}\n```'
        result = self.parser.extract_json_block(text)
        self.assertEqual(result['analysis']['summary'], "Bullish")

    def test_extract_json_block_invalid_json(self):
        text = '```json\n{"analysis": {"summary": "Bullish",}}\n```' # Trailing comma
        result = self.parser.extract_json_block(text)
        self.assertIsNone(result)

    def test_parse_ai_response_valid(self):
        text = '```json\n{"analysis": {"risk_ratio": "2.5"}}\n```'
        result = self.parser.parse_ai_response(text)
        self.assertEqual(result['analysis']['risk_ratio'], 2.5)

    def test_detect_coins_in_text(self):
        text = "Bitcoin (BTC) and Ethereum (ETH)"
        known = {"BTC", "ETH"}
        coins = self.parser.detect_coins_in_text(text, known)
        self.assertEqual(coins, {"BTC", "ETH"})

    def test_parse_article_categories_comma(self):
        categories = self.parser.parse_article_categories("tech, crypto, bitcoin")
        self.assertEqual(categories, {"tech", "crypto", "bitcoin"})

    def test_parse_article_categories_semicolon(self):
        categories = self.parser.parse_article_categories("tech; crypto; bitcoin")
        self.assertEqual(categories, {"tech", "crypto", "bitcoin"})

    def test_parse_article_categories_pipe(self):
        categories = self.parser.parse_article_categories("tech| crypto| bitcoin")
        self.assertEqual(categories, {"tech", "crypto", "bitcoin"})

    def test_parse_article_categories_mixed_priority(self):
        # Comma takes precedence
        categories = self.parser.parse_article_categories("tech, crypto; bitcoin")
        # Should split by comma: "tech", " crypto; bitcoin"
        self.assertEqual(categories, {"tech", "crypto; bitcoin"})

    def test_parse_article_categories_single(self):
        categories = self.parser.parse_article_categories("technology")
        self.assertEqual(categories, {"technology"})

    def test_parse_article_categories_empty(self):
        categories = self.parser.parse_article_categories("")
        self.assertEqual(categories, set())

if __name__ == '__main__':
    unittest.main()
