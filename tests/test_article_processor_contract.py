from unittest.mock import MagicMock

from src.rag.article_processor import ArticleProcessor


class _ParserStub:
    @staticmethod
    def detect_coins_in_text(text: str, known_crypto_tickers: set[str]) -> set[str]:
        lowered = text.lower()
        return {ticker for ticker in known_crypto_tickers if ticker.lower() in lowered}

    @staticmethod
    def extract_base_coin(symbol: str) -> str:
        return symbol.split("/")[0]


def test_detect_coins_in_article_scans_full_body_without_hard_cutoff():
    processor = ArticleProcessor(
        logger=MagicMock(),
        format_utils=MagicMock(),
        unified_parser=_ParserStub(),
    )
    long_prefix = "x" * 12050
    article = {
        "title": "No coin in title",
        "categories": "",
        "body": f"{long_prefix} btc appears late in body",
    }

    detected = processor.detect_coins_in_article(article, {"BTC"})

    assert "BTC" in detected
