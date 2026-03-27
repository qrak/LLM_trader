import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.dashboard.routers.brain import _build_current_market_context, _extract_market_status


def test_build_current_market_context_uses_legacy_response_indicators(tmp_path):
    trading_dir = tmp_path / "trading"
    trading_dir.mkdir()
    previous_response = trading_dir / "previous_response.json"
    previous_response.write_text(
        json.dumps(
            {
                "response": {
                    "text_analysis": "SIGNAL: UPDATE\nConfidence: 82%\nTrend remains bearish.",
                    "adx": 18.1,
                    "rsi": 40.5,
                    "atr_percent": 1.2,
                    "plus_di": 14.8,
                    "minus_di": 24.2,
                    "macd_line": -268.6,
                    "macd_signal": -28.0,
                    "obv_slope": -0.7,
                    "bb_upper": 72140.5,
                    "bb_lower": 68571.7,
                    "current_price": 68795.96,
                },
                "timestamp": "2026-03-27T00:00:50.843610+00:00",
            }
        ),
        encoding="utf-8",
    )

    config = SimpleNamespace(DATA_DIR=str(tmp_path))
    logger = MagicMock()

    display_context, query_document = _build_current_market_context(config, logger)

    assert display_context.startswith("BEARISH + Low ADX + LOW Volatility")
    assert "MACD BEARISH" in display_context
    assert "Indicators: ADX=18.1 (Low ADX)" in query_document


def test_extract_market_status_uses_text_and_indicators_without_json_parser():
    parser = MagicMock()
    data = {
        "response": {
            "text_analysis": "SIGNAL: UPDATE\nConfidence: 82%\nThe structure remains bearish.\n```json\n{bad json}\n```",
            "adx": 18.1,
            "rsi": 40.5,
            "plus_di": 14.8,
            "minus_di": 24.2,
        }
    }

    status = _extract_market_status(data, parser)

    parser.extract_json_block.assert_not_called()
    assert status["action"] == "UPDATE"
    assert status["confidence"] == 82
    assert status["trend"] == "BEARISH"
    assert status["adx"] == 18.1
    assert status["rsi"] == 40.5