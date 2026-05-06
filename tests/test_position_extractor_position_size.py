"""Position size normalization tests for AI trading responses."""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.parsing.unified_parser import UnifiedParser
from src.trading.position_extractor import PositionExtractor


@pytest.mark.parametrize(
    ("raw_position_size", "expected_position_size"),
    [
        ("5%", 0.05),
        ("0.5%", 0.005),
        (0.05, 0.05),
        (50, 0.5),
    ],
)
def test_json_position_size_normalization(raw_position_size: Any, expected_position_size: float) -> None:
    parser = UnifiedParser(logger=MagicMock())
    extractor = PositionExtractor(logger=MagicMock(), unified_parser=parser)
    payload = {
        "analysis": {
            "signal": "BUY",
            "confidence": 80,
            "stop_loss": 95,
            "take_profit": 110,
            "position_size": raw_position_size,
            "reasoning": "Test setup.",
        }
    }
    response = f"```json\n{json.dumps(payload)}\n```"

    _, _, _, _, position_size, _ = extractor.extract_trading_info(response)

    assert position_size == pytest.approx(expected_position_size)


def test_text_position_size_below_one_percent_is_not_treated_as_half_capital() -> None:
    extractor = PositionExtractor(logger=MagicMock())
    response = "signal: BUY confidence: HIGH stop_loss: 95 take_profit: 110 position_size: 0.5%"

    _, _, _, _, position_size, _ = extractor.extract_trading_info(response)

    assert position_size == pytest.approx(0.005)
