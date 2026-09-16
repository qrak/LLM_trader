"""Trading-field extraction contract: one parser (UnifiedParser), one normalization."""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.parsing.unified_parser import UnifiedParser
from src.trading.position_extractor import PositionExtractor
from src.utils.format_utils import FormatUtils


def _parse_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the real parse path (contract validation included) and return the analysis payload."""
    parser = UnifiedParser(logger=MagicMock(), format_utils=FormatUtils())
    return parser.parse_ai_response(f"```json\n{json.dumps(payload)}\n```")["analysis"]


@pytest.mark.parametrize(
    ("raw_position_size", "expected_position_size"),
    [
        ("5%", 0.05),
        ("0.5%", 0.005),
        (0.05, 0.05),
        (50, None),
    ],
)
def test_position_size_contract_is_owned_by_the_parser(
    raw_position_size: Any, expected_position_size: float | None
) -> None:
    analysis = _parse_analysis(
        {
            "analysis": {
                "signal": "BUY",
                "confidence": 80,
                "stop_loss": 95,
                "take_profit": 110,
                "position_size": raw_position_size,
                "reasoning": "Test setup.",
            }
        }
    )

    _, _, _, _, position_size, _ = PositionExtractor().extract_trading_info(analysis)

    if expected_position_size is None:
        assert position_size is None
    else:
        assert position_size == pytest.approx(expected_position_size)


def test_extractor_reads_every_field_from_the_parsed_payload() -> None:
    analysis = _parse_analysis(
        {
            "analysis": {
                "signal": "LONG",
                "confidence": 82,
                "stop_loss": 76500,
                "take_profit": 80640,
                "position_size": 0.08,
                "reasoning": "Trend continuation with confluence.",
            }
        }
    )

    signal, confidence, stop_loss, take_profit, position_size, reasoning = (
        PositionExtractor().extract_trading_info(analysis)
    )

    assert signal == "LONG"
    assert confidence == "HIGH"
    assert stop_loss == pytest.approx(76500.0)
    assert take_profit == pytest.approx(80640.0)
    assert position_size == pytest.approx(0.08)
    assert reasoning == "Trend continuation with confluence."


def test_narrative_only_response_yields_hold_instead_of_guessing_from_prose() -> None:
    """A reply without the JSON block must NOT be heuristically read from its text."""
    parser = UnifiedParser(logger=MagicMock())
    parsed = parser.parse_ai_response(
        "signal: BUY confidence: HIGH stop_loss: 95 take_profit: 110 position_size: 5%"
    )

    signal, confidence, stop_loss, take_profit, position_size, reasoning = (
        PositionExtractor().extract_trading_info(parsed["analysis"])
    )

    assert signal == "HOLD"
    assert confidence == "MEDIUM"
    assert stop_loss is None
    assert take_profit is None
    assert position_size is None
    assert reasoning == ""


def test_confidence_labels_are_mapped_from_numbers() -> None:
    extractor = PositionExtractor()

    assert extractor._confidence_label(82) == "HIGH"
    assert extractor._confidence_label(60) == "MEDIUM"
    assert extractor._confidence_label(40) == "LOW"
    assert extractor._confidence_label("HIGH") == "HIGH"


def test_long_signal_accepted() -> None:
    """Futures LONG signal is recognized as valid."""
    assert PositionExtractor().validate_signal("LONG") is True


def test_short_signal_accepted() -> None:
    """Futures SHORT signal is recognized as valid."""
    assert PositionExtractor().validate_signal("SHORT") is True


def test_unknown_signal_rejected() -> None:
    """Anything outside the executor's accepted signal set is rejected."""
    assert PositionExtractor().validate_signal("MAYBE") is False
