"""Decision contract: real recorded payloads stay usable; broken ones are named."""

import json
from pathlib import Path

import pytest

from src.analyzer.decision_contract import validate_decision_payload

FIXTURES = sorted((Path(__file__).parent.parent / "fixtures" / "decisions").glob("*.json"))


def _payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["payload"]


def test_fixtures_are_present():
    """A silently empty fixture set would make the parametrised test vacuous."""
    assert len(FIXTURES) >= 5


@pytest.mark.parametrize("path", FIXTURES, ids=[path.stem for path in FIXTURES])
def test_recorded_payloads_satisfy_the_contract(path: Path):
    assert validate_decision_payload(_payload(path)) == []


def test_missing_take_profit_is_named():
    problems = validate_decision_payload({
        "analysis": {
            "signal": "BUY", "confidence": 70, "entry_price": 100.0,
            "stop_loss": 95.0, "take_profit": None, "risk_reward_ratio": 2.0,
            "position_size": 0.05,
        }
    })

    assert any("take_profit" in problem for problem in problems)


def test_inverted_buy_levels_are_named():
    problems = validate_decision_payload({
        "analysis": {
            "signal": "BUY", "confidence": 70, "entry_price": 100.0,
            "stop_loss": 105.0, "take_profit": 95.0, "risk_reward_ratio": 1.0,
            "position_size": 0.05,
        }
    })

    assert any("stop_loss < entry_price < take_profit" in problem for problem in problems)


def test_unknown_signal_is_named():
    problems = validate_decision_payload({"analysis": {"signal": "MAYBE", "confidence": 10}})

    assert any("analysis.signal" in problem for problem in problems)


def test_missing_confidence_is_not_an_execution_contract_violation():
    problems = validate_decision_payload({"analysis": {"signal": "HOLD"}})

    assert problems == []


def test_bool_confidence_is_refused():
    problems = validate_decision_payload({"analysis": {"signal": "HOLD", "confidence": True}})

    assert any("confidence" in problem for problem in problems)


def test_non_object_payload_is_refused():
    assert validate_decision_payload(None) == ["payload must be an object"]
