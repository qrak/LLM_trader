"""Tests for non-blocking trading response contract validation."""

from unittest.mock import MagicMock

from src.parsing.unified_parser import UnifiedParser


def test_valid_trading_response_attaches_validation_metadata() -> None:
    parser = UnifiedParser(logger=MagicMock())
    response = """```json
{"analysis":{"signal":"BUY","confidence":82,"entry_price":77880,"stop_loss":76500,"take_profit":80640,"position_size":0.08,"risk_reward_ratio":2.0,"reasoning":"Valid setup."}}
```"""

    parsed = parser.parse_ai_response(response)

    validation = parsed["response_validation"]
    assert validation["schema"] == "trading-analysis-response-v1"
    assert validation["status"] == "valid"
    assert validation["valid"] is True
    assert validation["errors"] == []


def test_invalid_trading_response_reports_field_errors_without_breaking_structure() -> None:
    parser = UnifiedParser(logger=MagicMock())
    response = """```json
{"analysis":{"signal":"BUY","confidence":125,"entry_price":77880,"reasoning":"Missing risk fields."}}
```"""

    parsed = parser.parse_ai_response(response)

    validation = parsed["response_validation"]
    assert parser.validate_ai_response(parsed) is True
    assert validation["status"] == "invalid"
    assert validation["valid"] is False
    fields = {error["field"] for error in validation["errors"]}
    assert "analysis.confidence" in fields


def test_legacy_analysis_without_signal_skips_trading_contract_validation() -> None:
    parser = UnifiedParser(logger=MagicMock())
    response = """```json
{"analysis":{"summary":"Legacy fallback analysis."}}
```"""

    parsed = parser.parse_ai_response(response)

    validation = parsed["response_validation"]
    assert validation["status"] == "skipped"
    assert validation["valid"] is None
    assert validation["errors"] == []


def test_parse_fallback_reports_validation_error_metadata() -> None:
    parser = UnifiedParser(logger=MagicMock())

    parsed = parser.parse_ai_response("not json")

    validation = parsed["response_validation"]
    assert parsed["parse_error"] == "Failed to parse response"
    assert validation["status"] == "invalid"
    assert validation["valid"] is False
    assert validation["errors"][0]["type"] == "json_parse_error"
