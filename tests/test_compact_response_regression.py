"""Regression tests for compact AI response format used in trading decisions."""

from unittest.mock import MagicMock

from src.parsing.unified_parser import UnifiedParser
from src.trading.position_extractor import PositionExtractor


def _build_compact_response() -> str:
    return """1) MARKET STRUCTURE: MIXED trend with weak momentum.
2) INDICATOR ASSESSMENT: Overextension plus exhaustion risk.
4) DECISION: HOLD until conditional short trigger confirms.

```json
{
  "analysis": {
    "signal": "HOLD",
    "confidence": 72,
    "entry_price": 77900,
    "stop_loss": 79750,
    "take_profit": 73114,
    "position_size": 0.0,
    "reasoning": "Wait for confirmation before entry.",
    "risk_reward_ratio": 2.58,
    "trend": {
      "direction": "NEUTRAL",
      "strength_4h": 23,
      "strength_daily": 41,
      "timeframe_alignment": "DIVERGENT"
    },
    "confluence_factors": {
      "trend_alignment": 63,
      "momentum_strength": 71,
      "volume_support": 55,
      "pattern_quality": 67,
      "support_resistance_strength": 78
    },
    "key_levels": {
      "support": [77275.0, 76564.0],
      "resistance": [78930.57, 79515.0]
    }
  }
}
```
"""


def _build_verbose_response() -> str:
    return """4) RISK/REWARD:
Signal: HOLD (Waiting for confirmation)
Conditional Entry (Short): $77,900
Stop Loss: $79,750
Take Profit: $73,114
R/R Ratio: 2.58

```json
{
  "analysis": {
    "signal": "HOLD",
    "confidence": 72,
    "entry_price": 77900,
    "stop_loss": 79750,
    "take_profit": 73114,
    "position_size": 0.0,
    "reasoning": "Wait for confirmation before entry.",
    "risk_reward_ratio": 2.58
  }
}
```
"""


def test_unified_parser_parses_compact_response_with_json_block() -> None:
    parser = UnifiedParser(logger=MagicMock())
    parsed = parser.parse_ai_response(_build_compact_response())

    analysis = parsed["analysis"]
    assert analysis["signal"] == "HOLD"
    assert analysis["confidence"] == 72
    assert analysis["entry_price"] == 77900
    assert analysis["stop_loss"] == 79750
    assert analysis["take_profit"] == 73114
    assert analysis["position_size"] == 0.0
    assert analysis["risk_reward_ratio"] == 2.58


def test_position_extractor_uses_json_for_compact_response() -> None:
    parser = UnifiedParser(logger=MagicMock())
    extractor = PositionExtractor(logger=MagicMock(), unified_parser=parser)

    signal, confidence, stop_loss, take_profit, position_size, reasoning = extractor.extract_trading_info(
        _build_compact_response()
    )

    assert signal == "HOLD"
    assert confidence == "HIGH"
    assert stop_loss == 79750.0
    assert take_profit == 73114.0
    assert position_size == 0.0
    assert reasoning == "Wait for confirmation before entry."


def test_position_extractor_remains_compatible_with_verbose_response() -> None:
    parser = UnifiedParser(logger=MagicMock())
    extractor = PositionExtractor(logger=MagicMock(), unified_parser=parser)

    signal, confidence, stop_loss, take_profit, position_size, reasoning = extractor.extract_trading_info(
        _build_verbose_response()
    )

    assert signal == "HOLD"
    assert confidence == "HIGH"
    assert stop_loss == 79750.0
    assert take_profit == 73114.0
    assert position_size == 0.0
    assert reasoning == "Wait for confirmation before entry."
