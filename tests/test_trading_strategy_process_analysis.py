"""Integration-style regression tests for TradingStrategy.process_analysis."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.parsing.unified_parser import UnifiedParser
from src.trading.position_extractor import PositionExtractor
from src.trading.trading_strategy import TradingStrategy


def _make_compact_buy_response() -> str:
    return """1) MARKET STRUCTURE: ALIGNED uptrend.
2) INDICATOR ASSESSMENT: Momentum and volume support continuation.
4) DECISION: BUY with managed risk.

```json
{
  "analysis": {
    "signal": "BUY",
    "confidence": 82,
    "entry_price": 77880,
    "stop_loss": 76500,
    "take_profit": 80640,
    "position_size": 0.42,
    "reasoning": "Trend continuation with confluence.",
    "risk_reward_ratio": 2.0,
    "trend": {
      "direction": "BULLISH",
      "strength_4h": 68,
      "strength_daily": 61,
      "timeframe_alignment": "ALIGNED"
    },
    "confluence_factors": {
      "trend_alignment": 80,
      "momentum_strength": 77,
      "volume_support": 71,
      "pattern_quality": 64,
      "support_resistance_strength": 73
    }
  }
}
```
"""


def _make_compact_hold_response() -> str:
    return """1) MARKET STRUCTURE: MIXED.
4) DECISION: HOLD until confirmation.

```json
{
  "analysis": {
    "signal": "HOLD",
    "confidence": 69,
    "entry_price": 77900,
    "stop_loss": 79750,
    "take_profit": 73114,
    "position_size": 0.0,
    "reasoning": "No clear edge yet.",
    "risk_reward_ratio": 2.58
  }
}
```
"""


def _build_strategy() -> TradingStrategy:
    strategy = TradingStrategy.__new__(TradingStrategy)
    strategy.logger = MagicMock()
    strategy.persistence = MagicMock()
    strategy.persistence.async_save_position = AsyncMock()
    strategy.persistence.async_save_trade_decision = AsyncMock()
    strategy.persistence.load_position = MagicMock(return_value=None)

    strategy.brain_service = MagicMock()
    strategy.statistics_service = MagicMock()
    strategy.statistics_service.get_current_capital = MagicMock(return_value=10000.0)
    strategy.memory_service = MagicMock()

    strategy.risk_manager = MagicMock()
    strategy.position_factory = MagicMock()
    strategy.position_factory.create_position = MagicMock(return_value=SimpleNamespace(id="pos-1"))

    strategy.config = SimpleNamespace(
        DEMO_QUOTE_CAPITAL=10000.0,
        TIMEFRAME="4h",
        STOP_LOSS_TYPE="soft",
        STOP_LOSS_CHECK_INTERVAL="4h",
        TAKE_PROFIT_TYPE="soft",
        TAKE_PROFIT_CHECK_INTERVAL="4h",
    )
    strategy.current_position = None

    parser = UnifiedParser(logger=MagicMock())
    strategy.extractor = PositionExtractor(logger=MagicMock(), unified_parser=parser)
    return strategy


@pytest.mark.asyncio
async def test_process_analysis_compact_buy_uses_json_fields_for_risk_inputs() -> None:
    strategy = _build_strategy()

    risk_assessment = SimpleNamespace(
        stop_loss=76480.0,
        take_profit=80720.0,
        size_pct=0.40,
        quantity=0.051,
        entry_fee=2.97,
        sl_distance_pct=0.018,
        tp_distance_pct=0.036,
        rr_ratio=2.0,
        quote_amount=4000.0,
    )
    strategy.risk_manager.calculate_entry_parameters = MagicMock(return_value=risk_assessment)

    analysis_result = {
        "raw_response": _make_compact_buy_response(),
        "current_price": 77880.0,
        "analysis": {
            "trend": {
                "direction": "BULLISH",
                "strength_4h": 68,
                "timeframe_alignment": "ALIGNED",
            },
            "confluence_factors": {
                "trend_alignment": 80,
                "momentum_strength": 77,
            },
        },
        "technical_data": {
            "adx": 26,
            "rsi": 62,
            "atr_percentage": 1.8,
            "macd": {"signal": "BULLISH"},
            "volume": {"state": "ACCUMULATION"},
        },
    }

    decision = await strategy.process_analysis(analysis_result, "BTC/USDC")

    assert decision is not None
    assert decision.action == "BUY"
    assert decision.confidence == "HIGH"
    assert decision.price == 77880.0
    assert decision.stop_loss == 76480.0
    assert decision.take_profit == 80720.0
    assert decision.position_size == 0.40
    assert decision.reasoning == "Trend continuation with confluence."

    strategy.risk_manager.calculate_entry_parameters.assert_called_once()
    kwargs = strategy.risk_manager.calculate_entry_parameters.call_args.kwargs
    assert kwargs["signal"] == "BUY"
    assert kwargs["confidence"] == "HIGH"
    assert kwargs["stop_loss"] == 76500.0
    assert kwargs["take_profit"] == 80640.0
    assert kwargs["position_size"] == 0.42
    assert kwargs["current_price"] == 77880.0
    assert kwargs["capital"] == 10000.0

    strategy.persistence.async_save_trade_decision.assert_awaited_once()
    strategy.persistence.async_save_position.assert_awaited_once()
    assert strategy.current_position is not None


@pytest.mark.asyncio
async def test_process_analysis_compact_hold_does_not_open_position() -> None:
    strategy = _build_strategy()
    strategy.risk_manager.calculate_entry_parameters = MagicMock()

    analysis_result = {
        "raw_response": _make_compact_hold_response(),
        "current_price": 77910.0,
        "analysis": {
            "trend": {
                "direction": "NEUTRAL",
                "strength_4h": 32,
                "timeframe_alignment": "MIXED",
            }
        },
    }

    decision = await strategy.process_analysis(analysis_result, "BTC/USDC")

    assert decision is None
    strategy.risk_manager.calculate_entry_parameters.assert_not_called()
    strategy.persistence.async_save_trade_decision.assert_not_awaited()
    strategy.persistence.async_save_position.assert_not_awaited()
    assert strategy.current_position is None