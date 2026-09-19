"""Consolidated TradingStrategy decision-flow suite.

Covers the whole analysis-to-position path in one module: process_analysis
(parsed payload -> risk inputs -> persistence side effects), the SL/TP update
gate, the close path (P&L, memory, post-mortem, dashboard lifecycle), entry
sizing with the executor notional clamp, the price/market-conditions/confluence
readers, check_position exit detection, the parser -> PositionExtractor trading
field contract, the response-contract metadata and the trading-memory context
summary. Every double comes from tests/conftest.py.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone, tzinfo
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.analyzer.prompts.template_manager import TemplateManager
from src.dashboard.dashboard_state import DashboardState
from src.parsing.unified_parser import UnifiedParser
from src.trading.data_models import (
    MarketConditions,
    Position,
    TradeDecision,
    TradingMemory,
)
from src.trading.market_conditions_extractor import MarketConditionsExtractor
from src.trading.position_extractor import PositionExtractor
from src.trading.stop_loss_tightening_policy import StopLossTighteningPolicy
from src.trading.trading_strategy import TradingStrategy
from src.utils.format_utils import FormatUtils
from tests.conftest import (
    make_config,
    make_market_conditions,
    make_position,
    mock_brain,
    mock_persistence,
    mock_statistics,
    null_logger,
)

_PARSER = UnifiedParser(logger=null_logger(), format_utils=FormatUtils())

_COMPACT_BUY = """1) MARKET STRUCTURE: ALIGNED uptrend.
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

_COMPACT_HOLD = """1) MARKET STRUCTURE: MIXED trend with weak momentum.
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
    "trend": {"direction": "NEUTRAL", "strength_4h": 23, "timeframe_alignment": "DIVERGENT"},
    "confluence_factors": {"trend_alignment": 63, "momentum_strength": 71},
    "key_levels": {"support": [77275.0, 76564.0], "resistance": [78930.57, 79515.0]}
  }
}
```
"""

_VERBOSE_HOLD = """4) RISK/REWARD:
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

_NARRATIVE_ONLY = (
    "signal: BUY confidence: HIGH stop_loss: 95 take_profit: 110 position_size: 5%"
)

_ANALYSIS_DEFAULTS: dict[str, Any] = {
    "signal": "BUY",
    "confidence": 80,
    "entry_price": 100.0,
    "stop_loss": 95.0,
    "take_profit": 110.0,
    "position_size": 0.05,
    "reasoning": "Breakout continuation.",
    "risk_reward_ratio": 2.0,
}

_BUY_RESULT: dict[str, Any] = {"analysis": dict(_ANALYSIS_DEFAULTS), "current_price": 100.0}

_SMALL_SIZE: dict[str, Any] = {
    "quantity": 0.5,
    "size_pct": 0.005,
    "quote_amount": 50.0,
    "entry_fee": 0.05,
}


def _parse(raw_text: str) -> dict[str, Any]:
    """Parse a raw AI reply through the production parser (validation attached)."""
    return _PARSER.parse_ai_response(raw_text)


def _parse_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Parse a fenced JSON payload and return the normalized analysis dict."""
    return _parse(f"```json\n{json.dumps(payload)}\n```")["analysis"]


def _analysis_payload(**overrides: Any) -> dict[str, Any]:
    """Analysis payload carrying the valid-entry defaults plus explicit overrides."""
    values = dict(_ANALYSIS_DEFAULTS)
    values.update(overrides)
    return {"analysis": values}


def _risk(**overrides: Any) -> SimpleNamespace:
    """RiskManager result double: a valid 5x100 entry at 2.4 R/R unless overridden."""
    values: dict[str, Any] = {
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "take_profit": 112.0,
        "size_pct": 0.05,
        "quantity": 5.0,
        "entry_fee": 0.5,
        "sl_distance_pct": 0.05,
        "tp_distance_pct": 0.12,
        "rr_ratio": 2.4,
        "quote_amount": 500.0,
        "volatility_level": "MEDIUM",
        "regime_profile": "neutral",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _strategy(
    *,
    position: Position | None = None,
    policy: StopLossTighteningPolicy | None = None,
    extractor: Any = None,
    config: Any = None,
    thresholds: dict[str, Any] | None = None,
    capital: float = 10000.0,
) -> tuple[TradingStrategy, MagicMock, MagicMock, MagicMock, MagicMock]:
    """Real TradingStrategy on the shared doubles; returns strategy and collaborators.

    Order of the returned tuple: strategy, logger, persistence, brain, statistics.
    """
    logger = null_logger()
    persistence = mock_persistence()
    persistence.load_position.return_value = position
    brain = mock_brain()
    brain.get_dynamic_thresholds.return_value = (
        {"rr_borderline_min": 1.5} if thresholds is None else thresholds
    )
    statistics = mock_statistics()
    statistics.get_current_capital.return_value = capital
    risk_manager = MagicMock()
    risk_manager.get_and_clear_frictions.return_value = []
    strategy = TradingStrategy(
        logger=logger,
        persistence=persistence,
        brain_service=brain,
        statistics_service=statistics,
        memory_service=MagicMock(),
        risk_manager=risk_manager,
        config=config if config is not None else make_config(),
        position_extractor=extractor if extractor is not None else PositionExtractor(),
        conditions_extractor=MarketConditionsExtractor(logger),
        tightening_policy=policy,
    )
    return strategy, logger, persistence, brain, statistics


def _sync_brain_hook(brain: MagicMock) -> MagicMock:
    """The close path calls update_from_closed_trade through asyncio.to_thread."""
    brain.update_from_closed_trade = MagicMock()
    return brain


def _decision(**overrides: Any) -> TradeDecision:
    """TradeDecision on a fixed UTC stamp, with the unused fields zeroed."""
    values: dict[str, Any] = {
        "timestamp": datetime(2026, 4, 14, 0, 0, tzinfo=timezone.utc),
        "symbol": "BTC/USDC",
        "action": "HOLD",
        "confidence": "HIGH",
        "price": 74000.0,
        "quantity": 0.0,
        "reasoning": "",
    }
    values.update(overrides)
    return TradeDecision(**values)


class TestProcessAnalysisFlow:
    """The analysis -> risk -> persistence path of process_analysis."""

    async def test_buy_flow_passes_parsed_fields_to_risk_and_persistence(self) -> None:
        """A compact BUY reply drives the risk call, the decision and the entry writes."""
        strategy, _, persistence, _, _ = _strategy()
        strategy.risk_manager.calculate_entry_parameters.return_value = _risk(
            entry_price=77880.0,
            stop_loss=76480.0,
            take_profit=80720.0,
            size_pct=0.40,
            quantity=0.051,
            entry_fee=2.97,
            sl_distance_pct=0.018,
            tp_distance_pct=0.036,
            rr_ratio=2.0,
            quote_amount=4000.0,
            volatility_level="HIGH",
            regime_profile="aggressive",
        )
        analysis_result = {
            "raw_response": _COMPACT_BUY,
            "current_price": 77880.0,
            "analysis": _parse(_COMPACT_BUY)["analysis"],
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
        assert (decision.action, decision.confidence) == ("BUY", "HIGH")
        assert (decision.price, decision.stop_loss, decision.take_profit) == (
            77880.0,
            76480.0,
            80720.0,
        )
        assert decision.position_size == 0.40
        assert decision.reasoning == "Trend continuation with confluence."
        assert decision.order_id is not None and decision.order_id.startswith("order-")

        strategy.risk_manager.calculate_entry_parameters.assert_called_once()
        kwargs = strategy.risk_manager.calculate_entry_parameters.call_args.kwargs
        assert (kwargs["signal"], kwargs["confidence"]) == ("BUY", "HIGH")
        assert (kwargs["stop_loss"], kwargs["take_profit"]) == (76500.0, 80640.0)
        assert kwargs["position_size"] == 0.42
        assert (kwargs["current_price"], kwargs["capital"]) == (77880.0, 10000.0)
        assert kwargs["market_conditions"].adx == 26.0
        assert kwargs["choppiness"] is None

        persistence.async_save_trade_decision.assert_awaited_once_with(decision)
        persistence.async_save_position.assert_awaited_once()
        assert persistence.async_save_position.await_args.args[0] is strategy.current_position
        assert strategy.current_position is not None
        assert strategy.current_position.symbol == "BTC/USDC"

    async def test_buy_flow_pins_the_entry_snapshot_and_confluence_factors(self) -> None:
        """Market conditions reach the risk manager and stick to the Position verbatim."""
        strategy, _, _, _, _ = _strategy()
        strategy.risk_manager.calculate_entry_parameters.return_value = _risk(
            entry_price=99.0,
            volatility_level="HIGH",
            regime_profile="conservative",
        )
        analysis_result = {
            "raw_response": _COMPACT_BUY,
            "current_price": 99.0,
            "analysis": _parse(_COMPACT_BUY)["analysis"],
            "technical_data": {
                "adx": 28.0,
                "rsi": 61.0,
                "atr": 120.0,
                "atr_percent": 4.2,
                "macd_line": 2.0,
                "macd_signal": 1.0,
                "obv_slope": 0.8,
                "bb_upper": 100.0,
                "bb_lower": 80.0,
            },
            "sentiment": {"fear_greed_index": 80},
            "market_microstructure": {"order_book": {"imbalance": 0.25}},
        }

        await strategy.process_analysis(analysis_result, "BTC/USDC")

        conditions = strategy.risk_manager.calculate_entry_parameters.call_args.kwargs[
            "market_conditions"
        ]
        assert (conditions.adx, conditions.rsi, conditions.rsi_level) == (
            28.0,
            61.0,
            "STRONG",
        )
        assert (conditions.volatility, conditions.macd_signal, conditions.volume_state) == (
            "HIGH",
            "BULLISH",
            "ACCUMULATION",
        )
        assert (
            conditions.bb_position,
            conditions.market_sentiment,
            conditions.order_book_bias,
        ) == ("UPPER", "EXTREME_GREED", "BUY_PRESSURE")
        assert (conditions.atr, conditions.atr_percentage) == (120.0, 4.2)
        assert conditions.fear_greed_index == 80

        position = strategy.current_position
        assert position is not None
        assert position.conditions_at_entry is conditions
        assert (position.adx_at_entry, position.rsi_at_entry, position.atr_at_entry) == (
            28.0,
            61.0,
            120.0,
        )
        assert position.confluence_factors == (
            ("trend_alignment", 80.0),
            ("momentum_strength", 77.0),
            ("volume_support", 71.0),
            ("pattern_quality", 64.0),
            ("support_resistance_strength", 73.0),
        )

    @pytest.mark.parametrize(
        "raw_response",
        [_COMPACT_HOLD, _VERBOSE_HOLD],
        ids=["compact", "verbose"],
    )
    async def test_hold_signals_take_no_action(self, raw_response: str) -> None:
        """A parsed HOLD reply returns None and touches no dependency."""
        strategy, _, persistence, _, _ = _strategy()
        analysis_result = {
            "raw_response": raw_response,
            "current_price": 77910.0,
            "analysis": _parse(raw_response)["analysis"],
        }

        assert await strategy.process_analysis(analysis_result, "BTC/USDC") is None

        strategy.risk_manager.calculate_entry_parameters.assert_not_called()
        persistence.async_save_trade_decision.assert_not_awaited()
        persistence.async_save_position.assert_not_awaited()
        assert strategy.current_position is None

    async def test_reply_without_a_json_block_degrades_to_hold(self) -> None:
        """No regex rescue is left: a narrative-only reply must degrade, never guess."""
        parsed = _parse(_NARRATIVE_ONLY)
        assert parsed["parse_error"] == "Failed to parse response"
        assert PositionExtractor().extract_trading_info(parsed["analysis"]) == (
            "HOLD",
            "MEDIUM",
            None,
            None,
            None,
            "",
        )
        strategy, _, persistence, _, _ = _strategy()

        decision = await strategy.process_analysis(
            {"raw_response": _NARRATIVE_ONLY, "current_price": 95.0, **parsed},
            "BTC/USDC",
        )

        assert decision is None
        strategy.risk_manager.calculate_entry_parameters.assert_not_called()
        persistence.async_save_position.assert_not_awaited()
        assert strategy.current_position is None

    @pytest.mark.parametrize(
        "result",
        [
            pytest.param({}, id="empty-result"),
            pytest.param({"analysis": {}, "current_price": 100.0}, id="empty-analysis"),
            pytest.param(
                {k: v for k, v in _BUY_RESULT.items() if k != "current_price"},
                id="missing-price-key",
            ),
            pytest.param({**_BUY_RESULT, "current_price": 0.0}, id="zero-price"),
            pytest.param({**_BUY_RESULT, "current_price": -1.0}, id="negative-price"),
            pytest.param({**_BUY_RESULT, "current_price": float("nan")}, id="nan-price"),
            pytest.param({**_BUY_RESULT, "current_price": float("inf")}, id="infinite-price"),
            pytest.param({**_BUY_RESULT, "current_price": "n/a"}, id="unparseable-price"),
        ],
    )
    async def test_aborts_without_a_usable_price(self, result: dict[str, Any]) -> None:
        """Missing, non-finite, non-positive or corrupt prices stop the cycle."""
        strategy, _, persistence, _, _ = _strategy()
        strategy.risk_manager.calculate_entry_parameters.return_value = _risk()

        assert await strategy.process_analysis(result, "BTC/USDC") is None

        strategy.risk_manager.calculate_entry_parameters.assert_not_called()
        persistence.async_save_trade_decision.assert_not_awaited()
        persistence.async_save_position.assert_not_awaited()
        assert strategy.current_position is None

    async def test_a_conditions_extraction_crash_drops_the_cycle(self) -> None:
        """A null ADX raises inside the conditions reader and the whole cycle returns None."""
        strategy, _, persistence, _, _ = _strategy()
        strategy.risk_manager.calculate_entry_parameters.return_value = _risk()
        result = {
            "analysis": dict(_ANALYSIS_DEFAULTS),
            "current_price": 100.0,
            "technical_data": {"adx": None},
        }

        assert await strategy.process_analysis(result, "BTC/USDC") is None

        strategy.risk_manager.calculate_entry_parameters.assert_not_called()
        persistence.async_save_position.assert_not_awaited()
        assert strategy.current_position is None

    @pytest.mark.parametrize(
        "analysis",
        [
            pytest.param({**_ANALYSIS_DEFAULTS, "signal": "MAYBE"}, id="unknown-signal"),
            pytest.param({**_ANALYSIS_DEFAULTS, "signal": "CLOSE"}, id="close-without-position"),
            pytest.param(
                {key: value for key, value in _ANALYSIS_DEFAULTS.items() if key != "signal"},
                id="missing-signal",
            ),
        ],
    )
    async def test_drops_signals_it_cannot_execute(self, analysis: dict[str, Any]) -> None:
        """Signals outside the entry set (or absent) are logged and ignored."""
        strategy, _, persistence, _, _ = _strategy()
        strategy.risk_manager.calculate_entry_parameters.return_value = _risk()

        assert (
            await strategy.process_analysis(
                {"analysis": analysis, "current_price": 100.0}, "BTC/USDC"
            )
            is None
        )

        strategy.risk_manager.calculate_entry_parameters.assert_not_called()
        persistence.async_save_position.assert_not_awaited()
        assert strategy.current_position is None

    async def test_unknown_keys_and_wrong_typed_risk_fields_open_degraded(self) -> None:
        """Unconvertible risk fields become None and unknown keys are ignored, not fatal."""
        analysis = _parse_payload(
            _analysis_payload(
                stop_loss="not-a-number",
                take_profit=None,
                position_size={},
                reasoning="",
                confluence_factors=["not", "a", "dict"],
                unexpected_nested={"levels": [1, 2, 3]},
            )
        )
        assert (
            analysis["stop_loss"],
            analysis["take_profit"],
            analysis["position_size"],
        ) == (None, None, None)

        strategy, _, _, _, _ = _strategy()
        strategy.risk_manager.calculate_entry_parameters.return_value = _risk()

        decision = await strategy.process_analysis(
            {"analysis": analysis, "current_price": 100.0}, "BTC/USDC"
        )

        assert decision is not None
        assert decision.action == "BUY"
        assert decision.reasoning == ""
        kwargs = strategy.risk_manager.calculate_entry_parameters.call_args.kwargs
        assert (kwargs["stop_loss"], kwargs["take_profit"], kwargs["position_size"]) == (
            None,
            None,
            None,
        )
        position = strategy.current_position
        assert position is not None
        assert position.confluence_factors == ()


class TestUpdatePositionParameters:
    """The single gate that decides every SL/TP proposal."""

    @pytest.mark.parametrize(
        (
            "position_kwargs",
            "policy",
            "stop_loss",
            "take_profit",
            "current_price",
            "updated",
            "expected_sl",
            "expected_tp",
            "warning",
        ),
        [
            pytest.param(
                {"entry_price": 100.0, "stop_loss": 95.0, "take_profit": 110.0},
                StopLossTighteningPolicy(swing_threshold=0.20),
                95.0,
                110.0,
                105.0,
                False,
                95.0,
                110.0,
                None,
                id="unchanged",
            ),
            pytest.param(
                {"entry_price": 100.0, "stop_loss": 95.0, "take_profit": 110.0},
                StopLossTighteningPolicy(swing_threshold=0.20),
                99.0,
                None,
                101.5,
                False,
                95.0,
                110.0,
                None,
                id="premature-tightening",
            ),
            pytest.param(
                {"entry_price": 100.0, "stop_loss": 95.0, "take_profit": 110.0},
                StopLossTighteningPolicy(swing_threshold=0.15),
                99.0,
                None,
                102.0,
                True,
                99.0,
                110.0,
                None,
                id="allowed-tightening",
            ),
            pytest.param(
                {"entry_price": 100.0, "stop_loss": 95.0, "take_profit": 110.0},
                StopLossTighteningPolicy(swing_threshold=0.20),
                92.5,
                None,
                105.0,
                True,
                92.5,
                110.0,
                None,
                id="long-widening-exactly-at-cap",
            ),
            pytest.param(
                {"entry_price": 100.0, "stop_loss": 95.0, "take_profit": 110.0},
                StopLossTighteningPolicy(swing_threshold=0.20),
                93.0,
                None,
                105.0,
                True,
                93.0,
                110.0,
                None,
                id="long-widening",
            ),
            pytest.param(
                {"entry_price": 100.0, "stop_loss": 95.0, "take_profit": 110.0},
                StopLossTighteningPolicy(swing_threshold=0.20),
                92.0,
                None,
                105.0,
                False,
                95.0,
                110.0,
                "REJECTED SL widening",
                id="long-widening-beyond-cap",
            ),
            pytest.param(
                {"entry_price": 100.0, "stop_loss": 105.0, "take_profit": 90.0, "direction": "SHORT"},
                StopLossTighteningPolicy(swing_threshold=0.20),
                107.0,
                None,
                100.0,
                True,
                107.0,
                90.0,
                None,
                id="short-widening",
            ),
            pytest.param(
                {"entry_price": 100.0, "stop_loss": 105.0, "take_profit": 90.0, "direction": "SHORT"},
                StopLossTighteningPolicy(swing_threshold=0.20),
                108.0,
                None,
                100.0,
                False,
                105.0,
                90.0,
                "REJECTED SL widening",
                id="short-widening-beyond-cap",
            ),
            pytest.param(
                {"entry_price": 100.0, "stop_loss": 105.0, "take_profit": 90.0, "direction": "SHORT"},
                StopLossTighteningPolicy(swing_threshold=0.20),
                103.0,
                None,
                99.0,
                False,
                105.0,
                90.0,
                None,
                id="short-tightening-rejected",
            ),
            pytest.param(
                {"entry_price": 100.0, "stop_loss": 95.0, "take_profit": 110.0},
                StopLossTighteningPolicy(swing_threshold=0.20),
                None,
                115.0,
                105.0,
                True,
                95.0,
                115.0,
                None,
                id="take-profit-only",
            ),
            pytest.param(
                {"entry_price": 100.0, "stop_loss": 95.0, "take_profit": 110.0},
                StopLossTighteningPolicy(swing_threshold=0.20),
                93.0,
                115.0,
                105.0,
                True,
                93.0,
                115.0,
                None,
                id="stop-loss-and-take-profit",
            ),
            pytest.param(
                {"entry_price": 100.0, "stop_loss": 95.0, "take_profit": 100.0},
                StopLossTighteningPolicy(swing_threshold=0.20),
                99.0,
                None,
                102.0,
                False,
                95.0,
                100.0,
                None,
                id="entry-equals-take-profit",
            ),
            pytest.param(
                {"entry_price": 100.0, "stop_loss": 95.0, "take_profit": 110.0},
                StopLossTighteningPolicy(swing_threshold=0.20),
                96.0,
                None,
                None,
                False,
                95.0,
                110.0,
                None,
                id="missing-current-price",
            ),
        ],
    )
    async def test_each_proposal_is_gated_and_persisted(
        self,
        position_kwargs: dict[str, Any],
        policy: StopLossTighteningPolicy,
        stop_loss: float | None,
        take_profit: float | None,
        current_price: float | None,
        updated: bool,
        expected_sl: float,
        expected_tp: float,
        warning: str | None,
    ) -> None:
        """Tightening, widening caps, TP changes and the price guard in one matrix."""
        strategy, logger, persistence, _, _ = _strategy(
            position=make_position(**{"direction": "LONG", **position_kwargs}),
            policy=policy,
            config=make_config(TIMEFRAME="4h"),
        )

        result = await strategy._update_position_parameters(
            stop_loss=stop_loss, take_profit=take_profit, current_price=current_price
        )

        assert result is updated
        current = strategy.current_position
        assert current is not None
        assert (current.stop_loss, current.take_profit) == (expected_sl, expected_tp)
        assert persistence.async_save_position.await_count == (1 if updated else 0)
        logged = [str(call) for call in logger.warning.call_args_list]
        if warning is not None:
            assert any(warning in call for call in logged)
        else:
            assert not any("REJECTED SL widening" in call for call in logged)

    async def test_without_a_position_it_is_a_noop(self) -> None:
        """No open position means no evaluation and no write."""
        strategy, _, persistence, _, _ = _strategy()

        assert (
            await strategy._update_position_parameters(
                stop_loss=90.0, take_profit=120.0, current_price=105.0
            )
            is False
        )
        persistence.async_save_position.assert_not_awaited()


class TestClosePosition:
    """Closing a position: P&L bookkeeping, learning hooks and failure tolerance."""

    @pytest.mark.parametrize(
        ("direction", "close_price", "reason", "expected_action", "expected_pnl", "expected_fee"),
        [
            pytest.param("LONG", 94.0, "stop_loss", "CLOSE_LONG", -6.00, 0.000705, id="long-stop-loss"),
            pytest.param("LONG", 115.0, "take_profit", "CLOSE_LONG", 15.00, 0.0008625, id="long-take-profit"),
            pytest.param("SHORT", 106.0, "stop_loss", "CLOSE_SHORT", -6.00, 0.000795, id="short-stop-loss"),
            pytest.param("SHORT", 85.0, "take_profit", "CLOSE_SHORT", 15.00, 0.0006375, id="short-take-profit"),
        ],
    )
    async def test_records_pnl_and_updates_dependencies(
        self,
        direction: str,
        close_price: float,
        reason: str,
        expected_action: str,
        expected_pnl: float,
        expected_fee: float,
    ) -> None:
        """The close decision carries the right side, P&L, fee and exit reason."""
        position = make_position(
            direction=direction,
            entry_price=100.0,
            stop_loss=95.0 if direction == "LONG" else 105.0,
            take_profit=115.0 if direction == "LONG" else 85.0,
        )
        strategy, _, persistence, brain, statistics = _strategy(position=position)
        _sync_brain_hook(brain)
        conditions = make_market_conditions(trend_direction="BULLISH", adx=40.0)

        await strategy.close_position(reason, close_price, conditions)

        persistence.async_save_trade_decision.assert_awaited_once()
        decision = persistence.async_save_trade_decision.await_args.args[0]
        assert decision.action == expected_action
        assert decision.symbol == "BTC/USDT"
        assert decision.reasoning == (
            f"Position closed: {reason}. P&L: {expected_pnl:+.2f}%. "
            f"Fee: ${expected_fee:.4f}"
        )
        assert decision.fee == pytest.approx(expected_fee)
        assert (decision.price, decision.stop_loss, decision.take_profit) == (
            close_price,
            position.stop_loss,
            position.take_profit,
        )
        assert (decision.quantity, decision.position_size) == (
            position.size,
            position.size_pct,
        )
        strategy.memory_service.add_decision.assert_called_once_with(decision)
        brain.update_from_closed_trade.assert_called_once()
        assert brain.update_from_closed_trade.call_args.kwargs["market_conditions"] is conditions
        assert brain.update_from_closed_trade.call_args.kwargs["close_reason"] == reason
        statistics.recalculate.assert_called_once_with(10000.0)
        persistence.async_save_position.assert_awaited_with(None)
        assert strategy.current_position is None

    @pytest.mark.parametrize("failure", ["none", "brain", "stats"])
    async def test_lifecycle_survives_a_dependency_failure(self, failure: str) -> None:
        """A crashing brain or statistics service never blocks the close."""
        strategy, _, persistence, brain, statistics = _strategy(
            position=make_position(direction="LONG")
        )
        _sync_brain_hook(brain)
        dashboard_state = DashboardState()
        dashboard_state.mark_brain_rebuild_started = AsyncMock()
        dashboard_state.mark_brain_rebuild_completed = AsyncMock()
        dashboard_state.mark_brain_rebuild_failed = AsyncMock()
        strategy.set_dashboard_state(dashboard_state)
        if failure == "brain":
            brain.update_from_closed_trade.side_effect = RuntimeError("Brain crash")
        if failure == "stats":
            statistics.recalculate.side_effect = RuntimeError("Stats crash")

        await strategy.close_position("take_profit", 110.0, make_market_conditions())

        assert strategy.current_position is None
        persistence.async_save_position.assert_awaited_with(None)
        dashboard_state.mark_brain_rebuild_started.assert_awaited_once_with(
            "Learning from closed LONG trade"
        )
        if failure == "brain":
            dashboard_state.mark_brain_rebuild_completed.assert_not_awaited()
            dashboard_state.mark_brain_rebuild_failed.assert_awaited_once_with(
                "Brain rebuild failed after trade close"
            )
        else:
            dashboard_state.mark_brain_rebuild_completed.assert_awaited_once_with(
                "Brain state rebuilt from closed trade"
            )
            dashboard_state.mark_brain_rebuild_failed.assert_not_awaited()

    @pytest.mark.parametrize(
        "entry_decision_present",
        [True, False],
        ids=["entry-decision", "no-entry-decision"],
    )
    async def test_post_mortem_wiring(self, entry_decision_present: bool) -> None:
        """The CLOSE row id reaches the post-mortem journal, unless the entry is missing."""
        position = make_position(direction="LONG")
        strategy, _, persistence, _, _ = _strategy(position=position)
        persistence.async_save_trade_decision = AsyncMock(return_value=42)
        entry = MagicMock()
        entry.reasoning = "Expected breakout continuation."
        persistence.get_entry_decision_for_position = MagicMock(
            return_value=entry if entry_decision_present else None
        )
        post_mortem = MagicMock()
        post_mortem.analyze_closed_trade = AsyncMock(return_value=None)
        strategy.post_mortem_service = post_mortem

        await strategy.close_position("stop_loss", 90.0, make_market_conditions())

        assert strategy.current_position is None
        if not entry_decision_present:
            post_mortem.analyze_closed_trade.assert_not_awaited()
            return
        post_mortem.analyze_closed_trade.assert_awaited_once()
        kwargs = post_mortem.analyze_closed_trade.await_args.kwargs
        assert kwargs["trade_id"] == 42
        assert kwargs["pnl"] == pytest.approx(position.calculate_pnl(90.0))
        assert kwargs["reason"] == "stop_loss"
        assert kwargs["entry_decision"] is entry
        assert kwargs["exit_decision"].action == "CLOSE_LONG"

    async def test_without_a_position_it_is_a_noop(self) -> None:
        """An empty book means no decision, no learning and no statistics refresh."""
        strategy, _, persistence, brain, statistics = _strategy()
        _sync_brain_hook(brain)

        await strategy.close_position("stop_loss", 90.0, make_market_conditions())

        brain.update_from_closed_trade.assert_not_called()
        statistics.recalculate.assert_not_called()
        persistence.async_save_trade_decision.assert_not_awaited()
        persistence.async_save_position.assert_not_awaited()


class TestOpenNewPosition:
    """Entry construction: the R/R floor, the recorded decision and size clamping."""

    @pytest.mark.parametrize(
        ("thresholds", "config_min_rr", "rr_ratio", "expected_action"),
        [
            pytest.param({"rr_borderline_min": 2.0}, 1.0, 2.5, "BUY", id="above-both-floors"),
            pytest.param({"rr_borderline_min": 2.0}, 1.0, 1.5, "HOLD", id="brain-raises-the-config-floor"),
            pytest.param({"rr_borderline_min": 2.0}, 2.0, 2.0, "BUY", id="exactly-at-the-floor"),
            pytest.param({}, 1.5, 1.4, "HOLD", id="config-floor-without-a-brain-value"),
            pytest.param({"rr_borderline_min": 0.5}, 1.5, 1.0, "HOLD", id="brain-cannot-lower-the-config-floor"),
        ],
    )
    async def test_rr_floor_is_the_config_value_raised_by_the_brain(
        self,
        thresholds: dict[str, Any],
        config_min_rr: float,
        rr_ratio: float,
        expected_action: str,
    ) -> None:
        """Entries below max(brain floor, config floor) become HOLD and are remembered."""
        strategy, _, _, brain, _ = _strategy(
            thresholds=thresholds,
            config=make_config(MIN_RR_ENTRY=config_min_rr),
        )
        strategy.risk_manager.calculate_entry_parameters.return_value = _risk(
            rr_ratio=rr_ratio
        )

        decision = await strategy._open_new_position(
            signal="BUY",
            confidence="HIGH",
            stop_loss=95.0,
            take_profit=109.0,
            position_size=0.05,
            current_price=100.0,
            symbol="BTC/USDC",
            reasoning="Breakout continuation",
            market_conditions=make_market_conditions(adx=30.0),
        )

        assert decision.action == expected_action
        if expected_action == "HOLD":
            assert strategy.current_position is None
            assert decision.reasoning.startswith("Entry blocked: R/R")
            brain.vector_memory.store_blocked_trade.assert_called_once()
        else:
            position = strategy.current_position
            assert position is not None
            assert position.size == pytest.approx(5.0)
            brain.vector_memory.store_blocked_trade.assert_not_called()

    async def test_records_the_entry_in_persistence_and_memory(self) -> None:
        """A valid entry persists the Position, the decision and the memory entry."""
        strategy, _, persistence, _, _ = _strategy()
        strategy.risk_manager.calculate_entry_parameters.return_value = _risk()
        conditions = make_market_conditions(adx=30.0)

        decision = await strategy._open_new_position(
            signal="BUY",
            confidence="HIGH",
            stop_loss=95.0,
            take_profit=112.0,
            position_size=0.05,
            current_price=100.0,
            symbol="BTC/USDC",
            reasoning="Breakout continuation",
            market_conditions=conditions,
            confluence_factors=(("trend_alignment", 80.0),),
        )

        persistence.async_save_trade_decision.assert_awaited_once_with(decision)
        strategy.memory_service.add_decision.assert_called_once_with(decision)
        assert (decision.action, decision.confidence, decision.price) == ("BUY", "HIGH", 100.0)
        assert (decision.stop_loss, decision.take_profit) == (95.0, 112.0)
        assert (
            decision.quantity,
            decision.quote_amount,
            decision.position_size,
            decision.fee,
        ) == (5.0, 500.0, 0.05, 0.5)
        assert decision.indicators_json["adx_at_entry"] == 30.0
        assert decision.indicators_json["rr_ratio_at_entry"] == 2.4

        persistence.async_save_position.assert_awaited_once()
        position = persistence.async_save_position.await_args.args[0]
        assert position is strategy.current_position
        assert (position.direction, position.symbol, position.entry_price) == (
            "LONG",
            "BTC/USDC",
            100.0,
        )
        assert (position.size, position.size_pct, position.quote_amount) == (5.0, 0.05, 500.0)
        assert position.conditions_at_entry is conditions
        assert position.confluence_factors == (("trend_alignment", 80.0),)

    @pytest.mark.parametrize(
        ("executor_max", "risk_overrides", "expected", "clamped"),
        [
            pytest.param(100.0, {}, (1.0, 100.0, 0.01, 0.1), True, id="scaled-to-cap"),
            pytest.param(100.0, _SMALL_SIZE, (0.5, 50.0, 0.005, 0.05), False, id="under-cap"),
            pytest.param(500.0, {}, (5.0, 500.0, 0.05, 0.5), False, id="exactly-at-cap"),
            pytest.param(0.0, {}, (5.0, 500.0, 0.05, 0.5), False, id="cap-disabled"),
        ],
    )
    async def test_executor_notional_clamp(
        self,
        executor_max: float,
        risk_overrides: dict[str, Any],
        expected: tuple[float, float, float, float],
        clamped: bool,
    ) -> None:
        """Notional above the cap is scaled down; at or below it sizing is untouched."""
        strategy, logger, _, _, _ = _strategy(
            config=make_config(EXECUTOR_MAX_POSITION_USDC=executor_max)
        )
        strategy.risk_manager.calculate_entry_parameters.return_value = _risk(
            **risk_overrides
        )

        decision = await strategy._open_new_position(
            signal="BUY",
            confidence="HIGH",
            stop_loss=95.0,
            take_profit=112.0,
            position_size=risk_overrides.get("size_pct", 0.05),
            current_price=100.0,
            symbol="BTC/USDC",
            reasoning="Breakout continuation",
            market_conditions=make_market_conditions(adx=30.0),
        )

        assert decision.quantity == pytest.approx(expected[0])
        assert decision.quote_amount == pytest.approx(expected[1])
        assert decision.position_size == pytest.approx(expected[2])
        assert decision.fee == pytest.approx(expected[3])
        position = strategy.current_position
        assert position is not None
        assert (position.size, position.quote_amount, position.size_pct) == pytest.approx(
            expected[:3]
        )
        clamp_logs = [
            str(call) for call in logger.warning.call_args_list if "Executor notional clamp" in str(call)
        ]
        assert len(clamp_logs) == (1 if clamped else 0)


class TestConditionsAndPriceExtraction:
    """The pure readers behind the decision flow."""

    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            pytest.param({"current_price": 50000.0}, 50000.0, id="current-price-key"),
            pytest.param({"current_price": "50000"}, 50000.0, id="numeric-string"),
            pytest.param(
                {"context": SimpleNamespace(current_price=42500.0)},
                42500.0,
                id="context-object",
            ),
            pytest.param({"some_other_key": "value"}, 0.0, id="nothing-to-read"),
            pytest.param({"context": None}, 0.0, id="null-context"),
        ],
    )
    def test_extract_price_prefers_key_then_context_then_zero(
        self, result: dict[str, Any], expected: float
    ) -> None:
        """Price lookup order and the silent 0.0 fallback."""
        extractor = MarketConditionsExtractor(null_logger())

        assert extractor.extract_price(result) == expected

    def test_extract_price_propagates_an_unparseable_value(self) -> None:
        """A corrupt price is not swallowed into 0.0: it raises for the caller to guard."""
        extractor = MarketConditionsExtractor(null_logger())

        with pytest.raises(ValueError, match="could not convert string to float"):
            extractor.extract_price({"current_price": "n/a"})

    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            pytest.param(
                {
                    "analysis": {
                        "confluence_factors": {
                            "trend_alignment": 85,
                            "volume_support": 70,
                            "pattern_quality": 60,
                        }
                    }
                },
                (("trend_alignment", 85.0), ("volume_support", 70.0), ("pattern_quality", 60.0)),
                id="valid-factors",
            ),
            pytest.param(
                {"analysis": {"confluence_factors": {"str_score": "75"}}},
                (("str_score", 75.0),),
                id="numeric-strings",
            ),
            pytest.param(
                {"analysis": {"confluence_factors": {"valid": 50, "too_high": 150, "negative": -10}}},
                (("valid", 50.0),),
                id="out-of-range-excluded",
            ),
            pytest.param(
                {"analysis": {"confluence_factors": {"valid": 42, "bad_type": "high", "also_bad": None}}},
                (("valid", 42.0),),
                id="non-numeric-skipped",
            ),
            pytest.param({"analysis": {"confluence_factors": {}}}, (), id="empty-map"),
            pytest.param({"analysis": {"confluence_factors": ["not", "a", "dict"]}}, (), id="wrong-type"),
            pytest.param({"other": "data"}, (), id="missing-analysis"),
        ],
    )
    def test_extract_confluence_factors_filters_and_normalizes(
        self, result: dict[str, Any], expected: tuple[tuple[str, float], ...]
    ) -> None:
        """Only 0-100 numeric scores survive, in payload order."""
        factors = MarketConditionsExtractor(null_logger()).extract_confluence_factors(result)

        assert factors == expected

    @pytest.mark.parametrize(
        ("percent_b", "expected_bb"),
        [(0.97, "UPPER"), (0.75, "MIDDLE"), (0.03, "LOWER")],
        ids=["above-upper-band", "inside-bands", "below-lower-band"],
    )
    def test_extract_market_conditions_maps_the_full_payload(
        self, percent_b: float, expected_bb: str
    ) -> None:
        """Every field of a rich analysis result lands on the snapshot."""
        result = {
            "analysis": {
                "trend": {
                    "direction": "BULLISH",
                    "strength_4h": 45,
                    "timeframe_alignment": "ALIGNED",
                }
            },
            "technical_data": {
                "adx": 42.0,
                "rsi": 65.0,
                "atr": 150.0,
                "atr_percent": 2.5,
                "bollinger_bands": {"percent_b": percent_b},
                "vwap": 101.5,
                "mfi": 62.0,
                "cmf": 0.12,
                "bb_percent_b": 0.71,
                "chandelier_long": 90.0,
                "pfe": 8.0,
                "supertrend_direction": 1.0,
                "choppiness": 44.4,
                "macd": {"signal": "BEARISH"},
                "volume": {"state": "ACCUMULATION"},
            },
            "sentiment": {"fear_greed_index": 80},
            "market_microstructure": {"order_book": {"imbalance": 0.25}},
            "current_price": 102.0,
        }

        conditions = MarketConditionsExtractor(null_logger()).extract_market_conditions(result)

        assert (
            conditions.trend_direction,
            conditions.trend_strength,
            conditions.timeframe_alignment,
        ) == ("BULLISH", 45.0, "ALIGNED")
        assert (conditions.adx, conditions.rsi, conditions.rsi_level) == (42.0, 65.0, "STRONG")
        assert (conditions.atr, conditions.atr_percentage, conditions.volatility) == (
            150.0,
            2.5,
            "MEDIUM",
        )
        assert (conditions.macd_signal, conditions.volume_state) == (
            "BEARISH",
            "ACCUMULATION",
        )
        assert (
            conditions.vwap,
            conditions.mfi,
            conditions.cmf,
            conditions.bb_percent_b,
        ) == (101.5, 62.0, 0.12, 0.71)
        assert (
            conditions.chandelier_long,
            conditions.pfe,
            conditions.supertrend_direction,
        ) == (90.0, 8.0, "Bullish")
        assert (conditions.choppiness, conditions.fear_greed_index) == (44.4, 80)
        assert (
            conditions.bb_position,
            conditions.market_sentiment,
            conditions.order_book_bias,
        ) == (expected_bb, "EXTREME_GREED", "BUY_PRESSURE")

    @pytest.mark.parametrize(
        "result",
        [{}, {"analysis": None}],
        ids=["empty-result", "null-analysis"],
    )
    def test_extract_market_conditions_defaults_when_nothing_is_readable(
        self, result: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unreadable payload yields the neutral snapshot, never a partial one."""
        # is_weekend follows the UTC clock, so freeze it to a Wednesday: the neutral
        # snapshot must hold regardless of when the suite runs (it failed every weekend).
        class _Sroda(datetime):
            @classmethod
            def now(cls, tz: tzinfo | None = None) -> datetime:
                return datetime(2026, 9, 16, 12, 0, tzinfo=tz or timezone.utc)

        monkeypatch.setattr("src.trading.market_conditions_extractor.datetime", _Sroda)
        conditions = MarketConditionsExtractor(null_logger()).extract_market_conditions(result)

        assert conditions == MarketConditions()

    @pytest.mark.parametrize(
        ("trend", "technical_data", "expected"),
        [
            pytest.param(
                {},
                {"adx": 30, "rsi": "72.5"},
                (30.0, 72.5, "OVERBOUGHT", 2.0, "MEDIUM", 0.0),
                id="numeric-strings",
            ),
            pytest.param(
                {},
                {"adx": 30, "rsi": "abc"},
                (30.0, 50.0, "NEUTRAL", 2.0, "MEDIUM", 0.0),
                id="rsi-garbage",
            ),
            pytest.param(
                {},
                {"adx": 30, "rsi": 61, "atr_percent": "abc"},
                (30.0, 61.0, "STRONG", 2.0, "MEDIUM", 0.0),
                id="atr-percent-garbage",
            ),
            pytest.param(
                {},
                {"adx": 30, "rsi": 61, "atr_percentage": 1.8},
                (30.0, 61.0, "STRONG", 1.8, "MEDIUM", 0.0),
                id="atr-percentage-fallback-key",
            ),
            pytest.param(
                {"strength_4h": 42},
                {"adx": 30},
                (30.0, 50.0, "NEUTRAL", 2.0, "MEDIUM", 42.0),
                id="trend-strength",
            ),
            pytest.param(
                {"strength_4h": "abc"},
                {"adx": 30},
                (30.0, 50.0, "NEUTRAL", 2.0, "MEDIUM", 50.0),
                id="trend-strength-garbage",
            ),
        ],
    )
    def test_extract_market_conditions_sanitizes_unconvertible_fields(
        self,
        trend: dict[str, Any],
        technical_data: dict[str, Any],
        expected: tuple[float, float, str, float, str, float],
    ) -> None:
        """Bad numbers fall back to the documented defaults instead of raising."""
        result = {"analysis": {"trend": trend}, "technical_data": technical_data}

        conditions = MarketConditionsExtractor(null_logger()).extract_market_conditions(result)

        assert (
            conditions.adx,
            conditions.rsi,
            conditions.rsi_level,
            conditions.atr_percentage,
            conditions.volatility,
            conditions.trend_strength,
        ) == expected

    def test_extract_market_conditions_never_guesses_trend_from_prose(self) -> None:
        """trend_direction comes only from analysis["trend"]; prose is never scanned."""
        result = {
            "analysis": {"trend": {}},
            "technical_data": {},
            "raw_response": "Strong BULLISH momentum expected; signal: BUY, bearish risk",
        }

        conditions = MarketConditionsExtractor(null_logger()).extract_market_conditions(result)

        assert conditions.trend_direction == "NEUTRAL"

    def test_extract_market_conditions_propagates_a_null_numeric_field(self) -> None:
        """A null ADX is not sanitized: it escapes the reader as a TypeError."""
        with pytest.raises(TypeError, match="must be a string or a real number"):
            MarketConditionsExtractor(null_logger()).extract_market_conditions(
                {"analysis": {"trend": {}}, "technical_data": {"adx": None}}
            )

    def test_extract_market_conditions_passes_nan_through(self) -> None:
        """NaN is finite-checked for trend strength but never for ADX."""
        conditions = MarketConditionsExtractor(null_logger()).extract_market_conditions(
            {"analysis": {"trend": {"strength_4h": float("nan")}}, "technical_data": {"adx": float("nan")}}
        )

        assert math.isnan(conditions.adx)
        assert conditions.trend_strength == 50.0

    @pytest.mark.parametrize(
        ("frozen_day", "expected"),
        [
            (datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc), True),
            (datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc), False),
        ],
        ids=["saturday", "monday"],
    )
    def test_extract_market_conditions_marks_the_weekend_from_utc(
        self, frozen_day: datetime, expected: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """is_weekend drives brain matching, so it must follow the UTC weekday."""
        import src.trading.market_conditions_extractor as extractor_module

        class _FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz: Any = None) -> datetime:
                return frozen_day

        monkeypatch.setattr(extractor_module, "datetime", _FrozenDatetime)

        conditions = MarketConditionsExtractor(null_logger()).extract_market_conditions({})

        assert conditions.is_weekend is expected

    def test_build_conditions_from_position_returns_the_entry_snapshot(self) -> None:
        """The close path hands back the entry snapshot verbatim, not a reconstruction."""
        snapshot = make_market_conditions(
            trend_direction="BULLISH",
            adx=26.08,
            rsi=45.47,
            atr=912.07,
            atr_percentage=1.17,
            choppiness=62.7,
            mfi=38.0,
            cmf=-0.0559,
            fear_greed_index=23,
            is_weekend=True,
        )
        position = make_position(
            conditions_at_entry=snapshot,
            trend_direction_at_entry="BEARISH",
            adx_at_entry=99.0,
            atr_at_entry=1.0,
            atr_percentage_at_entry=0.0,
        )

        conditions = MarketConditionsExtractor.build_conditions_from_position(position)

        assert conditions is snapshot
        assert (conditions.trend_direction, conditions.adx, conditions.rsi) == (
            "BULLISH",
            26.08,
            45.47,
        )
        assert (conditions.atr, conditions.atr_percentage) == (912.07, 1.17)
        assert (conditions.choppiness, conditions.mfi, conditions.cmf) == (62.7, 38.0, -0.0559)
        assert (conditions.fear_greed_index, conditions.is_weekend) == (23, True)


class TestCheckPosition:
    """Exit detection for both directions, including exact-boundary prices."""

    @pytest.mark.parametrize(
        ("direction", "price", "expected_reason", "expected_profit", "expected_drawdown"),
        [
            pytest.param("LONG", 94.0, "stop_loss", None, None, id="long-below-stop"),
            pytest.param("LONG", 95.0, "stop_loss", None, None, id="long-at-stop"),
            pytest.param("LONG", 115.0, "take_profit", None, None, id="long-at-target"),
            pytest.param("LONG", 116.0, "take_profit", None, None, id="long-above-target"),
            pytest.param("LONG", 105.0, None, 5.0, 0.0, id="long-in-range"),
            pytest.param("SHORT", 106.0, "stop_loss", None, None, id="short-above-stop"),
            pytest.param("SHORT", 105.0, "stop_loss", None, None, id="short-at-stop"),
            pytest.param("SHORT", 84.0, "take_profit", None, None, id="short-below-target"),
            pytest.param("SHORT", 85.0, "take_profit", None, None, id="short-at-target"),
            pytest.param("SHORT", 95.0, None, 5.0, 0.0, id="short-in-range"),
            pytest.param("SHORT", 104.0, None, 0.0, -4.0, id="short-in-range-adverse"),
        ],
    )
    async def test_detects_exits_and_tracks_metrics(
        self,
        direction: str,
        price: float,
        expected_reason: str | None,
        expected_profit: float | None,
        expected_drawdown: float | None,
    ) -> None:
        """A hit closes and returns the reason; a miss only updates live metrics.

        The metrics are excursions, not signed returns: a favourable move lands in
        ``max_profit_pct`` on either side of the book, an adverse one in
        ``max_drawdown_pct``.
        """
        position = make_position(
            direction=direction,
            entry_price=100.0,
            stop_loss=95.0 if direction == "LONG" else 105.0,
            take_profit=115.0 if direction == "LONG" else 85.0,
        )
        strategy, _, persistence, brain, _ = _strategy(position=position)
        _sync_brain_hook(brain)

        reason = await strategy.check_position(price)

        assert reason == expected_reason
        if expected_reason is None:
            current = strategy.current_position
            assert current is not None
            assert persistence.async_save_position.await_count == 1
            assert current.max_profit_pct == pytest.approx(expected_profit)
            assert current.max_drawdown_pct == pytest.approx(expected_drawdown)
            return
        assert strategy.current_position is None
        decision = persistence.async_save_trade_decision.await_args.args[0]
        assert decision.action == f"CLOSE_{direction}"
        assert decision.reasoning.startswith(f"Position closed: {expected_reason}.")

    async def test_without_a_position_it_returns_none(self) -> None:
        """No position means no metrics write and no exit."""
        strategy, _, persistence, _, _ = _strategy()

        assert await strategy.check_position(100.0) is None
        persistence.async_save_position.assert_not_awaited()


class TestTradingMemoryContext:
    """The prompt-facing summary of recent decisions and closed-trade results."""

    @pytest.mark.parametrize(
        ("entry", "close", "expected_label", "expected_line", "expected_total", "win_rate"),
        [
            pytest.param(
                {
                    "timestamp": datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc),
                    "action": "BUY",
                    "price": 72000.0,
                    "quantity": 0.001,
                    "reasoning": "Entered long on breakout.",
                },
                {
                    "timestamp": datetime(2026, 4, 14, 20, 0, tzinfo=timezone.utc),
                    "action": "CLOSE_LONG",
                    "price": 74286.53,
                    "quantity": 0.001,
                    "reasoning": "Position closed: stop_loss. P&L: +3.30%. Fee: $0.0426",
                },
                "profit-protecting stop",
                "- [2026-04-14 20:00] CLOSE_LONG @ $74,286.53 (Conf: HIGH) - "
                "Position closed: profit-protecting stop. P&L: +3.30%. Fee: $0.0426",
                "- Total P&L: $+2.29 (+3.18%)",
                "100.0% (1/1 trades)",
                id="profitable-stop",
            ),
            pytest.param(
                {
                    "timestamp": datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc),
                    "action": "SELL",
                    "price": 72000.0,
                    "quantity": 0.001,
                    "reasoning": "Entered short on breakdown.",
                },
                {
                    "timestamp": datetime(2026, 4, 14, 20, 0, tzinfo=timezone.utc),
                    "action": "CLOSE_SHORT",
                    "price": 72720.0,
                    "quantity": 0.001,
                    "reasoning": "Position closed: stop_loss. P&L: -1.00%. Fee: $0.0426",
                },
                "loss-cutting stop",
                "- [2026-04-14 20:00] CLOSE_SHORT @ $72,720.00 (Conf: HIGH) - "
                "Position closed: loss-cutting stop. P&L: -1.00%. Fee: $0.0426",
                "- Total P&L: $-0.72 (-1.00%)",
                "0.0% (0/1 trades)",
                id="losing-stop",
            ),
            pytest.param(
                {
                    "timestamp": datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc),
                    "action": "BUY",
                    "price": 100.0,
                    "quantity": 1.0,
                    "quote_amount": 100.0,
                    "reasoning": "Entered long.",
                },
                {
                    "timestamp": datetime(2026, 4, 14, 20, 0, tzinfo=timezone.utc),
                    "action": "CLOSE_LONG",
                    "price": 100.0,
                    "quantity": 1.0,
                    "quote_amount": 100.0,
                    "reasoning": "Position closed: stop_loss. P&L: +0.00%. Fee: $0.0000",
                },
                "breakeven stop",
                "- [2026-04-14 20:00] CLOSE_LONG @ $100.00 (Conf: HIGH) - "
                "Position closed: breakeven stop. P&L: +0.00%. Fee: $0.0000",
                "- Total P&L: $+0.00 (+0.00%)",
                "0.0% (0/1 trades)",
                id="flat-stop",
            ),
            pytest.param(
                {
                    "timestamp": datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc),
                    "action": "BUY",
                    "price": 100.0,
                    "quantity": 1.0,
                    "quote_amount": 100.0,
                    "reasoning": "Entered long.",
                },
                {
                    "timestamp": datetime(2026, 4, 14, 20, 0, tzinfo=timezone.utc),
                    "action": "CLOSE_LONG",
                    "price": 110.0,
                    "quantity": 1.0,
                    "quote_amount": 100.0,
                    "reasoning": "Position closed: take_profit. P&L: +10.00%. Fee: $0.0000",
                },
                "take profit",
                "- [2026-04-14 20:00] CLOSE_LONG @ $110.00 (Conf: HIGH) - "
                "Position closed: take profit. P&L: +10.00%. Fee: $0.0000",
                "- Total P&L: $+10.00 (+10.00%)",
                "100.0% (1/1 trades)",
                id="take-profit-reason",
            ),
        ],
    )
    def test_close_outcomes_are_labelled_and_rendered(
        self,
        entry: dict[str, Any],
        close: dict[str, Any],
        expected_label: str,
        expected_line: str,
        expected_total: str,
        win_rate: str,
    ) -> None:
        """Paired entry/close decisions produce the outcome-aware history lines."""
        history = [_decision(**entry), _decision(**close)]

        summary = TradingMemory(decisions=history).get_context_summary(full_history=history)

        assert "## Recent Trading History (Last 5 Decisions):" in summary
        assert expected_line in summary
        assert expected_label in summary
        assert expected_total in summary
        assert f"- Win Rate: {win_rate}" in summary
        assert "## Overall Performance (1 Total Closed Trades):" in summary

    def test_total_pnl_uses_capital_not_the_sum_of_trade_returns(self) -> None:
        """With an initial capital given, the total percentage is capital-based."""
        history = [
            _decision(
                timestamp=datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc),
                action="BUY",
                price=100.0,
                quantity=1.0,
                quote_amount=100.0,
                reasoning="Entered long.",
            ),
            _decision(
                timestamp=datetime(2026, 4, 14, 13, 0, tzinfo=timezone.utc),
                action="CLOSE_LONG",
                price=110.0,
                quantity=1.0,
                quote_amount=100.0,
                reasoning="Position closed: take_profit. P&L: +10.00%. Fee: $0.0000",
            ),
            _decision(
                timestamp=datetime(2026, 4, 14, 14, 0, tzinfo=timezone.utc),
                action="BUY",
                price=100.0,
                quantity=20.0,
                quote_amount=2000.0,
                reasoning="Entered long.",
            ),
            _decision(
                timestamp=datetime(2026, 4, 14, 15, 0, tzinfo=timezone.utc),
                action="CLOSE_LONG",
                price=99.0,
                quantity=20.0,
                quote_amount=2000.0,
                reasoning="Position closed: stop_loss. P&L: -1.00%. Fee: $0.0000",
            ),
        ]

        summary = TradingMemory(decisions=history).get_context_summary(
            full_history=history, initial_capital=10000.0
        )

        assert "- Total P&L: $-10.00 (-0.10%)" in summary
        assert "- Average P&L per Trade: +4.50%" in summary
        assert "- Win Rate: 50.0% (1/2 trades)" in summary

    def test_window_rolls_over_and_empty_memory_summarizes_to_nothing(self) -> None:
        """max_decisions keeps only the newest entries; empty memory renders no block."""
        assert TradingMemory().get_context_summary() == ""
        memory = TradingMemory(max_decisions=2)
        for hour in range(4):
            memory.add_decision(
                _decision(timestamp=datetime(2026, 4, 14, hour, tzinfo=timezone.utc))
            )

        assert [decision.timestamp.hour for decision in memory.decisions] == [2, 3]


class TestParserAndExtractorContract:
    """One parser owns normalization; PositionExtractor only reads the payload."""

    @pytest.mark.parametrize(
        ("raw_response", "expected"),
        [
            pytest.param(
                _COMPACT_BUY,
                ("BUY", "HIGH", 76500.0, 80640.0, 0.42, "Trend continuation with confluence."),
                id="compact-buy",
            ),
            pytest.param(
                _COMPACT_HOLD,
                ("HOLD", "HIGH", 79750.0, 73114.0, 0.0, "Wait for confirmation before entry."),
                id="compact-hold",
            ),
            pytest.param(
                _VERBOSE_HOLD,
                ("HOLD", "HIGH", 79750.0, 73114.0, 0.0, "Wait for confirmation before entry."),
                id="verbose-hold",
            ),
            pytest.param(
                f"```json\n{json.dumps(_analysis_payload(signal='LONG', confidence=82, stop_loss=76500, take_profit=80640, position_size=0.08, reasoning='Trend continuation with confluence.'))}\n```",
                ("LONG", "HIGH", 76500.0, 80640.0, 0.08, "Trend continuation with confluence."),
                id="long-signal",
            ),
        ],
    )
    def test_extractor_reads_every_field_from_the_parsed_payload(
        self, raw_response: str, expected: tuple[Any, ...]
    ) -> None:
        """Signal, confidence label, levels, sizing and reasoning come from the parse."""
        analysis = _parse(raw_response)["analysis"]

        extracted = PositionExtractor().extract_trading_info(analysis)

        assert extracted == expected
        assert PositionExtractor().validate_signal(extracted[0]) is True

    @pytest.mark.parametrize(
        ("raw_position_size", "expected_position_size"),
        [
            pytest.param("5%", 0.05, id="percent-string"),
            pytest.param("0.5%", 0.005, id="fraction-percent-string"),
            pytest.param("100%", 1.0, id="full-capital"),
            pytest.param("0%", 0.0, id="zero-percent"),
            pytest.param(0.05, 0.05, id="fraction-float"),
            pytest.param(1, 1.0, id="bare-one"),
            pytest.param(1.01, None, id="just-above-one"),
            pytest.param(50, None, id="bare-percent-sized-int"),
            pytest.param("150%", None, id="percent-above-one"),
            pytest.param(-5, None, id="negative-rejected"),
            pytest.param("", None, id="empty-string"),
            pytest.param(None, None, id="null"),
        ],
    )
    def test_position_size_contract_is_owned_by_the_parser(
        self, raw_position_size: Any, expected_position_size: float | None
    ) -> None:
        """Percent strings divide by 100; everything outside the 0.0-1.0 contract is unset.

        A percent string above 100% and a negative number both break the sizing contract,
        so the parser drops them instead of forwarding an unsizeable fraction (F4 in
        ``docs/SRC_AUDIT_FINDINGS.md``).
        """
        analysis = _parse_payload(
            _analysis_payload(position_size=raw_position_size, reasoning="Test setup.")
        )

        _, _, _, _, position_size, _ = PositionExtractor().extract_trading_info(analysis)

        if expected_position_size is None:
            assert position_size is None
        else:
            assert position_size == pytest.approx(expected_position_size)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(82, "HIGH", id="high-score"),
            pytest.param(70, "HIGH", id="high-lower-bound"),
            pytest.param(69, "MEDIUM", id="medium-upper-neighbour"),
            pytest.param(60, "MEDIUM", id="medium-score"),
            pytest.param(50, "MEDIUM", id="medium-lower-bound"),
            pytest.param(49, "LOW", id="low-upper-neighbour"),
            pytest.param(40, "LOW", id="low-score"),
            pytest.param(True, "MEDIUM", id="bool-is-not-a-score"),
            pytest.param("HIGH", "HIGH", id="label-passthrough"),
            pytest.param("high", "HIGH", id="label-lowercased"),
            pytest.param(None, "NONE", id="null-becomes-a-label"),
        ],
    )
    def test_confidence_labels_are_mapped_from_numbers(self, value: Any, expected: str) -> None:
        """Numeric scores map to HIGH/MEDIUM/LOW; strings pass through upper-cased."""
        assert PositionExtractor()._confidence_label(value) == expected

    @pytest.mark.parametrize(
        "signal",
        ["LONG", "SHORT", "BUY", "SELL", "HOLD", "CLOSE", "CLOSE_LONG", "UPDATE", "close_short"],
    )
    def test_supported_signals_are_accepted(self, signal: str) -> None:
        """Every signal in the executor's accepted set validates."""
        assert PositionExtractor().validate_signal(signal) is True

    @pytest.mark.parametrize(
        "signal",
        ["MAYBE", "", "HOLDING", "LONGER", "STRONG_BUY"],
    )
    def test_unknown_signals_are_rejected(self, signal: str) -> None:
        """Near-misses outside the accepted set are rejected."""
        assert PositionExtractor().validate_signal(signal) is False


class TestResponseValidationMetadata:
    """The non-blocking response contract attached by the parser."""

    @pytest.mark.parametrize(
        ("raw_text", "status", "valid", "error_types", "fields"),
        [
            pytest.param(
                f"```json\n{json.dumps({'analysis': {'signal': 'BUY', 'confidence': 82, 'entry_price': 77880, 'stop_loss': 76500, 'take_profit': 80640, 'position_size': 0.08, 'risk_reward_ratio': 2.0, 'reasoning': 'Valid setup.'}})}\n```",
                "valid",
                True,
                frozenset(),
                frozenset(),
                id="valid",
            ),
            pytest.param(
                f"```json\n{json.dumps({'analysis': {'signal': 'BUY', 'confidence': 125, 'entry_price': 77880, 'reasoning': 'Missing risk fields.'}})}\n```",
                "invalid",
                False,
                frozenset({"less_than_equal"}),
                frozenset({"analysis.confidence"}),
                id="confidence-above-maximum",
            ),
            pytest.param(
                f"```json\n{json.dumps({'analysis': {'signal': 'BUY', 'confidence': 80}})}\n```",
                "invalid",
                False,
                frozenset({"value_error"}),
                frozenset({"analysis"}),
                id="buy-without-execution-fields",
            ),
            pytest.param(
                f"```json\n{json.dumps({'analysis': {'summary': 'Legacy fallback analysis.'}})}\n```",
                "skipped",
                None,
                frozenset(),
                frozenset(),
                id="legacy-without-signal",
            ),
            pytest.param(
                "not json",
                "invalid",
                False,
                frozenset({"json_parse_error"}),
                frozenset({"response"}),
                id="unparseable-reply",
            ),
        ],
    )
    def test_metadata_states(
        self,
        raw_text: str,
        status: str,
        valid: bool | None,
        error_types: frozenset[str],
        fields: frozenset[str],
    ) -> None:
        """Validation metadata never breaks the parsed structure, only annotates it."""
        parsed = _parse(raw_text)

        validation = parsed["response_validation"]
        assert validation["schema"] == "trading-analysis-response-v1"
        assert validation["status"] == status
        assert validation["valid"] is valid
        assert {error["type"] for error in validation["errors"]} == set(error_types)
        assert {error["field"] for error in validation["errors"]} == set(fields)
        assert _PARSER.validate_ai_response(parsed) is True


class TestTemplateManagerPreviousAnalysis:
    """Shared parser helper: the LAST parseable json block wins."""

    def test_extract_previous_analysis_uses_the_last_block(self) -> None:
        """A contract repair appends the recovered block, so the last one must win."""
        manager = TemplateManager(
            config=make_config(), logger=null_logger(), timeframe_validator=None
        )
        schema_example = {"analysis": {"signal": "HOLD", "confidence": 10}}
        actual_analysis = {
            "analysis": {"signal": "SELL", "confidence": 82, "reasoning": "Breakdown confirmed."}
        }
        previous_response = (
            "```json\n"
            + json.dumps(schema_example)
            + "\n```\n1) MARKET STRUCTURE: Bearish continuation.\n```json\n"
            + json.dumps(actual_analysis)
            + "\n```"
        )

        extracted = manager._extract_previous_analysis(previous_response)

        assert extracted == {
            "signal": "SELL",
            "confidence": 82,
            "reasoning": "Breakdown confirmed.",
        }
        assert manager._extract_previous_analysis("no json block at all") is None
