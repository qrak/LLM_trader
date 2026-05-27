"""Regression tests for order lifecycle governance primitives."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.trading.audit import AuditTrail
from src.trading.guards import GuardResult
from src.trading.guards.cooldown_window import CooldownWindowGuard
from src.trading.guards.max_position_size import MaxPositionSizeGuard
from src.trading.guards.pipeline import GuardPipeline
from src.trading.guards.configured_symbol import ConfiguredSymbolGuard
from src.trading.order_lifecycle import OrderIntent, OrderLifecycle
from src.trading.trading_strategy import TradingStrategy


class PassingGuard:
    name = "passing_guard"

    def check(self, intent: OrderIntent, /, *, capital: float, config) -> GuardResult:
        return GuardResult(
            guard_name=self.name,
            passed=True,
            reason="passed",
            metadata={"capital": capital, "symbol": intent.symbol},
        )


class RejectingGuard:
    name = "rejecting_guard"

    def check(self, intent: OrderIntent, /, *, capital: float, config) -> GuardResult:
        return GuardResult(
            guard_name=self.name,
            passed=False,
            reason="blocked by test guard",
            metadata={"capital": capital, "symbol": intent.symbol},
        )


def _make_intent() -> OrderIntent:
    return OrderIntent(
        order_id="order-test",
        signal="BUY",
        direction="LONG",
        symbol="BTC/USDC",
        confidence="HIGH",
        current_price=100.0,
        stop_loss=95.0,
        take_profit=115.0,
        position_size=0.05,
    )


def _make_strategy(
    *,
    guard_pipeline: GuardPipeline | None = None,
    audit_trail: AuditTrail | None = None,
    config_overrides: dict | None = None,
) -> TradingStrategy:
    persistence = MagicMock()
    persistence.load_position.return_value = None
    persistence.async_save_position = AsyncMock()
    persistence.async_save_trade_decision = AsyncMock()

    brain = MagicMock()
    brain.vector_memory = MagicMock()
    brain.get_dynamic_thresholds.return_value = {"rr_borderline_min": 1.5}

    statistics = MagicMock()
    statistics.get_current_capital.return_value = 10000.0

    risk_manager = MagicMock()
    risk_manager.get_and_clear_frictions.return_value = []
    risk_manager.calculate_entry_parameters.return_value = SimpleNamespace(
        stop_loss=95.0,
        take_profit=115.0,
        size_pct=0.05,
        quantity=5.0,
        entry_fee=0.375,
        sl_distance_pct=0.05,
        tp_distance_pct=0.15,
        rr_ratio=3.0,
        quote_amount=500.0,
        entry_price=100.0,
        volatility_level="MEDIUM",
    )

    config = SimpleNamespace(
        DEMO_QUOTE_CAPITAL=10000.0,
        TIMEFRAME="4h",
        CRYPTO_PAIR="BTC/USDC",
        MAX_POSITION_SIZE=0.10,
        DATA_DIR="data",
        STOP_LOSS_TYPE="soft",
        STOP_LOSS_CHECK_INTERVAL="4h",
        TAKE_PROFIT_TYPE="soft",
        TAKE_PROFIT_CHECK_INTERVAL="4h",
    )
    if config_overrides:
        for key, value in config_overrides.items():
            setattr(config, key, value)

    return TradingStrategy(
        logger=MagicMock(),
        persistence=persistence,
        brain_service=brain,
        statistics_service=statistics,
        memory_service=MagicMock(),
        risk_manager=risk_manager,
        config=config,
        position_extractor=MagicMock(),
        guard_pipeline=guard_pipeline,
        audit_trail=audit_trail,
    )


def test_order_intent_can_be_rejected_from_initial_intent_state() -> None:
    intent = _make_intent()

    assert intent.transition_to(OrderLifecycle.REJECTED, reason="guard failed") is True

    assert intent.state is OrderLifecycle.REJECTED


def test_guard_pipeline_returns_fail_fast_check_results_without_audit_side_effects() -> None:
    pipeline = GuardPipeline([PassingGuard(), RejectingGuard()])

    results = pipeline.evaluate(_make_intent(), capital=10000.0, config=SimpleNamespace())

    assert [result.guard_name for result in results] == ["passing_guard", "rejecting_guard"]


def test_max_position_guard_only_rejects_explicit_over_cap_size() -> None:
    guard = MaxPositionSizeGuard()
    config = SimpleNamespace(MAX_POSITION_SIZE=0.10)

    missing_size_intent = _make_intent()
    missing_size_intent.position_size = None
    missing_size_result = guard.check(missing_size_intent, capital=10000.0, config=config)

    invalid_size_intent = _make_intent()
    invalid_size_intent.position_size = 0.0
    invalid_size_result = guard.check(invalid_size_intent, capital=10000.0, config=config)

    string_max_size_result = guard.check(
        _make_intent(),
        capital=10000.0,
        config=SimpleNamespace(MAX_POSITION_SIZE="0.10"),
    )

    over_cap_intent = _make_intent()
    over_cap_intent.position_size = 0.42
    over_cap_result = guard.check(over_cap_intent, capital=10000.0, config=config)

    assert missing_size_result.passed is True
    assert invalid_size_result.passed is True
    assert string_max_size_result.passed is True
    assert string_max_size_result.metadata["max_size"] == 0.10
    assert over_cap_result.passed is False
    assert "exceeds maximum" in over_cap_result.reason


def test_max_position_guard_rejects_invalid_max_position_size_config() -> None:
    guard = MaxPositionSizeGuard()

    text_config_result = guard.check(
        _make_intent(),
        capital=10000.0,
        config=SimpleNamespace(MAX_POSITION_SIZE="not-a-number"),
    )
    nan_config_result = guard.check(
        _make_intent(),
        capital=10000.0,
        config=SimpleNamespace(MAX_POSITION_SIZE=float("nan")),
    )

    assert text_config_result.passed is False
    assert "not a valid number" in text_config_result.reason
    assert nan_config_result.passed is False
    assert "positive finite decimal" in nan_config_result.reason


def test_max_position_guard_lets_non_finite_requested_size_fall_back_to_risk_manager() -> None:
    guard = MaxPositionSizeGuard()
    config = SimpleNamespace(MAX_POSITION_SIZE=0.10)
    intent = _make_intent()
    intent.position_size = float("inf")

    result = guard.check(intent, capital=10000.0, config=config)

    assert result.passed is True
    assert "RiskManager fallback sizing" in result.reason


def test_configured_symbol_guard_only_allows_configured_pair() -> None:
    guard = ConfiguredSymbolGuard()
    config = SimpleNamespace(CRYPTO_PAIR="BTC/USDC")

    primary_intent = _make_intent()
    rejected_intent = _make_intent()
    rejected_intent.symbol = "ETH/USDC"

    assert guard.check(primary_intent, capital=10000.0, config=config).passed is True
    assert guard.check(rejected_intent, capital=10000.0, config=config).passed is False


@pytest.mark.asyncio
async def test_strategy_rejected_guard_records_audit_and_skips_risk_calculation() -> None:
    audit_trail = AuditTrail()
    strategy = _make_strategy(
        guard_pipeline=GuardPipeline([RejectingGuard()]),
        audit_trail=audit_trail,
    )

    decision = await strategy._open_new_position(
        signal="BUY",
        confidence="HIGH",
        stop_loss=95.0,
        take_profit=115.0,
        position_size=0.05,
        current_price=100.0,
        symbol="BTC/USDC",
        reasoning="test",
    )

    assert decision.action == "HOLD"
    strategy.risk_manager.calculate_entry_parameters.assert_not_called()
    assert [record.event_type for record in audit_trail.all_records] == [
        "intent_created",
        "guard_check",
        "rejection",
    ]
    assert audit_trail.all_records[-1].result == "rejected"


@pytest.mark.asyncio
async def test_strategy_with_production_guard_pipeline_rejects_over_cap_size(tmp_path) -> None:
    audit_trail = AuditTrail()
    pipeline = GuardPipeline(
        [
            ConfiguredSymbolGuard(),
            MaxPositionSizeGuard(),
            CooldownWindowGuard(),
        ]
    )
    strategy = _make_strategy(
        guard_pipeline=pipeline,
        audit_trail=audit_trail,
        config_overrides={"DATA_DIR": str(tmp_path)},
    )

    decision = await strategy._open_new_position(
        signal="BUY",
        confidence="HIGH",
        stop_loss=95.0,
        take_profit=115.0,
        position_size=0.42,
        current_price=100.0,
        symbol="BTC/USDC",
        reasoning="test",
    )

    assert decision.action == "HOLD"
    strategy.risk_manager.calculate_entry_parameters.assert_not_called()
    assert [record.actor for record in audit_trail.all_records] == [
        "TradingStrategy",
        "configured_symbol",
        "max_position_size",
        "GuardPipeline",
    ]
    assert audit_trail.all_records[-1].result == "rejected"


@pytest.mark.asyncio
async def test_strategy_without_guards_records_approval_and_execution_audit() -> None:
    audit_trail = AuditTrail()
    strategy = _make_strategy(audit_trail=audit_trail)

    decision = await strategy._open_new_position(
        signal="BUY",
        confidence="HIGH",
        stop_loss=95.0,
        take_profit=115.0,
        position_size=0.05,
        current_price=100.0,
        symbol="BTC/USDC",
        reasoning="test",
    )

    assert decision.action == "BUY"
    assert [record.event_type for record in audit_trail.all_records] == [
        "intent_created",
        "approval",
        "execution",
    ]
    assert audit_trail.all_records[-1].result == "executed"


@pytest.mark.asyncio
async def test_strategy_invalidates_cooldown_cache_after_execution() -> None:
    cooldown_guard = MagicMock()
    cooldown_guard.name = "cooldown_window"
    cooldown_guard.check.return_value = GuardResult(
        guard_name="cooldown_window",
        passed=True,
        reason="passed",
        metadata={},
    )
    cooldown_guard.invalidate_cache = MagicMock()
    strategy = _make_strategy(guard_pipeline=GuardPipeline([cooldown_guard]))

    decision = await strategy._open_new_position(
        signal="BUY",
        confidence="HIGH",
        stop_loss=95.0,
        take_profit=115.0,
        position_size=0.05,
        current_price=100.0,
        symbol="BTC/USDC",
        reasoning="test",
    )

    assert decision.action == "BUY"
    cooldown_guard.invalidate_cache.assert_called_once_with()
