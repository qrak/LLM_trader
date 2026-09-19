"""Order governance, stop-loss tightening policy and exit-monitor cadence.

Pre-entry guards gate the intent before RiskManager sizing, the tightening
policy gates every stop-loss move, and the monitor ticks the due exits.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.parsing.unified_parser import UnifiedParser
from src.trading.data_models import MarketConditions, Position
from src.trading.exit_monitor import ExitMonitor
from src.trading.guards import GuardResult
from src.trading.guards.configured_symbol import ConfiguredSymbolGuard
from src.trading.guards.cooldown_window import CooldownWindowGuard
from src.trading.guards.max_position_size import MaxPositionSizeGuard
from src.trading.guards.pipeline import GuardPipeline
from src.trading.market_conditions_extractor import MarketConditionsExtractor
from src.trading.order_lifecycle import OrderIntent, OrderLifecycle
from src.trading.position_extractor import PositionExtractor
from src.trading.position_status_monitor import PositionStatusMonitor
from src.trading.stop_loss_tightening_policy import StopLossTighteningPolicy
from src.trading.trading_strategy import TradingStrategy
from src.utils.format_utils import FormatUtils
from tests.conftest import (
    make_config,
    make_market_conditions,
    make_position,
    mock_brain,
    mock_extractor,
    mock_persistence,
    mock_statistics,
    null_logger,
)

CAPITAL = 10000.0
CONFIG_PAIR = "BTC/USDC"
NOW = datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc)
PARSER = UnifiedParser(logger=null_logger(), format_utils=FormatUtils())
EXTRACTOR = PositionExtractor()


class GuardDouble:
    """Guard double that counts the cache invalidations the pipeline requests."""

    name = "guard"

    def __init__(self) -> None:
        self.invalidations = 0

    def check(self, order: OrderIntent, /, *, capital: float, config: Any) -> GuardResult:
        return GuardResult(guard_name=self.name, passed=True, reason="passed", metadata={})

    def invalidate_cache(self) -> None:
        self.invalidations += 1


class PassingGuard(GuardDouble):
    name = "passing_guard"


class RejectingGuard(GuardDouble):
    name = "rejecting_guard"

    def check(self, order: OrderIntent, /, *, capital: float, config: Any) -> GuardResult:
        return GuardResult(guard_name=self.name, passed=False, reason="blocked by test guard")


class CooldownGuardDouble(GuardDouble):
    name = "cooldown_window"


class FaultyGuard(GuardDouble):
    name = "faulty_guard"

    def check(self, order: OrderIntent, /, *, capital: float, config: Any) -> GuardResult:
        raise RuntimeError("Simulated internal guard crash!")


class RecordingExitStrategy:
    """Exit-strategy double recording which exit check the monitor selected."""

    def __init__(self, close_reason: str | None = None) -> None:
        self.current_position: Any = object()
        self.close_reason = close_reason
        self.calls: list[tuple[str, float]] = []

    async def check_position(self, current_price: float) -> str | None:
        self.calls.append(("check_position", current_price))
        if self.close_reason:
            self.current_position = None
        return self.close_reason

    async def check_stop_loss(self, current_price: float) -> str | None:
        self.calls.append(("check_stop_loss", current_price))
        return self.close_reason

    async def check_take_profit(self, current_price: float) -> str | None:
        self.calls.append(("check_take_profit", current_price))
        return self.close_reason


class MonitorPersistence:
    """In-memory position monitor state store."""

    def __init__(self, state: dict[str, Any] | None = None) -> None:
        self.state = dict(state or {})

    async def async_load_position_monitor_state(self) -> dict[str, Any]:
        return dict(self.state)

    async def async_save_position_monitor_state(self, state: dict[str, Any]) -> None:
        self.state = dict(state)

    async def async_clear_position_monitor_state(self) -> None:
        self.state = {}

    def load_trade_history(self) -> list[dict[str, Any]]:
        return []


async def noop_sleep(_seconds: float, respect_force_analysis: bool = True) -> bool:
    """``interruptible_sleep`` double that returns control immediately."""
    return False


def intent(position_size: float | None = 0.05, symbol: str = CONFIG_PAIR) -> OrderIntent:
    """Order intent on the configured pair, LONG at 100 with a 95/115 bracket."""
    return OrderIntent(
        order_id="order-test",
        signal="BUY",
        direction="LONG",
        symbol=symbol,
        confidence="HIGH",
        current_price=100.0,
        stop_loss=95.0,
        take_profit=115.0,
        position_size=position_size,
    )


def guard_strategy(
    pipeline: GuardPipeline | None, **config_overrides: Any
) -> TradingStrategy:
    """TradingStrategy on conftest doubles whose only live policy is the guard pipeline."""
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
        regime_profile="neutral",
    )
    brain = mock_brain()
    brain.get_dynamic_thresholds.return_value = {"rr_borderline_min": 1.5}
    statistics = mock_statistics()
    statistics.get_current_capital.return_value = CAPITAL
    return TradingStrategy(
        logger=null_logger(),
        persistence=mock_persistence(),
        brain_service=brain,
        statistics_service=statistics,
        memory_service=MagicMock(),
        risk_manager=risk_manager,
        config=make_config(CRYPTO_PAIR=CONFIG_PAIR, TIMEFRAME="4h", **config_overrides),
        position_extractor=mock_extractor(),
        guard_pipeline=pipeline,
    )


async def open_position(
    strategy: TradingStrategy, position_size: float | None = 0.05
) -> Any:
    """Run the strategy entry path for a single BUY intent."""
    return await strategy._open_new_position(
        signal="BUY",
        confidence="HIGH",
        stop_loss=95.0,
        take_profit=115.0,
        position_size=position_size,
        current_price=100.0,
        symbol=CONFIG_PAIR,
        reasoning="test",
        market_conditions=make_market_conditions(),
    )


def long_position(stop_loss: float = 95.0, take_profit: float = 115.0) -> Position:
    """LONG position entered at 100 with a configurable bracket."""
    return make_position(
        entry_price=100.0, stop_loss=stop_loss, take_profit=take_profit, direction="LONG"
    )


def short_position(stop_loss: float = 105.0, take_profit: float = 85.0) -> Position:
    """SHORT position entered at 100 with a configurable bracket."""
    return make_position(
        entry_price=100.0,
        stop_loss=stop_loss,
        take_profit=take_profit,
        direction="SHORT",
    )


def exit_config(
    stop_loss_type: str = "hard",
    take_profit_type: str = "hard",
    stop_loss_interval: str = "5m",
    take_profit_interval: str = "15m",
) -> SimpleNamespace:
    """ExitMonitor config double on 5m/15m hard-exit cadences."""
    return SimpleNamespace(
        STOP_LOSS_TYPE=stop_loss_type,
        STOP_LOSS_CHECK_INTERVAL=stop_loss_interval,
        STOP_LOSS_CHECK_INTERVAL_SECONDS=300,
        TAKE_PROFIT_TYPE=take_profit_type,
        TAKE_PROFIT_CHECK_INTERVAL=take_profit_interval,
        TAKE_PROFIT_CHECK_INTERVAL_SECONDS=900,
        MAIN_CHANNEL_ID=123,
    )


def exit_strategy(direction: str = "LONG") -> TradingStrategy:
    """TradingStrategy shell for the exit checks: real Position, mocked venue seams."""
    strategy = TradingStrategy.__new__(TradingStrategy)
    strategy.current_position = (
        long_position(stop_loss=95.0, take_profit=110.0)
        if direction == "LONG"
        else short_position(stop_loss=105.0, take_profit=90.0)
    )
    persistence = MagicMock()
    persistence.async_save_position = AsyncMock()
    strategy.persistence = persistence
    strategy.close_position = AsyncMock()
    strategy.logger = null_logger()
    strategy._conditions = MarketConditionsExtractor(null_logger())
    return strategy


def monitor_context(
    state: dict[str, Any] | None = None,
    strategy: Any = None,
    notifier: Any = None,
    is_running: Callable[[], bool] | None = None,
    fetch_current_ticker: Any = None,
    interruptible_sleep: Any = None,
) -> SimpleNamespace:
    """PositionStatusMonitor wired to an in-memory monitor-state store."""
    config = exit_config()
    persistence = MonitorPersistence(state)
    exit_monitor = ExitMonitor(config, "1h", 3600)
    trading_strategy = strategy or RecordingExitStrategy()
    monitor = PositionStatusMonitor(
        logger=null_logger(),
        config=config,
        persistence=persistence,
        trading_strategy=trading_strategy,
        exit_monitor=exit_monitor,
        notifier=notifier,
        active_tasks=set(),
        is_running=is_running or (lambda: False),
        fetch_current_ticker=fetch_current_ticker or AsyncMock(return_value=None),
        interruptible_sleep=interruptible_sleep or noop_sleep,
        get_symbol=lambda: CONFIG_PAIR,
    )
    return SimpleNamespace(
        config=config,
        exit_monitor=exit_monitor,
        persistence=persistence,
        trading_strategy=trading_strategy,
        position_monitor=monitor,
    )


def test_order_intent_rejects_from_the_initial_state() -> None:
    """An intent that has not been reviewed yet can transition straight to REJECTED."""
    order = intent()

    assert order.transition_to(OrderLifecycle.REJECTED, reason="guard failed") is True
    assert order.state is OrderLifecycle.REJECTED


@pytest.mark.parametrize(
    ("guard_classes", "expected_names", "expected_passed", "expected_reasons"),
    [
        pytest.param(
            (PassingGuard, RejectingGuard),
            ["passing_guard", "rejecting_guard"],
            [True, False],
            ["passed", "blocked by test guard"],
            id="pass-then-reject",
        ),
        pytest.param(
            (RejectingGuard, PassingGuard),
            ["rejecting_guard"],
            [False],
            ["blocked by test guard"],
            id="reject-stops-the-pipeline",
        ),
        pytest.param(
            (PassingGuard, FaultyGuard),
            ["passing_guard", "faulty_guard"],
            [True, False],
            [
                "passed",
                (
                    "Guard 'faulty_guard' failed closed due to error: "
                    "Simulated internal guard crash!"
                ),
            ],
            id="fault-after-pass",
        ),
        pytest.param(
            (FaultyGuard, PassingGuard),
            ["faulty_guard"],
            [False],
            [
                (
                    "Guard 'faulty_guard' failed closed due to error: "
                    "Simulated internal guard crash!"
                )
            ],
            id="fault-stops-the-pipeline",
        ),
    ],
)
def test_guard_pipeline_stops_at_the_first_block(
    guard_classes: tuple[type, ...],
    expected_names: list[str],
    expected_passed: list[bool],
    expected_reasons: list[str],
) -> None:
    """A rejection ends the run and a guard fault becomes a fail-closed result."""
    pipeline = GuardPipeline([guard_class() for guard_class in guard_classes])

    results = pipeline.evaluate(intent(), capital=CAPITAL, config=SimpleNamespace())

    assert [result.guard_name for result in results] == expected_names
    assert [result.passed for result in results] == expected_passed
    assert [result.reason for result in results] == expected_reasons


@pytest.mark.parametrize(
    ("guard_classes", "expected_invalidations"),
    [
        pytest.param((), [], id="empty-pipeline"),
        pytest.param((PassingGuard, RejectingGuard), [0, 0], id="no-cooldown-guard"),
        pytest.param((PassingGuard, CooldownGuardDouble), [0, 1], id="cooldown-guard"),
    ],
)
def test_invalidate_cooldown_cache_targets_only_the_cooldown_guard(
    guard_classes: tuple[type, ...], expected_invalidations: list[int]
) -> None:
    """Invalidation reaches the cooldown guard only; any other pipeline is a silent no-op."""
    guards = [guard_class() for guard_class in guard_classes]
    pipeline = GuardPipeline(guards)

    pipeline.invalidate_cooldown_cache()

    assert [guard.invalidations for guard in guards] == expected_invalidations


@pytest.mark.parametrize(
    ("history", "expected_reason", "expected_metadata"),
    [
        pytest.param(
            None,
            "Cooldown guard is not wired with persistence (fail-closed)",
            {"error": "persistence_not_configured"},
            id="no-persistence",
        ),
        pytest.param(
            RuntimeError("db unavailable"),
            "Cooldown guard could not read execution history (fail-closed)",
            {"error": "Cooldown guard could not read execution history"},
            id="unreadable-history",
        ),
    ],
)
def test_cooldown_guard_fails_closed(
    history: Exception | None, expected_reason: str, expected_metadata: dict[str, str]
) -> None:
    """An absent or failing history store blocks the order instead of allowing it."""
    guard = CooldownWindowGuard()
    if history is not None:
        persistence = MagicMock()
        persistence.get_last_execution_timestamp.side_effect = history
        guard = CooldownWindowGuard(persistence=persistence)

    result = guard.check(intent(), capital=CAPITAL, config=SimpleNamespace(TIMEFRAME="4h"))

    assert result.passed is False
    assert result.reason == expected_reason
    assert result.metadata == expected_metadata


@pytest.mark.parametrize(
    ("timeframe", "elapsed_minutes", "expected_passed", "expected_cooldown"),
    [
        pytest.param("15m", 60.0, True, 60, id="scalping-4x-opens-exactly"),
        pytest.param("15m", 59.9, False, 60, id="scalping-4x-one-tick-inside"),
        pytest.param("1h", 180.0, True, 180, id="intraday-3x-opens-exactly"),
        pytest.param("4h", 480.0, True, 480, id="swing-2x-opens-exactly"),
        pytest.param("1d", 1440.0, True, 1440, id="position-1x-opens-exactly"),
    ],
)
def test_cooldown_guard_window_scales_with_the_configured_timeframe(
    timeframe: str,
    elapsed_minutes: float,
    expected_passed: bool,
    expected_cooldown: int,
) -> None:
    """The per-timeframe cooldown multiple, with equality at the window opening the gate."""
    persistence = MagicMock()
    persistence.get_last_execution_timestamp.return_value = (
        datetime.now(timezone.utc) - timedelta(minutes=elapsed_minutes)
    )

    result = CooldownWindowGuard(persistence=persistence).check(
        intent(), capital=CAPITAL, config=SimpleNamespace(TIMEFRAME=timeframe)
    )

    assert result.passed is expected_passed
    assert result.metadata["cooldown_minutes"] == expected_cooldown
    if expected_passed:
        assert result.reason.startswith("Cooldown expired:")
    else:
        assert result.reason.startswith("Cooldown active:")
        assert result.metadata["remaining_minutes"] == pytest.approx(0.1)


def test_cooldown_guard_caches_history_until_invalidated() -> None:
    """A cached history read is served until invalidate_cache forces a re-read."""
    persistence = MagicMock()
    persistence.get_last_execution_timestamp.return_value = None
    guard = CooldownWindowGuard(persistence=persistence)
    config = SimpleNamespace(TIMEFRAME="4h")

    first = guard.check(intent(), capital=CAPITAL, config=config)
    persistence.get_last_execution_timestamp.return_value = (
        datetime.now(timezone.utc) - timedelta(minutes=1)
    )
    cached = guard.check(intent(), capital=CAPITAL, config=config)
    guard.invalidate_cache()
    refreshed = guard.check(intent(), capital=CAPITAL, config=config)

    assert first.passed is True
    assert first.reason == "No prior execution — cooldown not applicable"
    assert cached.passed is True
    assert cached.reason == first.reason
    assert refreshed.passed is False
    assert refreshed.metadata["cooldown_minutes"] == 480
    assert persistence.get_last_execution_timestamp.call_count == 2


@pytest.mark.parametrize(
    ("configured_max", "requested", "expected_passed", "expected_reason", "valid_config"),
    [
        pytest.param(
            0.10,
            0.10,
            True,
            "Position size 10.0% within limit 10.0%",
            True,
            id="requested-exactly-at-the-cap",
        ),
        pytest.param(
            0.10,
            0.42,
            False,
            "Position size 42.0% exceeds maximum 10.0%",
            True,
            id="requested-above-the-cap",
        ),
        pytest.param(
            "0.10",
            0.05,
            True,
            "Position size 5.0% within limit 10.0%",
            True,
            id="cap-configured-as-text",
        ),
        pytest.param(
            0.10,
            None,
            True,
            "No position_size provided; RiskManager fallback sizing will apply",
            True,
            id="size-unset",
        ),
        pytest.param(
            0.10,
            0.0,
            True,
            "Invalid position_size provided; RiskManager fallback sizing will apply",
            True,
            id="size-zero",
        ),
        pytest.param(
            0.10,
            float("inf"),
            True,
            "Invalid position_size provided; RiskManager fallback sizing will apply",
            True,
            id="size-infinite",
        ),
        pytest.param(
            float("nan"),
            0.05,
            False,
            "Configured MAX_POSITION_SIZE must be a positive finite decimal",
            False,
            id="cap-not-finite",
        ),
        pytest.param(
            "not-a-number",
            0.05,
            False,
            "Configured MAX_POSITION_SIZE is not a valid number",
            False,
            id="cap-not-a-number",
        ),
    ],
)
def test_max_position_guard_cap_and_config_contract(
    configured_max: Any,
    requested: float | None,
    expected_passed: bool,
    expected_reason: str,
    valid_config: bool,
) -> None:
    """Only an explicit over-cap size is rejected; an unusable cap rejects every intent."""
    result = MaxPositionSizeGuard().check(
        intent(position_size=requested),
        capital=CAPITAL,
        config=SimpleNamespace(MAX_POSITION_SIZE=configured_max),
    )

    assert result.passed is expected_passed
    assert result.reason == expected_reason
    if valid_config:
        assert result.metadata["max_size"] == float(configured_max)
    else:
        assert str(result.metadata["max_size"]) == str(configured_max)
    if "requested" in result.metadata:
        recorded = result.metadata["requested"]
        assert math.isfinite(recorded) == math.isfinite(requested)
        if math.isfinite(requested):
            assert recorded == pytest.approx(requested)
    else:
        assert sorted(result.metadata) == ["max_size"]


@pytest.mark.parametrize(
    ("symbol", "expected_passed", "expected_reason"),
    [
        pytest.param(
            "BTC/USDC",
            True,
            "Symbol 'BTC/USDC' matches configured trading pair",
            id="configured-pair",
        ),
        pytest.param(
            "btc/usdc",
            False,
            "Symbol 'btc/usdc' does not match configured trading pair 'BTC/USDC'",
            id="case-is-not-normalized",
        ),
        pytest.param(
            "BTC/USDT",
            False,
            "Symbol 'BTC/USDT' does not match configured trading pair 'BTC/USDC'",
            id="quote-currency-mismatch",
        ),
    ],
)
def test_configured_symbol_guard_rejects_any_other_pair(
    symbol: str, expected_passed: bool, expected_reason: str
) -> None:
    """The guard compares the raw symbol string against CRYPTO_PAIR."""
    result = ConfiguredSymbolGuard().check(
        intent(symbol=symbol),
        capital=CAPITAL,
        config=SimpleNamespace(CRYPTO_PAIR=CONFIG_PAIR),
    )

    assert result.passed is expected_passed
    assert result.reason == expected_reason
    assert result.metadata == {"symbol": symbol, "configured_symbol": CONFIG_PAIR}


@pytest.mark.parametrize(
    ("guard_classes", "position_size", "expected_block_reason"),
    [
        pytest.param(
            (RejectingGuard,),
            0.05,
            "rejecting_guard: blocked by test guard",
            id="rejecting-guard",
        ),
        pytest.param(
            (ConfiguredSymbolGuard, MaxPositionSizeGuard, CooldownWindowGuard),
            0.42,
            "max_position_size: Position size 42.0% exceeds maximum 10.0%",
            id="production-pipeline-over-cap",
        ),
    ],
)
async def test_strategy_pipeline_blocks_entry_and_skips_risk_sizing(
    guard_classes: tuple[type, ...], position_size: float, expected_block_reason: str
) -> None:
    """A blocked intent returns HOLD with the guard reason and never reaches RiskManager."""
    strategy = guard_strategy(GuardPipeline([guard_class() for guard_class in guard_classes]))

    decision = await open_position(strategy, position_size=position_size)

    assert decision.action == "HOLD"
    assert decision.reasoning.endswith(f"rejected by guard pipeline: {expected_block_reason}")
    assert strategy.risk_manager.calculate_entry_parameters.call_count == 0
    assert strategy.persistence.async_save_position.await_count == 0
    assert strategy.current_position is None
    strategy.logger.warning.assert_called_once_with(
        "Order REJECTED by guard pipeline: %s", expected_block_reason
    )


@pytest.mark.parametrize(
    ("guard_classes", "expected_invalidations"),
    [
        pytest.param(None, None, id="no-guard-pipeline"),
        pytest.param((CooldownGuardDouble,), [1], id="cooldown-cache-dropped-after-execution"),
    ],
)
async def test_strategy_executes_and_invalidates_the_cooldown_cache(
    guard_classes: tuple[type, ...] | None, expected_invalidations: list[int] | None
) -> None:
    """An accepted entry is sized, persisted, and drops the cooldown cache."""
    guards = [] if guard_classes is None else [guard_class() for guard_class in guard_classes]
    strategy = guard_strategy(None if guard_classes is None else GuardPipeline(guards))

    decision = await open_position(strategy)

    assert decision.action == "BUY"
    assert decision.reasoning == "test"
    assert strategy.risk_manager.calculate_entry_parameters.call_count == 1
    assert strategy.persistence.async_save_position.await_count == 1
    assert strategy.current_position.size == pytest.approx(5.0)
    if guard_classes is None:
        assert strategy.guard_pipeline is None
    else:
        assert [guard.invalidations for guard in guards] == expected_invalidations


@pytest.mark.parametrize(
    ("tf_minutes", "expected_threshold"),
    [
        pytest.param(0, 0.25, id="zero-minutes"),
        pytest.param(59, 0.25, id="scalping-upper-edge"),
        pytest.param(60, 0.20, id="intraday-lower-edge"),
        pytest.param(240, 0.15, id="swing-lower-edge"),
        pytest.param(1440, 0.10, id="position-lower-edge"),
    ],
)
def test_base_threshold_buckets_at_the_exact_timeframe_edges(
    tf_minutes: int, expected_threshold: float
) -> None:
    """Bucket edges are exclusive on the left: 60 minutes is already intraday."""
    assert StopLossTighteningPolicy().get_base_threshold(tf_minutes) == expected_threshold


def test_policy_thresholds_from_config_and_from_defaults() -> None:
    """from_config maps every SL_TIGHTENING_* key; the bare constructor keeps the defaults."""
    from_config = StopLossTighteningPolicy.from_config(
        make_config(
            SL_TIGHTENING_SCALPING=0.30,
            SL_TIGHTENING_INTRADAY=0.22,
            SL_TIGHTENING_SWING=0.18,
            SL_TIGHTENING_POSITION=0.12,
            SL_TIGHTENING_FLOOR=0.06,
            SL_TIGHTENING_CEILING=0.45,
            SL_TIGHTENING_MIN_SAMPLES=15,
        )
    )
    defaults = StopLossTighteningPolicy()

    assert (
        from_config._scalping,
        from_config._intraday,
        from_config._swing,
        from_config._position,
        from_config._floor,
        from_config._ceiling,
        from_config._min_brain_samples,
    ) == (0.30, 0.22, 0.18, 0.12, 0.06, 0.45, 15)
    assert (
        defaults._scalping,
        defaults._intraday,
        defaults._swing,
        defaults._position,
        defaults._floor,
        defaults._ceiling,
        defaults._min_brain_samples,
    ) == (0.25, 0.20, 0.15, 0.10, 0.05, 0.40, 10)


@pytest.mark.parametrize(
    ("position", "proposed_sl"),
    [
        pytest.param(long_position(stop_loss=95.0), 90.0, id="long-widening"),
        pytest.param(long_position(stop_loss=95.0), 95.0, id="long-unchanged"),
        pytest.param(short_position(stop_loss=105.0), 106.0, id="short-widening"),
    ],
)
def test_non_tightening_moves_are_allowed_without_progress(
    position: Position, proposed_sl: float
) -> None:
    """Widening or holding the stop is never gated by price progress."""
    result = StopLossTighteningPolicy(swing_threshold=0.15).evaluate_update(
        position, proposed_sl=proposed_sl, current_price=103.0, tf_minutes=240
    )

    assert result.is_tightening is False
    assert result.allowed is True
    assert result.price_progress == 0.0
    assert result.base_min_progress == 0.15
    assert result.effective_min_progress == 0.15
    assert result.source == "config"
    assert result.reason == "Not a tightening move — SL is widening or unchanged."


@pytest.mark.parametrize(
    ("position", "price", "expected_allowed", "expected_progress"),
    [
        pytest.param(long_position(), 101.0, False, 1 / 15, id="long-below-threshold"),
        pytest.param(long_position(), 102.25, True, 0.15, id="long-exactly-at-threshold"),
        pytest.param(short_position(), 97.76, False, 0.14933333333333298, id="short-below"),
        pytest.param(short_position(), 97.75, True, 0.15, id="short-exactly-at-threshold"),
    ],
)
def test_tightening_gate_at_the_exact_progress_threshold(
    position: Position,
    price: float,
    expected_allowed: bool,
    expected_progress: float,
) -> None:
    """Progress is measured against the entry-to-target distance and allows equality."""
    result = StopLossTighteningPolicy(swing_threshold=0.15).evaluate_update(
        position, proposed_sl=97.0 if position.direction == "LONG" else 103.0,
        current_price=price, tf_minutes=240,
    )

    assert result.is_tightening is True
    assert result.allowed is expected_allowed
    assert result.price_progress == pytest.approx(expected_progress)
    assert result.base_min_progress == 0.15
    assert result.effective_min_progress == 0.15
    assert result.source == "config"
    if expected_allowed:
        assert result.reason.endswith("(source: config). Tightening allowed.")
    else:
        assert result.reason.endswith("(source: config). Premature tightening rejected.")


@pytest.mark.parametrize(
    ("take_profit", "price", "expected_allowed", "expected_reason"),
    [
        pytest.param(
            115.0,
            0.0,
            False,
            "No valid current price — tightening rejected as a safety measure.",
            id="zero-price",
        ),
        pytest.param(
            115.0,
            None,
            False,
            "No valid current price — tightening rejected as a safety measure.",
            id="missing-price",
        ),
        pytest.param(
            115.0,
            -5.0,
            False,
            "No valid current price — tightening rejected as a safety measure.",
            id="negative-price",
        ),
        pytest.param(
            115.0,
            float("inf"),
            True,
            "Price progress inf% >= required 15.0% (source: config). Tightening allowed.",
            id="infinite-price",
        ),
        pytest.param(
            100.0,
            102.0,
            False,
            "Cannot compute progress: entry equals take-profit.",
            id="flat-target",
        ),
    ],
)
def test_tightening_rejects_without_a_usable_price_or_distance(
    take_profit: float, price: float | None, expected_allowed: bool, expected_reason: str
) -> None:
    """A tightening is only evaluated against a positive price and a non-flat target."""
    result = StopLossTighteningPolicy(swing_threshold=0.15).evaluate_update(
        long_position(take_profit=take_profit),
        proposed_sl=97.0,
        current_price=price,
        tf_minutes=240,
    )

    assert result.is_tightening is True
    assert result.allowed is expected_allowed
    assert result.reason == expected_reason


@pytest.mark.parametrize(
    ("brain_thresholds", "min_samples", "expected_source", "expected_effective", "allowed"),
    [
        pytest.param(
            {"sl_tightening": {"sample_count": 5, "learned_threshold": 0.35}},
            5,
            "brain",
            0.35,
            True,
            id="samples-exactly-at-minimum",
        ),
        pytest.param(
            {"sl_tightening": {"sample_count": 4, "learned_threshold": 0.35}},
            5,
            "config",
            0.50,
            False,
            id="samples-one-below-minimum",
        ),
        pytest.param(
            {"sl_tightening": {"sample_count": 20, "learned_threshold": 0.01}},
            5,
            "brain",
            0.05,
            True,
            id="learned-below-floor",
        ),
        pytest.param(
            {"sl_tightening": {"sample_count": 20, "learned_threshold": 0.99}},
            5,
            "brain",
            0.40,
            True,
            id="learned-above-ceiling",
        ),
        pytest.param(
            {"sl_tightening": {"sample_count": 20, "learned_threshold": "bad"}},
            5,
            "config",
            0.50,
            False,
            id="learned-not-numeric",
        ),
        pytest.param(
            {"rr_borderline_min": 1.5}, 5, "config", 0.50, False, id="no-sl-tightening-key"
        ),
        pytest.param(None, 5, "config", 0.50, False, id="no-brain-thresholds"),
    ],
)
def test_brain_threshold_needs_samples_and_clamps(
    brain_thresholds: dict[str, Any] | None,
    min_samples: int,
    expected_source: str,
    expected_effective: float,
    allowed: bool,
) -> None:
    """The brain override applies only past the sample gate and stays inside floor/ceiling."""
    policy = StopLossTighteningPolicy(
        swing_threshold=0.50, floor=0.05, ceiling=0.40, min_brain_samples=min_samples
    )

    result = policy.evaluate_update(
        long_position(take_profit=120.0),
        proposed_sl=97.0,
        current_price=108.0,
        tf_minutes=240,
        brain_thresholds=brain_thresholds,
    )

    assert result.source == expected_source
    assert result.effective_min_progress == pytest.approx(expected_effective)
    assert result.base_min_progress == 0.50
    assert result.price_progress == pytest.approx(0.4)
    assert result.allowed is allowed


@pytest.mark.parametrize(
    ("stop_loss_type", "stop_loss_interval", "take_profit_type", "take_profit_interval",
     "expected_error"),
    [
        pytest.param(
            "hard", "2h", "hard", "15m",
            "stop loss interval '2h' must not be greater than timeframe '1h'",
            id="stop-loss-interval-above-timeframe",
        ),
        pytest.param(
            "hard", "5m", "hard", "2h",
            "take profit interval '2h' must not be greater than timeframe '1h'",
            id="take-profit-interval-above-timeframe",
        ),
        pytest.param(
            "trailing", "5m", "hard", "15m",
            "Invalid stop loss type 'trailing'. Expected soft or hard.",
            id="unknown-stop-loss-type",
        ),
        pytest.param(
            "hard", "5m", "grid", "15m",
            "Invalid take profit type 'grid'. Expected soft or hard.",
            id="unknown-take-profit-type",
        ),
        pytest.param(
            "hard", "0m", "hard", "15m",
            "stop loss interval must be positive",
            id="zero-interval",
        ),
        pytest.param("hard", "1h", "hard", "15m", None, id="interval-at-timeframe"),
    ],
)
def test_exit_monitor_validates_type_and_interval(
    stop_loss_type: str,
    stop_loss_interval: str,
    take_profit_type: str,
    take_profit_interval: str,
    expected_error: str | None,
) -> None:
    """Intervals may not exceed the timeframe; only soft/hard exit types are accepted."""
    monitor = ExitMonitor(
        exit_config(stop_loss_type, take_profit_type, stop_loss_interval, take_profit_interval),
        "1h",
        3600,
    )

    if expected_error is None:
        assert monitor.validate() is None
        assert monitor.exit_type(ExitMonitor.STOP_LOSS) == "hard"
        return

    with pytest.raises(ValueError) as excinfo:
        monitor.validate()

    assert str(excinfo.value) == expected_error


@pytest.mark.parametrize(
    ("state", "expected_due"),
    [
        pytest.param(
            {
                "last_stop_loss_check_at": (NOW - timedelta(seconds=299)).isoformat(),
                "last_take_profit_check_at": NOW.isoformat(),
            },
            [],
            id="one-second-short-of-the-interval",
        ),
        pytest.param(
            {
                "last_stop_loss_check_at": (NOW - timedelta(seconds=300)).isoformat(),
                "last_take_profit_check_at": NOW.isoformat(),
            },
            ["stop_loss"],
            id="exactly-at-the-interval",
        ),
        pytest.param(
            {
                "last_stop_loss_check_at": NOW.isoformat(),
                "last_take_profit_check_at": (NOW - timedelta(seconds=900)).isoformat(),
            },
            ["take_profit"],
            id="take-profit-exactly-at-its-interval",
        ),
        pytest.param({}, ["stop_loss", "take_profit"], id="no-persisted-timestamps"),
        pytest.param(
            {"last_stop_loss_check_at": "yesterday", "last_take_profit_check_at": "yesterday"},
            ["stop_loss", "take_profit"],
            id="unreadable-timestamps",
        ),
        pytest.param(
            {"last_take_profit_check_at": "2026-04-30T12:00:00"},
            ["stop_loss"],
            id="naive-timestamp-is-fresh-in-utc",
        ),
    ],
)
def test_hard_exits_are_due_at_the_exact_interval_boundary(
    state: dict[str, Any], expected_due: list[str]
) -> None:
    """A check is due at exactly its interval; a missing or unreadable timestamp is due at once."""
    monitor = ExitMonitor(exit_config(), "1h", 3600)

    assert monitor.due_hard_exits(NOW, state) == expected_due
    assert monitor.is_due("stop_loss", NOW, state) is ("stop_loss" in expected_due)


@pytest.mark.parametrize(
    ("stop_loss_type", "take_profit_type", "expected_status_seconds", "expected_due"),
    [
        pytest.param("soft", "soft", 3600, [], id="all-soft-uses-the-default-cadence"),
        pytest.param("soft", "hard", 900, ["take_profit"], id="take-profit-hard-only"),
        pytest.param("hard", "soft", 300, ["stop_loss"], id="stop-loss-hard-only"),
    ],
)
def test_soft_exits_are_never_due_and_hard_ones_set_the_status_cadence(
    stop_loss_type: str,
    take_profit_type: str,
    expected_status_seconds: int,
    expected_due: list[str],
) -> None:
    """Only hard exits are ticked; status updates run at the fastest hard interval."""
    config = exit_config(stop_loss_type, take_profit_type)
    monitor = ExitMonitor(config, "1h", 3600)
    state = {
        "last_status_sent_at": NOW.isoformat(),
        "last_stop_loss_check_at": (NOW - timedelta(minutes=5)).isoformat(),
        "last_take_profit_check_at": (NOW - timedelta(minutes=15)).isoformat(),
    }

    assert monitor.status_interval_seconds() == expected_status_seconds
    assert monitor.due_hard_exits(NOW, state) == expected_due
    assert monitor.is_status_due(NOW, state) is False


@pytest.mark.parametrize(
    ("state", "expected_delay"),
    [
        pytest.param(
            {
                "last_status_sent_at": (NOW - timedelta(minutes=2)).isoformat(),
                "last_stop_loss_check_at": (NOW - timedelta(minutes=2)).isoformat(),
                "last_take_profit_check_at": (NOW - timedelta(minutes=2)).isoformat(),
            },
            180.0,
            id="shortest-of-status-and-exits",
        ),
        pytest.param({}, 0.0, id="nothing-recorded-yet"),
    ],
)
def test_seconds_until_next_tick_uses_the_shortest_delay(
    state: dict[str, Any], expected_delay: float
) -> None:
    """The tick sleeps until the earliest of the status and hard-exit deadlines."""
    monitor = ExitMonitor(exit_config(), "1h", 3600)

    assert monitor.seconds_until_next_tick(state, NOW) == expected_delay


@pytest.mark.parametrize(
    ("state", "price", "expected_calls", "expected_stamps"),
    [
        pytest.param(
            {
                "last_stop_loss_check_at": NOW.isoformat(),
                "last_take_profit_check_at": (NOW - timedelta(minutes=15)).isoformat(),
            },
            100.0,
            [("check_take_profit", 100.0)],
            {"last_take_profit_check_at": NOW},
            id="take-profit-only",
        ),
        pytest.param(
            {
                "last_stop_loss_check_at": NOW.isoformat(),
                "last_take_profit_check_at": NOW.isoformat(),
            },
            100.0,
            [],
            {},
            id="nothing-due",
        ),
        pytest.param(
            {
                "last_stop_loss_check_at": (NOW - timedelta(minutes=5)).isoformat(),
                "last_take_profit_check_at": (NOW - timedelta(minutes=15)).isoformat(),
            },
            None,
            [],
            {},
            id="no-price",
        ),
    ],
)
async def test_hard_exit_routing_selects_only_the_due_side(
    state: dict[str, Any],
    price: float | None,
    expected_calls: list[tuple[str, float]],
    expected_stamps: dict[str, datetime],
) -> None:
    """Exactly the due side is checked, and an unavailable price skips the tick entirely."""
    strategy = RecordingExitStrategy()
    monitor = ExitMonitor(exit_config(), "1h", 3600)

    close_reason, timestamps = await monitor.check_hard_exits(
        strategy, price, NOW, state, asyncio.Lock()
    )

    assert close_reason is None
    assert strategy.calls == expected_calls
    assert timestamps == expected_stamps


@pytest.mark.parametrize(
    ("state", "price", "close_reason", "expected_reason", "expected_keys"),
    [
        pytest.param(
            {
                "last_stop_loss_check_at": (NOW - timedelta(minutes=5)).isoformat(),
                "last_take_profit_check_at": (NOW - timedelta(minutes=10)).isoformat(),
            },
            100.0,
            None,
            None,
            ["last_stop_loss_check_at"],
            id="stop-loss-due-only",
        ),
        pytest.param(
            {
                "last_stop_loss_check_at": (NOW - timedelta(minutes=5)).isoformat(),
                "last_take_profit_check_at": (NOW - timedelta(minutes=15)).isoformat(),
            },
            100.0,
            None,
            None,
            ["last_stop_loss_check_at", "last_take_profit_check_at"],
            id="both-exits-due",
        ),
        pytest.param(
            {
                "last_stop_loss_check_at": (NOW - timedelta(minutes=5)).isoformat(),
                "last_take_profit_check_at": (NOW - timedelta(minutes=15)).isoformat(),
            },
            94.0,
            "stop_loss",
            "stop_loss",
            ["last_stop_loss_check_at", "last_take_profit_check_at"],
            id="both-due-and-the-position-closes",
        ),
    ],
)
async def test_position_monitor_persists_due_exit_timestamps(
    state: dict[str, Any],
    price: float,
    close_reason: str | None,
    expected_reason: str | None,
    expected_keys: list[str],
) -> None:
    """The monitor stores each executed hard-exit check under its persisted key."""
    strategy = RecordingExitStrategy(close_reason=close_reason) if close_reason else None
    context = monitor_context(state=state, strategy=strategy)

    reason = await context.position_monitor.run_hard_exit_checks(price, NOW, state)

    assert reason == expected_reason
    stamps = {
        key: value
        for key, value in context.persistence.state.items()
        if key.startswith("last_")
    }
    assert stamps == {
        key: (NOW.isoformat() if key in expected_keys else value)
        for key, value in state.items()
    }
    assert context.persistence.state["symbol"] == CONFIG_PAIR
    assert context.persistence.state["stop_loss_type"] == "hard"
    assert context.persistence.state["take_profit_check_interval"] == "15m"


async def test_position_status_loop_waits_then_runs_due_checks_and_status() -> None:
    """The loop sleeps once, then checks the due exit and sends the due status update."""
    started = datetime.now(timezone.utc)
    state = {
        "last_status_sent_at": (started - timedelta(minutes=2)).isoformat(),
        "last_stop_loss_check_at": (started - timedelta(minutes=2)).isoformat(),
        "last_take_profit_check_at": (started - timedelta(minutes=2)).isoformat(),
    }
    strategy = MagicMock()
    strategy.current_position = long_position(stop_loss=95.0, take_profit=110.0)
    strategy.check_stop_loss = AsyncMock(return_value=None)
    strategy.check_take_profit = AsyncMock(return_value=None)
    notifier = MagicMock()
    notifier.send_position_status = AsyncMock()
    loop_checks = {"count": 0}
    sleep_calls: list[tuple[float, bool]] = []
    written: dict[str, str] = {}

    def is_running() -> bool:
        loop_checks["count"] += 1
        return loop_checks["count"] <= 2

    async def fake_sleep(seconds: float, respect_force_analysis: bool = True) -> bool:
        sleep_calls.append((seconds, respect_force_analysis))
        due_time = datetime.now(timezone.utc)
        context.persistence.state.update(
            {
                "last_status_sent_at": (due_time - timedelta(minutes=5, seconds=1)).isoformat(),
                "last_stop_loss_check_at": (
                    due_time - timedelta(minutes=5, seconds=1)
                ).isoformat(),
                "last_take_profit_check_at": (due_time - timedelta(minutes=2)).isoformat(),
            }
        )
        written["take_profit"] = context.persistence.state["last_take_profit_check_at"]
        return False

    context = monitor_context(
        state=state,
        strategy=strategy,
        notifier=notifier,
        is_running=is_running,
        fetch_current_ticker=AsyncMock(return_value={"last": 100.0}),
        interruptible_sleep=fake_sleep,
    )

    await context.position_monitor._loop()

    assert len(sleep_calls) == 1
    assert 170 <= sleep_calls[0][0] <= 190
    assert sleep_calls[0][1] is False
    strategy.check_stop_loss.assert_awaited_once_with(100.0)
    strategy.check_take_profit.assert_not_called()
    notifier.send_position_status.assert_awaited_once_with(
        position=strategy.current_position, current_price=100.0, channel_id=123
    )
    assert context.persistence.state["last_stop_loss_check_at"] != state["last_stop_loss_check_at"]
    assert context.persistence.state["last_status_sent_at"] != state["last_status_sent_at"]
    assert context.persistence.state["last_take_profit_check_at"] == written["take_profit"]


@pytest.mark.parametrize(
    ("method", "direction", "price", "expected_reason"),
    [
        pytest.param(
            TradingStrategy.check_stop_loss, "LONG", 111.0, None, id="long-stop-not-hit"
        ),
        pytest.param(
            TradingStrategy.check_stop_loss, "LONG", 94.0, "stop_loss", id="long-stop-hit"
        ),
        pytest.param(
            TradingStrategy.check_take_profit, "LONG", 94.0, None, id="long-target-not-hit"
        ),
        pytest.param(
            TradingStrategy.check_take_profit,
            "LONG",
            111.0,
            "take_profit",
            id="long-target-hit",
        ),
        pytest.param(
            TradingStrategy.check_stop_loss, "SHORT", 106.0, "stop_loss", id="short-stop-hit"
        ),
        pytest.param(
            TradingStrategy.check_take_profit,
            "SHORT",
            89.0,
            "take_profit",
            id="short-target-hit",
        ),
    ],
)
async def test_stop_loss_and_take_profit_checks_are_side_specific(
    method: Any, direction: str, price: float, expected_reason: str | None
) -> None:
    """Each check only evaluates its own exit while still persisting live metrics."""
    strategy = exit_strategy(direction)

    reason = await method(strategy, price)

    assert reason == expected_reason
    assert strategy.persistence.async_save_position.await_count == 1
    if expected_reason is None:
        strategy.close_position.assert_not_awaited()
        return
    close_args = strategy.close_position.await_args.args
    assert close_args[:2] == (expected_reason, price)
    assert type(close_args[2]) is MarketConditions


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        pytest.param(
            {
                "signal": "BUY",
                "confidence": "HIGH",
                "entry_price": float("nan"),
                "stop_loss": float("inf"),
                "take_profit": "not_a_number",
                "position_size": float("-inf"),
                "reasoning": "Corrupted text test",
            },
            ("BUY", "MEDIUM", None, None, None, "Corrupted text test"),
            id="nan-inf-and-text",
        ),
        pytest.param(
            {"signal": "BUY", "confidence": 82, "stop_loss": None, "take_profit": None,
             "position_size": None},
            ("BUY", "HIGH", None, None, None, ""),
            id="null-levels",
        ),
        pytest.param(
            {"signal": "BUY", "confidence": float("nan"), "stop_loss": 95.0,
             "position_size": 0.05},
            ("BUY", "MEDIUM", 95.0, None, 0.05, ""),
            id="nan-confidence",
        ),
        pytest.param(
            {"signal": "SELL", "confidence": "LOW", "stop_loss": float("inf"),
             "take_profit": -float("inf"), "position_size": 0.05},
            ("SELL", "MEDIUM", None, None, 0.05, ""),
            id="infinite-levels",
        ),
        pytest.param(
            {"stop_loss": 95.0},
            ("HOLD", "MEDIUM", 95.0, None, None, ""),
            id="signal-missing",
        ),
    ],
)
def test_corrupted_numeric_payload_normalizes_to_none(
    payload: dict[str, Any], expected: tuple[Any, ...]
) -> None:
    """The parser is the only normalizer: non-finite levels and sizing arrive as None."""
    raw = f"```json\n{json.dumps({'analysis': payload})}\n```"
    analysis = PARSER.parse_ai_response(raw)["analysis"]

    assert EXTRACTOR.extract_trading_info(analysis) == expected
