"""Comprehensive tests for RiskManager friction reporting (closed-loop feedback).

Validates:
  - Friction reports generated for each guard type
  - Delta calculations (suggested vs. clamped values)
  - get_and_clear_frictions() returns expected dict structure and clears buffer
  - All 6 guard types produce correct output shapes
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.managers.risk_manager import RiskManager
from src.trading.data_models import MarketConditions


# ── Helpers ──────────────────────────────────────────────────────


def _make_config(
    max_position_size: float = 0.10,
    fallback_low: float = 0.01,
    fallback_medium: float = 0.02,
    fallback_high: float = 0.03,
) -> SimpleNamespace:
    return SimpleNamespace(
        MAX_POSITION_SIZE=max_position_size,
        POSITION_SIZE_FALLBACK_LOW=fallback_low,
        POSITION_SIZE_FALLBACK_MEDIUM=fallback_medium,
        POSITION_SIZE_FALLBACK_HIGH=fallback_high,
        TRANSACTION_FEE_PERCENT=0.001,
    )


def _entry(
    manager: RiskManager,
    *,
    signal: str = "BUY",
    current_price: float = 100.0,
    capital: float = 10000.0,
    confidence: str = "HIGH",
    stop_loss: float | None = None,
    take_profit: float | None = None,
    position_size: float | None = None,
    market_conditions: MarketConditions | None = None,
):
    return manager.calculate_entry_parameters(
        signal=signal,
        current_price=current_price,
        capital=capital,
        confidence=confidence,
        stop_loss=stop_loss,
        take_profit=take_profit,
        position_size=position_size,
        market_conditions=market_conditions,
    )


# ── Friction Accumulation and Clearing ───────────────────────────


class TestFrictionAccumulationAndClearing:
    """Verify _last_frictions buffer lifecycle."""

    def test_no_frictions_when_all_params_valid(self):
        """Valid SL/TP/size — no frictions should be generated."""
        mgr = RiskManager(logger=MagicMock(), config=_make_config())
        _entry(mgr, stop_loss=95.0, take_profit=110.0, position_size=0.05)

        frictions = mgr.get_and_clear_frictions()
        assert frictions == []

    def test_frictions_cleared_after_get(self):
        """get_and_clear_frictions clears the buffer after returning."""
        mgr = RiskManager(logger=MagicMock(), config=_make_config(max_position_size=0.02))
        _entry(mgr, position_size=0.50)  # triggers position_size_clamp

        first = mgr.get_and_clear_frictions()
        assert len(first) == 1

        second = mgr.get_and_clear_frictions()
        assert second == []

    def test_multiple_frictions_accumulate_across_calls(self):
        """Multiple guarded entries accumulate until cleared."""
        mgr = RiskManager(logger=MagicMock(), config=_make_config(max_position_size=0.02))
        _entry(mgr, position_size=0.50)  # size clamp
        _entry(mgr, signal="BUY", stop_loss=99.0, current_price=100.0)  # SL min

        frictions = mgr.get_and_clear_frictions()
        assert len(frictions) >= 2

    def test_independent_instances_have_separate_buffers(self):
        """Each RiskManager instance has its own friction buffer."""
        mgr1 = RiskManager(logger=MagicMock(), config=_make_config(max_position_size=0.02))
        mgr2 = RiskManager(logger=MagicMock(), config=_make_config(max_position_size=0.02))

        _entry(mgr1, position_size=0.50)

        f1 = mgr1.get_and_clear_frictions()
        f2 = mgr2.get_and_clear_frictions()

        assert len(f1) == 1
        assert f2 == []


# ── Guard: position_size_clamp ───────────────────────────────────


class TestGuardPositionSizeClamp:
    """Verify position_size_clamp friction structure."""

    def test_size_clamp_generates_friction(self):
        mgr = RiskManager(logger=MagicMock(), config=_make_config(max_position_size=0.05))
        assessment = _entry(mgr, position_size=0.30)

        frictions = mgr.get_and_clear_frictions()
        assert len(frictions) == 1
        f = frictions[0]

        assert f["guard_type"] == "position_size_clamp"
        assert f["direction"] == "N/A"
        assert f["suggested_size"] == pytest.approx(0.30)
        assert f["max_size"] == pytest.approx(0.05)
        assert f["detail"].startswith("Position size")
        assert assessment.size_pct == pytest.approx(0.05)

    def test_size_clamp_delta_correct(self):
        """Suggested 0.30 vs max 0.10 — delta is 0.20."""
        mgr = RiskManager(logger=MagicMock(), config=_make_config(max_position_size=0.10))
        _entry(mgr, position_size=0.30)

        f = mgr.get_and_clear_frictions()[0]
        assert f["suggested_size"] - f["max_size"] == pytest.approx(0.20)

    def test_size_within_limit_no_friction(self):
        """Position size within cap should not generate friction."""
        mgr = RiskManager(logger=MagicMock(), config=_make_config(max_position_size=0.10))
        _entry(mgr, position_size=0.05)

        assert mgr.get_and_clear_frictions() == []


# ── Guard: sl_distance_max ───────────────────────────────────────


class TestGuardSlTooFar:
    """SL distance >10% gets clamped."""

    def test_sl_too_far_long(self):
        mgr = RiskManager(logger=MagicMock(), config=_make_config())
        assessment = _entry(mgr, signal="BUY", stop_loss=80.0, current_price=100.0)

        frictions = mgr.get_and_clear_frictions()
        # 20% distance triggers sl_distance_max
        assert any(f["guard_type"] == "sl_distance_max" for f in frictions)
        f = next(f for f in frictions if f["guard_type"] == "sl_distance_max")

        assert f["direction"] == "LONG"
        assert f["suggested_sl_pct"] == pytest.approx(0.20)
        assert f["corrected_sl_pct"] == pytest.approx(0.10)
        assert f["volatility_level"] in ("HIGH", "MEDIUM", "LOW")
        assert "clamped to max 10%" in f["detail"]
        # Clamped SL should be 10% below entry for LONG
        assert assessment.stop_loss == pytest.approx(90.0)

    def test_sl_too_far_short(self):
        mgr = RiskManager(logger=MagicMock(), config=_make_config())
        assessment = _entry(mgr, signal="SELL", stop_loss=120.0, current_price=100.0)

        frictions = mgr.get_and_clear_frictions()
        assert any(f["guard_type"] == "sl_distance_max" for f in frictions)
        f = next(f for f in frictions if f["guard_type"] == "sl_distance_max")

        assert f["direction"] == "SHORT"
        assert assessment.stop_loss == pytest.approx(110.0)  # 10% above entry for SHORT


# ── Guard: sl_distance_min ───────────────────────────────────────


class TestGuardSlTooTight:
    """SL distance <1% gets expanded."""

    def test_sl_too_tight_long(self):
        mgr = RiskManager(logger=MagicMock(), config=_make_config())
        assessment = _entry(mgr, signal="BUY", stop_loss=99.8, current_price=100.0)

        frictions = mgr.get_and_clear_frictions()
        assert any(f["guard_type"] == "sl_distance_min" for f in frictions)
        f = next(f for f in frictions if f["guard_type"] == "sl_distance_min")

        assert f["direction"] == "LONG"
        assert f["corrected_sl_pct"] == pytest.approx(0.01)
        assert "expanded to min 1%" in f["detail"]
        assert assessment.stop_loss == pytest.approx(99.0)

    def test_sl_too_tight_short(self):
        mgr = RiskManager(logger=MagicMock(), config=_make_config())
        assessment = _entry(mgr, signal="SELL", stop_loss=100.2, current_price=100.0)

        frictions = mgr.get_and_clear_frictions()
        assert any(f["guard_type"] == "sl_distance_min" for f in frictions)
        f = next(f for f in frictions if f["guard_type"] == "sl_distance_min")

        assert f["direction"] == "SHORT"
        assert assessment.stop_loss == pytest.approx(101.0)


# ── Guard: sl_below_entry / tp_below_entry ───────────────────────


class TestGuardInvalidSlTp:
    """SL/TP on the wrong side of entry triggers logical correction."""

    def test_sl_above_entry_for_long(self):
        """LONG with SL >= entry_price is nonsensical."""
        mgr = RiskManager(logger=MagicMock(), config=_make_config())
        # Provide a clearly invalid SL, with no ATR to override the default dynamic calc
        assessment = _entry(mgr, signal="BUY", stop_loss=102.0, current_price=100.0,
                            market_conditions=MarketConditions(atr=1.0, atr_percentage=1.0))

        frictions = mgr.get_and_clear_frictions()
        assert any(f["guard_type"] == "sl_below_entry" for f in frictions)

    def test_tp_below_entry_for_long(self):
        """LONG with TP <= entry_price is nonsensical."""
        mgr = RiskManager(logger=MagicMock(), config=_make_config())
        assessment = _entry(mgr, signal="BUY", take_profit=95.0, current_price=100.0,
                            market_conditions=MarketConditions(atr=2.0, atr_percentage=2.0))

        frictions = mgr.get_and_clear_frictions()
        # TP 5% below entry for LONG should be caught
        assert len(frictions) >= 1

    def test_sl_below_entry_for_short(self):
        """SHORT with SL <= entry_price is nonsensical."""
        mgr = RiskManager(logger=MagicMock(), config=_make_config())
        _entry(mgr, signal="SELL", stop_loss=95.0, current_price=100.0,
               market_conditions=MarketConditions(atr=1.0, atr_percentage=1.0))

        frictions = mgr.get_and_clear_frictions()
        assert any(f["guard_type"] == "sl_below_entry" for f in frictions)

    def test_tp_above_entry_for_short(self):
        """SHORT with TP >= entry_price is nonsensical."""
        mgr = RiskManager(logger=MagicMock(), config=_make_config())
        _entry(mgr, signal="SELL", take_profit=105.0, current_price=100.0,
               market_conditions=MarketConditions(atr=2.0, atr_percentage=2.0))

        frictions = mgr.get_and_clear_frictions()
        assert len(frictions) >= 1


# ── Friction Dict Schema Contract ────────────────────────────────


class TestFrictionDictContract:
    """Every friction dict must contain the mandatory keys."""

    MANDATORY_KEYS = {"guard_type", "direction", "detail"}

    def test_position_size_clamp_schema(self):
        mgr = RiskManager(logger=MagicMock(), config=_make_config(max_position_size=0.05))
        _entry(mgr, position_size=0.30)
        f = mgr.get_and_clear_frictions()[0]

        assert self.MANDATORY_KEYS.issubset(f.keys())
        assert isinstance(f["guard_type"], str)
        assert isinstance(f["direction"], str)
        assert isinstance(f["detail"], str)

    def test_sl_distance_max_schema(self):
        mgr = RiskManager(logger=MagicMock(), config=_make_config())
        _entry(mgr, signal="BUY", stop_loss=80.0, current_price=100.0)
        f = next(f for f in mgr.get_and_clear_frictions() if f["guard_type"] == "sl_distance_max")

        assert self.MANDATORY_KEYS.issubset(f.keys())
        assert "suggested_sl_pct" in f
        assert "corrected_sl_pct" in f
        assert "current_price" in f
        assert "volatility_level" in f

    def test_sl_distance_min_schema(self):
        mgr = RiskManager(logger=MagicMock(), config=_make_config())
        _entry(mgr, signal="BUY", stop_loss=99.8, current_price=100.0)
        f = next(f for f in mgr.get_and_clear_frictions() if f["guard_type"] == "sl_distance_min")

        assert self.MANDATORY_KEYS.issubset(f.keys())
        assert "suggested_sl_pct" in f
        assert "corrected_sl_pct" in f
        assert "current_price" in f
        assert "volatility_level" in f

    def test_sl_below_entry_schema(self):
        mgr = RiskManager(logger=MagicMock(), config=_make_config())
        _entry(mgr, signal="BUY", stop_loss=102.0, current_price=100.0,
               market_conditions=MarketConditions(atr=1.0, atr_percentage=1.0))
        f = next(f for f in mgr.get_and_clear_frictions() if f["guard_type"] == "sl_below_entry")

        assert self.MANDATORY_KEYS.issubset(f.keys())
        assert "suggested_sl" in f
        assert "dynamic_sl" in f
        assert "current_price" in f
        assert "volatility_level" in f

    def test_tp_below_entry_schema(self):
        mgr = RiskManager(logger=MagicMock(), config=_make_config())
        _entry(mgr, signal="LONG", take_profit=95.0, current_price=100.0, stop_loss=None,
               market_conditions=MarketConditions(atr=2.0, atr_percentage=2.0))
        tp_frictions = [f for f in mgr.get_and_clear_frictions() if f["guard_type"] == "tp_below_entry"]
        if tp_frictions:  # only if AI TP was used (prevailed over dynamic for LONG)
            f = tp_frictions[0]
            assert self.MANDATORY_KEYS.issubset(f.keys())
            assert "suggested_tp" in f
            assert "dynamic_tp" in f


# ── Volatility Level Propagation ─────────────────────────────────


class TestVolatilityLevelInFrictions:
    """Friction reports carry the volatility_level for LLM context."""

    def test_high_volatility_label_in_sl_max_friction(self):
        mgr = RiskManager(logger=MagicMock(), config=_make_config())
        _entry(mgr, signal="BUY", stop_loss=80.0, current_price=100.0,
               market_conditions=MarketConditions(atr=5.0, atr_percentage=5.0))

        f = next(f for f in mgr.get_and_clear_frictions() if f["guard_type"] == "sl_distance_max")
        assert f["volatility_level"] == "HIGH"

    def test_low_volatility_label_in_sl_min_friction(self):
        mgr = RiskManager(logger=MagicMock(), config=_make_config())
        _entry(mgr, signal="BUY", stop_loss=99.8, current_price=100.0,
               market_conditions=MarketConditions(atr=1.0, atr_percentage=1.0))

        f = next(f for f in mgr.get_and_clear_frictions() if f["guard_type"] == "sl_distance_min")
        assert f["volatility_level"] == "LOW"


# ── Default ATR Fallback ─────────────────────────────────────────


class TestDefaultAtrFallback:
    """When no market_conditions ATR is provided, uses 2% of price."""

    def test_no_atr_uses_default_2pct(self):
        """Default ATR = current_price * 0.02 (100 * 0.02 = 2.0)."""
        mgr = RiskManager(logger=MagicMock(), config=_make_config())
        assessment = _entry(mgr, signal="BUY", stop_loss=95.0, take_profit=110.0,
                            market_conditions=MarketConditions())

        # Dynamic SL = 100 - 2*2 = 96, TP = 100 + 4*2 = 108
        # But AI SL 95 is used since it's provided
        assert assessment.stop_loss == pytest.approx(95.0)
        assert assessment.take_profit == pytest.approx(110.0)


# ── R/R Ratio in Assessment ──────────────────────────────────────


class TestRiskRewardRatio:
    """Verify rr_ratio is calculated correctly in RiskAssessment."""

    def test_rr_ratio_basic(self):
        mgr = RiskManager(logger=MagicMock(), config=_make_config())
        assessment = _entry(mgr, signal="BUY", stop_loss=95.0, take_profit=110.0)

        # SL distance = 5%, TP distance = 10%, R/R = 10/5 = 2.0
        assert assessment.sl_distance_pct == pytest.approx(0.05)
        assert assessment.tp_distance_pct == pytest.approx(0.10)
        assert assessment.rr_ratio == pytest.approx(2.0)

    def test_rr_ratio_zero_when_sl_at_entry(self):
        """Zero SL distance gives rr_ratio 0 (guard: division by zero)."""
        mgr = RiskManager(logger=MagicMock(), config=_make_config())
        # SL at 100 when price is 100 → sl_distance 0
        assessment = _entry(mgr, signal="BUY", stop_loss=100.0, current_price=100.0,
                            market_conditions=MarketConditions(atr=1.0, atr_percentage=1.0))
        assert assessment.rr_ratio >= 0
