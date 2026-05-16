"""Unit tests for StopLossTighteningPolicy."""

import pytest

from src.trading.data_models import Position
from src.trading.stop_loss_tightening_policy import StopLossTighteningPolicy, TighteningEvaluation


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────


def _make_position(
    entry_price: float = 100.0,
    stop_loss: float = 95.0,
    take_profit: float = 115.0,
    direction: str = "LONG",
) -> Position:
    return Position(
        symbol="BTC/USDT",
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        size=0.01,
        entry_time="2024-01-01T00:00:00+00:00",
        confidence="MEDIUM",
    )


def _make_config(**overrides):
    """Return a minimal config-like namespace for from_config()."""
    from types import SimpleNamespace
    defaults = {
        "SL_TIGHTENING_SCALPING": 0.25,
        "SL_TIGHTENING_INTRADAY": 0.20,
        "SL_TIGHTENING_SWING": 0.15,
        "SL_TIGHTENING_POSITION": 0.10,
        "SL_TIGHTENING_FLOOR": 0.05,
        "SL_TIGHTENING_CEILING": 0.40,
        "SL_TIGHTENING_MIN_SAMPLES": 10,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ─────────────────────────────────────────────────────────────────
# Timeframe bucket thresholds
# ─────────────────────────────────────────────────────────────────


class TestGetBaseThreshold:
    def test_scalping_bucket(self):
        policy = StopLossTighteningPolicy()
        assert policy.get_base_threshold(1) == 0.25
        assert policy.get_base_threshold(59) == 0.25

    def test_intraday_bucket(self):
        policy = StopLossTighteningPolicy()
        assert policy.get_base_threshold(60) == 0.20
        assert policy.get_base_threshold(239) == 0.20

    def test_swing_bucket(self):
        policy = StopLossTighteningPolicy()
        assert policy.get_base_threshold(240) == 0.15
        assert policy.get_base_threshold(1439) == 0.15

    def test_position_bucket(self):
        policy = StopLossTighteningPolicy()
        assert policy.get_base_threshold(1440) == 0.10
        assert policy.get_base_threshold(10080) == 0.10


# ─────────────────────────────────────────────────────────────────
# from_config classmethod
# ─────────────────────────────────────────────────────────────────


class TestFromConfig:
    def test_reads_all_keys(self):
        cfg = _make_config(
            SL_TIGHTENING_SCALPING=0.30,
            SL_TIGHTENING_INTRADAY=0.22,
            SL_TIGHTENING_SWING=0.18,
            SL_TIGHTENING_POSITION=0.12,
            SL_TIGHTENING_FLOOR=0.06,
            SL_TIGHTENING_CEILING=0.45,
            SL_TIGHTENING_MIN_SAMPLES=15,
        )
        policy = StopLossTighteningPolicy.from_config(cfg)
        assert policy._scalping == 0.30
        assert policy._intraday == 0.22
        assert policy._swing == 0.18
        assert policy._position == 0.12
        assert policy._floor == 0.06
        assert policy._ceiling == 0.45
        assert policy._min_brain_samples == 15

    def test_defaults_are_stable(self):
        """Default constructor matches expected hardcoded values."""
        policy = StopLossTighteningPolicy()
        assert policy._scalping == 0.25
        assert policy._intraday == 0.20
        assert policy._swing == 0.15
        assert policy._position == 0.10
        assert policy._floor == 0.05
        assert policy._ceiling == 0.40
        assert policy._min_brain_samples == 10


# ─────────────────────────────────────────────────────────────────
# Non-tightening moves
# ─────────────────────────────────────────────────────────────────


class TestNonTighteningMoves:
    def test_long_sl_widening_always_allowed(self):
        pos = _make_position(stop_loss=95.0, direction="LONG")
        policy = StopLossTighteningPolicy()
        result = policy.evaluate_update(pos, proposed_sl=90.0, current_price=105.0, tf_minutes=240)
        assert result.is_tightening is False
        assert result.allowed is True

    def test_short_sl_widening_always_allowed(self):
        pos = _make_position(stop_loss=105.0, direction="SHORT", take_profit=85.0)
        policy = StopLossTighteningPolicy()
        result = policy.evaluate_update(pos, proposed_sl=110.0, current_price=95.0, tf_minutes=240)
        assert result.is_tightening is False
        assert result.allowed is True

    def test_sl_unchanged_is_not_tightening(self):
        pos = _make_position(stop_loss=95.0, direction="LONG")
        policy = StopLossTighteningPolicy()
        result = policy.evaluate_update(pos, proposed_sl=95.0, current_price=105.0, tf_minutes=240)
        assert result.is_tightening is False
        assert result.allowed is True


# ─────────────────────────────────────────────────────────────────
# LONG tightening
# ─────────────────────────────────────────────────────────────────


class TestLongTightening:
    def test_rejected_when_progress_below_threshold(self):
        # progress = (101.0 - 100) / (115 - 100) = 1/15 ≈ 0.067  <  0.15
        pos = _make_position(entry_price=100.0, stop_loss=95.0, take_profit=115.0, direction="LONG")
        policy = StopLossTighteningPolicy(swing_threshold=0.15)
        result = policy.evaluate_update(pos, proposed_sl=97.0, current_price=101.0, tf_minutes=240)
        assert result.is_tightening is True
        assert result.allowed is False
        assert result.price_progress == pytest.approx(1 / 15)

    def test_allowed_when_progress_at_threshold(self):
        # progress = (102.25 - 100) / (115 - 100) = 2.25/15 = 0.15 == threshold
        pos = _make_position(entry_price=100.0, stop_loss=95.0, take_profit=115.0, direction="LONG")
        policy = StopLossTighteningPolicy(swing_threshold=0.15)
        result = policy.evaluate_update(pos, proposed_sl=97.0, current_price=102.25, tf_minutes=240)
        assert result.allowed is True

    def test_allowed_when_progress_above_threshold(self):
        # progress = (103 - 100) / (115 - 100) = 3/15 = 0.20  >  0.15
        pos = _make_position(entry_price=100.0, stop_loss=95.0, take_profit=115.0, direction="LONG")
        policy = StopLossTighteningPolicy(swing_threshold=0.15)
        result = policy.evaluate_update(pos, proposed_sl=97.0, current_price=103.0, tf_minutes=240)
        assert result.is_tightening is True
        assert result.allowed is True


# ─────────────────────────────────────────────────────────────────
# SHORT tightening
# ─────────────────────────────────────────────────────────────────


class TestShortTightening:
    def test_rejected_when_progress_below_threshold(self):
        # entry=100, sl=105, tp=85 (SHORT), current=99
        # progress = (100 - 99) / (100 - 85) = 1/15 ≈ 0.067  <  0.15
        pos = _make_position(entry_price=100.0, stop_loss=105.0, take_profit=85.0, direction="SHORT")
        policy = StopLossTighteningPolicy(swing_threshold=0.15)
        result = policy.evaluate_update(pos, proposed_sl=103.0, current_price=99.0, tf_minutes=240)
        assert result.is_tightening is True
        assert result.allowed is False

    def test_allowed_when_progress_sufficient(self):
        # progress = (100 - 97) / (100 - 85) = 3/15 = 0.20  >  0.15
        pos = _make_position(entry_price=100.0, stop_loss=105.0, take_profit=85.0, direction="SHORT")
        policy = StopLossTighteningPolicy(swing_threshold=0.15)
        result = policy.evaluate_update(pos, proposed_sl=103.0, current_price=97.0, tf_minutes=240)
        assert result.is_tightening is True
        assert result.allowed is True


# ─────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_no_current_price_rejects_tightening(self):
        pos = _make_position(stop_loss=95.0, direction="LONG")
        policy = StopLossTighteningPolicy()
        result = policy.evaluate_update(pos, proposed_sl=97.0, current_price=0.0, tf_minutes=240)
        assert result.is_tightening is True
        assert result.allowed is False
        assert "No valid current price" in result.reason

    def test_none_current_price_rejects_tightening(self):
        pos = _make_position(stop_loss=95.0, direction="LONG")
        policy = StopLossTighteningPolicy()
        result = policy.evaluate_update(pos, proposed_sl=97.0, current_price=None, tf_minutes=240)  # type: ignore[arg-type]
        assert result.allowed is False

    def test_zero_tp_distance_rejects_tightening(self):
        # entry == take_profit → tp_distance_total == 0
        pos = _make_position(entry_price=100.0, stop_loss=95.0, take_profit=100.0, direction="LONG")
        policy = StopLossTighteningPolicy()
        result = policy.evaluate_update(pos, proposed_sl=97.0, current_price=102.0, tf_minutes=240)
        assert result.is_tightening is True
        assert result.allowed is False
        assert "entry equals take-profit" in result.reason


# ─────────────────────────────────────────────────────────────────
# Brain override
# ─────────────────────────────────────────────────────────────────


class TestBrainOverride:
    def _long_pos(self) -> Position:
        return _make_position(entry_price=100.0, stop_loss=95.0, take_profit=120.0, direction="LONG")

    def test_brain_override_applied_when_samples_sufficient(self):
        # progress = (108 - 100) / (120 - 100) = 8/20 = 0.40
        # config swing = 0.50 (would reject), brain learned = 0.35 (would accept)
        pos = self._long_pos()
        policy = StopLossTighteningPolicy(swing_threshold=0.50, min_brain_samples=5)
        brain = {"sl_tightening": {"sample_count": 10, "learned_threshold": 0.35}}
        result = policy.evaluate_update(pos, proposed_sl=97.0, current_price=108.0, tf_minutes=240, brain_thresholds=brain)
        assert result.source == "brain"
        assert result.effective_min_progress == pytest.approx(0.35)
        assert result.allowed is True

    def test_brain_override_ignored_below_min_samples(self):
        # same scenario but only 3 samples → config threshold used (0.50 → reject)
        pos = self._long_pos()
        policy = StopLossTighteningPolicy(swing_threshold=0.50, min_brain_samples=5)
        brain = {"sl_tightening": {"sample_count": 3, "learned_threshold": 0.35}}
        result = policy.evaluate_update(pos, proposed_sl=97.0, current_price=108.0, tf_minutes=240, brain_thresholds=brain)
        assert result.source == "config"
        assert result.effective_min_progress == pytest.approx(0.50)
        assert result.allowed is False

    def test_brain_override_clamped_to_floor(self):
        pos = self._long_pos()
        policy = StopLossTighteningPolicy(floor=0.05, ceiling=0.40, min_brain_samples=1)
        brain = {"sl_tightening": {"sample_count": 5, "learned_threshold": 0.01}}
        result = policy.evaluate_update(pos, proposed_sl=97.0, current_price=101.0, tf_minutes=240, brain_thresholds=brain)
        assert result.source == "brain"
        assert result.effective_min_progress == pytest.approx(0.05)

    def test_brain_override_clamped_to_ceiling(self):
        pos = self._long_pos()
        policy = StopLossTighteningPolicy(floor=0.05, ceiling=0.40, min_brain_samples=1)
        brain = {"sl_tightening": {"sample_count": 5, "learned_threshold": 0.99}}
        result = policy.evaluate_update(pos, proposed_sl=97.0, current_price=109.0, tf_minutes=240, brain_thresholds=brain)
        assert result.source == "brain"
        assert result.effective_min_progress == pytest.approx(0.40)

    def test_brain_missing_sl_tightening_key_falls_back_to_config(self):
        pos = self._long_pos()
        policy = StopLossTighteningPolicy(swing_threshold=0.15, min_brain_samples=1)
        brain = {"rr_borderline_min": 1.5}  # no sl_tightening key
        result = policy.evaluate_update(pos, proposed_sl=97.0, current_price=103.0, tf_minutes=240, brain_thresholds=brain)
        assert result.source == "config"

    def test_brain_learned_threshold_not_numeric_falls_back(self):
        pos = self._long_pos()
        policy = StopLossTighteningPolicy(swing_threshold=0.15, min_brain_samples=1)
        brain = {"sl_tightening": {"sample_count": 20, "learned_threshold": "bad"}}
        result = policy.evaluate_update(pos, proposed_sl=97.0, current_price=103.0, tf_minutes=240, brain_thresholds=brain)
        assert result.source == "config"

    def test_empty_brain_thresholds_falls_back_to_config(self):
        pos = self._long_pos()
        policy = StopLossTighteningPolicy(swing_threshold=0.15, min_brain_samples=1)
        result = policy.evaluate_update(pos, proposed_sl=97.0, current_price=103.0, tf_minutes=240, brain_thresholds={})
        assert result.source == "config"

    def test_none_brain_thresholds_falls_back_to_config(self):
        pos = self._long_pos()
        policy = StopLossTighteningPolicy(swing_threshold=0.15, min_brain_samples=1)
        result = policy.evaluate_update(pos, proposed_sl=97.0, current_price=103.0, tf_minutes=240, brain_thresholds=None)
        assert result.source == "config"
