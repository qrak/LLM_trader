

import math
from dataclasses import dataclass
from typing import Any

from .data_models import Position

MIN_STOP_BUFFER_ATR = 0.5
MIN_STOP_BUFFER_PCT = 0.005
MIN_STOP_BUFFER_TP_SHARE = 0.5


@dataclass(slots=True)
class TighteningEvaluation:
    is_tightening: bool
    price_progress: float
    base_min_progress: float
    effective_min_progress: float
    allowed: bool
    source: str
    reason: str


class StopLossTighteningPolicy:
    _SCALPING_CEILING = 60
    _INTRADAY_CEILING = 240
    _SWING_CEILING = 1440

    def __init__(
        self,
        scalping_threshold: float = 0.25,
        intraday_threshold: float = 0.20,
        swing_threshold: float = 0.15,
        position_threshold: float = 0.10,
        floor: float = 0.05,
        ceiling: float = 0.40,
        min_brain_samples: int = 10,
        breakeven_progress: float = 0.0,
    ) -> None:

        self._scalping = scalping_threshold
        self._intraday = intraday_threshold
        self._swing = swing_threshold
        self._position = position_threshold
        self._floor = floor
        self._ceiling = ceiling
        self._min_brain_samples = min_brain_samples
        self._breakeven = breakeven_progress

    @classmethod
    def from_config(cls, config: Any) -> "StopLossTighteningPolicy":

        return cls(
            scalping_threshold=config.SL_TIGHTENING_SCALPING,
            intraday_threshold=config.SL_TIGHTENING_INTRADAY,
            swing_threshold=config.SL_TIGHTENING_SWING,
            position_threshold=config.SL_TIGHTENING_POSITION,
            floor=config.SL_TIGHTENING_FLOOR,
            ceiling=config.SL_TIGHTENING_CEILING,
            min_brain_samples=config.SL_TIGHTENING_MIN_SAMPLES,
            breakeven_progress=config.SL_BREAKEVEN_PROGRESS,
        )

    @property
    def breakeven_progress(self) -> float:

        return self._breakeven

    def get_base_threshold(self, tf_minutes: int) -> float:

        if tf_minutes < self._SCALPING_CEILING:
            return self._scalping
        if tf_minutes < self._INTRADAY_CEILING:
            return self._intraday
        if tf_minutes < self._SWING_CEILING:
            return self._swing
        return self._position

    def breakeven_floor(self, position: Position, current_price: float | None) -> float | None:

        if self._breakeven <= 0 or not current_price or current_price <= 0:
            return None
        price_progress = self._price_progress(position, current_price)
        if price_progress is None or price_progress < self._breakeven:
            return None
        if abs(position.entry_price - current_price) < self.entry_buffer(position):
            return None
        return position.entry_price

    @staticmethod
    def entry_buffer(position: Position) -> float:

        atr = position.atr_at_entry
        if math.isfinite(atr) and atr > 0:
            buffer = MIN_STOP_BUFFER_ATR * atr
        else:
            atr_pct = position.atr_percentage_at_entry
            if math.isfinite(atr_pct) and atr_pct > 0:
                buffer = MIN_STOP_BUFFER_ATR * atr_pct / 100.0 * position.entry_price
            else:
                buffer = MIN_STOP_BUFFER_PCT * position.entry_price
        tp_distance = abs(position.take_profit - position.entry_price)
        if tp_distance <= 0:
            return buffer
        return min(buffer, MIN_STOP_BUFFER_TP_SHARE * tp_distance)

    @staticmethod
    def _inside_entry_buffer(position: Position, stop: float, buffer: float) -> bool:

        if position.direction == "SHORT":
            return stop <= position.entry_price and position.entry_price - stop < buffer
        return stop >= position.entry_price and stop - position.entry_price < buffer

    @staticmethod
    def _price_progress(position: Position, current_price: float) -> float | None:

        tp_distance_total = abs(position.take_profit - position.entry_price)
        if tp_distance_total <= 0:
            return None
        if position.direction == "LONG":
            return (current_price - position.entry_price) / tp_distance_total
        return (position.entry_price - current_price) / tp_distance_total

    def evaluate_update(
        self,
        position: Position,
        proposed_sl: float,
        current_price: float,
        tf_minutes: int,
        brain_thresholds: dict[str, Any] | None = None,
    ) -> TighteningEvaluation:

        direction = position.direction
        old_sl = position.stop_loss

        is_tightening = (
            (direction == "LONG" and proposed_sl > old_sl)
            or (direction == "SHORT" and proposed_sl < old_sl)
        )

        if not is_tightening:
            return TighteningEvaluation(
                is_tightening=False,
                price_progress=0.0,
                base_min_progress=self.get_base_threshold(tf_minutes),
                effective_min_progress=self.get_base_threshold(tf_minutes),
                allowed=True,
                source="config",
                reason="Not a tightening move — SL is widening or unchanged.",
            )

        if not current_price or current_price <= 0:
            base = self.get_base_threshold(tf_minutes)
            return TighteningEvaluation(
                is_tightening=True,
                price_progress=0.0,
                base_min_progress=base,
                effective_min_progress=base,
                allowed=False,
                source="config",
                reason="No valid current price — tightening rejected as a safety measure.",
            )

        price_progress = self._price_progress(position, current_price)
        if price_progress is None:
            base = self.get_base_threshold(tf_minutes)
            return TighteningEvaluation(
                is_tightening=True,
                price_progress=0.0,
                base_min_progress=base,
                effective_min_progress=base,
                allowed=False,
                source="config",
                reason="Cannot compute progress: entry equals take-profit.",
            )

        base = self.get_base_threshold(tf_minutes)
        effective, source = self.resolve_effective_threshold(base, brain_thresholds)

        if price_progress >= effective:
            buffer = self.entry_buffer(position)
            if self._inside_entry_buffer(position, proposed_sl, buffer):
                return TighteningEvaluation(
                    is_tightening=True,
                    price_progress=price_progress,
                    base_min_progress=base,
                    effective_min_progress=effective,
                    allowed=False,
                    source=source,
                    reason=(
                        f"Proposed stop {proposed_sl:,.2f} sits "
                        f"{abs(proposed_sl - position.entry_price):,.2f} from the entry "
                        f"{position.entry_price:,.2f}, inside the {buffer:,.2f} noise buffer. "
                        f"Tightening rejected."
                    ),
                )
            reason = (
                f"Price progress {price_progress:.1%} >= required {effective:.1%} "
                f"(source: {source}). Tightening allowed."
            )
            return TighteningEvaluation(
                is_tightening=True,
                price_progress=price_progress,
                base_min_progress=base,
                effective_min_progress=effective,
                allowed=True,
                source=source,
                reason=reason,
            )

        reason = (
            f"Price progress {price_progress:.1%} < required {effective:.1%} "
            f"(source: {source}). Premature tightening rejected."
        )
        return TighteningEvaluation(
            is_tightening=True,
            price_progress=price_progress,
            base_min_progress=base,
            effective_min_progress=effective,
            allowed=False,
            source=source,
            reason=reason,
        )

    def resolve_effective_threshold(
        self,
        base: float,
        brain_thresholds: dict[str, Any] | None,
    ) -> tuple[float, str]:

        if not brain_thresholds:
            return base, "config"

        sl_tightening = brain_thresholds.get("sl_tightening")
        if not isinstance(sl_tightening, dict):
            return base, "config"

        sample_count = sl_tightening.get("sample_count", 0)
        if sample_count < self._min_brain_samples:
            return base, "config"

        effective_count = sl_tightening.get("effective_sample_count", 0)
        if effective_count is True or effective_count is False:
            return base, "config"
        try:
            effective_count = float(effective_count)
        except (TypeError, ValueError, OverflowError):
            return base, "config"
        if not math.isfinite(effective_count) or effective_count + 1e-9 < self._min_brain_samples:
            return base, "config"

        learned = sl_tightening.get("learned_threshold")
        if learned is None:
            return base, "config"

        try:
            learned_float = float(learned)
        except (TypeError, ValueError):
            return base, "config"

        if not math.isfinite(learned_float):
            return base, "config"
        clamped = max(self._floor, min(self._ceiling, learned_float))
        return clamped, "brain"
