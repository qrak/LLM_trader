"""Stop-loss tightening policy — single authoritative gate for SL update decisions."""

from dataclasses import dataclass
from typing import Any

from .data_models import Position


@dataclass(slots=True)
class TighteningEvaluation:
    """Result of a stop-loss tightening evaluation.

    Attributes:
        is_tightening: True when the proposed SL moves toward the entry price.
        price_progress: Fraction of entry-to-TP distance covered by current price.
        base_min_progress: Config/timeframe default minimum progress required.
        effective_min_progress: Actual threshold used (may be brain-adjusted).
        allowed: Whether the tightening is permitted.
        source: Where effective_min_progress came from ("config" or "brain").
        reason: Human-readable justification for the decision.
    """
    is_tightening: bool
    price_progress: float
    base_min_progress: float
    effective_min_progress: float
    allowed: bool
    source: str
    reason: str


class StopLossTighteningPolicy:
    """Determines whether a proposed SL change constitutes premature tightening.

    Acts as the single authoritative gate shared by the executor and prompt layer.
    The brain may supply an advisory override; config-based timeframe defaults act
    as the safety floor/ceiling.  The policy is deterministic and side-effect free.
    """

    # Timeframe bucket boundaries in minutes
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
    ) -> None:
        """Initialise policy with per-timeframe thresholds and safety clamps.

        Args:
            scalping_threshold: Minimum progress for sub-1h timeframes.
            intraday_threshold: Minimum progress for 1h–4h timeframes.
            swing_threshold: Minimum progress for 4h–1d timeframes.
            position_threshold: Minimum progress for daily+ timeframes.
            floor: Lower clamp — brain cannot push below this.
            ceiling: Upper clamp — brain cannot push above this.
            min_brain_samples: Minimum paired samples before brain override is trusted.
        """
        self._scalping = scalping_threshold
        self._intraday = intraday_threshold
        self._swing = swing_threshold
        self._position = position_threshold
        self._floor = floor
        self._ceiling = ceiling
        self._min_brain_samples = min_brain_samples

    @classmethod
    def from_config(cls, config: Any) -> "StopLossTighteningPolicy":
        """Construct policy from a ConfigProtocol-compatible config object."""
        return cls(
            scalping_threshold=config.SL_TIGHTENING_SCALPING,
            intraday_threshold=config.SL_TIGHTENING_INTRADAY,
            swing_threshold=config.SL_TIGHTENING_SWING,
            position_threshold=config.SL_TIGHTENING_POSITION,
            floor=config.SL_TIGHTENING_FLOOR,
            ceiling=config.SL_TIGHTENING_CEILING,
            min_brain_samples=config.SL_TIGHTENING_MIN_SAMPLES,
        )

    def get_base_threshold(self, tf_minutes: int) -> float:
        """Return the config-based minimum progress threshold for a timeframe."""
        if tf_minutes < self._SCALPING_CEILING:
            return self._scalping
        if tf_minutes < self._INTRADAY_CEILING:
            return self._intraday
        if tf_minutes < self._SWING_CEILING:
            return self._swing
        return self._position

    def evaluate_update(
        self,
        position: Position,
        proposed_sl: float,
        current_price: float,
        tf_minutes: int,
        brain_thresholds: dict[str, Any] | None = None,
    ) -> TighteningEvaluation:
        """Evaluate whether the proposed SL change is a premature tightening.

        Args:
            position: Active position (must not be None).
            proposed_sl: Requested new stop-loss price.
            current_price: Current market price.
            tf_minutes: Analysis timeframe in minutes.
            brain_thresholds: Optional learned thresholds from TradingBrainService.

        Returns:
            TighteningEvaluation with all decision metadata.
        """
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

        tp_distance_total = abs(position.take_profit - position.entry_price)
        if tp_distance_total <= 0:
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

        if direction == "LONG":
            price_progress = (current_price - position.entry_price) / tp_distance_total
        else:
            price_progress = (position.entry_price - current_price) / tp_distance_total

        base = self.get_base_threshold(tf_minutes)
        effective, source = self._resolve_effective_threshold(base, brain_thresholds)

        if price_progress >= effective:
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

    def _resolve_effective_threshold(
        self,
        base: float,
        brain_thresholds: dict[str, Any] | None,
    ) -> tuple[float, str]:
        """Resolve the effective threshold from config base and optional brain override.

        Returns:
            Tuple of (effective_threshold, source_label).
        """
        if not brain_thresholds:
            return base, "config"

        sl_tightening = brain_thresholds.get("sl_tightening")
        if not isinstance(sl_tightening, dict):
            return base, "config"

        sample_count = sl_tightening.get("sample_count", 0)
        if sample_count < self._min_brain_samples:
            return base, "config"

        learned = sl_tightening.get("learned_threshold")
        if learned is None:
            return base, "config"

        try:
            learned_float = float(learned)
        except (TypeError, ValueError):
            return base, "config"

        clamped = max(self._floor, min(self._ceiling, learned_float))
        return clamped, "brain"
