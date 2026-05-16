"""Risk Manager for converting signals into actionable trade parameters."""

import math
from typing import TYPE_CHECKING, Any

from src.logger.logger import Logger
from src.contracts.risk_contract import RiskManagerProtocol

if TYPE_CHECKING:
    from src.config.protocol import ConfigProtocol
    from src.trading.data_models import RiskAssessment, MarketConditions

class RiskManager(RiskManagerProtocol):
    """
    Manages risk calculations including position sizing, stop-loss/take-profit dynamic adjustment,
    and circuit breakers.
    """

    def __init__(self, logger: Logger, config: "ConfigProtocol"):
        self.logger = logger
        self.config = config
        self._last_frictions: list[dict[str, Any]] = []

    def validate_signal(self, signal: str) -> bool:
        """Validate if a signal is actionable."""
        return signal in ("BUY", "SELL", "CLOSE", "CLOSE_LONG", "CLOSE_SHORT")

    def _is_valid_position_size(self, position_size: float | None) -> bool:
        """Return whether position size is a usable capital fraction."""
        return position_size is not None and math.isfinite(position_size) and position_size > 0

    def _get_confidence_fallback_size(self, confidence: str) -> float:
        """Get configured fallback size for a confidence level."""
        fallback_sizes = {
            "HIGH": self.config.POSITION_SIZE_FALLBACK_HIGH,
            "MEDIUM": self.config.POSITION_SIZE_FALLBACK_MEDIUM,
            "LOW": self.config.POSITION_SIZE_FALLBACK_LOW,
        }
        fallback_size = fallback_sizes.get(confidence.upper(), self.config.POSITION_SIZE_FALLBACK_MEDIUM)
        if math.isfinite(fallback_size) and fallback_size > 0:
            return fallback_size
        self.logger.warning(
            "Configured fallback position size for %s confidence is invalid, using MEDIUM fallback",
            confidence,
        )
        return self.config.POSITION_SIZE_FALLBACK_MEDIUM

    def _resolve_position_size_pct(self, position_size: float | None, confidence: str) -> float:
        """Resolve final position size from AI request or configured confidence fallback."""
        max_size = self.config.MAX_POSITION_SIZE
        if not math.isfinite(max_size) or max_size <= 0:
            raise ValueError("MAX_POSITION_SIZE must be a positive finite decimal")

        if self._is_valid_position_size(position_size):
            requested_size = position_size
            source = "AI"
        else:
            requested_size = self._get_confidence_fallback_size(confidence)
            source = f"configured {confidence.upper()} confidence fallback"
            self.logger.info("Using %s position size: %.2f%%", source, requested_size * 100)

        final_size_pct = min(requested_size, max_size)
        if requested_size > max_size:
            self.logger.warning(
                "%s position size %.2f%% exceeds cap %.2f%%, clamping",
                source,
                requested_size * 100,
                max_size * 100,
            )
            self._last_frictions.append({
                "guard_type": "position_size_clamp",
                "direction": "N/A",
                "suggested_size": requested_size,
                "max_size": max_size,
                "detail": f"Position size {requested_size*100:.1f}% clamped to max {max_size*100:.1f}%",
            })
        self.logger.debug(
            "Resolved position size: source=%s requested=%.4f cap=%.4f final=%.4f",
            source,
            requested_size,
            max_size,
            final_size_pct,
        )
        return final_size_pct

    def calculate_entry_parameters(
        self,
        signal: str,
        current_price: float,
        capital: float,
        confidence: str,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        position_size: float | None = None,
        market_conditions: "MarketConditions | None" = None
    ) -> "RiskAssessment":
        """
        Calculate all risk parameters for a new position entry.
        """
        from src.trading.data_models import RiskAssessment
        mc = market_conditions
        direction = "LONG" if signal == "BUY" else "SHORT"

        # 1. Extract or Default ATR/Volatility
        atr = mc.atr if mc and mc.atr > 0 else current_price * 0.02
        atr_pct = mc.atr_percentage if mc and mc.atr_percentage > 0 else (atr / current_price) * 100

        # Determine volatility level
        if atr_pct > 3:
            volatility_level = "HIGH"
        elif atr_pct < 1.5:
            volatility_level = "LOW"
        else:
            volatility_level = "MEDIUM"

        # 2. Dynamic SL/TP Calculation (Dynamic Defaults)
        # Use 2x ATR for SL, 4x ATR for TP (2:1 R/R default)
        dynamic_sl_distance = atr * 2
        dynamic_tp_distance = atr * 4

        if direction == "LONG":
            dynamic_sl = current_price - dynamic_sl_distance
            dynamic_tp = current_price + dynamic_tp_distance
        else:  # SHORT
            dynamic_sl = current_price + dynamic_sl_distance
            dynamic_tp = current_price - dynamic_tp_distance

        # 3. Resolve Final SL/TP (AI vs Dynamic)
        if stop_loss and stop_loss > 0:
            final_sl = stop_loss
            self.logger.debug("Using AI-provided SL: $%s", f"{final_sl:,.2f}")
        else:
            final_sl = dynamic_sl
            self.logger.info("Using dynamic SL (2x ATR): $%s", f"{final_sl:,.2f}")

        if take_profit and take_profit > 0:
            final_tp = take_profit
            self.logger.debug("Using AI-provided TP: $%s", f"{final_tp:,.2f}")
        else:
            final_tp = dynamic_tp
            self.logger.info("Using dynamic TP (4x ATR): $%s", f"{final_tp:,.2f}")

        # 4. Circuit Breakers (Clamp Extreme Values)
        sl_distance_raw = abs(current_price - final_sl) / current_price

        # Clamp SL: min 1.0%, max 10%
        if sl_distance_raw > 0.10:
            self.logger.warning("SL distance %s exceeds 10%% max, clamping", f"{sl_distance_raw:.1%}")
            self._last_frictions.append({
                "guard_type": "sl_distance_max",
                "direction": direction,
                "suggested_sl_pct": sl_distance_raw,
                "corrected_sl_pct": 0.10,
                "current_price": current_price,
                "volatility_level": volatility_level,
                "detail": f"SL distance {sl_distance_raw*100:.1f}% clamped to max 10%",
            })
            if direction == "LONG":
                final_sl = current_price * 0.90
            else:
                final_sl = current_price * 1.10
        elif sl_distance_raw < 0.01:
            self.logger.warning("SL distance %s below 1.0%% min, expanding", f"{sl_distance_raw:.1%}")
            self._last_frictions.append({
                "guard_type": "sl_distance_min",
                "direction": direction,
                "suggested_sl_pct": sl_distance_raw,
                "corrected_sl_pct": 0.01,
                "current_price": current_price,
                "volatility_level": volatility_level,
                "detail": f"SL distance {sl_distance_raw*100:.1f}% expanded to min 1%",
            })
            if direction == "LONG":
                final_sl = current_price * 0.99
            else:
                final_sl = current_price * 1.01

        # Validate Logical Consistency
        if direction == "LONG":
            if final_sl >= current_price:
                self.logger.warning("Invalid SL for LONG (%s >= %s), using dynamic", final_sl, current_price)
                self._last_frictions.append({
                    "guard_type": "sl_below_entry",
                    "direction": direction,
                    "suggested_sl": final_sl,
                    "dynamic_sl": dynamic_sl,
                    "current_price": current_price,
                    "volatility_level": volatility_level,
                    "detail": f"SL ${final_sl:,.2f} was above/at entry ${current_price:,.2f}, using dynamic",
                })
                final_sl = dynamic_sl
            if final_tp <= current_price:
                self.logger.warning("Invalid TP for LONG (%s <= %s), using dynamic", final_tp, current_price)
                self._last_frictions.append({
                    "guard_type": "tp_below_entry",
                    "direction": direction,
                    "suggested_tp": final_tp,
                    "dynamic_tp": dynamic_tp,
                    "current_price": current_price,
                    "volatility_level": volatility_level,
                    "detail": f"TP ${final_tp:,.2f} was below/at entry ${current_price:,.2f}, using dynamic",
                })
                final_tp = dynamic_tp
        else:  # SHORT
            if final_sl <= current_price:
                self.logger.warning("Invalid SL for SHORT (%s <= %s), using dynamic", final_sl, current_price)
                self._last_frictions.append({
                    "guard_type": "sl_below_entry",
                    "direction": direction,
                    "suggested_sl": final_sl,
                    "dynamic_sl": dynamic_sl,
                    "current_price": current_price,
                    "volatility_level": volatility_level,
                    "detail": f"SL ${final_sl:,.2f} was below/at entry ${current_price:,.2f}, using dynamic",
                })
                final_sl = dynamic_sl
            if final_tp >= current_price:
                self.logger.warning("Invalid TP for SHORT (%s >= %s), using dynamic", final_tp, current_price)
                self._last_frictions.append({
                    "guard_type": "tp_below_entry",
                    "direction": direction,
                    "suggested_tp": final_tp,
                    "dynamic_tp": dynamic_tp,
                    "current_price": current_price,
                    "volatility_level": volatility_level,
                    "detail": f"TP ${final_tp:,.2f} was above/at entry ${current_price:,.2f}, using dynamic",
                })
                final_tp = dynamic_tp

        # 5. Position Sizing
        final_size_pct = self._resolve_position_size_pct(position_size, confidence)

        # 6. Calculate Financials
        allocation = capital * final_size_pct
        quantity = allocation / current_price
        entry_fee = allocation * self.config.TRANSACTION_FEE_PERCENT

        # 7. Metrics
        sl_distance_pct = abs(current_price - final_sl) / current_price
        tp_distance_pct = abs(final_tp - current_price) / current_price
        rr_ratio = tp_distance_pct / sl_distance_pct if sl_distance_pct > 0 else 0

        return RiskAssessment(
            direction=direction,
            entry_price=current_price,
            stop_loss=final_sl,
            take_profit=final_tp,
            quantity=quantity,
            size_pct=final_size_pct,
            quote_amount=allocation,
            entry_fee=entry_fee,
            sl_distance_pct=sl_distance_pct,
            tp_distance_pct=tp_distance_pct,
            rr_ratio=rr_ratio,
            volatility_level=volatility_level
        )

    def get_and_clear_frictions(self) -> list[dict[str, Any]]:
        """Return accumulated friction reports and clear the buffer.

        Called by TradingStrategy after `calculate_entry_parameters` to retrieve
        any clamping/rejection events for persistence as blocked trade feedback.

        Returns:
            List of friction dicts (guard_type, direction, deltas, etc.).
        """
        frictions = self._last_frictions
        self._last_frictions = []
        return frictions
