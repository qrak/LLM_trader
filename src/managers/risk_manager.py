"""Risk Manager for converting signals into actionable trade parameters."""

from typing import Optional, Dict, Any, TYPE_CHECKING
from src.logger.logger import Logger
from src.contracts.risk_contract import RiskManagerProtocol

if TYPE_CHECKING:
    from src.config.protocol import ConfigProtocol
    from src.trading.dataclasses import RiskAssessment

class RiskManager(RiskManagerProtocol):
    """
    Manages risk calculations including position sizing, stop-loss/take-profit dynamic adjustment,
    and circuit breakers.
    """

    def __init__(self, logger: Logger, config: "ConfigProtocol"):
        self.logger = logger
        self.config = config

    def validate_signal(self, signal: str) -> bool:
        """Validate if a signal is actionable."""
        return signal in ("BUY", "SELL", "CLOSE", "CLOSE_LONG", "CLOSE_SHORT")

    def calculate_entry_parameters(
        self,
        signal: str,
        current_price: float,
        capital: float,
        confidence: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        position_size: Optional[float] = None,
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> "RiskAssessment":
        """
        Calculate all risk parameters for a new position entry.
        """
        from src.trading.dataclasses import RiskAssessment
        market_conditions = market_conditions or {}
        direction = "LONG" if signal == "BUY" else "SHORT"

        # 1. Extract or Default ATR/Volatility
        atr = market_conditions.get("atr", current_price * 0.02)
        atr_pct = market_conditions.get("atr_percentage", (atr / current_price) * 100)

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
            self.logger.debug(f"Using AI-provided SL: ${final_sl:,.2f}")
        else:
            final_sl = dynamic_sl
            self.logger.info(f"Using dynamic SL (2x ATR): ${final_sl:,.2f}")

        if take_profit and take_profit > 0:
            final_tp = take_profit
            self.logger.debug(f"Using AI-provided TP: ${final_tp:,.2f}")
        else:
            final_tp = dynamic_tp
            self.logger.info(f"Using dynamic TP (4x ATR): ${final_tp:,.2f}")

        # 4. Circuit Breakers (Clamp Extreme Values)
        sl_distance_raw = abs(current_price - final_sl) / current_price

        # Clamp SL: min 0.5%, max 10%
        if sl_distance_raw > 0.10:
            self.logger.warning(f"SL distance {sl_distance_raw:.1%} exceeds 10% max, clamping")
            if direction == "LONG":
                final_sl = current_price * 0.90
            else:
                final_sl = current_price * 1.10
        elif sl_distance_raw < 0.005:
            self.logger.warning(f"SL distance {sl_distance_raw:.1%} below 0.5% min, expanding")
            if direction == "LONG":
                final_sl = current_price * 0.995
            else:
                final_sl = current_price * 1.005

        # Validate Logical Consistency
        if direction == "LONG":
            if final_sl >= current_price:
                self.logger.warning(f"Invalid SL for LONG ({final_sl} >= {current_price}), using dynamic")
                final_sl = dynamic_sl
            if final_tp <= current_price:
                self.logger.warning(f"Invalid TP for LONG ({final_tp} <= {current_price}), using dynamic")
                final_tp = dynamic_tp
        else:  # SHORT
            if final_sl <= current_price:
                self.logger.warning(f"Invalid SL for SHORT ({final_sl} <= {current_price}), using dynamic")
                final_sl = dynamic_sl
            if final_tp >= current_price:
                self.logger.warning(f"Invalid TP for SHORT ({final_tp} >= {current_price}), using dynamic")
                final_tp = dynamic_tp

        # 5. Position Sizing
        if position_size and position_size > 0:
            final_size_pct = position_size
        else:
            # Dynamic sizing based on confidence
            confidence_map = {"HIGH": 0.03, "MEDIUM": 0.02, "LOW": 0.01}
            final_size_pct = confidence_map.get(confidence.upper(), 0.02)
            self.logger.info(f"Using confidence-based size: {final_size_pct*100:.1f}%")

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
