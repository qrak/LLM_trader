"""Experience recording behavior for the trading brain."""

from datetime import datetime
from uuid import uuid4

from src.logger.logger import Logger
from src.utils.indicator_classifier import (
    build_context_string_from_classified_values,
    build_exit_execution_context_from_position,
)

from .brain_patterns import TradePatternAnalyzer
from .data_models import ExitExecutionContext, MarketConditions, Position, TradeDecision
from .stop_loss_tightening_policy import TighteningEvaluation
from .vector_memory import VectorMemoryService


class BrainExperienceRecorder:
    """Store closed-trade and position-update experiences in vector memory."""

    def __init__(
        self,
        logger: Logger,
        vector_memory: VectorMemoryService,
        pattern_analyzer: TradePatternAnalyzer,
        default_exit_execution_context: ExitExecutionContext,
    ):
        """Initialize recorder dependencies."""
        self.logger = logger
        self.vector_memory = vector_memory
        self.pattern_analyzer = pattern_analyzer
        self.default_exit_execution_context = default_exit_execution_context

    @staticmethod
    def _timeframe_bucket(tf_minutes: int | None) -> str:
        """Map timeframe minutes to a stable bucket label."""
        if tf_minutes is None or tf_minutes <= 0:
            return "unknown"
        if tf_minutes < 60:
            return "scalping"
        if tf_minutes < 240:
            return "intraday"
        if tf_minutes < 1440:
            return "swing"
        return "position"

    @staticmethod
    def build_rich_context_string(
        trend_direction: str = "NEUTRAL",
        adx: float = 0,
        volatility_level: str = "MEDIUM",
        rsi_level: str = "NEUTRAL",
        macd_signal: str = "NEUTRAL",
        volume_state: str = "NORMAL",
        bb_position: str = "MIDDLE",
        is_weekend: bool = False,
        market_sentiment: str = "NEUTRAL",
        order_book_bias: str = "BALANCED",
        exit_execution_context: ExitExecutionContext | None = None,
    ) -> str:
        """Build rich semantic context string for vector storage and retrieval."""
        return build_context_string_from_classified_values(
            trend_direction=trend_direction,
            adx=adx,
            volatility_level=volatility_level,
            rsi_level=rsi_level,
            macd_signal=macd_signal,
            volume_state=volume_state,
            bb_position=bb_position,
            is_weekend=is_weekend,
            market_sentiment=market_sentiment,
            order_book_bias=order_book_bias,
            exit_execution_context=exit_execution_context,
        )

    def record_closed_trade(
        self,
        position: Position,
        close_price: float,
        close_reason: str,
        entry_decision: TradeDecision | None = None,
        market_conditions: MarketConditions | None = None,
    ) -> float:
        """Extract insights from a closed trade and store them in vector memory."""
        pnl_pct = position.calculate_pnl(close_price)
        is_win = pnl_pct > 0
        conditions = market_conditions or MarketConditions()
        exit_execution_context = build_exit_execution_context_from_position(position).with_defaults(
            self.default_exit_execution_context
        )
        entry_confidence = entry_decision.confidence if entry_decision else position.confidence
        entry_action = entry_decision.action if entry_decision else position.direction
        reasoning = entry_decision.reasoning if entry_decision else "N/A"
        condition_str = self.build_rich_context_string(
            trend_direction=conditions.trend_direction,
            adx=float(conditions.adx),
            volatility_level=conditions.volatility,
            rsi_level=conditions.rsi_level,
            macd_signal=conditions.macd_signal,
            volume_state=conditions.volume_state,
            bb_position=conditions.bb_position,
            is_weekend=conditions.is_weekend,
            market_sentiment=conditions.market_sentiment,
            order_book_bias=conditions.order_book_bias,
            exit_execution_context=exit_execution_context,
        )
        trade_id = f"trade_{position.entry_time.isoformat()}"
        position_id = f"{position.symbol}|{position.entry_time.isoformat()}"
        self.vector_memory.store_experience(
            trade_id=trade_id,
            market_context=condition_str,
            outcome="WIN" if is_win else "LOSS",
            pnl_pct=pnl_pct,
            direction=position.direction,
            confidence=entry_confidence,
            reasoning=reasoning,
            symbol=position.symbol,
            close_reason=close_reason,
            metadata={
                "entry_action": entry_action,
                "entry_confidence": entry_confidence,
                "ai_reasoning": reasoning if reasoning != "N/A" else "",
                "adx_at_entry": position.adx_at_entry,
                "rsi_at_entry": position.rsi_at_entry,
                "atr_at_entry": position.atr_at_entry,
                "volatility_level": position.volatility_level,
                "sl_distance_pct": position.sl_distance_pct,
                "tp_distance_pct": position.tp_distance_pct,
                "rr_ratio": position.rr_ratio_at_entry,
                "max_drawdown_pct": position.max_drawdown_pct,
                "max_profit_pct": position.max_profit_pct,
                "fear_greed_index": conditions.fear_greed_index,
                "market_regime": conditions.trend_direction,
                "is_weekend": conditions.is_weekend,
                "position_size_pct": position.size_pct,
                "confluence_count": self.pattern_analyzer.count_strong_confluences(position.confluence_factors),
                "timeframe_alignment": conditions.timeframe_alignment,
                "market_sentiment": conditions.market_sentiment,
                "order_book_bias": conditions.order_book_bias,
                "macd_signal": conditions.macd_signal,
                "bb_pos": conditions.bb_position,
                "position_entry_timestamp": position.entry_time.isoformat(),
                "position_entry_trade_id": trade_id,
                "position_id": position_id,
                **exit_execution_context.to_dict(),
                **self.pattern_analyzer.extract_factor_scores(position.confluence_factors),
            }
        )
        self.logger.info(
            "Updated brain from %s trade (%s, P&L: %s%%)",
            position.direction,
            close_reason,
            f"{pnl_pct:+.2f}",
        )
        return pnl_pct

    def track_position_update(
        self,
        position: Position,
        old_sl: float,
        old_tp: float,
        new_sl: float,
        new_tp: float,
        current_price: float,
        current_pnl_pct: float,
        market_conditions: MarketConditions | None = None,
        tightening_evaluation: TighteningEvaluation | None = None,
        timeframe_minutes: int | None = None,
    ) -> None:
        """Track position update decisions for learning."""
        conditions = market_conditions or MarketConditions()
        exit_execution_context = build_exit_execution_context_from_position(position)
        sl_moved = new_sl != old_sl
        tp_moved = new_tp != old_tp
        if sl_moved and not tp_moved:
            action_type = "SL_TRAIL"
        elif tp_moved and not sl_moved:
            action_type = "TP_EXTEND"
        elif sl_moved and tp_moved:
            action_type = "BOTH"
        else:
            return
        market_context = self.build_rich_context_string(
            trend_direction=conditions.trend_direction,
            adx=float(conditions.adx),
            volatility_level=conditions.volatility,
            rsi_level=conditions.rsi_level,
            macd_signal=conditions.macd_signal,
            volume_state=conditions.volume_state,
            bb_position=conditions.bb_position,
            is_weekend=conditions.is_weekend,
            market_sentiment=conditions.market_sentiment,
            order_book_bias=conditions.order_book_bias,
            exit_execution_context=exit_execution_context,
        )
        update_id = f"update_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid4().hex[:8]}"
        reasoning_str = f"Moved {action_type}: SL {old_sl:.2f}→{new_sl:.2f}, TP {old_tp:.2f}→{new_tp:.2f}"
        position_entry_timestamp = position.entry_time.isoformat()
        position_entry_trade_id = f"trade_{position_entry_timestamp}"
        position_id = f"{position.symbol}|{position_entry_timestamp}"
        entry_price = position.entry_price
        old_sl_dist = abs(old_sl - entry_price) / entry_price if entry_price else 0.0
        new_sl_dist = abs(new_sl - entry_price) / entry_price if entry_price else 0.0
        policy_meta: dict = {}
        if tightening_evaluation is not None:
            policy_meta = {
                "is_tightening": tightening_evaluation.is_tightening,
                "price_progress": round(tightening_evaluation.price_progress, 4),
                "base_min_progress": round(tightening_evaluation.base_min_progress, 4),
                "effective_min_progress": round(tightening_evaluation.effective_min_progress, 4),
                "policy_source": tightening_evaluation.source,
                "policy_allowed": tightening_evaluation.allowed,
                "policy_reason": tightening_evaluation.reason[:200],
            }
        self.vector_memory.store_experience(
            trade_id=update_id,
            market_context=market_context,
            outcome="UPDATE",
            pnl_pct=current_pnl_pct,
            direction=position.direction,
            confidence=position.confidence,
            reasoning=reasoning_str,
            metadata={
                "action_type": action_type,
                "current_price": current_price,
                "sl_change": new_sl - old_sl,
                "tp_change": new_tp - old_tp,
                "pnl_at_update": current_pnl_pct,
                "adx_at_update": conditions.adx,
                "volatility": conditions.volatility,
                "position_entry_timestamp": position_entry_timestamp,
                "position_entry_trade_id": position_entry_trade_id,
                "position_id": position_id,
                "entry_price": entry_price,
                "old_stop_loss": old_sl,
                "new_stop_loss": new_sl,
                "old_take_profit": old_tp,
                "new_take_profit": new_tp,
                "old_sl_distance_pct": round(old_sl_dist, 4),
                "new_sl_distance_pct": round(new_sl_dist, 4),
                "timeframe_minutes": timeframe_minutes if timeframe_minutes is not None else 0,
                "timeframe_bucket": self._timeframe_bucket(timeframe_minutes),
                **policy_meta,
                **exit_execution_context.to_dict(),
            }
        )
        self.logger.debug("Tracked position update: %s at %s%% PnL", action_type, f"{current_pnl_pct:+.1f}")