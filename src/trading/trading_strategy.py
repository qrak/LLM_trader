"""Trading strategy that wraps analysis with position management."""

import asyncio
import dataclasses
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from src.logger.logger import Logger
from src.utils.indicator_classifier import (
    build_exit_execution_context_from_config,
)
from .data_models import MarketConditions, Position, TradeDecision
from .brain import TradingBrainService
from .statistics import TradingStatisticsService
from .memory import TradingMemoryService
from .stop_loss_tightening_policy import StopLossTighteningPolicy, TighteningEvaluation
from .order_lifecycle import OrderIntent, OrderLifecycle
from .audit import AuditTrail
from .guards.pipeline import GuardPipeline

if TYPE_CHECKING:
    from src.dashboard.dashboard_state import DashboardState
    from src.managers.risk_manager import RiskManager
    from src.managers.persistence_manager import PersistenceManager
    from .market_conditions_extractor import MarketConditionsExtractor


class TradingStrategy:
    """Manages trading positions and decision execution based on AI analysis."""

    def __init__(
        self,
        logger: Logger,
        persistence: "PersistenceManager",
        brain_service: TradingBrainService,
        statistics_service: TradingStatisticsService,
        memory_service: TradingMemoryService,
        risk_manager: "RiskManager",
        config: Any = None,
        position_extractor=None,
        conditions_extractor: "MarketConditionsExtractor | None" = None,
        dashboard_state: "DashboardState | None" = None,
        tightening_policy: StopLossTighteningPolicy | None = None,
        guard_pipeline: GuardPipeline | None = None,
        audit_trail: AuditTrail | None = None,
        post_mortem_service: Any | None = None,
    ):
        """Initialize the trading strategy with DI pattern.

        Args:
            logger: Logger instance
            persistence: Persistence service for loading/saving data
            brain_service: Brain service for learning and insights
            statistics_service: Statistics service for performance metrics
            memory_service: Memory service for recent decision context
            risk_manager: Risk Manager for position sizing and SL/TP
            config: Configuration module
            position_extractor: PositionExtractor instance (injected from app.py)
            conditions_extractor: MarketConditionsExtractor (injected from start.py)
            dashboard_state: Optional dashboard state for UI lifecycle notifications
            tightening_policy: Stop-loss tightening policy (injected from start.py)
            guard_pipeline: Pre-execution guard pipeline (injected from start.py)
            audit_trail: Optional audit collector for governance events
        """
        self.logger = logger
        self.persistence = persistence
        self.brain_service = brain_service
        self.statistics_service = statistics_service
        self.memory_service = memory_service
        self.risk_manager = risk_manager
        self.config = config
        self.extractor = position_extractor
        self.dashboard_state = dashboard_state

        self._conditions = conditions_extractor
        if self._conditions is None:
            from .market_conditions_extractor import MarketConditionsExtractor
            self._conditions = MarketConditionsExtractor(logger)

        self.audit_trail = audit_trail if audit_trail is not None else AuditTrail()
        self.guard_pipeline = guard_pipeline
        self.post_mortem_service = post_mortem_service

        self.current_position: Position | None = self.persistence.load_position()

        self._last_position_update_time: datetime | None = None

        try:
            from src.utils.timeframe_validator import TimeframeValidator
            self._tf_minutes: int = TimeframeValidator.to_minutes(config.TIMEFRAME) if config else 240
        except Exception:
            self._tf_minutes = 240

        self._tightening_policy: StopLossTighteningPolicy = (
            tightening_policy if tightening_policy is not None else StopLossTighteningPolicy()
        )

        self._last_sl_tightening_evaluation: TighteningEvaluation | None = None

        tf = self._tf_minutes
        if tf < 60:
            self._min_update_interval_hours: float = (tf * 4) / 60.0
        elif tf < 240:
            self._min_update_interval_hours = (tf * 3) / 60.0
        elif tf < 1440:
            self._min_update_interval_hours = (tf * 2) / 60.0
        else:
            self._min_update_interval_hours = tf / 60.0

        if self.current_position:
            self.logger.info("Loaded existing position: %s %s @ $%s", self.current_position.direction, self.current_position.symbol, f"{self.current_position.entry_price:,.2f}")

        # Validate loaded position against current config — warn about mismatches
        # but don't discard the position (operator should decide).
        try:
            expected_symbol = config.CRYPTO_PAIR if config else None
            state_warnings = self.persistence.validate_loaded_position(expected_symbol)
            for warning in state_warnings:
                self.logger.warning("STARTUP STATE WARNING: %s", warning)
        except Exception as e:
            self.logger.warning("Could not validate loaded position: %s", e)

    def set_dashboard_state(self, dashboard_state: "DashboardState | None") -> None:
        """Inject dashboard state after dashboard server construction."""
        self.dashboard_state = dashboard_state

    async def _record_trade_decision(self, decision: TradeDecision) -> None:
        """Persist a decision and refresh short-term memory."""
        await self.persistence.async_save_trade_decision(decision)
        self.memory_service.add_decision(decision)

    async def _update_live_metrics(self, current_price: float) -> bool:
        """Update live position metrics before evaluating an exit."""
        if not self.current_position:
            return False
        self.current_position.update_metrics(current_price)
        await self.persistence.async_save_position(self.current_position)
        return True

    async def check_position(self, current_price: float) -> str | None:
        """Check if current position hit stop loss or take profit.

        Args:
            current_price: Current market price

        Returns:
            Reason for closing position if hit, else None
        """
        if not await self._update_live_metrics(current_price):
            return None

        if self.current_position.is_stop_hit(current_price):
            conditions = self._conditions.build_conditions_from_position(self.current_position)
            await self.close_position("stop_loss", current_price, conditions)
            return "stop_loss"

        if self.current_position.is_target_hit(current_price):
            conditions = self._conditions.build_conditions_from_position(self.current_position)
            await self.close_position("take_profit", current_price, conditions)
            return "take_profit"

        return None

    async def check_stop_loss(self, current_price: float) -> str | None:
        """Check only the configured stop loss exit."""
        if not await self._update_live_metrics(current_price):
            return None

        if self.current_position.is_stop_hit(current_price):
            conditions = self._conditions.build_conditions_from_position(self.current_position)
            await self.close_position("stop_loss", current_price, conditions)
            return "stop_loss"

        return None

    async def check_take_profit(self, current_price: float) -> str | None:
        """Check only the configured take profit exit."""
        if not await self._update_live_metrics(current_price):
            return None

        if self.current_position.is_target_hit(current_price):
            conditions = self._conditions.build_conditions_from_position(self.current_position)
            await self.close_position("take_profit", current_price, conditions)
            return "take_profit"

        return None

    async def close_position(
        self,
        reason: str,
        current_price: float,
        market_conditions: MarketConditions | None = None
    ) -> None:
        """Close the current position and update trading brain.

        Args:
            reason: Reason for closing (stop_loss, take_profit, signal)
            current_price: Current market price
            market_conditions: Optional market conditions for brain learning
        """
        if not self.current_position:
            return

        closed_position = self.current_position
        pnl = closed_position.calculate_pnl(current_price)

        closing_fee = closed_position.calculate_closing_fee(
            current_price,
            self.config.TRANSACTION_FEE_PERCENT
        )

        decision = TradeDecision(
            timestamp=datetime.now(timezone.utc),
            symbol=closed_position.symbol,
            action=f"CLOSE_{closed_position.direction}",
            confidence=closed_position.confidence,
            price=current_price,
            stop_loss=closed_position.stop_loss,
            take_profit=closed_position.take_profit,
            position_size=closed_position.size_pct,
            quote_amount=closed_position.quote_amount,
            quantity=closed_position.size,
            fee=closing_fee,
            reasoning=f"Position closed: {reason}. P&L: {pnl:+.2f}%. Fee: ${closing_fee:.4f}",
        )

        self.logger.info("Closing %s position (%s) @ $%s, P&L: %s%%, Fee: $%.4f", closed_position.direction, reason, f"{current_price:,.2f}", f"{pnl:+.2f}", closing_fee)

        # Retrieve entry decision from trade history for brain learning
        entry_decision = None
        try:
            entry_decision = self.persistence.get_entry_decision_for_position(
                closed_position.entry_time,
                symbol=closed_position.symbol,
            )
            if entry_decision:
                reasoning_preview = entry_decision.reasoning[:500] if entry_decision.reasoning else "(no reasoning)"
                self.logger.debug("Retrieved entry decision with reasoning: %s...", reasoning_preview)
            else:
                self.logger.warning("Could not retrieve entry decision from trade history")
        except Exception as e:
            self.logger.error("Error retrieving entry decision: %s", e)

        await self._record_trade_decision(decision)

        # --- Post-Mortem Analysis ---
        # Trigger LLM post-mortem after the CLOSE row is persisted to SQLite.
        # Skip if no entry_decision (can't analyze without original reasoning).
        # Graceful degradation: any failure is logged and swallowed.
        if self.post_mortem_service and entry_decision:
            try:
                await self.post_mortem_service.analyze_closed_trade(
                    closed_position=closed_position,
                    entry_decision=entry_decision,
                    exit_decision=decision,
                    pnl=pnl,
                    reason=reason,
                    market_conditions=market_conditions,
                )
            except Exception:
                self.logger.warning("Post-mortem analysis failed", exc_info=True)

        try:
            self.statistics_service.recalculate(self.config.DEMO_QUOTE_CAPITAL)
        except Exception as e:
            self.logger.error("Error recalculating statistics: %s", e)
        await self.persistence.async_save_position(None)
        self.current_position = None

        try:
            if self.dashboard_state:
                await self.dashboard_state.mark_brain_rebuild_started(
                    f"Learning from closed {closed_position.direction} trade"
                )
            await asyncio.to_thread(
                self.brain_service.update_from_closed_trade,
                position=closed_position,
                close_price=current_price,
                close_reason=reason,
                entry_decision=entry_decision,
                market_conditions=market_conditions,
            )
            if self.dashboard_state:
                await self.dashboard_state.mark_brain_rebuild_completed("Brain state rebuilt from closed trade")
        except Exception as e:
            self.logger.error("Error updating trading brain: %s", e)
            if self.dashboard_state:
                await self.dashboard_state.mark_brain_rebuild_failed("Brain rebuild failed after trade close")

    async def process_analysis(self, analysis_result: dict, symbol: str) -> TradeDecision | None:
        """Process AI analysis result and execute trading decision.

        Args:
            analysis_result: Result from AnalysisEngine.analyze_market()
            symbol: Trading symbol

        Returns:
            TradeDecision if action taken, else None
        """
        try:
            raw_response = analysis_result.get("raw_response", "")
            current_price = self._conditions.extract_price(analysis_result)

            if not raw_response:
                self.logger.warning("No response to process")
                return None

            if current_price <= 0:
                self.logger.error("Invalid current_price extracted, cannot process trade")
                return None

            signal, confidence, stop_loss, take_profit, position_size, reasoning = \
                self.extractor.extract_trading_info(raw_response)

            self.logger.info("Extracted Signal: %s, Confidence: %s", signal, confidence)

            if not self.extractor.validate_signal(signal):
                self.logger.warning("Invalid signal: %s", signal)
                return None

            market_conditions = self._conditions.extract_market_conditions(analysis_result)

            confluence_factors = self._conditions.extract_confluence_factors(analysis_result)

            if self.current_position:
                return await self._handle_existing_position(
                    signal, confidence, stop_loss, take_profit,
                    current_price, symbol, reasoning, market_conditions
                )

            if signal in ("BUY", "SELL"):
                return await self._open_new_position(
                    signal, confidence, stop_loss, take_profit,
                    position_size, current_price, symbol, reasoning,
                    confluence_factors, market_conditions
                )

            if reasoning:
                self.logger.info("No action taken. Signal: %s. Reasoning: %s", signal, reasoning[:200])
            else:
                self.logger.info("No action taken. Signal: %s", signal)
            return None

        except Exception as e:
            self.logger.error("Error processing analysis: %s", e)
            return None

    async def _handle_existing_position(
        self,
        signal: str,
        confidence: str,
        stop_loss: float | None,
        take_profit: float | None,
        current_price: float,
        symbol: str,
        reasoning: str,
        market_conditions: MarketConditions | None = None,
    ) -> TradeDecision | None:
        """Handle trading decision when position exists.

        Args:
            signal: Trading signal
            confidence: Confidence level
            stop_loss: New stop loss (for update)
            take_profit: New take profit (for update)
            current_price: Current price
            symbol: Trading symbol
            reasoning: AI reasoning
            market_conditions: Market state for brain learning

        Returns:
            TradeDecision if action taken
        """
        if signal == "CLOSE" or signal.startswith("CLOSE_"):
            self.logger.info("Closing position based on analysis signal...")
            await self.close_position("analysis_signal", current_price, market_conditions)
            return TradeDecision(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                action="CLOSE",
                confidence=confidence,
                price=current_price,
                fee=0.0,
                reasoning=reasoning,
            )

        old_sl = self.current_position.stop_loss
        old_tp = self.current_position.take_profit

        now = datetime.now(timezone.utc)
        if self._last_position_update_time is not None:
            hours_since_last = (now - self._last_position_update_time).total_seconds() / 3600
            if hours_since_last < self._min_update_interval_hours:
                self.logger.info(
                    "REJECTED UPDATE: only %.1fh since last update (min %.1fh for %s). "
                    "Letting trade breathe.",
                    hours_since_last, self._min_update_interval_hours, self.config.TIMEFRAME,
                )
                return None

        self._last_sl_tightening_evaluation = None
        updated = await self._update_position_parameters(stop_loss, take_profit, current_price)

        if updated:
            self._last_position_update_time = now
            try:
                current_pnl = self.current_position.calculate_pnl(current_price)
                self.brain_service.track_position_update(
                    position=self.current_position,
                    old_sl=old_sl,
                    old_tp=old_tp,
                    new_sl=stop_loss if stop_loss else old_sl,
                    new_tp=take_profit if take_profit else old_tp,
                    current_price=current_price,
                    current_pnl_pct=current_pnl,
                    market_conditions=market_conditions,
                    tightening_evaluation=self._last_sl_tightening_evaluation,
                )
            except Exception as e:
                self.logger.warning("Failed to track position update: %s", e)

            decision = TradeDecision(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                action="UPDATE",
                confidence=confidence,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                fee=0.0,
                reasoning=f"Updated position parameters. {reasoning}",
            )
            await self._record_trade_decision(decision)
            self.logger.info("Position updated: New SL=$%s, TP=$%s",
                             f"{stop_loss:,.2f}" if stop_loss else "unchanged",
                             f"{take_profit:,.2f}" if take_profit else "unchanged")
            return decision

        return None

    def _audit(
        self,
        order_id: str,
        event_type: str,
        actor: str,
        result: str,
        reason: str = "",
        **metadata: Any,
    ) -> None:
        """Record an audit event. Keyword args become metadata."""
        self.audit_trail.record(
            order_id=order_id,
            event_type=event_type,
            actor=actor,
            result=result,
            reason=reason,
            metadata=metadata,
        )

    async def _open_new_position(
        self,
        signal: str,
        confidence: str,
        stop_loss: float | None,
        take_profit: float | None,
        position_size: float | None,
        current_price: float,
        symbol: str,
        reasoning: str,
        confluence_factors: tuple = (),
        market_conditions: MarketConditions | None = None,
    ) -> TradeDecision:
        """Open a new trading position with guard-governed lifecycle."""
        direction = "LONG" if signal == "BUY" else "SHORT"
        market_conditions = market_conditions or {}
        order_id = f"order-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"

        intent = OrderIntent(
            order_id=order_id,
            signal=signal, direction=direction, symbol=symbol,
            confidence=confidence, current_price=current_price,
            stop_loss=stop_loss, take_profit=take_profit,
            position_size=position_size, reasoning=reasoning,
            confluence_factors=confluence_factors, market_conditions=market_conditions,
        )
        self.logger.info("Order intent created: %s %s @ $%.2f (order_id=%s)", signal, symbol, current_price, order_id)
        self._audit(order_id, "intent_created", "TradingStrategy", "created",
                    f"{signal} {symbol} @ {current_price}",
                    signal=signal, direction=direction, symbol=symbol)

        if self.guard_pipeline is not None:
            capital = self.statistics_service.get_current_capital(self.config.DEMO_QUOTE_CAPITAL)
            guard_results = self.guard_pipeline.evaluate(intent, capital=capital, config=self.config)

            for result in guard_results:
                self._audit(
                    order_id,
                    "guard_check",
                    result.guard_name,
                    "passed" if result.passed else "failed",
                    result.reason,
                    **result.metadata,
                )

            if not all(r.passed for r in guard_results):
                failed = [r for r in guard_results if not r.passed]
                failure_reasons = "; ".join(f"{r.guard_name}: {r.reason}" for r in failed)
                intent.transition_to(OrderLifecycle.REJECTED, reason=failure_reasons)
                self._audit(
                    order_id, "rejection", "GuardPipeline", "rejected", failure_reasons,
                    failed_guards=[r.guard_name for r in failed]
                )
                self.logger.warning("Order REJECTED by guard pipeline: %s", failure_reasons)
                return TradeDecision(
                    timestamp=datetime.now(timezone.utc), symbol=symbol,
                    action="HOLD", confidence=confidence, price=current_price, fee=0.0,
                    reasoning=f"Order {order_id} rejected by guard pipeline: {failure_reasons}")

            intent.transition_to(OrderLifecycle.READY_FOR_REVIEW, reason="Passed guard pipeline")
        else:
            intent.transition_to(OrderLifecycle.READY_FOR_REVIEW, reason="No guard pipeline configured")

        capital = self.statistics_service.get_current_capital(self.config.DEMO_QUOTE_CAPITAL)

        risk_assessment = self.risk_manager.calculate_entry_parameters(
            signal=signal,
            current_price=current_price,
            capital=capital,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            market_conditions=market_conditions
        )

        try:
            for friction in self.risk_manager.get_and_clear_frictions():
                guard_type = friction.get("guard_type", "unknown")
                friction_dir = friction.get("direction", direction)
                friction_vol = friction.get("volatility_level", risk_assessment.volatility_level)
                self.brain_service.vector_memory.store_blocked_trade(
                    guard_type=guard_type,
                    direction=friction_dir,
                    confidence=confidence,
                    suggested_rr=risk_assessment.rr_ratio,
                    required_rr=risk_assessment.rr_ratio,
                    suggested_sl_pct=friction.get("suggested_sl_pct", risk_assessment.sl_distance_pct),
                    suggested_tp_pct=risk_assessment.tp_distance_pct,
                    suggested_sl=friction.get("suggested_sl", risk_assessment.stop_loss),
                    suggested_tp=friction.get("suggested_tp", risk_assessment.take_profit),
                    current_price=current_price,
                    volatility_level=friction_vol,
                    reasoning_snippet=friction.get("detail", ""),
                    metadata={"friction": friction},
                )
        except Exception:
            self.logger.warning("Failed to store friction event from RiskManager", exc_info=True)

        final_sl = risk_assessment.stop_loss
        final_tp = risk_assessment.take_profit
        final_size_pct = risk_assessment.size_pct
        quantity = risk_assessment.quantity
        entry_fee = risk_assessment.entry_fee
        sl_distance_pct = risk_assessment.sl_distance_pct
        tp_distance_pct = risk_assessment.tp_distance_pct
        rr_ratio = risk_assessment.rr_ratio

        self.logger.info("Position sizing: Capital=$%s, Size=%.2f%%, Allocation=$%s, Quantity=%.6f", f"{capital:,.2f}", final_size_pct * 100, f"{risk_assessment.quote_amount:,.2f}", quantity)
        self.logger.info("Risk metrics: SL=%.2f%%, TP=%.2f%%, R/R=%.2f", sl_distance_pct * 100, tp_distance_pct * 100, rr_ratio)

        brain_thresholds = self.brain_service.get_dynamic_thresholds()
        try:
            min_rr_for_entry = float(brain_thresholds.get("rr_borderline_min", 1.5))
        except (TypeError, ValueError):
            min_rr_for_entry = 1.5
        if rr_ratio < min_rr_for_entry:
            self.logger.warning(
                "REJECTED entry: R/R %.2f below minimum %.1f. "
                "Trade has unfavorable risk/reward. Signal: %s, Confidence: %s",
                rr_ratio, min_rr_for_entry, signal, confidence,
            )
            try:
                self.brain_service.vector_memory.store_blocked_trade(
                    guard_type="rr_minimum", direction=direction, confidence=confidence,
                    suggested_rr=rr_ratio, required_rr=min_rr_for_entry,
                    suggested_sl_pct=sl_distance_pct, suggested_tp_pct=tp_distance_pct,
                    suggested_sl=risk_assessment.stop_loss, suggested_tp=risk_assessment.take_profit,
                    current_price=current_price, volatility_level=risk_assessment.volatility_level,
                    reasoning_snippet=reasoning[:200] if reasoning else "",
                )
            except Exception:
                self.logger.warning("Failed to store blocked trade event", exc_info=True)

            intent.transition_to(OrderLifecycle.REJECTED, reason=f"R/R {rr_ratio:.2f} below minimum")
            self._audit(
                order_id,
                "rejection",
                "TradingStrategy",
                "rejected",
                f"R/R {rr_ratio:.2f} below minimum {min_rr_for_entry}",
                rr_ratio=rr_ratio,
                min_rr_for_entry=min_rr_for_entry,
            )
            return TradeDecision(
                timestamp=datetime.now(timezone.utc), symbol=symbol,
                action="HOLD", confidence=confidence, price=current_price, fee=0.0,
                reasoning=f"Entry blocked: R/R {rr_ratio:.2f} below minimum {min_rr_for_entry}. {reasoning[:150]}" if reasoning else f"Entry blocked: R/R {rr_ratio:.2f} below minimum {min_rr_for_entry}.",
            )

        self._audit(
            order_id,
            "approval",
            "TradingStrategy",
            "approved",
            f"R/R {rr_ratio:.2f} >= minimum {min_rr_for_entry}",
            rr_ratio=rr_ratio,
            min_rr_for_entry=min_rr_for_entry,
        )

        _mc = market_conditions or MarketConditions()
        _ec = build_exit_execution_context_from_config(self.config, self.config.TIMEFRAME)
        self.current_position = Position(
            entry_price=risk_assessment.entry_price,
            stop_loss=risk_assessment.stop_loss,
            take_profit=risk_assessment.take_profit,
            size=risk_assessment.quantity,
            entry_time=datetime.now(timezone.utc),
            confidence=confidence,
            direction=direction,
            symbol=symbol,
            confluence_factors=confluence_factors,
            entry_fee=risk_assessment.entry_fee,
            quote_amount=risk_assessment.quote_amount,
            size_pct=risk_assessment.size_pct,
            atr_at_entry=_mc.atr,
            volatility_level=risk_assessment.volatility_level,
            sl_distance_pct=risk_assessment.sl_distance_pct,
            tp_distance_pct=risk_assessment.tp_distance_pct,
            rr_ratio_at_entry=risk_assessment.rr_ratio,
            adx_at_entry=_mc.adx,
            rsi_at_entry=_mc.rsi,
            trend_direction_at_entry=_mc.trend_direction,
            macd_signal_at_entry=_mc.macd_signal,
            bb_position_at_entry=_mc.bb_position,
            volume_state_at_entry=_mc.volume_state,
            market_sentiment_at_entry=_mc.market_sentiment,
            order_book_bias_at_entry=_mc.order_book_bias,
            stop_loss_type_at_entry=_ec.stop_loss_type,
            stop_loss_check_interval_at_entry=_ec.stop_loss_check_interval,
            take_profit_type_at_entry=_ec.take_profit_type,
            take_profit_check_interval_at_entry=_ec.take_profit_check_interval,
            max_drawdown_pct=0.0,
            max_profit_pct=0.0,
        )

        # Invalidate cooldown guard cache now that a new position was opened
        if self.guard_pipeline is not None:
            self.guard_pipeline.invalidate_cooldown_cache()

        await self.persistence.async_save_position(self.current_position)
        self.logger.info("Opened %s position @ $%s (SL: $%s, TP: $%s, Qty: %.6f, Fee: $%.4f)", direction, f"{current_price:,.2f}", f"{final_sl:,.2f}", f"{final_tp:,.2f}", quantity, entry_fee)

        intent.transition_to(OrderLifecycle.EXECUTED, reason="Position persisted")
        self._audit(
            order_id,
            "execution",
            "TradingStrategy",
            "executed",
            f"Order {order_id} executed: {direction} {symbol} @ {current_price}",
            direction=direction,
            symbol=symbol,
            entry_price=current_price,
            stop_loss=final_sl,
            take_profit=final_tp,
            position_size_pct=final_size_pct,
            quantity=quantity,
        )

        decision = TradeDecision(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            action=signal,
            confidence=confidence,
            price=current_price,
            stop_loss=final_sl,
            take_profit=final_tp,
            position_size=final_size_pct,
            quote_amount=risk_assessment.quote_amount,
            quantity=quantity,
            fee=entry_fee,
            reasoning=reasoning,
        )

        await self._record_trade_decision(decision)

        return decision

    async def _update_position_parameters(
        self,
        stop_loss: float | None,
        take_profit: float | None,
        current_price: float | None = None,
    ) -> bool:
        """Update position stop loss and take profit.

        Args:
            stop_loss: New stop loss
            take_profit: New take profit
            current_price: Current price for SL tightening validation

        Returns:
            True if anything was updated
        """
        if not self.current_position:
            return False

        updated = False
        new_sl = self.current_position.stop_loss
        new_tp = self.current_position.take_profit

        if stop_loss and stop_loss != self.current_position.stop_loss:
            direction = self.current_position.direction
            old_sl = self.current_position.stop_loss
            brain_thresholds = self.brain_service.get_dynamic_thresholds()

            evaluation = self._tightening_policy.evaluate_update(
                position=self.current_position,
                proposed_sl=stop_loss,
                current_price=current_price or 0.0,
                tf_minutes=self._tf_minutes,
                brain_thresholds=brain_thresholds,
            )

            if evaluation.is_tightening:
                if not evaluation.allowed:
                    self.logger.info(
                        "REJECTED premature SL tightening: %s. "
                        "Keeping SL at $%s (AI requested $%s)",
                        evaluation.reason,
                        f"{old_sl:,.2f}",
                        f"{stop_loss:,.2f}",
                    )
                    try:
                        pos = self.current_position
                        self.brain_service.vector_memory.store_blocked_trade(
                            guard_type="sl_tightening",
                            direction=direction,
                            confidence=pos.confidence,
                            suggested_rr=0.0,
                            required_rr=0.0,
                            suggested_sl_pct=abs(stop_loss - pos.entry_price) / pos.entry_price if pos.entry_price else 0.0,
                            suggested_tp_pct=pos.tp_distance_pct,
                            suggested_sl=stop_loss,
                            suggested_tp=pos.take_profit,
                            current_price=current_price or 0.0,
                            volatility_level=pos.volatility_level,
                            reasoning_snippet=evaluation.reason[:200],
                            metadata={
                                "price_progress": evaluation.price_progress,
                                "effective_min_progress": evaluation.effective_min_progress,
                                "base_min_progress": evaluation.base_min_progress,
                                "policy_source": evaluation.source,
                                "tf_minutes": self._tf_minutes,
                                "position_entry_timestamp": pos.entry_time.isoformat(),
                                "position_entry_trade_id": f"trade_{pos.entry_time.isoformat()}",
                                "position_id": f"{pos.symbol}|{pos.entry_time.isoformat()}",
                            },
                        )
                    except Exception:
                        self.logger.warning("Failed to store sl_tightening blocked event", exc_info=True)
                else:
                    new_sl = stop_loss
                    self._last_sl_tightening_evaluation = evaluation
                    self.logger.info(
                        "Tightening Stop Loss: $%s -> $%s (%s)",
                        f"{old_sl:,.2f}",
                        f"{stop_loss:,.2f}",
                        evaluation.reason,
                    )
                    updated = True
            else:
                # SL is widening or unchanged — allow unconditionally
                if direction == "LONG" and stop_loss < old_sl:
                    self.logger.info(
                        "AI Widening Stop Loss for LONG: $%.2f -> $%.2f (Risk Increased)",
                        old_sl, stop_loss
                    )
                elif direction == "SHORT" and stop_loss > old_sl:
                    self.logger.info(
                        "AI Widening Stop Loss for SHORT: $%.2f -> $%.2f (Risk Increased)",
                        old_sl, stop_loss
                    )
                else:
                    self.logger.info("Updated Stop Loss: $%s", f"{stop_loss:,.2f}")
                new_sl = stop_loss
                updated = True

        if take_profit and take_profit != self.current_position.take_profit:
            new_tp = take_profit
            self.logger.info("Updated Take Profit: $%s", f"{take_profit:,.2f}")
            updated = True

        if updated:
            self.current_position = dataclasses.replace(
                self.current_position,
                stop_loss=new_sl,
                take_profit=new_tp,
            )
            await self.persistence.async_save_position(self.current_position)

        return updated

    def _get_last_closed_position_info(self) -> str | None:
        """Query trade history for the most recent closed position.

        Returns:
            Formatted string like 'LONG, closed 3.2 hours ago' or None if no history.
        """
        try:
            rows = self.persistence.sqlite_history.query(
                action=None,
                limit=20,
                order="DESC",
            )
            for row in rows:
                action = row.get("action", "")
                if action in ("CLOSE_LONG", "CLOSE_SHORT"):
                    direction = "LONG" if action == "CLOSE_LONG" else "SHORT"
                    ts_str = row.get("timestamp", "")
                    if not ts_str:
                        continue
                    try:
                        close_time = datetime.fromisoformat(ts_str)
                        if close_time.tzinfo is None:
                            close_time = close_time.replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        continue
                    now = datetime.now(timezone.utc)
                    delta = now - close_time
                    total_seconds = delta.total_seconds()
                    if total_seconds < 3600:
                        time_ago = f"{total_seconds / 60:.0f} minutes ago"
                    elif total_seconds < 86400:
                        time_ago = f"{total_seconds / 3600:.1f} hours ago"
                    else:
                        time_ago = f"{total_seconds / 86400:.1f} days ago"
                    return f"{direction}, closed {time_ago}"
            return None
        except Exception:
            return None

    def get_position_context(self, current_price: float | None = None) -> str:
        """Get formatted context about current position for prompts.

        Args:
            current_price: Current market price for P&L calculation

        Returns:
            Formatted position context string with capital status
        """
        capital = self.statistics_service.get_current_capital(self.config.DEMO_QUOTE_CAPITAL)
        currency = self.config.QUOTE_CURRENCY

        capital_header = [
            "## Capital Status",
            f"- Total Capital: ${capital:,.2f} {currency}",
        ]
        position_header = "## Current Position"

        if not self.current_position:
            lines = capital_header + [
                f"- Available: ${capital:,.2f} (100%)",
                "",
                position_header,
                "- Status: None",
            ]
            last_closed = self._get_last_closed_position_info()
            if last_closed:
                lines.append(f"- Last Position: {last_closed}")
            return "\n".join(lines)

        pos = self.current_position
        now = datetime.now(timezone.utc)
        entry_time = pos.entry_time
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        duration = now - entry_time
        hours = duration.total_seconds() / 3600
        allocated = pos.quote_amount
        available = capital - allocated
        allocation_pct = (allocated / capital) * 100 if capital > 0 else 0

        lines = capital_header + [
            f"- Allocated: ${allocated:,.2f} ({allocation_pct:.1f}%)",
            f"- Available: ${available:,.2f} ({100 - allocation_pct:.1f}%)",
            "",
            position_header,
            f"- Direction: {pos.direction}",
            f"- Symbol: {pos.symbol}",
            f"- Entry Price: ${pos.entry_price:,.2f}",
        ]
        if current_price and current_price > 0:
            lines.append(f"- Current Price: ${current_price:,.2f}")
        lines.extend([
            f"- Stop Loss: ${pos.stop_loss:,.2f}",
            f"- Take Profit: ${pos.take_profit:,.2f}",
            f"- Position Size: {pos.size_pct * 100:.2f}%",
            f"- Quantity: {pos.size:.6f}",
            f"- Entry Fee: ${pos.entry_fee:.4f}",
            f"- Duration: {hours:.1f} hours",
            f"- Confidence: {pos.confidence}",
        ])
        if current_price and current_price > 0:
            pnl_pct = pos.calculate_pnl(current_price)
            pnl_quote = (current_price - pos.entry_price) * pos.size if pos.direction == 'LONG' else (pos.entry_price - current_price) * pos.size
            lines.append(f"- Unrealized P&L: {pnl_pct:+.2f}% (${pnl_quote:+,.2f} {currency})")

        brain_thresholds = self.brain_service.get_dynamic_thresholds()
        sentinel_sl = pos.stop_loss - 1e-8 if pos.direction == "SHORT" else pos.stop_loss + 1e-8
        sl_eval = self._tightening_policy.evaluate_update(
            position=pos,
            proposed_sl=sentinel_sl,
            current_price=current_price or 0.0,
            tf_minutes=self._tf_minutes,
            brain_thresholds=brain_thresholds,
        )
        effective_pct = sl_eval.effective_min_progress * 100
        progress_pct = sl_eval.price_progress * 100 if current_price and current_price > 0 else None
        if progress_pct is not None:
            eligible = progress_pct >= effective_pct
            lines.extend([
                "",
                "## SL Tightening Policy",
                f"- Effective minimum progress: {effective_pct:.0f}% of entry-to-TP (source: {sl_eval.source})",
                f"- Current price progress: {progress_pct:.1f}%",
                f"- Tightening eligible: {'YES' if eligible else 'NO — wait until price progress reaches the minimum'}",
            ])
        else:
            lines.extend([
                "",
                "## SL Tightening Policy",
                f"- Effective minimum progress: {effective_pct:.0f}% of entry-to-TP (source: {sl_eval.source})",
            ])

        return "\n".join(lines)
