"""Position lifecycle for the trading strategy.

Opening a position, updating its SL/TP, and deciding what to do with an existing position.
Split out of trading_strategy.py; the methods use ``self`` state provided by TradingStrategy
at MRO resolution time.
"""

import asyncio
import dataclasses
from datetime import datetime, timezone
from typing import Any

from src.utils.indicator_classifier import build_exit_execution_context_from_config

from .data_models import MarketConditions, Position, TradeDecision, entry_direction
from .order_lifecycle import OrderIntent, OrderLifecycle
from .rr_policy import format_rr_floor, resolve_entry_rr_floor


class PositionManagementMixin:
    """Entry, parameter-update and existing-position decision handling."""

    logger: Any
    config: Any
    persistence: Any
    statistics_service: Any
    risk_manager: Any
    brain_service: Any
    guard_pipeline: Any
    current_position: Position | None
    _executor_has_position: Any
    _record_trade_decision: Any
    close_position: Any
    _tf_minutes: int
    _tightening_policy: Any
    _last_position_update_time: Any
    _last_sl_tightening_evaluation: Any
    _min_update_interval_hours: float

    async def _handle_existing_position(
        self,
        signal: str,
        confidence: str,
        stop_loss: float | None,
        take_profit: float | None,
        current_price: float,
        symbol: str,
        reasoning: str,
        market_conditions: MarketConditions,
    ) -> TradeDecision | None:
        """Handle trading decision when position exists.
        Returns:
            TradeDecision if action taken
        """
        executor_pos_state = await self._executor_has_position(symbol)

        if signal == "CLOSE" or signal.startswith("CLOSE_"):
            if executor_pos_state is False:
                self.logger.warning(
                    "CLOSE signal for %s but executor confirmed no open position — "
                    "position was closed on exchange or rejected on entry. Resetting local position state.",
                    symbol,
                )
                self.current_position = None
                await self.persistence.async_save_position(None)
                return None
            if executor_pos_state is None:
                self.logger.warning(
                    "CLOSE signal for %s skipped — failed to verify executor position state.",
                    symbol,
                )
                return None

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

        old_sl = self.current_position.stop_loss  # type: ignore
        old_tp = self.current_position.take_profit  # type: ignore

        if executor_pos_state is False:
            self.logger.warning(
                "UPDATE for %s skipped — executor confirmed no open position. "
                "The position was closed on exchange or rejected on entry. Clearing local ghost position state.",
                symbol,
            )
            self.current_position = None
            await self.persistence.async_save_position(None)
            return None
        if executor_pos_state is None:
            self.logger.warning(
                "UPDATE for %s skipped — failed to verify executor position state.",
                symbol,
            )
            return None

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
                current_pnl = self.current_position.calculate_pnl(current_price)  # type: ignore
                self.brain_service.track_position_update(
                    position=self.current_position,  # type: ignore
                    old_sl=old_sl,
                    old_tp=old_tp,
                    new_sl=stop_loss if stop_loss else old_sl,
                    new_tp=take_profit if take_profit else old_tp,
                    current_price=current_price,
                    current_pnl_pct=current_pnl,
                    market_conditions=market_conditions,
                    tightening_evaluation=self._last_sl_tightening_evaluation,
                )
            except Exception as e:  # noqa: BLE001
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

    def _reject_intent(
        self, intent: OrderIntent, confidence: str, current_price: float, order_id: str
    ) -> TradeDecision | None:
        """Run the pre-execution guard pipeline; return the HOLD decision when it blocks."""
        if self.guard_pipeline is None:
            intent.transition_to(OrderLifecycle.READY_FOR_REVIEW, reason="No guard pipeline configured")
            return None
        capital = self.statistics_service.get_current_capital(self.config.DEMO_QUOTE_CAPITAL)
        guard_results = self.guard_pipeline.evaluate(intent, capital=capital, config=self.config)
        failed = [result for result in guard_results if not result.passed]
        if not failed:
            intent.transition_to(OrderLifecycle.READY_FOR_REVIEW, reason="Passed guard pipeline")
            return None
        failure_reasons = "; ".join(f"{result.guard_name}: {result.reason}" for result in failed)
        intent.transition_to(OrderLifecycle.REJECTED, reason=failure_reasons)
        self.logger.warning("Order REJECTED by guard pipeline: %s", failure_reasons)
        return TradeDecision(
            timestamp=datetime.now(timezone.utc), symbol=intent.symbol,
            action="HOLD", confidence=confidence, price=current_price, fee=0.0,
            reasoning=f"Order {order_id} rejected by guard pipeline: {failure_reasons}",
        )

    async def _store_risk_frictions(
        self, risk, direction: str, confidence: str, current_price: float, min_rr_for_entry: float
    ) -> None:
        """Persist RiskManager SL/TP clamping frictions so the brain learns from them."""
        try:
            for friction in self.risk_manager.get_and_clear_frictions():
                await asyncio.to_thread(
                    self.brain_service.vector_memory.store_blocked_trade,
                    guard_type=friction.get("guard_type", "unknown"),
                    direction=friction.get("direction", direction),
                    confidence=confidence,
                    suggested_rr=risk.rr_ratio,
                    required_rr=min_rr_for_entry,
                    suggested_sl_pct=friction.get("suggested_sl_pct", risk.sl_distance_pct),
                    suggested_tp_pct=friction.get("suggested_tp_pct", risk.tp_distance_pct),
                    suggested_sl=friction.get("suggested_sl", risk.stop_loss),
                    suggested_tp=friction.get("suggested_tp", risk.take_profit),
                    current_price=current_price,
                    volatility_level=friction.get("volatility_level", risk.volatility_level),
                    reasoning_snippet=friction.get("detail", ""),
                    metadata={"friction": friction},
                )
        except Exception:
            self.logger.warning("Failed to store friction event from RiskManager", exc_info=True)

    def _resolve_min_rr_for_entry(
        self,
        brain_thresholds: dict[str, Any] | None = None,
    ) -> float:
        """Return the single R/R floor shared with the prompt."""
        thresholds = (
            brain_thresholds
            if brain_thresholds is not None
            else self.brain_service.get_dynamic_thresholds()
        )
        return resolve_entry_rr_floor(self.config, thresholds)

    async def _check_entry_thresholds(
        self,
        risk,
        intent: OrderIntent,
        direction: str,
        signal: str,
        confidence: str,
        current_price: float,
        reasoning: str,
        min_rr_for_entry: float,
    ) -> TradeDecision | None:
        """Enforce the config-driven R/R floor; store the rejection and return a HOLD decision."""
        if risk.rr_ratio >= min_rr_for_entry:
            return None

        min_rr_text = format_rr_floor(min_rr_for_entry)
        self.logger.warning(
            "REJECTED entry: R/R %.2f below minimum %s. "
            "Trade has unfavorable risk/reward. Signal: %s, Confidence: %s",
            risk.rr_ratio, min_rr_text, signal, confidence,
        )
        try:
            await asyncio.to_thread(
                self.brain_service.vector_memory.store_blocked_trade,
                guard_type="rr_minimum", direction=direction, confidence=confidence,
                suggested_rr=risk.rr_ratio, required_rr=min_rr_for_entry,
                suggested_sl_pct=risk.sl_distance_pct, suggested_tp_pct=risk.tp_distance_pct,
                suggested_sl=risk.stop_loss, suggested_tp=risk.take_profit,
                current_price=current_price, volatility_level=risk.volatility_level,
                reasoning_snippet=reasoning or "",
            )
        except Exception:
            self.logger.warning("Failed to store blocked trade event", exc_info=True)

        intent.transition_to(OrderLifecycle.REJECTED, reason=f"R/R {risk.rr_ratio:.2f} below minimum")
        detail = f" {reasoning[:150]}" if reasoning else ""
        return TradeDecision(
            timestamp=datetime.now(timezone.utc), symbol=intent.symbol,
            action="HOLD", confidence=confidence, price=current_price, fee=0.0,
            reasoning=f"Entry blocked: R/R {risk.rr_ratio:.2f} below minimum {min_rr_text}.{detail}",
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
        market_conditions: MarketConditions,
        confluence_factors: tuple = (),
    ) -> TradeDecision:
        """Open a new trading position with guard-governed lifecycle."""
        direction = entry_direction(signal)
        order_id = f"order-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        intent = OrderIntent(
            order_id=order_id, signal=signal, direction=direction, symbol=symbol,
            confidence=confidence, current_price=current_price,
            stop_loss=stop_loss, take_profit=take_profit,
            position_size=position_size, reasoning=reasoning,
            confluence_factors=confluence_factors, market_conditions=market_conditions,
        )
        self.logger.info("Order intent created: %s %s @ $%.2f (order_id=%s)", signal, symbol, current_price, order_id)

        rejected = self._reject_intent(intent, confidence, current_price, order_id)
        if rejected is not None:
            return rejected

        capital = self.statistics_service.get_current_capital(self.config.DEMO_QUOTE_CAPITAL)
        risk = self.risk_manager.calculate_entry_parameters(
            signal=signal,
            current_price=current_price,
            capital=capital,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            market_conditions=market_conditions,
            choppiness=market_conditions.choppiness,
        )
        min_rr_for_entry = self._resolve_min_rr_for_entry()
        await self._store_risk_frictions(risk, direction, confidence, current_price, min_rr_for_entry)

        blocked = await self._check_entry_thresholds(
            risk, intent, direction, signal, confidence, current_price, reasoning, min_rr_for_entry
        )
        if blocked is not None:
            return blocked

        quantity = risk.quantity
        size_pct = risk.size_pct
        quote_amount = risk.quote_amount
        entry_fee = risk.entry_fee

        executor_max = float(self.config.EXECUTOR_MAX_POSITION_USDC or 0.0)
        notional = quantity * current_price
        if executor_max > 0 and notional > executor_max:
            scale = executor_max / notional
            self.logger.warning(
                "Executor notional clamp: $%.2f exceeds max $%.2f — "
                "scaling qty %.6f→%.6f, size %.2f%%→%.2f%%",
                notional, executor_max, quantity, quantity * scale,
                risk.size_pct * 100, risk.size_pct * scale * 100,
            )
            quantity *= scale
            size_pct *= scale
            quote_amount *= scale
            entry_fee *= scale

        self.logger.info(
            "Position sizing: Capital=$%s, Size=%.2f%%, Allocation=$%s, Quantity=%.6f",
            f"{capital:,.2f}", size_pct * 100, f"{quote_amount:,.2f}", quantity,
        )
        self.logger.info(
            "Risk metrics: SL=%.2f%%, TP=%.2f%%, R/R=%.2f",
            risk.sl_distance_pct * 100, risk.tp_distance_pct * 100, risk.rr_ratio,
        )

        mc = market_conditions
        ec = build_exit_execution_context_from_config(self.config, self.config.TIMEFRAME)
        self.current_position = Position(
            entry_price=risk.entry_price,
            stop_loss=risk.stop_loss,
            take_profit=risk.take_profit,
            size=quantity,
            entry_time=datetime.now(timezone.utc),
            confidence=confidence,
            direction=direction,
            symbol=symbol,
            confluence_factors=confluence_factors,
            entry_fee=entry_fee,
            quote_amount=quote_amount,
            size_pct=size_pct,
            atr_at_entry=mc.atr,
            atr_percentage_at_entry=mc.atr_percentage,
            conditions_at_entry=mc,
            volatility_level=risk.volatility_level,
            sl_distance_pct=risk.sl_distance_pct,
            tp_distance_pct=risk.tp_distance_pct,
            rr_ratio_at_entry=risk.rr_ratio,
            adx_at_entry=mc.adx,
            rsi_at_entry=mc.rsi,
            trend_direction_at_entry=mc.trend_direction,
            macd_signal_at_entry=mc.macd_signal,
            bb_position_at_entry=mc.bb_position,
            volume_state_at_entry=mc.volume_state,
            market_sentiment_at_entry=mc.market_sentiment,
            order_book_bias_at_entry=mc.order_book_bias,
            stop_loss_type_at_entry=ec.stop_loss_type,
            stop_loss_check_interval_at_entry=ec.stop_loss_check_interval,
            take_profit_type_at_entry=ec.take_profit_type,
            take_profit_check_interval_at_entry=ec.take_profit_check_interval,
            max_drawdown_pct=0.0,
            max_profit_pct=0.0,
            regime_profile=risk.regime_profile,
        )

        if self.guard_pipeline is not None:
            self.guard_pipeline.invalidate_cooldown_cache()

        await self.persistence.async_save_position(self.current_position)
        self.logger.info(
            "Opened %s position @ $%s (SL: $%s, TP: $%s, Qty: %.6f, Fee: $%.4f)",
            direction, f"{current_price:,.2f}", f"{risk.stop_loss:,.2f}",
            f"{risk.take_profit:,.2f}", quantity, entry_fee,
        )
        intent.transition_to(OrderLifecycle.EXECUTED, reason="Position persisted")

        decision = TradeDecision(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            action=signal,
            confidence=confidence,
            price=current_price,
            stop_loss=risk.stop_loss,
            take_profit=risk.take_profit,
            position_size=size_pct,
            quote_amount=quote_amount,
            quantity=quantity,
            fee=entry_fee,
            reasoning=reasoning,
            indicators_json={
                "adx_at_entry": mc.adx,
                "rsi_at_entry": mc.rsi,
                "volatility_level": risk.volatility_level,
                "macd_signal_at_entry": mc.macd_signal,
                "bb_position_at_entry": mc.bb_position,
                "volume_state_at_entry": mc.volume_state,
                "market_sentiment_at_entry": mc.market_sentiment,
                "order_book_bias_at_entry": mc.order_book_bias,
                "sl_distance_pct": risk.sl_distance_pct,
                "tp_distance_pct": risk.tp_distance_pct,
                "rr_ratio_at_entry": risk.rr_ratio,
                "trend_direction_at_entry": mc.trend_direction,
            },
            order_id=order_id,
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
                        await asyncio.to_thread(
                            self.brain_service.vector_memory.store_blocked_trade,
                            guard_type="sl_tightening",
                            direction=direction,
                            confidence=pos.confidence,
                            suggested_rr=pos.rr_ratio_at_entry,
                            required_rr=self._resolve_min_rr_for_entry(brain_thresholds=brain_thresholds),
                            suggested_sl_pct=abs(stop_loss - pos.entry_price) / pos.entry_price if pos.entry_price else 0.0,
                            suggested_tp_pct=pos.tp_distance_pct,
                            suggested_sl=stop_loss,
                            suggested_tp=pos.take_profit,
                            current_price=current_price or 0.0,
                            volatility_level=pos.volatility_level,
                            reasoning_snippet=evaluation.reason,
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
                entry_price = self.current_position.entry_price
                original_sl_distance = abs(entry_price - old_sl)
                proposed_sl_distance = abs(entry_price - stop_loss)
                max_allowed_distance = original_sl_distance * 1.5

                if (
                    original_sl_distance > 0
                    and proposed_sl_distance > max_allowed_distance
                ):
                    self.logger.warning(
                        "REJECTED SL widening: proposed distance %.2f%% exceeds "
                        "150%% of original %.2f%%. Keeping SL at $%.2f "
                        "(AI requested $%.2f)",
                        proposed_sl_distance / entry_price * 100,
                        original_sl_distance / entry_price * 100,
                        old_sl,
                        stop_loss,
                    )
                else:
                    if direction == "LONG" and stop_loss < old_sl:
                        self.logger.info(
                            "AI Widening Stop Loss for LONG: $%.2f -> $%.2f "
                            "(Risk Increased)",
                            old_sl, stop_loss,
                        )
                    elif direction == "SHORT" and stop_loss > old_sl:
                        self.logger.info(
                            "AI Widening Stop Loss for SHORT: $%.2f -> $%.2f "
                            "(Risk Increased)",
                            old_sl, stop_loss,
                        )
                    else:
                        self.logger.info(
                            "Updated Stop Loss: $%s", f"{stop_loss:,.2f}",
                        )
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
