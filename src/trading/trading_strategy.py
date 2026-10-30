

import asyncio
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from src.logger.logger import Logger
from src.utils.timeframe_validator import TimeframeValidator

from .brain import TradingBrainService
from .data_models import (
    FILL_DIVERGENCE_MARKER,
    LocalExitRequest,
    MarketConditions,
    Position,
    TradeDecision,
    UnconfirmedIntentAlert,
    market_verification_is_verified,
)
from .executor_reconciliation import (
    INTENT_ACTION_CLOSE,
    INTENT_CONFIRMED,
    INTENT_PENDING,
    INTENT_REFUSED,
    INTENT_UNKNOWN,
    LESSON_SKIPPED_ALREADY_LEARNED,
    LESSON_SKIPPED_FILL_DIVERGENCE,
    LESSON_SKIPPED_NO_EVIDENCE,
    LESSON_SKIPPED_NO_POST_MORTEM,
    MAX_LEARNED_EXIT_EVENTS,
    ExecutorReconciliationMixin,
    ExitRecords,
    LocalPositionReconciliation,
    PositionIntentLedger,
    position_identity,
)
from .guards.pipeline import GuardPipeline
from .memory import TradingMemoryService
from .position_management import PositionManagementMixin
from .statistics import TradingStatisticsService
from .stop_loss_tightening_policy import StopLossTighteningPolicy, TighteningEvaluation

if TYPE_CHECKING:
    from src.dashboard.dashboard_state import DashboardState
    from src.managers.persistence_manager import PersistenceManager
    from src.managers.risk_manager import RiskManager

    from .market_conditions_extractor import MarketConditionsExtractor


class TradingStrategy(ExecutorReconciliationMixin, PositionManagementMixin):

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
        post_mortem_service: Any | None = None,
    ):

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

        self.guard_pipeline = guard_pipeline
        self.post_mortem_service = post_mortem_service
        self._http_client = None
        self._last_executor_position_payload: dict | None = None

        self.current_position: Position | None = self.persistence.load_position()

        self._executor_side_exit_reason: str | None = None
        self._state_divergence: str | None = None
        self._local_exit_request: LocalExitRequest | None = None
        self._pending_local_close_decision: TradeDecision | None = None

        self._position_transition_lock: asyncio.Lock | None = None
        self._booked_exit_events: set[str] = set()
        self._position_intent_ledger: PositionIntentLedger | None = None
        self._unconfirmed_intent_alert: UnconfirmedIntentAlert | None = None
        self._unconfirmed_intent_alert_at: datetime | None = None
        self._rejected_intent_alert: str | None = None
        self._last_booked_exit: dict[str, Any] | None = None
        self._last_reconciliation: LocalPositionReconciliation | None = None
        self._learned_exit_events: set[str] = set()

        self._tf_minutes: int = TimeframeValidator.to_minutes(config.TIMEFRAME)

        self._tightening_policy: StopLossTighteningPolicy = (
            tightening_policy if tightening_policy is not None else StopLossTighteningPolicy()
        )

        self._last_sl_tightening_evaluation: TighteningEvaluation | None = None

        if self.current_position:
            self.logger.info("Loaded existing position: %s %s @ $%s", self.current_position.direction, self.current_position.symbol, f"{self.current_position.entry_price:,.2f}")

        try:
            expected_symbol = config.CRYPTO_PAIR if config else None
            state_warnings = self.persistence.validate_loaded_position(expected_symbol)
            for warning in state_warnings:
                self.logger.warning("STARTUP STATE WARNING: %s", warning)
        except Exception as e:  # noqa: BLE001
            self.logger.warning("Could not validate loaded position: %s", e)

    def set_dashboard_state(self, dashboard_state: "DashboardState | None") -> None:

        self.dashboard_state = dashboard_state

    async def _record_trade_decision(self, decision: TradeDecision) -> int:

        row_id = await self.persistence.async_save_trade_decision(decision)
        self.memory_service.add_decision(decision)
        return row_id

    async def _update_live_metrics(self, current_price: float) -> bool:

        if not self.current_position:
            return False
        self.current_position.update_metrics(current_price)
        await self.persistence.async_save_position(self.current_position)
        return True

    async def check_position(self, current_price: float) -> str | None:

        if not await self._update_live_metrics(current_price):
            return None

        if self.current_position.is_stop_hit(current_price):  # type: ignore
            return await self._close_on_exit("stop_loss", current_price)

        if self.current_position.is_target_hit(current_price):  # type: ignore
            return await self._close_on_exit("take_profit", current_price)

        return None

    async def check_stop_loss(self, current_price: float) -> str | None:

        if not await self._update_live_metrics(current_price):
            return None

        if self.current_position.is_stop_hit(current_price):  # type: ignore
            return await self._close_on_exit("stop_loss", current_price)

        return None

    async def check_take_profit(self, current_price: float) -> str | None:

        if not await self._update_live_metrics(current_price):
            return None

        if self.current_position.is_target_hit(current_price):  # type: ignore
            return await self._close_on_exit("take_profit", current_price)

        return None

    async def check_reference_exit(self, current_price: float) -> str | None:

        position = self.current_position
        if position is None:
            return None
        if position.is_stop_hit(current_price):  # type: ignore
            return await self._close_on_exit("stop_loss", current_price)
        if position.is_target_hit(current_price):  # type: ignore
            return await self._close_on_exit("take_profit", current_price)
        return None

    def executor_query_available(self) -> bool:

        return self._executor_query_available()

    async def _close_on_exit(self, reason: str, current_price: float) -> str | None:

        position = self.current_position
        if position is None:
            return None

        position_key = position_identity(position)
        symbol = position.symbol
        ledger = self.position_intents()
        intent_key = self.intent_identity(INTENT_ACTION_CLOSE, symbol, position_id=position_key)
        previous_intent = ledger.get(intent_key)
        self._local_exit_request_was_pending = bool(
            previous_intent is not None and previous_intent.state == INTENT_PENDING
        )

        if self._executor_query_available():
            existing = previous_intent
            order_id = (
                existing.order_id
                if existing is not None
                and existing.state in (INTENT_PENDING, INTENT_UNKNOWN)
                and existing.order_id
                else None
            ) or f"close-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
            self.register_position_intent(
                INTENT_ACTION_CLOSE,
                symbol,
                order_id=order_id,
                position_id=position_key,
                detail=(
                    f"the local exit condition ({reason}) triggered at price "
                    f"{current_price:,.2f} — CLOSE intent; the ticker price is NOT the exit "
                    f"price and books nothing"
                ),
                payload={
                    "local_exit_reason": reason,
                    "observed_price": current_price,
                    "source": "local_monitor",
                },
            )
            self._pending_local_close_decision = TradeDecision(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                action="CLOSE",
                confidence=position.confidence,
                price=current_price,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                quantity=position.size,
                fee=0.0,
                reasoning=(
                    f"The local exit condition ({reason}) triggered at price "
                    f"{current_price:,.2f}. The exit is NOT booked: booking happens "
                    f"only from the executor's fill evidence."
                ),
                order_id=order_id,
            )
            state = INTENT_PENDING
            note = (
                f"the local exit condition {reason} triggered at price {current_price:,.2f} — "
                f"a CLOSE intent was sent to the executor; the local position STAYS, history and "
                f"statistics unchanged, until fill evidence appears (the ticker price does not "
                f"book an exit)"
            )
        else:
            ledger.record(
                INTENT_ACTION_CLOSE,
                symbol,
                key=intent_key,
                position_id=position_key,
                state=INTENT_UNKNOWN,
                evidence="local_monitor",
                detail=(
                    "no executor integration — the local exit condition cannot "
                    "book a close; there is no fill evidence at all"
                ),
                payload={"local_exit_reason": reason, "observed_price": current_price},
            )
            order_id = None
            state = INTENT_UNKNOWN
            note = (
                f"the local exit condition {reason} triggered at price {current_price:,.2f}, but "
                f"executor integration is DISABLED — there is no fill evidence, so the exit was NOT "
                f"booked (state: {INTENT_UNKNOWN}/unresolved); the local position "
                f"STAYS without clearing and without zeroing — check the exchange state"
            )
            self._state_divergence = note

        self._local_exit_request = LocalExitRequest(
            reason=reason,
            observed_price=current_price,
            state=state,
            intent_key=intent_key,
            order_id=order_id,
            detail=note,
        )
        self._log_local_exit_condition(reason, current_price, state)
        self._raise_unconfirmed_intent_alert(note, intent_key)
        return None

    def _log_local_exit_condition(self, reason: str, current_price: float, state: str) -> None:

        if self._local_exit_request_was_pending:
            self.logger.debug(
                "Local exit condition %s still pending @ %s (state=%s, position kept)",
                reason, current_price, state,
            )
            return
        self.logger.critical(
            "LOCAL EXIT CONDITION %s hit @ %s — NOTHING booked locally (state=%s, position kept)",
            reason, current_price, state,
        )

    def take_local_exit_request(self) -> "LocalExitRequest | None":

        request = self._local_exit_request
        self._local_exit_request = None
        return request

    def take_pending_local_close_decision(self) -> TradeDecision | None:

        decision = self._pending_local_close_decision
        self._pending_local_close_decision = None
        return decision

    @staticmethod
    def _booked_quantity(
        position: Position, filled_quantity: float | None
    ) -> tuple[float, str]:

        local = position.size
        if isinstance(filled_quantity, bool):
            return local, ""
        try:
            filled = float(filled_quantity)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return local, ""
        if not math.isfinite(filled) or filled <= 0:
            return local, ""
        if abs(filled - local) > max(1e-9, abs(local) * 0.01):
            return filled, (
                f" WARNING: the executor fill quantity {filled:.8f} differs from the local "
                f"size {local:.8f} — the fill quantity was booked."
            )
        return filled, ""

    async def close_position(
        self,
        reason: str,
        current_price: float,
        market_conditions: MarketConditions,
        *,
        exit_fee: float | None = None,
        fee_source: str | None = None,
        filled_quantity: float | None = None,
        evidence: Any | None = None,
        exit_time: datetime | None = None,
    ) -> None:

        if not self.current_position:
            return

        closed_position = self.current_position
        pnl = closed_position.calculate_pnl(current_price)

        fee_text = self._commission_text(exit_fee, fee_source)
        learnable, refusal_reason = ExitRecords.is_learnable(evidence)
        entry_quality = closed_position.entry_execution_quality
        if (entry_quality is not None or self.config.EXECUTOR_API_ENABLED) and not market_verification_is_verified(entry_quality):
            learnable = False
            refusal_reason = f"entry {ExitRecords.verification_gap_reason(entry_quality)}"
        exit_quality = evidence.to_dict() if evidence is not None and not isinstance(evidence, dict) else dict(evidence or {})
        execution_quality = {"eligible": learnable, "entry": entry_quality, "exit": exit_quality, "reason": refusal_reason}
        divergent = ExitRecords.close_fill_is_divergent(entry_quality, evidence)
        if not learnable:
            marker = FILL_DIVERGENCE_MARKER if divergent else "EXECUTION UNVERIFIED"
            if marker not in reason:
                reason = f"{marker}: {refusal_reason}. {reason}"
        if filled_quantity is None:
            filled_quantity = getattr(evidence, "quantity", None)
        booked_quantity, quantity_note = self._booked_quantity(closed_position, filled_quantity)
        if quantity_note:
            self.logger.critical(
                "QUANTITY DIVERGENCE for %s on close (%s): booked the FILL amount",
                closed_position.symbol, reason,
            )

        pnl_text = f"P&L: {pnl:+.2f}%." if learnable else "Strategy P&L: excluded (unverified execution)."
        decision = TradeDecision(
            timestamp=exit_time or datetime.now(timezone.utc),
            symbol=closed_position.symbol,
            action=f"CLOSE_{closed_position.direction}",
            confidence=closed_position.confidence,
            price=current_price,
            stop_loss=closed_position.stop_loss,
            take_profit=closed_position.take_profit,
            position_size=closed_position.size_pct,
            quote_amount=closed_position.quote_amount,
            quantity=booked_quantity,
            fee=exit_fee,
            execution_quality=execution_quality,
            reasoning=(
                f"Position closed: {reason}. {pnl_text} {fee_text}{quantity_note}"
            ),
        )

        self.logger.info("Closing %s position (%s) @ $%s, P&L: %s%%, %s", closed_position.direction, reason, f"{current_price:,.2f}", f"{pnl:+.2f}", fee_text)

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
        except Exception as e:  # noqa: BLE001
            self.logger.error("Error retrieving entry decision: %s", e)

        close_row_id = await self._record_trade_decision(decision)

        post_mortem_result: Any | None = None
        if not learnable:
            self.logger.warning(
                "%s (%s) | %s %s",
                LESSON_SKIPPED_FILL_DIVERGENCE if divergent else LESSON_SKIPPED_NO_EVIDENCE,
                refusal_reason, closed_position.symbol, reason,
            )
        elif not entry_decision:
            self.logger.warning(
                "%s (no entry decision for this trade) | %s",
                LESSON_SKIPPED_NO_POST_MORTEM, closed_position.symbol,
            )
        elif not self.post_mortem_service:
            self.logger.warning(
                "%s (post-mortem service is not connected) | %s",
                LESSON_SKIPPED_NO_POST_MORTEM, closed_position.symbol,
            )
        else:
            try:
                post_mortem_result = await self.post_mortem_service.analyze_closed_trade(
                    closed_position=closed_position,
                    entry_decision=entry_decision,
                    exit_decision=decision,
                    trade_id=close_row_id,
                    pnl=pnl,
                    reason=reason,
                    market_conditions=market_conditions,
                )
            except Exception:
                self.logger.warning("Post-mortem analysis failed", exc_info=True)
                post_mortem_result = None
            if post_mortem_result is None:
                self.logger.warning(
                    "%s (post-mortem returned no confirmed analysis) | %s %s",
                    LESSON_SKIPPED_NO_POST_MORTEM, closed_position.symbol, reason,
                )

        try:
            self.statistics_service.recalculate(self.config.DEMO_QUOTE_CAPITAL)
        except Exception as e:  # noqa: BLE001
            self.logger.error("Error recalculating statistics: %s", e)
        await self.persistence.async_save_position(None)
        self.current_position = None

        learn_key = str(getattr(evidence, "event_id", None) or "")
        if not learnable:
            self.logger.info(
                "Brain NOT updated (%s) | %s", closed_position.symbol, reason
            )
        elif post_mortem_result is None:
            self.logger.info(
                "Brain NOT updated: no confirmed post-mortem | %s %s",
                closed_position.symbol, reason,
            )
        elif learn_key in self._learned_exit_events:
            self.logger.warning(
                "%s (%s) | %s", LESSON_SKIPPED_ALREADY_LEARNED, learn_key,
                closed_position.symbol,
            )
        else:
            self._learned_exit_events.add(learn_key)
            while len(self._learned_exit_events) > MAX_LEARNED_EXIT_EVENTS:
                self._learned_exit_events.pop()
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
                    evidence=evidence,
                )
                if self.dashboard_state:
                    await self.dashboard_state.mark_brain_rebuild_completed("Brain state rebuilt from closed trade")
            except Exception as e:  # noqa: BLE001
                self._learned_exit_events.discard(learn_key)
                self.logger.error("Error updating trading brain: %s", e)
                if self.dashboard_state:
                    await self.dashboard_state.mark_brain_rebuild_failed("Brain rebuild failed after trade close")

    async def process_analysis(
        self, analysis_result: dict, symbol: str, market_price: float | None = None
    ) -> TradeDecision | None:

        try:
            analysis = analysis_result.get("analysis") or {}
            current_price = self._conditions.extract_price(analysis_result)  # type: ignore

            if not analysis:
                self.logger.warning("No parsed analysis to process")
                return None

            if current_price is None or not math.isfinite(current_price) or current_price <= 0:
                self.logger.error("Invalid current_price extracted, cannot process trade")
                return None

            signal, confidence, stop_loss, take_profit, position_size, reasoning = \
                self.extractor.extract_trading_info(analysis)  # type: ignore

            self.logger.info("Extracted Signal: %s, Confidence: %s", signal, confidence)

            if not self.extractor.validate_signal(signal):  # type: ignore
                self.logger.warning("Invalid signal: %s", signal)
                return None

            market_conditions = self._conditions.extract_market_conditions(analysis_result)  # type: ignore

            confluence_factors = self._conditions.extract_confluence_factors(analysis_result)  # type: ignore

            if self.current_position:
                return await self._handle_existing_position(
                    signal, confidence, stop_loss, take_profit,
                    current_price, symbol, reasoning, market_conditions,
                    market_price,
                )

            if signal in ("BUY", "SELL", "LONG", "SHORT"):
                return await self._open_new_position(
                    signal, confidence, stop_loss, take_profit,
                    position_size, current_price, symbol, reasoning,
                    market_conditions, confluence_factors
                )

            if reasoning:
                self.logger.info("No action taken. Signal: %s. Reasoning: %s", signal, reasoning)
            else:
                self.logger.info("No action taken. Signal: %s", signal)
            return None

        except Exception as e:  # noqa: BLE001
            self.logger.error("Error processing analysis: %s", e)
            return None


    def _get_last_closed_position_info(self) -> str | None:

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
        except Exception:  # noqa: BLE001
            return None

    def _last_action_outcome_lines(self) -> list[str]:

        ledger = self.position_intents()
        position = self.current_position
        if position is not None:
            intent = ledger.latest_for_position(position_identity(position))
        else:
            recent = ledger.recent(1)
            intent = recent[0] if recent else None
        if intent is None:
            return []
        stamp = (intent.created_at or intent.updated_at or "")[:16].replace("T", " ")
        evidence = f" (evidence: {intent.evidence})" if intent.evidence else ""
        lines = [
            "",
            "### Last Action Outcome",
            f"- {intent.action} {intent.symbol} at {stamp} UTC — state: {intent.state}{evidence}",
        ]
        if intent.detail:
            lines.append(f"- outcome detail: {intent.detail}")
        payload = intent.payload or {}
        proposed_sl = payload.get("requested_stop_loss")
        if isinstance(proposed_sl, (int, float)):
            in_force = payload.get("old_stop_loss")
            in_force_text = f"${in_force:,.2f}" if isinstance(in_force, (int, float)) else "unchanged"
            lines.append(
                f"- proposed stop loss ${proposed_sl:,.2f} — stop loss in force stays {in_force_text}"
            )
        if intent.state == INTENT_CONFIRMED:
            lines.append("- Executor evidence confirms this command. This outcome supersedes earlier missing-confirmation alerts.")
        elif intent.state == INTENT_REFUSED:
            lines.append("- This command was refused; do not treat its requested levels as executed.")
        else:
            lines.append("- Execution is not confirmed. Local SL/TP values are a mirror, not proof of exchange protection.")
        return lines

    def get_position_context(self, current_price: float | None = None) -> str:

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
            lines.extend(self._last_action_outcome_lines())
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
            f"- Entry Fee: {self._commission_text(pos.entry_fee)}",
            f"- Duration: {hours:.1f} hours",
            f"- Confidence: {pos.confidence}",
        ])
        if current_price and current_price > 0:
            pnl_pct = pos.calculate_pnl(current_price)
            pnl_quote = (current_price - pos.entry_price) * pos.size if pos.direction == "LONG" else (pos.entry_price - current_price) * pos.size
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

        lines.extend(self._last_action_outcome_lines())
        return "\n".join(lines)
