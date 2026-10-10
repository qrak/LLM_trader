"""Trading strategy that wraps analysis with position management."""

import asyncio
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from src.logger.logger import Logger
from src.utils.timeframe_validator import TimeframeValidator

from .brain import TradingBrainService
from .data_models import LocalExitRequest, MarketConditions, Position, TradeDecision
from .executor_reconciliation import (
    INTENT_ACTION_CLOSE,
    INTENT_PENDING,
    INTENT_UNKNOWN,
    LESSON_SKIPPED_ALREADY_LEARNED,
    LESSON_SKIPPED_NO_EVIDENCE,
    LESSON_SKIPPED_NO_POST_MORTEM,
    MAX_LEARNED_EXIT_EVENTS,
    ExecutorReconciliationMixin,
    LocalPositionReconciliation,
    exit_evidence_is_learnable,
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
        """Initialize the trading strategy with DI pattern."""
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
        self._last_booked_exit: dict[str, Any] | None = None
        self._last_reconciliation: LocalPositionReconciliation | None = None
        self._learned_exit_events: set[str] = set()

        self._tf_minutes: int = TimeframeValidator.to_minutes(config.TIMEFRAME) if config else 240

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
        """Inject dashboard state after dashboard server construction."""
        self.dashboard_state = dashboard_state

    async def _record_trade_decision(self, decision: TradeDecision) -> int:
        """Persist a decision, refresh short-term memory, return SQLite row ID."""
        row_id = await self.persistence.async_save_trade_decision(decision)
        self.memory_service.add_decision(decision)
        return row_id

    async def _update_live_metrics(self, current_price: float) -> bool:
        """Update live position metrics before evaluating an exit."""
        if not self.current_position:
            return False
        self.current_position.update_metrics(current_price)
        await self.persistence.async_save_position(self.current_position)
        return True

    async def check_position(self, current_price: float) -> str | None:
        """Check if current position hit stop loss or take profit.

        Wave 3: a hit does NOT close anything locally. It registers a CLOSE intent
        (see ``_close_on_exit``) and this returns None — ``take_local_exit_request``
        carries the unbooked condition to the caller.
        """
        if not await self._update_live_metrics(current_price):
            return None

        if self.current_position.is_stop_hit(current_price):  # type: ignore
            return await self._close_on_exit("stop_loss", current_price)

        if self.current_position.is_target_hit(current_price):  # type: ignore
            return await self._close_on_exit("take_profit", current_price)

        return None

    async def check_stop_loss(self, current_price: float) -> str | None:
        """Check only the configured stop loss exit (never books the close locally)."""
        if not await self._update_live_metrics(current_price):
            return None

        if self.current_position.is_stop_hit(current_price):  # type: ignore
            return await self._close_on_exit("stop_loss", current_price)

        return None

    async def check_take_profit(self, current_price: float) -> str | None:
        """Check only the configured take profit exit (never books the close locally)."""
        if not await self._update_live_metrics(current_price):
            return None

        if self.current_position.is_target_hit(current_price):  # type: ignore
            return await self._close_on_exit("take_profit", current_price)

        return None

    async def _close_on_exit(self, reason: str, current_price: float) -> str | None:
        """Local SL/TP condition hit: REQUEST a close — never book one locally.

        Wave 3: a ticker price that trips the bracket is NOT a fill. Booking a CLOSE
        here (at the ticker price) is the exact fabrication class the 2026-09-21
        incident covers, so this method books nothing, touches no statistics, never
        clears the local position and never rewrites the exit price. What it does:

        * executor integration live: registers a pending CLOSE intent and queues a
          CLOSE recommendation for the app to forward. The local position stays until
          real fill evidence (price AND quantity) arrives from the executor;
        * integration off: no command can be sent, so no exit may be invented. The
          intent is recorded as ``unknown``, the position stays and a divergence alarm
          is raised for the operator — no silent reset, no zeroing.

        Returns None on purpose: the caller's contract is "a close was booked", and one
        is never booked from this path.
        """
        position = self.current_position
        if position is None:
            return None

        position_key = position_identity(position)
        symbol = position.symbol
        ledger = self.position_intents()
        intent_key = self.intent_identity(INTENT_ACTION_CLOSE, symbol, position_id=position_key)

        if self._executor_query_available():
            existing = ledger.get(intent_key)
            order_id = (
                existing.order_id
                if existing is not None and existing.state == INTENT_PENDING and existing.order_id
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
        self.logger.critical(
            "LOCAL EXIT CONDITION %s hit @ %s — NOTHING booked locally (state=%s, position kept)",
            reason, current_price, state,
        )
        self._unconfirmed_intent_alert = note
        return None

    def take_local_exit_request(self) -> "LocalExitRequest | None":
        """One-shot read of the last unbooked local exit condition (or None)."""
        request = getattr(self, "_local_exit_request", None)
        self._local_exit_request = None
        return request

    def take_pending_local_close_decision(self) -> TradeDecision | None:
        """One-shot read of the CLOSE recommendation a local exit condition queued.

        The monitor detects the bracket but has no forward path; the app drains this
        recommendation and hands it to the executor. Nothing local is booked either
        way — only the executor's fill evidence closes the trade.
        """
        decision = getattr(self, "_pending_local_close_decision", None)
        self._pending_local_close_decision = None
        return decision

    @staticmethod
    def _commission_text(fee: float | None, fee_source: str | None = None) -> str:
        """Operator-facing commission text that never invents a number.

        A commission may only be stated when real fill/order fee data provided it. With
        no such data the text says UNKNOWN explicitly: ``0.0`` and the configured
        0.075% rate are both claims the bot cannot back with evidence.
        """
        if fee is None:
            return "Fee: unknown (no fill/order fee data — no rate was assumed)"
        origin = f" from {fee_source}" if fee_source else ""
        return f"Fee: ${fee:.4f}{origin}"

    @staticmethod
    def _booked_quantity(
        position: Position, filled_quantity: float | None
    ) -> tuple[float, str]:
        """The amount a CLOSE row may carry, plus a divergence note for the operator.

        Wave 5: when the executor's evidence carries the ACTUAL filled amount, that is
        what gets booked — the local size can be a corrupted restore (the 2026-09-21
        incident left ``size=0.045`` against a real 0.00554 fill). A mismatch larger
        than 1% of the local size is NOT silently fixed: it is logged at critical level
        and stated on the row itself.

        Without a filled amount (an older caller, no executor fields) the local size is
        kept exactly as before — the caller in that case has no evidence at all, so the
        close never reaches the learning path (see the guard in ``close_position``).
        """
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
        """Close the current position and update trading brain.

        ``current_price`` is the ACTUAL exit price (a validated fill/order average),
        never a ticker read and never the entry price. ``exit_fee`` may only carry real
        fill/order fee data (e.g. the executor exit journal's ``fees``); it defaults to
        ``None`` = UNKNOWN, in which case the booked row says so and no rate (the
        configured ``TRANSACTION_FEE_PERCENT``, 0.075%) is ever substituted for it.

        ``filled_quantity`` is the ACTUAL filled base amount from the same evidence; when
        present it is booked instead of the local size (see :meth:`_booked_quantity`).

        ``evidence`` (:class:`~src.trading.executor_reconciliation.ExitEvidence`) is the
        PROOF this close rests on. Wave-5 brain guard: without usable evidence (a real
        price AND amount from a confirming source, with a stable event id) the
        post-mortem is not even asked and the trading brain is NOT updated — instead the
        skip is logged explicitly (``LESSON_SKIPPED_NO_EVIDENCE``). The same holds when
        the post-mortem runs but produces no validated analysis: a close never turns into
        a lesson the bot cannot back with evidence (the 2026-09-21 false lesson).
        """
        if not self.current_position:
            return

        closed_position = self.current_position
        pnl = closed_position.calculate_pnl(current_price)

        fee_text = self._commission_text(exit_fee, fee_source)
        learnable, refusal_reason = exit_evidence_is_learnable(evidence)
        if filled_quantity is None:
            filled_quantity = getattr(evidence, "quantity", None)
        booked_quantity, quantity_note = self._booked_quantity(closed_position, filled_quantity)
        if quantity_note:
            self.logger.critical(
                "QUANTITY DIVERGENCE for %s on close (%s): booked the FILL amount",
                closed_position.symbol, reason,
            )

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
            reasoning=(
                f"Position closed: {reason}. P&L: {pnl:+.2f}%. {fee_text}{quantity_note}"
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
                "%s (%s) | %s %s", LESSON_SKIPPED_NO_EVIDENCE, refusal_reason,
                closed_position.symbol, reason,
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
        """Process AI analysis result and execute trading decision.

        market_price is the live price the prompt showed the model (the position
        context advertises it). The SL tightening gate evaluates with it, so the
        prompt and the gate cannot disagree about the same cycle.

        Returns:
            TradeDecision if action taken, else None
        """
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
        except Exception:  # noqa: BLE001
            return None

    def _last_action_outcome_lines(self) -> list[str]:
        """Render the newest recorded command outcome — what the bot did, or refused to do.

        One analysis per cycle means the next prompt is the only place where a refused
        UPDATE/CLOSE/ENTRY can still change a decision, so the outcome travels with the
        position context instead of living only in the logs.
        """
        intents = self.position_intents().recent(1)
        if not intents:
            return []
        intent = intents[0]
        stamp = (intent.updated_at or intent.created_at or "")[:16].replace("T", " ")
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
        lines.append(
            "- Only state 'confirmed' means the command reached the exchange: 'refused' means "
            "it did NOT run, so the levels above are what the exchange still holds."
        )
        return lines

    def get_position_context(self, current_price: float | None = None) -> str:
        """Get formatted context about current position for prompts.
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
