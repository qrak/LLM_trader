"""Trading strategy that wraps analysis with position management."""

import asyncio
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from src.logger.logger import Logger
from src.utils.timeframe_validator import TimeframeValidator

from .brain import TradingBrainService
from .data_models import MarketConditions, Position, TradeDecision
from .executor_reconciliation import ExecutorReconciliationMixin
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

        self.current_position: Position | None = self.persistence.load_position()

        self._last_position_update_time: datetime | None = None

        self._tf_minutes: int = TimeframeValidator.to_minutes(config.TIMEFRAME) if config else 240

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
        """Check if current position hit stop loss or take profit."""
        if not await self._update_live_metrics(current_price):
            return None

        if self.current_position.is_stop_hit(current_price):  # type: ignore
            return await self._close_on_exit("stop_loss", current_price)

        if self.current_position.is_target_hit(current_price):  # type: ignore
            return await self._close_on_exit("take_profit", current_price)

        return None

    async def check_stop_loss(self, current_price: float) -> str | None:
        """Check only the configured stop loss exit."""
        if not await self._update_live_metrics(current_price):
            return None

        if self.current_position.is_stop_hit(current_price):  # type: ignore
            return await self._close_on_exit("stop_loss", current_price)

        return None

    async def check_take_profit(self, current_price: float) -> str | None:
        """Check only the configured take profit exit."""
        if not await self._update_live_metrics(current_price):
            return None

        if self.current_position.is_target_hit(current_price):  # type: ignore
            return await self._close_on_exit("take_profit", current_price)

        return None

    async def _close_on_exit(self, reason: str, current_price: float) -> str:
        """Close the open position on a hit exit and return the close reason."""
        conditions = self._conditions.build_conditions_from_position(self.current_position)  # type: ignore
        await self.close_position(reason, current_price, conditions)
        return reason

    async def close_position(
        self,
        reason: str,
        current_price: float,
        market_conditions: MarketConditions,
    ) -> None:
        """Close the current position and update trading brain."""
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

        if self.post_mortem_service and entry_decision:
            try:
                await self.post_mortem_service.analyze_closed_trade(
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

        try:
            self.statistics_service.recalculate(self.config.DEMO_QUOTE_CAPITAL)
        except Exception as e:  # noqa: BLE001
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
        except Exception as e:  # noqa: BLE001
            self.logger.error("Error updating trading brain: %s", e)
            if self.dashboard_state:
                await self.dashboard_state.mark_brain_rebuild_failed("Brain rebuild failed after trade close")

    async def process_analysis(self, analysis_result: dict, symbol: str) -> TradeDecision | None:
        """Process AI analysis result and execute trading decision.
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
                    current_price, symbol, reasoning, market_conditions
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

        return "\n".join(lines)
