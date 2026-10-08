"""Main entry point for the Crypto Trading Bot application.

This module defines the `CryptoTradingBot` class, which orchestrates the interaction
between various components like the market analyzer, trading strategy, and external APIs.
"""
import asyncio
import inspect
import io
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from src.logger.logger import Logger
from src.managers.persistence_manager import PersistenceManager
from src.trading import (
    ExitMonitor,
    PositionStatusMonitor,
    TradingBrainService,
    TradingMemoryService,
    TradingStatisticsService,
)
from src.trading.data_models import TradeDecision
from src.trading.executor_reconciliation import (
    NON_BOOKED_INTENT_STATES,
    RECONCILE_UNVERIFIED,
    LocalPositionReconciliation,
)
from src.utils.decorators import retry_async
from src.utils.timeframe_validator import TimeframeValidator

if TYPE_CHECKING:
    from src.managers.model_manager import ModelManager


POSITION_UPDATE_INTERVAL = 3600
SLEEP_CHUNK_SIZE = 1.0
CANDLE_BUFFER_SECONDS = 2
ERROR_WAIT_SHORT = 60
ERROR_WAIT_LONG = 300

ACTIONABLE_RECOMMENDATIONS = frozenset(
    {"BUY", "SELL", "LONG", "SHORT", "UPDATE", "CLOSE", "CLOSE_LONG", "CLOSE_SHORT"}
)


_MARKET_KNOWLEDGE_FALLBACK_MSG = "continuing with cached/partial market knowledge"


@dataclass
class BotServices:
    """Runtime services required by CryptoTradingBot."""

    logger: Logger
    config: Any
    shutdown_manager: Any | None
    exchange_manager: Any
    market_analyzer: Any
    trading_strategy: Any
    discord_notifier: Any
    keyboard_handler: Any
    rag_engine: Any
    coingecko_api: Any
    market_api: Any
    alternative_me_api: Any
    http_session: Any
    persistence: PersistenceManager
    model_manager: "ModelManager"
    brain_service: TradingBrainService
    statistics_service: TradingStatisticsService
    memory_service: TradingMemoryService
    exit_monitor: ExitMonitor
    sentiment_analyst: Any = None
    executor_handler: Any = None
    ev_formatter: Any = None
    dashboard_state: Any = None
    discord_task: asyncio.Task | None = None
    position_monitor_factory: Callable[[Any], PositionStatusMonitor] | None = None
    force_analysis_event: asyncio.Event | None = None


class CryptoTradingBot:
    """Automated crypto trading bot - TRADING MODE ONLY."""

    @staticmethod
    def _format_utc_and_local(dt: datetime, local_tz=None) -> str:
        """Format a timestamp with both UTC and local time for operator-facing logs."""
        utc_dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
        local_dt = utc_dt.astimezone(local_tz)
        local_zone = local_dt.tzname() or "local"
        return (
            f"{utc_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC "
            f"/ {local_dt.strftime('%Y-%m-%d %H:%M:%S')} {local_zone}"
        )

    def __init__(self, services: BotServices):
        """Initialize bot with runtime services assembled by the composition root."""
        self.logger = services.logger
        self.config = services.config
        self.shutdown_manager = services.shutdown_manager

        self.exchange_manager = services.exchange_manager
        self.market_analyzer = services.market_analyzer
        self.trading_strategy = services.trading_strategy
        self.discord_notifier = services.discord_notifier
        self.keyboard_handler = services.keyboard_handler
        self.rag_engine = services.rag_engine

        self.coingecko_api = services.coingecko_api
        self.market_api = services.market_api
        self.alternative_me_api = services.alternative_me_api
        self.http_session = services.http_session

        self.persistence = services.persistence
        self.model_manager = services.model_manager
        self.brain_service = services.brain_service
        self.statistics_service = services.statistics_service
        self.memory_service = services.memory_service
        self.exit_monitor = services.exit_monitor
        self.executor_handler = services.executor_handler
        self.sentiment_analyst = services.sentiment_analyst
        self.ev_formatter = services.ev_formatter
        self.dashboard_state = services.dashboard_state

        self.tasks = []
        self.running = False
        self._active_tasks = set()
        self._force_analysis = services.force_analysis_event or asyncio.Event()
        self._discord_task = services.discord_task

        self.current_symbol: str | None = None
        self.current_timeframe: str | None = None
        self._reddit_sentiment_label = "NEUTRAL"
        self.position_monitor = (
            services.position_monitor_factory(self)
            if services.position_monitor_factory is not None
            else None
        )

    @property
    def active_tasks(self) -> set[asyncio.Task]:
        """Return bot-managed background tasks for composition-root wiring."""
        return self._active_tasks

    def _require_position_monitor(self) -> PositionStatusMonitor:
        if self.position_monitor is None:
            raise RuntimeError("Position monitor dependency is not configured")
        return self.position_monitor

    async def initialize(self):
        """Initialize all components."""
        if self.shutdown_manager:
            self.shutdown_manager.register_shutdown_callback(self.shutdown)

            if self.keyboard_handler:
                self.shutdown_manager.register_shutdown_callback(self.keyboard_handler.stop_listening)

            if self.model_manager:
                self.shutdown_manager.register_shutdown_callback(self.model_manager.close)

            if self.market_analyzer:
                self.shutdown_manager.register_shutdown_callback(self.market_analyzer.close)

            if self.rag_engine:
                self.shutdown_manager.register_shutdown_callback(self.rag_engine.close)

            if self.exchange_manager:
                self.shutdown_manager.register_shutdown_callback(self.exchange_manager.shutdown)

            if self.http_session:
                self.shutdown_manager.register_shutdown_callback(self.http_session.close)

            for client in [self.alternative_me_api, self.coingecko_api, self.market_api]:
                if client:
                    try:
                        self.shutdown_manager.register_shutdown_callback(client.close)
                    except AttributeError:
                        pass

        self.keyboard_handler.register_command("a", self._force_analysis_now, "Force immediate analysis")
        self.keyboard_handler.register_command("h", self._show_help, "Show available keyboard commands")
        self.keyboard_handler.register_command("q", self._request_shutdown, "Quit the application")
        self.keyboard_handler.register_command(
            "R", self._request_reload, "Reload (in-place restart)"
        )

        keyboard_task = asyncio.create_task(
            self.keyboard_handler.start_listening(),
            name="Keyboard-Handler"
        )
        self._active_tasks.add(keyboard_task)
        keyboard_task.add_done_callback(self._active_tasks.discard)
        self.tasks.append(keyboard_task)

    async def shutdown(self):
        """Callback for graceful shutdown."""
        self.logger.info("Signaling trading loops to stop...")
        self.running = False

        pending_tasks = list(self._active_tasks)
        if pending_tasks:
            self.logger.info("Cancelling %s bot-specific tasks...", len(pending_tasks))
            for task in pending_tasks:
                if not task.done():
                    task.cancel()
            try:
                await asyncio.wait(pending_tasks, timeout=3.0)
            except asyncio.TimeoutError:
                self.logger.warning("Bot tasks shutdown timed out")

        if self.discord_notifier:
            try:
                await self.discord_notifier.shutdown()
            except Exception as e:  # noqa: BLE001
                self.logger.warning("Error shutting down Discord notifier: %s", e)

        if self._discord_task and not self._discord_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._discord_task), timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("Discord task did not exit after bot shutdown; cancelling it")
                self._discord_task.cancel()
                try:
                    await self._discord_task
                except asyncio.CancelledError:
                    pass

        self.logger.info("Bot shutdown signaling complete.")


    async def run(self, symbol: str, timeframe: str | None = None):
        """Run the trading bot in continuous mode."""
        self.current_symbol = symbol
        requested_timeframe = timeframe or self.config.TIMEFRAME
        try:
            self.current_timeframe = TimeframeValidator.validate_and_normalize(requested_timeframe)
            self.exit_monitor.timeframe = self.current_timeframe
            self.exit_monitor.validate()
        except ValueError as e:
            self.logger.error("Invalid timeframe or exit monitoring config '%s': %s", requested_timeframe, e)
            return

        exchange, exchange_id = await self.exchange_manager.find_symbol_exchange(symbol)
        if not exchange:
            self.logger.error("Symbol %s not found on any configured exchange", symbol)
            return

        self.current_exchange = exchange
        self.logger.info("Starting trading for %s on %s", symbol, exchange_id)
        self.logger.info("Timeframe: %s", self.current_timeframe)

        self.market_analyzer.initialize_for_symbol(
            symbol=symbol,
            exchange=exchange,
            timeframe=self.current_timeframe
        )

        self.running = True
        check_count = 0

        if self.trading_strategy.current_position:
            position = self.trading_strategy.current_position
            self.logger.info("Existing position: %s @ $%s", position.direction, f"{position.entry_price:,.2f}")
            await self._require_position_monitor().start()
        else:
            self.logger.info("No existing position")

        try:
            await self._fetch_current_ticker()  # type: ignore
        except Exception:  # noqa: BLE001
            self.logger.warning("Initial ticker fetch failed, continuing without price")

        last_analysis_time = self.persistence.get_last_analysis_time()
        if last_analysis_time:
            # The single "Resuming from last analysis ..." line is logged by the wait helper.
            await self._wait_until_next_timeframe_after(last_analysis_time)
        else:
            self.logger.info("Crypto Trading Bot ready")

        is_regular_run = True

        while self.running:
            try:
                check_count += 1
                await self._execute_trading_check(check_count, force_news_update=is_regular_run, is_candle_close=is_regular_run)

                if not self.running:
                    break

                was_forced_wait = await self._wait_for_next_timeframe()
                is_regular_run = not was_forced_wait

            except asyncio.CancelledError:
                self.logger.info("Trading cancelled")
                self.running = False
                break
            except Exception as e:  # noqa: BLE001
                self.logger.error("Error in trading loop: %s", e)
                await self._interruptible_sleep(ERROR_WAIT_SHORT)

    async def _execute_trading_check(self, check_count: int, force_news_update: bool = True, is_candle_close: bool = True):
        """Execute a single trading check iteration."""
        self._log_check_header(check_count)

        current_ticker, current_price = await self._fetch_ticker_data()
        await self._check_position_status(current_price, is_candle_close=is_candle_close)
        await self._forward_local_exit_request()
        await self._execute_market_knowledge_update(force_news_update)

        reconcile_outcome = await self._reconcile_position_state(source="pre_analysis")
        snapshot_before = self._position_snapshot_token()
        protection_before = self._protection_snapshot()
        protection_version_before = await self._protection_version()

        self.logger.info("Running market analysis...")
        context_data = await self._build_analysis_context(current_price, current_ticker)
        result = await self.market_analyzer.analyze_market(**context_data)

        if "error" in result:
            self.logger.error("Analysis failed: %s", result["error"])
            return

        snapshot_after_analysis = self._position_snapshot_token()
        stale_recommendation = (
            snapshot_before is not None
            and snapshot_after_analysis is not None
            and snapshot_before != snapshot_after_analysis
        )

        result["_social_sentiment_reddit"] = self._reddit_sentiment_label
        demo_capital = float(self.config.DEMO_QUOTE_CAPITAL)
        current_capital = self.statistics_service.get_current_capital(demo_capital)
        result["_portfolio_pnl_pct"] = ((current_capital - demo_capital) / demo_capital * 100) if demo_capital > 0 else 0.0

        await self.persistence.async_save_last_analysis_time()
        if stale_recommendation:
            action = str((result.get("analysis") or {}).get("signal") or "UNKNOWN").upper()
            self.logger.warning(
                "Discarding stale %s recommendation: the position changed during the "
                "analysis (%s -> %s) — not forwarding it to the executor.",
                action, snapshot_before, snapshot_after_analysis,
            )
            decision = None
        else:
            decision = await self.trading_strategy.process_analysis(
                result, self.current_symbol, market_price=current_price
            )

        protection_after = self._protection_snapshot()
        protection_guard_note = self._protection_guard_note(
            decision, protection_before, protection_after
        )
        if decision is not None and protection_guard_note:
            self.logger.warning(
                "Discarding %s recommendation: %s", decision.action, protection_guard_note
            )
            decision = None

        protection_version_after = await self._protection_version()
        protection_version_note = self._protection_version_guard(
            decision, protection_version_before, protection_version_after
        )
        if decision is not None and protection_version_note:
            self.logger.warning(
                "Discarding stale %s recommendation: %s", decision.action, protection_version_note
            )
            decision = None

        if decision:
            await self._handle_new_position(decision, current_price)
        else:
            self.logger.info("No trading action taken")
        await self._report_executor_side_exit()
        await self._report_state_divergence()

        intent_state: str | None = None

        if decision is not None and decision.action == "HOLD" and result.get("analysis"):
            self._patch_rejected_signal_in_response(result, decision)

        analysis = result.get("analysis")
        if self.executor_handler is not None and analysis and decision is not None:
            forward_delivered = await self.executor_handler.handle(analysis, decision, self.current_symbol)
            if decision.action in ("BUY", "SELL"):
                await self.trading_strategy.rollback_blocked_entry(
                    self.current_symbol, forward_delivered,
                    order_id=decision.order_id,
                )
            intent_state = await self._resolve_position_intents(decision, forward_delivered)

        execution_note = self._build_execution_note(
            result, decision, stale_recommendation, reconcile_outcome,
            protection_guard_note=protection_guard_note,
            protection_version_note=protection_version_note,
            intent_state=intent_state,
        )
        await self._send_discord_notification(result, execution_note=execution_note)
        await self._report_unconfirmed_intent()
        await self._report_rejected_intent()
        await self._save_analysis_data(result)

    async def _forward_local_exit_request(self) -> str | None:
        """Forward a CLOSE the local SL/TP monitor requested to the executor.

        The monitor can detect that the bracket was reached but it cannot close the
        trade on the exchange, and a ticker price is not a fill — so the only thing
        this does is hand the already-recorded intent to the executor. With no
        executor handler (or no queued recommendation) nothing is sent: the intent
        stays pending/unknown, the local position stays, and never a local close at
        the ticker price.

        Returns the resulting intent state, or None when there was nothing to send.
        """
        if self.executor_handler is None:
            return None
        taker = getattr(self.trading_strategy, "take_pending_local_close_decision", None)
        if not callable(taker):
            return None
        try:
            decision = taker()
        except Exception as e:  # noqa: BLE001
            self.logger.warning("Failed to read the pending local exit recommendation: %s", e)
            return None
        if decision is None:
            return None

        try:
            delivered = await self.executor_handler.handle(
                {"signal": "CLOSE"}, decision, self.current_symbol
            )
        except Exception as e:  # noqa: BLE001
            self.logger.error("Forwarding the local exit recommendation failed: %s", e)
            delivered = False
        self.logger.warning(
            "Local exit recommendation forwarded for %s (order_id=%s, delivered=%s) — the "
            "local position is NOT closed until the executor's fill evidence arrives.",
            self.current_symbol, getattr(decision, "order_id", None), delivered,
        )
        return await self._resolve_position_intents(decision, delivered)

    async def _reconcile_position_state(self, source: str) -> LocalPositionReconciliation | None:
        """Run the strategy's public reconciliation hook (no-op when not wired).

        The hook books a validated exchange-side exit exactly once; an unwired or
        non-conforming strategy is left untouched (a test double must not decide the
        trading state).
        """
        hook = getattr(self.trading_strategy, "reconcile_local_position", None)
        if not callable(hook):
            return None
        try:
            pending = hook(source=source)
            if not inspect.isawaitable(pending):
                return None
            outcome = await pending
        except Exception as e:  # noqa: BLE001
            self.logger.warning("Position reconciliation (%s) failed: %s", source, e)
            return None
        return outcome if isinstance(outcome, LocalPositionReconciliation) else None

    async def _resolve_position_intents(self, decision: Any, delivered: bool) -> str | None:
        """Hand the forwarded command's intent to the strategy for evidence-based closure.

        Only the strategy's executor evidence may confirm it; a missing hook (test double,
        no intent layer) leaves the state unknown here instead of inventing one.
        """
        resolver = getattr(self.trading_strategy, "resolve_position_intents_after_forward", None)
        state: Any = None
        if callable(resolver):
            try:
                pending = resolver(
                    order_id=getattr(decision, "order_id", None),
                    delivered=delivered,
                    symbol=self.current_symbol,
                )
                if inspect.isawaitable(pending):
                    state = await pending
            except Exception as e:  # noqa: BLE001
                self.logger.warning("Failed to resolve the forwarded position intent: %s", e)
        elif callable(getattr(self.trading_strategy, "position_intent_state", None)):
            state = self.trading_strategy.position_intent_state(
                order_id=getattr(decision, "order_id", None)
            )
        return state if isinstance(state, str) else None

    async def _report_unconfirmed_intent(self) -> None:
        """Alert the operator about a command whose execution the executor never confirmed.

        Wave 2: an entry/UPDATE/CLOSE the bot sent is an INTENT. Silence is not a fill:
        the local state stays pending/unknown, the position is left alone, and this is
        the only place that turns that into a visible alert (never a silent close).
        """
        alert = None
        hook = getattr(self.trading_strategy, "take_unconfirmed_intent_alert", None)
        if callable(hook):
            try:
                alert = hook()
            except Exception as e:  # noqa: BLE001
                self.logger.warning("Failed to read the unconfirmed-intent alert: %s", e)
        if not isinstance(alert, str) or not alert:
            return
        self.logger.critical("UNCONFIRMED POSITION INTENT: %s", alert)
        if self.discord_notifier is None:
            return
        try:
            await self.discord_notifier.send_message(
                "⚠️ **Unconfirmed command** — it was sent to the executor, but there is no "
                f"proof it was executed: {alert}. I am not changing the local state and not "
                "closing the position silently — check the exchange.",
                channel_id=self.config.MAIN_CHANNEL_ID,
            )
        except Exception as e:  # noqa: BLE001
            self.logger.warning("Failed to send the unconfirmed-intent alert: %s", e)

    async def _report_rejected_intent(self) -> None:
        """Alert the operator about a command the bot's own policy refused to send.

        A refusal is not a missing confirmation: nothing was sent, so the exchange
        never saw the command and the model's request silently died inside the bot.
        The next analysis prompt carries it too (see the intent ledger).
        """
        alert = None
        hook = getattr(self.trading_strategy, "take_rejected_intent_alert", None)
        if callable(hook):
            try:
                alert = hook()
            except Exception as e:  # noqa: BLE001
                self.logger.warning("Failed to read the rejected-intent alert: %s", e)
        if not isinstance(alert, str) or not alert:
            return
        self.logger.warning("REFUSED POSITION INTENT: %s", alert)
        if self.discord_notifier is None:
            return
        try:
            await self.discord_notifier.send_message(
                "🚫 **Command refused by the bot** — it never reached the executor and "
                f"nothing changed on the exchange: {alert} The next analysis sees this "
                "refusal, so the model will not keep assuming the order went through.",
                channel_id=self.config.MAIN_CHANNEL_ID,
            )
        except Exception as e:  # noqa: BLE001
            self.logger.warning("Failed to send the rejected-intent alert: %s", e)

    def _position_snapshot_token(self) -> str | None:
        """Current position revision token, or None when the strategy has no contract."""
        token_fn = getattr(self.trading_strategy, "position_snapshot_token", None)
        if not callable(token_fn):
            return None
        token = token_fn()
        return token if isinstance(token, str) else None

    def _protection_snapshot(self) -> tuple[Any, Any] | None:
        """(stop_loss, take_profit) of the current position, or None without a contract."""
        snapshot_fn = getattr(self.trading_strategy, "protection_snapshot", None)
        if not callable(snapshot_fn):
            return None
        snapshot = snapshot_fn()
        return tuple(snapshot) if isinstance(snapshot, tuple) and len(snapshot) == 2 else None

    async def _protection_version(self) -> str | None:
        """Protection revision the executor reports, or None without that contract.

        Wave-5 requirement (3). The strategy's ``executor_protection_version`` asks the
        executor's ``/position`` for its ``protection_version`` (monotonic counter of
        protection changes; ``0`` = none observed, returned verbatim). An older executor
        without the field gives None, in which case the guard below stays inert (it
        never guesses a version). A strategy double / older strategy without the method
        also yields None — the guard must not fail a trading cycle over a missing hook.
        """
        version_fn = getattr(self.trading_strategy, "executor_protection_version", None)
        if not callable(version_fn):
            return None
        try:
            version = await version_fn()
        except Exception as e:  # noqa: BLE001
            self.logger.warning("Protection version query failed: %s", e)
            return None
        return str(version) if version is not None else None

    def _protection_version_guard(
        self,
        decision: TradeDecision | None,
        version_before: str | None,
        version_after: str | None,
    ) -> str | None:
        """Refuse an UPDATE computed against a protection set that was replaced.

        Only meaningful when BOTH reads returned a revision (the executor's additive
        ``protection_version``): a different value means the protection was replaced while
        the model was working, so the UPDATE describes a state that no longer exists. A
        missing revision is not treated as "unchanged" — it is "unversioned", and the
        wave-1 loosening guard remains the substitute (see ``_protection_guard_note``).
        """
        if decision is None or str(getattr(decision, "action", "") or "").upper() != "UPDATE":
            return None
        if not version_before or not version_after:
            return None
        if version_before == version_after:
            return None
        return (
            f"the protection version changed during the analysis ({version_before} -> "
            f"{version_after}) — the UPDATE describes outdated protection"
        )

    def _protection_guard_note(
        self,
        decision: TradeDecision | None,
        protection_before: tuple[Any, Any] | None,
        protection_after: tuple[Any, Any] | None,
    ) -> str | None:
        """Refuse an UPDATE whose protection snapshot is no longer current, or None.

        Only consulted when the protection actually changed during the analysis (this
        token pairs SL/TP and cannot version a REPLACED set; the executor's
        ``protection_version`` is the versioned guard — see ``_protection_version_guard``,
        which runs on top of this one). The strategy's monotone guard then decides:
        tightening passes, loosening is refused. Without that guard the safe answer is to
        refuse, not to forward.
        """
        if decision is None or str(getattr(decision, "action", "")).upper() != "UPDATE":
            return None
        if protection_before is None or protection_after is None:
            return None
        if protection_before == protection_after:
            return None
        guard = getattr(self.trading_strategy, "protection_loosening_reason", None)
        if not callable(guard):
            return (
                f"protection changed during the analysis ({protection_before} -> "
                f"{protection_after}) and the strategy exposes no protection guard"
            )
        reason = guard(decision)
        return reason if isinstance(reason, str) else None

    def _build_execution_note(
        self,
        result: dict[str, Any],
        decision: TradeDecision | None,
        stale_recommendation: bool,
        reconcile_outcome: LocalPositionReconciliation | None,
        protection_guard_note: str | None = None,
        protection_version_note: str | None = None,
        intent_state: str | None = None,
    ) -> str | None:
        """Explain WHY an actionable recommendation was not executed, or None.

        Returns None only when the recommendation WAS performed with executor evidence
        (or when there was nothing actionable to perform). A command that was forwarded
        but not yet confirmed keeps its note: the card must say pending/unknown instead
        of showing an execution. The text is shown verbatim on the Discord card.
        """
        analysis = result.get("analysis") or {}
        signal = str(analysis.get("signal") or "").upper()
        if signal not in ACTIONABLE_RECOMMENDATIONS:
            return None

        action = str(getattr(decision, "action", "") or "").upper() if decision is not None else ""
        if decision is not None and action in ACTIONABLE_RECOMMENDATIONS:
            if intent_state in NON_BOOKED_INTENT_STATES:
                return (
                    f"Recommendation {signal} was SENT to the executor, but its execution "
                    f"is UNCONFIRMED (state: {intent_state}) — no proof from the executor. "
                    f"The local state and statistics do NOT count it as executed; the local "
                    f"position stays unchanged."
                )
            return None

        if protection_version_note:
            reason = (
                "the position protection was replaced during the analysis, so the proposed "
                f"UPDATE describes an outdated state: {protection_version_note}"
            )
        elif protection_guard_note:
            reason = (
                "the position protection changed during the analysis and the proposed UPDATE "
                f"would loosen the protection: {protection_guard_note}"
            )
        elif stale_recommendation:
            reason = (
                "the position changed during the analysis (an exit or an exchange close "
                "was detected) — the result describes an outdated position, so it was not "
                "executed and not turned into a new entry"
            )
        elif decision is not None and action == "HOLD":
            detail = str(getattr(decision, "reasoning", "") or "no reason given")
            reason = f"the strategy rejected the recommendation: {detail}"
        elif getattr(reconcile_outcome, "state", None) == RECONCILE_UNVERIFIED:
            detail = getattr(reconcile_outcome, "detail", None) or "no reply from the executor"
            reason = f"the exchange did not confirm the position state ({detail}) — not executed"
        elif getattr(self.trading_strategy, "current_position", None) is None:
            reason = (
                "no open local position (exit detected/settled) — "
                "there is nothing to update or close"
            )
        else:
            reason = (
                "the strategy did not execute the recommendation (update interval, guard or "
                "missing confirmation) — the local position is still open"
            )
        return f"Recommendation {signal} NOT executed: {reason}."

    def _log_check_header(self, check_count: int):
        """Log trading check header"""
        current_time = datetime.now(timezone.utc)
        self.logger.info("=" * 60)
        self.logger.info("Trading Check #%s at %s", check_count, current_time.strftime("%Y-%m-%d %H:%M:%S"))
        self.logger.info("=" * 60)

    async def _fetch_ticker_data(self):
        """Fetch current ticker and price"""
        try:
            current_ticker = await self._fetch_current_ticker()  # type: ignore
            if current_ticker:
                current_price = float(current_ticker.get("last", current_ticker.get("close", 0)))
                return current_ticker, current_price
        except Exception as e:  # noqa: BLE001
            self.logger.warning("Could not fetch current ticker: %s", e)
        return None, None

    async def _check_position_status(self, current_price: float | None, *, is_candle_close: bool = True):
        """Check if existing position hit stop/target.

        Soft exit mode: SL/TP evaluation only runs on candle close.
        Forced analysis (keyboard 'a') skips automated stop checks
        but the AI can still consciously signal CLOSE.
        """
        await self._require_position_monitor().check_soft_exit_status(current_price, is_candle_close=is_candle_close)

    async def _execute_market_knowledge_update(self, force_news_update: bool):
        """Update market knowledge based on analysis type"""
        timeout_seconds = max(1, int(self.config.RAG_UPDATE_TIMEOUT))
        analysis_type = "Regular Analysis" if force_news_update else "Forced Analysis"
        self.logger.info(
            "Updating market knowledge (%s, timeout=%ss)...",
            analysis_type,
            timeout_seconds,
        )

        update_start = time.perf_counter()
        try:
            updated = await asyncio.wait_for(
                self.rag_engine.update_if_needed(force_update=force_news_update),
                timeout=timeout_seconds,
            )
            self.logger.debug(
                "Market knowledge update completed in %.1fs (updated=%s)",
                time.perf_counter() - update_start,
                updated,
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                "Market knowledge update timed out after %.1fs (limit=%ss); %s",
                time.perf_counter() - update_start,
                timeout_seconds,
                _MARKET_KNOWLEDGE_FALLBACK_MSG,
            )
        except Exception as e:  # noqa: BLE001
            self.logger.error(
                "Market knowledge update failed after %.1fs: %s; %s",
                time.perf_counter() - update_start,
                e,
                _MARKET_KNOWLEDGE_FALLBACK_MSG,
            )

    async def _build_analysis_context(self, current_price: float | None, current_ticker) -> dict[str, Any]:
        """Build context data for market analysis"""
        position_context = self.trading_strategy.get_position_context(current_price)
        memory_context = self.memory_service.get_context_summary()
        statistics_context = self.statistics_service.get_context()

        if statistics_context:
            position_context = f"{position_context}\n\n{statistics_context}"

        previous_data = await self.persistence.async_load_previous_response()
        previous_response = previous_data.get("response") if previous_data else None
        previous_indicators = previous_data.get("technical_indicators") if previous_data else None

        last_analysis_time_str = self._get_formatted_last_analysis_time()
        dynamic_thresholds = self.brain_service.get_dynamic_thresholds()

        additional_context = ""
        self._reddit_sentiment_label = "NEUTRAL"
        if (
            self.config.SOCIAL_SENTIMENT_ENABLED
            and self.sentiment_analyst is not None
        ):
            try:
                sentiment_data = await self.sentiment_analyst.fetch_sentiment()
                additional_context += self.sentiment_analyst.format_sentiment_section(
                    sentiment_data
                )
                self._reddit_sentiment_label = sentiment_data.get("overall_sentiment", "NEUTRAL")
            except Exception as e:  # noqa: BLE001
                self.logger.warning("Failed to fetch Reddit sentiment: %s", e)

        return {
            "previous_response": previous_response,
            "previous_indicators": previous_indicators,
            "position_context": position_context,
            "performance_context": memory_context,
            "brain_service": self.brain_service,
            "last_analysis_time": last_analysis_time_str,
            "current_ticker": current_ticker,
            "dynamic_thresholds": dynamic_thresholds,
            "ev_context": self._build_ev_context(),
            "additional_context": additional_context,
        }

    def _get_formatted_last_analysis_time(self) -> str | None:
        """Get last analysis time formatted as UTC string"""
        last_analysis_time_obj = self.persistence.get_last_analysis_time()
        if not last_analysis_time_obj:
            return None

        if last_analysis_time_obj.tzinfo is None:
            last_analysis_time_obj = last_analysis_time_obj.replace(tzinfo=timezone.utc)
        else:
            last_analysis_time_obj = last_analysis_time_obj.astimezone(timezone.utc)

        return last_analysis_time_obj.strftime("%Y-%m-%d %H:%M:%S")

    def _build_ev_context(self) -> str:
        """Build the EV framework context string with dynamic capital tracking."""
        if self.ev_formatter is None:
            return ""
        demo_capital = float(self.config.DEMO_QUOTE_CAPITAL)
        current_capital = self.statistics_service.get_current_capital(demo_capital)
        return self.ev_formatter.build_ev_framework_section(current_capital)

    async def _report_executor_side_exit(self) -> None:
        """Send the closing performance summary after an exchange-side exit.

        The exchange closed the position (SL/TP filled), so the local monitor never
        saw a close and reports nothing. The strategy books the trade from the
        executor's exit journal and flags the reason for us here.
        """
        reason = self.trading_strategy.take_executor_side_exit_reason()
        if not isinstance(reason, str) or not reason:
            return

        self.logger.info("Exchange-side exit booked (%s) — sending performance summary", reason)
        try:
            await self._require_position_monitor().handle_position_closed(reason)
        except Exception as e:  # noqa: BLE001
            self.logger.warning("Failed to report exchange-side exit: %s", e)

    async def _handle_new_position(self, decision, current_price: float | None):
        """Handle new position creation and status updates"""
        if decision.action not in ("BUY", "SELL"):
            self.logger.debug(
                "_handle_new_position skipped for non-entry action: %s", decision.action
            )
            return
        if not self.trading_strategy.current_position:
            self.logger.warning(
                "Expected current_position after %s action but position is None — "
                "skipping position monitor start",
                decision.action,
            )
            return

        await self._require_position_monitor().handle_new_position(current_price)

    async def _send_discord_notification(self, result: dict[str, Any], execution_note: str | None = None):
        """Send Discord notification with analysis results.

        ``execution_note`` marks a recommendation the bot did NOT perform, so the card
        never shows a bare signal that reads as an executed action.
        """
        if self.discord_notifier:
            chart_image = None
            last_chart_buffer = self.market_analyzer.last_chart_buffer if self.market_analyzer else None
            if last_chart_buffer is not None:
                last_chart_buffer.seek(0)
                chart_image = io.BytesIO(last_chart_buffer.getvalue())
                chart_image.seek(0)

            await self.discord_notifier.send_analysis_notification(
                result=result,
                symbol=self.current_symbol,
                timeframe=self.current_timeframe,
                channel_id=self.config.MAIN_CHANNEL_ID,
                chart_image=chart_image,
                execution_note=execution_note
            )

    async def _save_analysis_data(self, result: dict[str, Any]):
        """Save analysis response and technical data (off the event loop)"""
        raw_response = result.get("raw_response", "")
        if raw_response:
            technical_data = result.get("technical_data")
            generated_prompt = result.get("generated_prompt")
            await self.persistence.async_save_previous_response(raw_response, technical_data, generated_prompt)

    def _patch_rejected_signal_in_response(
        self, result: dict[str, Any], decision: TradeDecision
    ) -> None:
        """Rewrite ``result["raw_response"]`` when the strategy vetoed a BUY/SELL.

        The LLM sees its own previous response as context next cycle.  If it
        output BUY but the strategy blocked it (R/R guard, size cap, etc.),
        the LLM gets confused: "I said BUY, why is there no position?"

        This patches the JSON block — replacing BUY/SELL with HOLD and
        appending the rejection reason — so the next prompt shows the LLM
        that its trade was blocked and why.
        """
        import re

        raw = result.get("raw_response", "")
        if not raw:
            return
        signal: str = ""
        try:
            analysis = result.get("analysis") or {}
            signal = str(analysis.get("signal", ""))
        except (AttributeError, TypeError, KeyError) as exc:
            self.logger.debug("Failed to extract signal from result analysis: %s", exc)
        if signal not in ("BUY", "SELL"):
            return
        reason = decision.reasoning or "Blocked by trading strategy guard"
        raw = re.sub(
            rf'"signal"\s*:\s*"{re.escape(signal)}"',
            '"signal": "HOLD"',
            raw,
            count=1,
            flags=re.IGNORECASE,
        )
        match = re.search(r"```json\s*\{.*?\}\s*```", raw, re.DOTALL)
        if match:
            pos = match.end()
            raw = raw[:pos] + "\n\n⚠️  REJECTED: " + reason + raw[pos:]
        result["raw_response"] = raw

    @retry_async(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def _fetch_current_ticker(self) -> dict[str, Any] | None:
        """Fetch current ticker from exchange."""
        if self.current_exchange is None or self.current_symbol is None:
            return None
        ticker = await self.current_exchange.fetch_ticker(self.current_symbol)
        if ticker and self.dashboard_state:
            price = float(ticker.get("last", ticker.get("close", 0)))
            if price > 0:
                await self.dashboard_state.update_price(price)
        return ticker

    async def fetch_current_ticker(self) -> dict[str, Any] | None:
        """Fetch current ticker from exchange (public API)."""
        return await self._fetch_current_ticker()  # type: ignore

    def _calculate_next_check(self, source_time_ms: int) -> tuple[float, datetime]:
        """Calculate delay and next-check time for a timeframe candle.

        Shared by ``_wait_for_next_timeframe`` and
        ``_wait_until_next_timeframe_after`` to eliminate duplicated
        timeframe-arithmetic logic.

        Returns:
            (delay_seconds, next_check_time_utc)
        """
        if self.current_timeframe is None:
            raise ValueError("current timeframe is not set")
        timeframe = self.current_timeframe
        current_time_ms = int(time.time() * 1000)
        next_candle_ms = TimeframeValidator.calculate_next_candle_time(source_time_ms, timeframe)
        delay_ms = next_candle_ms - current_time_ms + (CANDLE_BUFFER_SECONDS * 1000)
        delay_seconds = max(0, delay_ms / 1000)
        next_check_time = datetime.fromtimestamp(next_candle_ms / 1000, timezone.utc)
        return delay_seconds, next_check_time

    async def _wait_for_next_timeframe(self):
        """Wait until the next timeframe candle starts."""
        try:
            current_time_ms = int(time.time() * 1000)
            delay_seconds, next_check_time = self._calculate_next_check(current_time_ms)

            self.logger.info(
                "Next check at %s (in %.0fs)",
                self._format_utc_and_local(next_check_time),
                delay_seconds
            )
            if self.dashboard_state:
                await self.dashboard_state.update_next_check(next_check_time)
            return await self._interruptible_sleep(delay_seconds)

        except Exception as e:  # noqa: BLE001
            self.logger.error("Error calculating next timeframe: %s", e)
            await self._interruptible_sleep(ERROR_WAIT_LONG)
            return False

    async def _wait_until_next_timeframe_after(self, last_time: datetime):
        """Wait until the next timeframe candle after a specific timestamp.

        Owns the single "Resuming from last analysis ..." startup line, so callers
        must not log a resume message of their own.
        """
        try:
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=timezone.utc)

            current_time_ms = int(time.time() * 1000)
            last_time_ms = int(last_time.timestamp() * 1000)

            delay_seconds, next_check_time = self._calculate_next_check(last_time_ms)

            next_candle_ms = int(next_check_time.timestamp() * 1000) - (CANDLE_BUFFER_SECONDS * 1000)
            if current_time_ms >= next_candle_ms:
                self.logger.info(
                    "Resuming from last analysis at %s. Next candle already passed - proceeding immediately",
                    self._format_utc_and_local(last_time)
                )
                self.logger.info("Crypto Trading Bot ready")
                return

            is_same = TimeframeValidator.is_same_candle(current_time_ms, last_time_ms, self.current_timeframe)  # type: ignore

            context_msg = "Still in same candle" if is_same else "Waiting for next candle"
            self.logger.info(
                "Resuming from last analysis at %s. %s - next check at %s (in %.0fs)",
                self._format_utc_and_local(last_time),
                context_msg,
                self._format_utc_and_local(next_check_time),
                delay_seconds
            )
            self.logger.info("Crypto Trading Bot ready")

            if self.dashboard_state:
                await self.dashboard_state.update_next_check(next_check_time)
            await self._interruptible_sleep(delay_seconds)

        except Exception as e:  # noqa: BLE001
            self.logger.error("Error calculating wait time: %s", e)
            await self._interruptible_sleep(ERROR_WAIT_SHORT)

    async def interruptible_sleep(self, seconds: float, respect_force_analysis: bool = True) -> bool:
        """Interruptible sleep (public API)."""
        return await self._interruptible_sleep(seconds, respect_force_analysis=respect_force_analysis)

    async def _interruptible_sleep(self, seconds: float, respect_force_analysis: bool = True):
        """Sleep in small chunks to allow responsive shutdown and force analysis.

        Uses SLEEP_CHUNK_SIZE to check for interruptions periodically.
        Properly handles cancellation for graceful shutdown.
        Returns:
            bool: True if sleep was interrupted by force_analysis, False otherwise
        """
        start_time = time.monotonic()

        if respect_force_analysis:
            self._force_analysis.clear()

        try:
            while self.running:
                elapsed = time.monotonic() - start_time
                if elapsed >= seconds:
                    break

                if respect_force_analysis and self._force_analysis.is_set():
                    self._force_analysis.clear()
                    self.logger.info("Force analysis triggered - interrupting wait")
                    return True

                remaining = seconds - elapsed
                sleep_time = min(SLEEP_CHUNK_SIZE, remaining)
                await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            self.logger.debug("Interruptible sleep cancelled during shutdown")
            return False

        return False

    async def _force_analysis_now(self):
        """Force immediate analysis by interrupting the wait."""
        self.logger.info("Forcing immediate analysis...")
        self._force_analysis.set()

    async def _show_help(self):
        """Show help information about available commands."""
        self.keyboard_handler.display_help()

    async def _request_shutdown(self):
        """Request application shutdown via keyboard."""
        self.logger.info("Shutdown requested via keyboard command")
        if self.shutdown_manager:
            await self.shutdown_manager.shutdown_gracefully()
        else:
            self.running = False
            for task in self.tasks:
                if not task.done():
                    task.cancel()

    async def _request_reload(self):
        """Request an in-place reload: graceful shutdown, then the launcher restarts the bot.

        Only available when the bot was started by a launcher that supports it
        (scripts/start_script_*.ps1 set LLM_TRADER_RELOAD_SUPPORTED=1).
        """
        if not self.shutdown_manager:
            self.logger.warning("Reload unavailable: shutdown manager missing")
            return
        if os.environ.get("LLM_TRADER_RELOAD_SUPPORTED") != "1":
            self.logger.warning(
                "Reload requested, but this launch cannot restart in place "
                "(start the bot via scripts/start_script_main.ps1). Use 'q' to quit."
            )
            return
        if not self.shutdown_manager.request_reload():
            self.logger.info("Reload ignored - shutdown already in progress")
            return
        self.logger.info("Reload requested - shutting down for in-place restart...")
        self.running = False

    async def _report_state_divergence(self) -> None:
        """Alert the operator when the executor's view and the local position disagree.

        The strategy KEEPS its local position when the executor reports flat without
        an exit record (a false "flat" is possible — see ``book_executor_side_exit``),
        so the mismatch needs a human look at the exchange instead of a silent guess.
        """
        message = self.trading_strategy.take_state_divergence()
        if not isinstance(message, str) or not message:
            return
        self.logger.critical("State divergence reported to the operator: %s", message)
        if self.discord_notifier is None:
            return
        try:
            await self.discord_notifier.send_message(
                "⚠️ **Bot/exchange state divergence** — I did not book the close, "
                f"because the executor has no exit record. {message}. Check the exchange "
                "for SL/TP orders and the balance, then let me know — I will fix the state.",
                channel_id=self.config.MAIN_CHANNEL_ID,
            )
        except Exception as e:  # noqa: BLE001
            self.logger.warning("Failed to send state-divergence alert: %s", e)
