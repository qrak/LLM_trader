

import asyncio
import math
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .data_models import LocalExitRequest
from .executor_reconciliation import (
    RECONCILE_EXIT_BOOKED,
    RECONCILE_NO_POSITION,
    RECONCILE_OPEN_EXECUTOR_REPORTED,
    VERIFICATION_EXCHANGE_VERIFIED,
    VERIFICATION_EXECUTOR_REPORTED,
    VERIFICATION_UNVERIFIED,
    LocalPositionReconciliation,
)
from .exit_monitor import ExitMonitor

if TYPE_CHECKING:
    from .trading_strategy import TradingStrategy

DEFAULT_RECONCILE_INTERVAL_SECONDS = 120.0
VERIFICATION_FRESHNESS_FACTOR = 2.0
DEFAULT_REFERENCE_CHECK_SECONDS = 10.0
REFERENCE_FORWARD_RETRY_SECONDS = 60.0
REFERENCE_MISSING_ALERT_SECONDS = 300.0


class PositionStatusMonitor:
    """Owns the open-position status loop and its persisted monitor state.

    When ``manages_exits`` reports False (the executor holds and watches the SL/TP
    orders on the exchange), this loop keeps publishing position status but never
    closes the position locally: the exchange stays the single source of truth.

    Independently of that policy, when ``reference_exits_enabled`` reports True the
    loop also evaluates the position's SL/TP against the REAL market on its own fast
    cadence (``reference_exit_check_seconds``) and hands a hit to the executor at
    once. The exchange's protective leg is a catastrophe backstop placed beyond the
    venue's own price noise (the executor's ``PROTECTION_BACKSTOP_MULTIPLIER``), so
    this check — not that leg — is the stop that actually runs.

    Independently of both, when ``reconcile_position`` is wired the loop also
    runs the local-position reconciliation on its own short cadence (<= 120s on a
    healthy API): no ticker, no LLM, no 4h schedule involved.
    """

    @staticmethod
    def _always_manages_exits() -> bool:

        return True

    @staticmethod
    def _never_checks_reference() -> bool:

        return False

    def __init__(
        self,
        logger: Any,
        config: Any,
        persistence: Any,
        trading_strategy: "TradingStrategy",
        exit_monitor: ExitMonitor,
        notifier: Any,
        active_tasks: set[asyncio.Task],
        is_running: Callable[[], bool],
        fetch_current_ticker: Callable[[], Awaitable[dict[str, Any] | None]],
        interruptible_sleep: Callable[..., Awaitable[Any]],
        get_symbol: Callable[[], str | None],
        manages_exits: Callable[[], bool] | None = None,
        reconcile_position: Callable[[], Awaitable[LocalPositionReconciliation]] | None = None,
        reconcile_interval_seconds: float = DEFAULT_RECONCILE_INTERVAL_SECONDS,
        reference_exits_enabled: Callable[[], bool] | None = None,
        reference_exit_check_seconds: float = DEFAULT_REFERENCE_CHECK_SECONDS,
        forward_local_close: Callable[[], Awaitable[Any]] | None = None,
    ) -> None:
        self.logger = logger
        self.config = config
        self.persistence = persistence
        self.trading_strategy = trading_strategy
        self.exit_monitor = exit_monitor
        self.notifier = notifier
        self.active_tasks = active_tasks
        self.is_running = is_running
        self.fetch_current_ticker = fetch_current_ticker
        self.interruptible_sleep = interruptible_sleep
        self.get_symbol = get_symbol
        self.manages_exits = manages_exits if manages_exits is not None else self._always_manages_exits
        self.reconcile_position = reconcile_position
        self.reconcile_interval_seconds = float(reconcile_interval_seconds)
        self.reference_exits_enabled = (
            reference_exits_enabled if reference_exits_enabled is not None else self._never_checks_reference
        )
        self.reference_exit_check_seconds = max(0.0, float(reference_exit_check_seconds))
        self.forward_local_close = forward_local_close
        self._task: asyncio.Task | None = None
        self._position_close_lock = asyncio.Lock()
        self._last_reconcile_at: datetime | None = None
        self._last_reconcile_outcome: LocalPositionReconciliation | None = None
        self._last_executor_reported_open_at: datetime | None = None
        self._last_reference_check_at: datetime | None = None
        self._last_reference_forward_at: datetime | None = None
        self._reference_exit_reported_key: tuple[str, str] | None = None
        self._reference_exit_position_key: str | None = None
        self._last_reference_missing_alert_at: datetime | None = None

    def seconds_until_next_reference_check(self, now: datetime) -> float:

        if self.reference_exit_check_seconds <= 0 or not self.reference_exits_enabled():
            return math.inf
        if self._last_reference_check_at is None:
            return 0.0
        elapsed = (now - self._last_reference_check_at).total_seconds()
        return max(0.0, self.reference_exit_check_seconds - elapsed)

    def seconds_until_next_reconcile(self, now: datetime) -> float:

        if self.reconcile_position is None:
            return math.inf
        if self._last_reconcile_at is None:
            return 0.0
        elapsed = (now - self._last_reconcile_at).total_seconds()
        return max(0.0, self.reconcile_interval_seconds - elapsed)

    async def sync_position_state(self, *, force: bool = False) -> LocalPositionReconciliation | None:

        if self.reconcile_position is None:
            return None
        now = datetime.now(timezone.utc)
        if not force and self.seconds_until_next_reconcile(now) > 0:
            return self._last_reconcile_outcome

        try:
            outcome = await self.reconcile_position()
        except Exception as e:  # noqa: BLE001
            self.logger.warning("Position reconciliation failed: %s", e)
            self._last_reconcile_at = now
            self._last_reconcile_outcome = None
            return None

        self._last_reconcile_at = datetime.now(timezone.utc)
        self._last_reconcile_outcome = outcome
        if outcome.state == RECONCILE_OPEN_EXECUTOR_REPORTED:
            self._last_executor_reported_open_at = outcome.checked_at or now
        return outcome

    def verification(self) -> tuple[str, datetime | None, str | None]:

        outcome = self._last_reconcile_outcome
        if outcome is None:
            detail = (
                "no exchange verification performed yet"
                if self._last_executor_reported_open_at is None
                else None
            )
            return VERIFICATION_UNVERIFIED, self._last_executor_reported_open_at, detail
        state = outcome.state
        checked_at = outcome.checked_at
        detail = outcome.detail
        exchange_verified_at = outcome.exchange_verified_at

        if state == RECONCILE_OPEN_EXECUTOR_REPORTED and exchange_verified_at is not None:
            freshness = max(
                self.reconcile_interval_seconds * VERIFICATION_FRESHNESS_FACTOR,
                self.reconcile_interval_seconds,
            )
            if (datetime.now(timezone.utc) - exchange_verified_at).total_seconds() <= freshness:
                return VERIFICATION_EXCHANGE_VERIFIED, exchange_verified_at, detail
            return (
                VERIFICATION_UNVERIFIED,
                exchange_verified_at,
                "exchange verification is stale",
            )

        if state == RECONCILE_OPEN_EXECUTOR_REPORTED and checked_at is not None:
            freshness = max(
                self.reconcile_interval_seconds * VERIFICATION_FRESHNESS_FACTOR,
                self.reconcile_interval_seconds,
            )
            if (datetime.now(timezone.utc) - checked_at).total_seconds() <= freshness:
                return VERIFICATION_EXECUTOR_REPORTED, checked_at, detail
            return (
                VERIFICATION_UNVERIFIED,
                checked_at,
                "executor tracker report is stale (no recent confirmation)",
            )

        if state in (None, RECONCILE_NO_POSITION) and self._last_executor_reported_open_at is None:
            return VERIFICATION_UNVERIFIED, None, detail or "no exchange verification performed yet"
        return VERIFICATION_UNVERIFIED, self._last_executor_reported_open_at, detail

    def _take_booked_exit_reason(self) -> str | None:

        return self.trading_strategy.take_executor_side_exit_reason() or None

    def _take_local_exit_request(self) -> LocalExitRequest | None:

        return self.trading_strategy.take_local_exit_request()

    async def _publish_local_exit_request(
        self, request: LocalExitRequest | None, *, forwarded: bool | None = None
    ) -> None:

        if request is None:
            return
        self.logger.critical(
            "LOCAL EXIT CONDITION %s @ %s NOT BOOKED (state=%s, forwarded=%s) — local "
            "position kept, no statistics entry; only executor fill evidence may close "
            "this trade.",
            request.reason,
            request.observed_price,
            request.state,
            "yes" if forwarded else "no",
        )
        if not self.notifier:
            return
        detail = request.detail
        forwarded_text = (
            "The CLOSE was handed to the executor; the position stays until its fill "
            "evidence arrives."
            if forwarded
            else "The CLOSE was NOT handed to the executor (see the log) — check the "
            "executor and the exchange."
        )
        message = (
            "🚨 **Local exit condition — no evidence, no booking** — "
            f"state: {request.state}. {detail} {forwarded_text} "
            "I am not closing the position silently."
        )
        try:
            await self.notifier.send_message(message, channel_id=self.config.MAIN_CHANNEL_ID)
        except Exception as e:  # noqa: BLE001
            self.logger.warning("Failed to send the local-exit alert: %s", e)

    async def _report_local_exit_request(self) -> None:

        await self._publish_local_exit_request(self._take_local_exit_request())

    def _current_position_key(self) -> str:

        position = self.trading_strategy.current_position
        if position is None:
            return str(self.get_symbol() or "?")
        return f"{self.get_symbol()}|{position.entry_time.isoformat()}"

    def _reference_exit_key(self, request: LocalExitRequest) -> tuple[str, str]:

        return (self._current_position_key(), request.reason)

    def _reference_forward_due(self, now: datetime) -> bool:

        if self.forward_local_close is None:
            return False
        if self._last_reference_forward_at is None:
            return True
        elapsed = (now - self._last_reference_forward_at).total_seconds()
        return elapsed >= REFERENCE_FORWARD_RETRY_SECONDS

    def _reference_exit_already_requested(self, now: datetime) -> bool:

        if self._reference_forward_due(now):
            return False
        return self._reference_exit_position_key == self._current_position_key()

    def _warn_missing_reference_price(self, now: datetime) -> None:

        last = self._last_reference_missing_alert_at
        if last is not None and (now - last).total_seconds() < REFERENCE_MISSING_ALERT_SECONDS:
            return
        self._last_reference_missing_alert_at = now
        self.logger.warning(
            "Real-market exit check SKIPPED: no current price for %s — the position is "
            "protected only by the exchange's backstop leg; retrying every %.0fs",
            self.get_symbol(), self.reference_exit_check_seconds,
        )

    async def check_reference_exits(self, current_price: float | None) -> None:

        if not self.reference_exits_enabled():
            return
        if not self.trading_strategy.current_position:
            return

        now = datetime.now(timezone.utc)
        self._last_reference_check_at = now
        if current_price is None:
            self._warn_missing_reference_price(now)
            return

        if self._reference_exit_already_requested(now):
            return

        try:
            async with self._position_close_lock:
                await self.trading_strategy.check_reference_exit(current_price)
        except Exception as e:  # noqa: BLE001
            self.logger.error("Real-market exit check failed: %s", e)
            return

        request = self._take_local_exit_request()
        if request is None:
            return

        key = self._reference_exit_key(request)
        first_report = key != self._reference_exit_reported_key
        self._reference_exit_reported_key = key
        self._reference_exit_position_key = self._current_position_key()

        forwarded: bool | None = None
        forwarder = self.forward_local_close
        if forwarder is not None and self._reference_forward_due(now):
            self._last_reference_forward_at = now
            try:
                forwarded = bool(await forwarder())
            except Exception as e:  # noqa: BLE001
                self.logger.error("Forwarding the real-market exit failed: %s", e)
                forwarded = False

        if first_report:
            await self._publish_local_exit_request(request, forwarded=forwarded)

    async def _send_position_status(self, position: Any, current_price: float) -> None:

        if not self.notifier:
            return
        verification, verified_at, detail = self.verification()
        try:
            await self.notifier.send_position_status(
                position=position,
                current_price=current_price,
                channel_id=self.config.MAIN_CHANNEL_ID,
                verification=verification,
                verified_at=verified_at,
                verification_detail=detail,
            )
        except TypeError:
            await self.notifier.send_position_status(
                position=position,
                current_price=current_price,
                channel_id=self.config.MAIN_CHANNEL_ID,
            )

    async def check_soft_exit_status(self, current_price: float | None, *, is_candle_close: bool = True) -> None:

        if not self.manages_exits():
            self.logger.debug(
                "Exchange owns the exits (executor enabled): skipping local soft SL/TP evaluation"
            )
            return

        if not (self.trading_strategy.current_position and current_price is not None):
            if self.trading_strategy.current_position and current_price is None:
                self.logger.warning(
                    "Soft exit check SKIPPED: open %s position but current price unavailable",
                    self.trading_strategy.current_position.direction,
                )
            return

        if not is_candle_close:
            self.logger.info("Intra-candle check: skipping soft SL/TP evaluation")
            return

        try:
            close_reason = await self.exit_monitor.check_soft_exits(
                self.trading_strategy,
                current_price,
                self._position_close_lock,
            )
            if close_reason:
                await self.handle_position_closed(close_reason)
            else:
                await self._report_local_exit_request()
        except Exception as e:  # noqa: BLE001
            self.logger.error("Error checking position: %s", e)

    async def handle_new_position(self, current_price: float | None) -> None:

        if not self.trading_strategy.current_position:
            return

        try:
            outcome = await self.sync_position_state(force=True)
            if outcome is not None and (outcome.state == RECONCILE_EXIT_BOOKED or outcome.exit_booked):
                reason = self._take_booked_exit_reason() or "exchange-side exit detected"
                self.logger.info("Position already exited on the exchange — skipping the OPEN card")
                await self.handle_position_closed(reason)
                return
            if not self.trading_strategy.current_position:
                self.logger.info("No local position after the pre-status sync — skipping the OPEN card")
                return

            if current_price is None:
                ticker = await self.fetch_current_ticker()
                if ticker:
                    current_price = float(ticker.get("last", ticker.get("close", 0)))
                else:
                    self.logger.warning("No ticker available for initial position status, skipping")
                    return

            await self._send_position_status(self.trading_strategy.current_position, current_price)

            now = datetime.now(timezone.utc)
            await self.save_state(
                last_stop_loss_check_at=now,
                last_take_profit_check_at=now,
                last_status_sent_at=now,
            )
        except Exception as e:  # noqa: BLE001
            self.logger.warning("Error sending initial position status: %s", e)

        await self.start()

    async def handle_position_closed(self, close_reason: str) -> None:

        self.logger.info("Position closed: %s", close_reason)
        await self.exit_monitor.clear_state(self.persistence)
        if asyncio.current_task() is not self._task:
            await self.stop()

        symbol = self.get_symbol()
        if self.notifier and symbol:
            history = await asyncio.to_thread(self.persistence.load_trade_history)
            await self.notifier.send_performance_stats(
                trade_history=history,
                symbol=symbol,
                channel_id=self.config.MAIN_CHANNEL_ID,
            )

    async def start(self) -> None:

        if self._task and not self._task.done():
            return

        self._task = asyncio.create_task(self._loop(), name="Position-Status-Updates")
        self.active_tasks.add(self._task)
        self._task.add_done_callback(self.active_tasks.discard)
        self.logger.debug("Started position status and exit monitor")

    async def stop(self) -> None:

        if self._task and not self._task.done():
            if asyncio.current_task() is self._task:
                return
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            self.logger.debug("Stopped position status and exit monitor")

    async def save_state(self, **timestamps: datetime) -> None:

        symbol = self.get_symbol()
        if not symbol:
            self.logger.warning("Skipping position monitor state save because current symbol is unset")
            return
        await self.exit_monitor.save_state(self.persistence, symbol, **timestamps)

    async def run_hard_exit_checks(
        self,
        current_price: float | None,
        now: datetime,
        state: dict[str, Any],
    ) -> str | None:

        if not self.manages_exits():
            due_exits = self.exit_monitor.due_hard_exits(now, state)
            if due_exits:
                self.logger.debug(
                    "Exchange owns the exits (executor enabled): skipping local %s evaluation",
                    ", ".join(due_exits),
                )
                await self.save_state(
                    **{self.exit_monitor.last_check_key(kind): now for kind in due_exits}
                )
            return None

        due_hard_exits = self.exit_monitor.due_hard_exits(now, state)
        close_reason, timestamps = await self.exit_monitor.check_hard_exits(
            self.trading_strategy,
            current_price,
            now,
            state,
            self._position_close_lock,
        )
        if current_price is None and due_hard_exits:
            self.logger.warning("Skipping hard exit checks because current ticker price is unavailable")

        if timestamps:
            await self.save_state(**timestamps)
        if close_reason is None:
            await self._report_local_exit_request()
        return close_reason

    async def _loop(self) -> None:

        try:
            while self.is_running():
                state = await self.exit_monitor.load_state(self.persistence)
                now = datetime.now(timezone.utc)
                delay_seconds = min(
                    self.exit_monitor.seconds_until_next_tick(state, now),
                    self.seconds_until_next_reconcile(now),
                    self.seconds_until_next_reference_check(now),
                )
                if delay_seconds > 0:
                    await self.interruptible_sleep(delay_seconds, respect_force_analysis=False)

                if not self.is_running():
                    break

                if not self.trading_strategy.current_position:
                    self.logger.debug("Position closed, stopping status updates")
                    break

                try:
                    outcome = await self.sync_position_state()
                except Exception as e:  # noqa: BLE001
                    self.logger.warning("Independent position sync failed: %s", e)
                    outcome = None

                if outcome is not None and (outcome.exit_booked or outcome.state == RECONCILE_EXIT_BOOKED):
                    reason = self._take_booked_exit_reason() or "exchange-side exit detected"
                    await self.handle_position_closed(reason)
                    break
                if not self.trading_strategy.current_position:
                    self.logger.debug("Position closed after the exchange sync, stopping status updates")
                    break

                try:
                    now = datetime.now(timezone.utc)
                    state = await self.exit_monitor.load_state(self.persistence)
                    ticker = await self.fetch_current_ticker()
                    current_price = float(ticker.get("last", ticker.get("close", 0))) if ticker else None

                    close_reason = await self.run_hard_exit_checks(current_price, now, state)
                    if close_reason:
                        await self.handle_position_closed(close_reason)
                        break

                    await self.check_reference_exits(current_price)
                    if not self.trading_strategy.current_position:
                        self.logger.debug("Position closed after the exit checks, stopping status updates")
                        break

                    if self.notifier and self.trading_strategy.current_position and current_price is not None and self.exit_monitor.is_status_due(now, state):
                        await self._send_position_status(self.trading_strategy.current_position, current_price)
                        await self.save_state(last_status_sent_at=now)
                        self.logger.debug("Sent position status update")
                except Exception as e:  # noqa: BLE001
                    self.logger.warning("Error running position monitor update: %s", e)
        except asyncio.CancelledError:
            self.logger.debug("Position status loop cancelled")
            raise

