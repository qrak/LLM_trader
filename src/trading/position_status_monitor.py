"""Background position status and hard-exit monitoring loop."""

import asyncio
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from .exit_monitor import ExitMonitor


class PositionStatusMonitor:
    """Owns the open-position status loop and its persisted monitor state."""

    def __init__(
        self,
        logger: Any,
        config: Any,
        persistence: Any,
        trading_strategy: Any,
        exit_monitor: ExitMonitor,
        notifier: Any,
        active_tasks: set[asyncio.Task],
        is_running: Callable[[], bool],
        fetch_current_ticker: Callable[[], Awaitable[dict[str, Any] | None]],
        interruptible_sleep: Callable[..., Awaitable[Any]],
        get_symbol: Callable[[], str | None],
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
        self._task: asyncio.Task | None = None
        self._position_close_lock = asyncio.Lock()

    async def check_soft_exit_status(self, current_price: float | None, *, is_candle_close: bool = True) -> None:
        """Evaluate soft exits at candle close and handle a closed position."""
        if not (self.trading_strategy.current_position and current_price is not None):
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
        except Exception as e:
            self.logger.error("Error checking position: %s", e)

    async def handle_new_position(self, current_price: float | None) -> None:
        """Send the first status message, seed monitor timestamps, and start the loop."""
        if not self.trading_strategy.current_position:
            return

        try:
            if current_price is None:
                ticker = await self.fetch_current_ticker()
                current_price = float(ticker.get('last', ticker.get('close', 0))) if ticker else 0.0

            if self.notifier:
                await self.notifier.send_position_status(
                    position=self.trading_strategy.current_position,
                    current_price=current_price,
                    channel_id=self.config.MAIN_CHANNEL_ID,
                )

            now = datetime.now(timezone.utc)
            await self.save_state(
                last_stop_loss_check_at=now,
                last_take_profit_check_at=now,
                last_status_sent_at=now,
            )
        except Exception as e:
            self.logger.warning("Error sending initial position status: %s", e)

        await self.start()

    async def handle_position_closed(self, close_reason: str) -> None:
        """Clear monitor state, stop the loop, and send performance stats."""
        self.logger.info("Position closed: %s", close_reason)
        await self.clear_state()
        if asyncio.current_task() is not self._task:
            await self.stop()

        symbol = self.get_symbol()
        if self.notifier and symbol:
            history = self.persistence.load_trade_history()
            await self.notifier.send_performance_stats(
                trade_history=history,
                symbol=symbol,
                channel_id=self.config.MAIN_CHANNEL_ID,
            )

    async def start(self) -> None:
        """Start periodic position status and hard-exit monitoring."""
        if self._task and not self._task.done():
            return

        self._task = asyncio.create_task(self._loop(), name="Position-Status-Updates")
        self.active_tasks.add(self._task)
        self._task.add_done_callback(self.active_tasks.discard)
        self.logger.debug("Started position status and exit monitor")

    async def stop(self) -> None:
        """Stop periodic position status and hard-exit monitoring."""
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

    async def load_state(self) -> dict[str, Any]:
        """Load persisted position monitor state."""
        return await self.exit_monitor.load_state(self.persistence)

    async def save_state(self, **timestamps: datetime) -> None:
        """Persist monitor config metadata plus timestamp updates."""
        symbol = self.get_symbol()
        if not symbol:
            self.logger.warning("Skipping position monitor state save because current symbol is unset")
            return
        await self.exit_monitor.save_state(self.persistence, symbol, **timestamps)

    async def clear_state(self) -> None:
        """Clear persisted position monitor state."""
        await self.exit_monitor.clear_state(self.persistence)

    async def run_hard_exit_checks(
        self,
        current_price: float | None,
        now: datetime,
        state: dict[str, Any],
    ) -> str | None:
        """Run due hard-exit checks and persist due timestamps."""
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
        return close_reason

    async def _loop(self) -> None:
        """Send position status updates and evaluate configured hard exits."""
        try:
            while self.is_running():
                state = await self.load_state()
                delay_seconds = self.exit_monitor.seconds_until_next_tick(state, datetime.now(timezone.utc))
                if delay_seconds > 0:
                    await self.interruptible_sleep(delay_seconds, respect_force_analysis=False)

                if not self.is_running():
                    break

                if not self.trading_strategy.current_position:
                    self.logger.debug("Position closed, stopping status updates")
                    break

                try:
                    now = datetime.now(timezone.utc)
                    state = await self.load_state()
                    ticker = await self.fetch_current_ticker()
                    current_price = float(ticker.get('last', ticker.get('close', 0))) if ticker else None

                    close_reason = await self.run_hard_exit_checks(current_price, now, state)
                    if close_reason:
                        await self.handle_position_closed(close_reason)
                        break

                    if self.notifier and self.trading_strategy.current_position and self.exit_monitor.is_status_due(now, state):
                        await self.notifier.send_position_status(
                            position=self.trading_strategy.current_position,
                            current_price=current_price if current_price is not None else 0.0,
                            channel_id=self.config.MAIN_CHANNEL_ID,
                        )
                        await self.save_state(last_status_sent_at=now)
                        self.logger.debug("Sent position status update")
                except Exception as e:
                    self.logger.warning("Error running position monitor update: %s", e)
        except asyncio.CancelledError:
            self.logger.debug("Position status loop cancelled")
            raise
