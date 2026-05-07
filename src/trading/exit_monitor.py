"""Exit monitoring configuration and cadence helpers."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol

from src.utils.timeframe_validator import TimeframeValidator


class PositionExitStrategy(Protocol):
    """Trading strategy surface required by exit monitoring."""
    current_position: Any

    async def check_position(self, current_price: float) -> Optional[str]: ...

    async def check_stop_loss(self, current_price: float) -> Optional[str]: ...

    async def check_take_profit(self, current_price: float) -> Optional[str]: ...


class ExitMonitor:
    """Keeps SL/TP monitor configuration and persisted cadence logic out of app orchestration."""

    VALID_EXIT_TYPES = {"soft", "hard"}
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

    def __init__(self, config: Any, timeframe: str, default_status_interval_seconds: int) -> None:
        self.config = config
        self.timeframe = timeframe
        self.default_status_interval_seconds = default_status_interval_seconds

    @classmethod
    def exit_kinds(cls) -> tuple[str, str]:
        """Return supported exit kinds in enforcement priority order."""
        return cls.STOP_LOSS, cls.TAKE_PROFIT

    @staticmethod
    def last_check_key(exit_kind: str) -> str:
        """Return persisted timestamp key for an exit kind."""
        return "last_stop_loss_check_at" if exit_kind == ExitMonitor.STOP_LOSS else "last_take_profit_check_at"

    @staticmethod
    def _normalize_exit_type(value: Any, label: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in ExitMonitor.VALID_EXIT_TYPES:
            raise ValueError(f"Invalid {label} type '{value}'. Expected soft or hard.")
        return normalized

    @staticmethod
    def _parse_monitor_timestamp(value: Any) -> Optional[datetime]:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def seconds_until_due(last_check: Optional[datetime], interval_seconds: int, now: datetime) -> float:
        """Calculate how many seconds remain until an interval is due."""
        if last_check is None:
            return 0.0
        elapsed = (now - last_check).total_seconds()
        return max(0.0, interval_seconds - elapsed)

    def validate(self) -> None:
        """Validate all configured exit modes and intervals against the active timeframe."""
        for exit_kind in self.exit_kinds():
            self.exit_type(exit_kind)
            self._validate_interval(self.interval(exit_kind), exit_kind)

    def exit_type(self, exit_kind: str) -> str:
        """Return normalized exit execution type for SL or TP."""
        if exit_kind == self.STOP_LOSS:
            return self._normalize_exit_type(self.config.STOP_LOSS_TYPE, "stop loss")
        return self._normalize_exit_type(self.config.TAKE_PROFIT_TYPE, "take profit")

    def interval(self, exit_kind: str) -> str:
        """Return configured monitor interval label for an exit kind."""
        if exit_kind == self.STOP_LOSS:
            return str(self.config.STOP_LOSS_CHECK_INTERVAL).strip().lower()
        return str(self.config.TAKE_PROFIT_CHECK_INTERVAL).strip().lower()

    def interval_seconds(self, exit_kind: str) -> int:
        """Return configured monitor interval in seconds for an exit kind."""
        if exit_kind == self.STOP_LOSS:
            return self.config.STOP_LOSS_CHECK_INTERVAL_SECONDS
        return self.config.TAKE_PROFIT_CHECK_INTERVAL_SECONDS

    def status_interval_seconds(self) -> int:
        """Use the fastest hard-exit interval for status cadence, else the legacy interval."""
        hard_intervals = [
            self.interval_seconds(exit_kind)
            for exit_kind in self.exit_kinds()
            if self.exit_type(exit_kind) == "hard"
        ]
        return min(hard_intervals) if hard_intervals else self.default_status_interval_seconds

    def is_due(self, exit_kind: str, now: datetime, state: Dict[str, Any]) -> bool:
        """Return whether a hard exit check is due."""
        if self.exit_type(exit_kind) != "hard":
            return False
        last_check = self._parse_monitor_timestamp(state.get(self.last_check_key(exit_kind)))
        return self.seconds_until_due(last_check, self.interval_seconds(exit_kind), now) <= 0

    def due_hard_exits(self, now: datetime, state: Dict[str, Any]) -> list[str]:
        """Return hard exits that should be checked now."""
        return [exit_kind for exit_kind in self.exit_kinds() if self.is_due(exit_kind, now, state)]

    def is_status_due(self, now: datetime, state: Dict[str, Any]) -> bool:
        """Return whether a periodic status update is due."""
        last_status_sent_at = self._parse_monitor_timestamp(state.get("last_status_sent_at"))
        return self.seconds_until_due(last_status_sent_at, self.status_interval_seconds(), now) <= 0

    async def check_soft_exits(self, strategy: PositionExitStrategy, current_price: float, close_lock: Any) -> Optional[str]:
        """Evaluate only exits configured as soft at candle close."""
        async with close_lock:
            if not strategy.current_position:
                return None

            stop_loss_type = self.exit_type(self.STOP_LOSS)
            take_profit_type = self.exit_type(self.TAKE_PROFIT)

            if stop_loss_type == "soft" and take_profit_type == "soft":
                return await strategy.check_position(current_price)
            if stop_loss_type == "soft":
                return await strategy.check_stop_loss(current_price)
            if take_profit_type == "soft":
                return await strategy.check_take_profit(current_price)

        return None

    async def check_hard_exits(
        self,
        strategy: PositionExitStrategy,
        current_price: Optional[float],
        now: datetime,
        state: Dict[str, Any],
        close_lock: Any,
    ) -> tuple[Optional[str], Dict[str, datetime]]:
        """Evaluate due hard exits and return close reason plus timestamp updates."""
        if current_price is None:
            return None, {}

        close_reason = None
        timestamps: Dict[str, datetime] = {}
        due_exits = self.due_hard_exits(now, state)
        async with close_lock:
            if not strategy.current_position:
                return None, {}

            if self.STOP_LOSS in due_exits and self.TAKE_PROFIT in due_exits:
                close_reason = await strategy.check_position(current_price)
                timestamps[self.last_check_key(self.STOP_LOSS)] = now
                timestamps[self.last_check_key(self.TAKE_PROFIT)] = now
                return close_reason, timestamps

            for exit_kind in due_exits:
                if exit_kind == self.STOP_LOSS:
                    close_reason = await strategy.check_stop_loss(current_price)
                else:
                    close_reason = await strategy.check_take_profit(current_price)
                timestamps[self.last_check_key(exit_kind)] = now
                if close_reason or not strategy.current_position:
                    break

        return close_reason, timestamps

    def seconds_until_next_tick(self, state: Dict[str, Any], now: datetime) -> float:
        """Return delay until next status or hard-exit check is due."""
        delays = [
            self.seconds_until_due(
                self._parse_monitor_timestamp(state.get("last_status_sent_at")),
                self.status_interval_seconds(),
                now,
            )
        ]
        for exit_kind in self.exit_kinds():
            if self.exit_type(exit_kind) == "hard":
                delays.append(
                    self.seconds_until_due(
                        self._parse_monitor_timestamp(state.get(self.last_check_key(exit_kind))),
                        self.interval_seconds(exit_kind),
                        now,
                    )
                )
        return min(delays) if delays else self.default_status_interval_seconds

    async def load_state(self, persistence: Any) -> Dict[str, Any]:
        """Load persisted monitor state."""
        return await persistence.async_load_position_monitor_state()

    async def save_state(self, persistence: Any, symbol: str, **timestamps: datetime) -> None:
        """Save monitor config metadata plus any timestamp updates."""
        state = await self.load_state(persistence)
        state.update({
            "symbol": symbol,
            "stop_loss_type": self.exit_type(self.STOP_LOSS),
            "stop_loss_check_interval": self.interval(self.STOP_LOSS),
            "take_profit_type": self.exit_type(self.TAKE_PROFIT),
            "take_profit_check_interval": self.interval(self.TAKE_PROFIT),
        })
        for key, value in timestamps.items():
            if value is not None:
                state[key] = value.astimezone(timezone.utc).isoformat()
        await persistence.async_save_position_monitor_state(state)

    async def clear_state(self, persistence: Any) -> None:
        """Clear persisted monitor state."""
        await persistence.async_clear_position_monitor_state()

    def _validate_interval(self, interval: Any, exit_kind: str) -> int:
        label = "stop loss" if exit_kind == self.STOP_LOSS else "take profit"
        minutes = TimeframeValidator.parse_period_to_minutes(str(interval).strip().lower())
        if minutes <= 0:
            raise ValueError(f"{label} interval must be positive")
        timeframe_minutes = TimeframeValidator.to_minutes(self.timeframe)
        if minutes > timeframe_minutes:
            raise ValueError(f"{label} interval '{interval}' must not be greater than timeframe '{self.timeframe}'")
        return minutes
