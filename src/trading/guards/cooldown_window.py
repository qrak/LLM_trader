"""Cooldown Window Guard — prevents rapid double-execution of trades."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from . import GuardResult

if TYPE_CHECKING:
    from src.managers.persistence_manager import PersistenceManager


class CooldownWindowGuard:
    """Reject intents that arrive too soon after the last executed order.

    The cooldown duration is derived from the configured timeframe to
    prevent the bot from opening positions on every single analysis cycle
    without time for price action to develop.

    Cooldown scaling:
      - Scalping (< 1h): 4x timeframe
      - Intraday (1h-3h): 3x timeframe
      - Swing (4h-12h): 2x timeframe
      - Position (1D+): 1x timeframe

    The last execution timestamp is cached in-process after the first
    read to avoid I/O on every guard evaluation. Call ``invalidate_cache()``
    after a new trade is executed to force a re-read.
    """

    name = "cooldown_window"

    def __init__(self, persistence: "PersistenceManager" | None = None) -> None:
        self.persistence = persistence
        self._cached_timestamp: datetime | None = None
        self._cache_populated: bool = False

    def invalidate_cache(self) -> None:
        """Force the next ``check()`` call to re-read from persistence."""
        self._cache_populated = False

    def check(self, intent, /, *, capital: float, config) -> GuardResult:
        if self.persistence is None:
            return GuardResult(
                guard_name=self.name,
                passed=False,
                reason="Cooldown guard is not wired with persistence (fail-closed)",
                metadata={"error": "persistence_not_configured"},
            )

        try:
            last_execution_timestamp = self._get_last_execution_timestamp(config)
        except RuntimeError as exc:
            return GuardResult(
                guard_name=self.name,
                passed=False,
                reason="Cooldown guard could not read execution history (fail-closed)",
                metadata={"error": str(exc)},
            )

        if last_execution_timestamp is None:
            return GuardResult(
                guard_name=self.name,
                passed=True,
                reason="No prior execution — cooldown not applicable",
                metadata={},
            )

        cooldown_minutes = self._compute_cooldown_minutes(config)
        elapsed = (datetime.now(timezone.utc) - last_execution_timestamp).total_seconds() / 60.0

        if elapsed < cooldown_minutes:
            remaining = cooldown_minutes - elapsed
            return GuardResult(
                guard_name=self.name,
                passed=False,
                reason=(
                    f"Cooldown active: {elapsed:.1f}m elapsed, "
                    f"need {cooldown_minutes:.0f}m ({remaining:.1f}m remaining)"
                ),
                metadata={
                    "elapsed_minutes": round(elapsed, 1),
                    "cooldown_minutes": cooldown_minutes,
                    "remaining_minutes": round(remaining, 1),
                    "last_execution": last_execution_timestamp.isoformat(),
                },
            )

        # Cooldown expired — invalidate cache so next execution re-reads the
        # (now-stale) timestamp, which will be ahead of this check.
        self.invalidate_cache()

        return GuardResult(
            guard_name=self.name,
            passed=True,
            reason=(
                f"Cooldown expired: {elapsed:.1f}m elapsed "
                f">= {cooldown_minutes:.0f}m required"
            ),
            metadata={
                "elapsed_minutes": round(elapsed, 1),
                "cooldown_minutes": cooldown_minutes,
                "last_execution": last_execution_timestamp.isoformat(),
            },
        )

    def _get_last_execution_timestamp(self, config) -> datetime | None:
        """Retrieve the timestamp of the last executed buy/sell decision.

        Uses a process-local cache; only re-reads from disk after an
        explicit invalidation or on first call.

        Subclasses or integration layers can override this to read from
        persistence, trade history, or in-memory state. The default
        implementation reads from TradeHistory in the data directory.
        """
        if self._cache_populated:
            return self._cached_timestamp

        last_ts = self._read_last_execution_from_persistence(config)
        self._cached_timestamp = last_ts
        self._cache_populated = True
        return last_ts

    def _read_last_execution_from_persistence(self, config) -> datetime | None:
        """Read the last BUY/SELL timestamp from persistence (SQLite-backed)."""
        if self.persistence is None:
            return None

        try:
            return self.persistence.get_last_execution_timestamp(actions=("BUY", "SELL"))
        except Exception as exc:  # pragma: no cover - defensive fail-closed path
            raise RuntimeError("Cooldown guard could not read execution history") from exc

    @staticmethod
    def _compute_cooldown_minutes(config) -> float:
        """Derive cooldown duration from configured timeframe."""
        timeframe = config.TIMEFRAME
        try:
            from src.utils.timeframe_validator import TimeframeValidator
            tf_minutes = TimeframeValidator.to_minutes(timeframe)
        except Exception:
            # Fallback: parse common formats
            tf_minutes = _fallback_tf_to_minutes(timeframe)

        if tf_minutes < 60:
            return tf_minutes * 4
        if tf_minutes < 240:
            return tf_minutes * 3
        if tf_minutes < 1440:
            return tf_minutes * 2
        return tf_minutes


def _fallback_tf_to_minutes(timeframe: str) -> int:
    """Best-effort timeframe string to minutes conversion."""
    timeframe = str(timeframe).strip().lower()
    multipliers = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    for suffix, mult in multipliers.items():
        if timeframe.endswith(suffix):
            try:
                return int(timeframe[:-1]) * mult
            except ValueError:
                pass
    return 240  # default to 4h
