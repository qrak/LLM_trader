"""Shared dashboard state for real-time updates.

This module holds state that is updated by the trading bot and read by the dashboard.
It enables WebSocket broadcasts and API endpoints to share live data.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import asyncio
import time
from src.dashboard.routers.ws_router import broadcast


@dataclass
class DashboardState:
    """Shared state between bot and dashboard."""
    # pylint: disable=too-many-instance-attributes
    next_check_utc: datetime | None = None
    current_price: float | None = None
    _cache: dict[str, Any] = field(default_factory=dict)
    cache_timestamps: dict[str, float] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def update_price(self, price: float) -> None:
        """Update current price (no broadcast to avoid spam)."""
        async with self._lock:
            self.current_price = price

    async def update_next_check(self, next_time: datetime) -> None:
        """Update next check time and broadcast to clients."""
        async with self._lock:
            self.next_check_utc = next_time
        await self._broadcast({"type": "countdown", "next_check_utc": next_time.isoformat()})

    async def _broadcast(self, data: dict[str, Any]) -> None:
        """Broadcast data to all connected WebSocket clients."""
        await broadcast(data)

    def get_countdown_data(self) -> dict[str, Any]:
        """Get current countdown state for REST API."""
        if not self.next_check_utc:
            return {"next_check_utc": None, "seconds_remaining": None}
        now = datetime.now(timezone.utc)
        next_check = (
            self.next_check_utc.replace(tzinfo=timezone.utc)
            if self.next_check_utc.tzinfo is None
            else self.next_check_utc
        )
        remaining = (next_check - now).total_seconds()
        return {
            "next_check_utc": self.next_check_utc.isoformat(),
            "seconds_remaining": max(0, int(remaining))
        }

    def get_cached(self, key: str, ttl_seconds: float = 30.0) -> Any | None:
        """Retrieve a cached value if it is within TTL."""
        cached_time = self.cache_timestamps.get(key, 0)
        if time.time() - cached_time > ttl_seconds:
            return None
        return self._cache.get(key)

    def set_cached(self, key: str, value: Any) -> None:
        """Store a value in cache with current timestamp, enforcing max size."""
        if len(self._cache) >= 100 and key not in self._cache:
            if self.cache_timestamps:
                oldest_key = min(self.cache_timestamps, key=lambda cache_key: self.cache_timestamps[cache_key])
                self.invalidate_cache(oldest_key)
        self._cache[key] = value
        self.cache_timestamps[key] = time.time()

    def invalidate_cache(self, key: str) -> None:
        """Remove cached value."""
        self._cache.pop(key, None)
        self.cache_timestamps.pop(key, None)


dashboard_state = DashboardState()
