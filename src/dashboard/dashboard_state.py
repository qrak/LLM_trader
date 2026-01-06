"""Shared dashboard state for real-time updates.

This module holds state that is updated by the trading bot and read by the dashboard.
It enables WebSocket broadcasts and API endpoints to share live data.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, Set
import asyncio


@dataclass
class DashboardState:
    """Shared state between bot and dashboard."""
    next_check_utc: Optional[datetime] = None
    bot_status: str = "running"
    last_analysis_time: Optional[datetime] = None
    current_position: Optional[Dict[str, Any]] = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def update_next_check(self, next_time: datetime) -> None:
        """Update next check time and broadcast to clients."""
        async with self._lock:
            self.next_check_utc = next_time
        await self._broadcast({"type": "countdown", "next_check_utc": next_time.isoformat()})

    async def update_position(self, position_data: Optional[Dict[str, Any]]) -> None:
        """Update current position and broadcast to clients."""
        async with self._lock:
            self.current_position = position_data
        await self._broadcast({"type": "position", "data": position_data})

    async def update_analysis_complete(self) -> None:
        """Signal that analysis has completed."""
        async with self._lock:
            self.last_analysis_time = datetime.utcnow()
        await self._broadcast({"type": "analysis_complete"})

    async def _broadcast(self, data: Dict[str, Any]) -> None:
        """Broadcast data to all connected WebSocket clients."""
        from src.dashboard.routers.ws_router import broadcast
        await broadcast(data)

    def get_countdown_data(self) -> Dict[str, Any]:
        """Get current countdown state for REST API."""
        if not self.next_check_utc:
            return {"next_check_utc": None, "seconds_remaining": None}
        now = datetime.utcnow()
        remaining = (self.next_check_utc.replace(tzinfo=None) - now).total_seconds()
        return {
            "next_check_utc": self.next_check_utc.isoformat(),
            "seconds_remaining": max(0, int(remaining))
        }


dashboard_state = DashboardState()
