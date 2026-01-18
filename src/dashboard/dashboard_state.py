"""Shared dashboard state for real-time updates.

This module holds state that is updated by the trading bot and read by the dashboard.
It enables WebSocket broadcasts and API endpoints to share live data.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import asyncio


@dataclass
class DashboardState:
    """Shared state between bot and dashboard."""
    next_check_utc: Optional[datetime] = None
    bot_status: str = "running"
    last_analysis_time: Optional[datetime] = None
    current_position: Optional[Dict[str, Any]] = None
    current_price: Optional[float] = None
    api_costs: Dict[str, float] = field(default_factory=lambda: {"openrouter": 0.0, "google": 0.0, "lmstudio": 0.0})
    last_request_cost: Optional[float] = None
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

    async def update_position(self, position_data: Optional[Dict[str, Any]]) -> None:
        """Update current position and broadcast to clients."""
        async with self._lock:
            self.current_position = position_data
        await self._broadcast({"type": "position", "data": position_data})

    async def update_analysis_complete(self) -> None:
        """Signal that analysis has completed."""
        async with self._lock:
            self.last_analysis_time = datetime.now(timezone.utc)
        await self._broadcast({"type": "analysis_complete"})

    async def update_api_costs(self, provider: str, cost: float) -> None:
        """Update API costs for a provider and broadcast to clients."""
        async with self._lock:
            if provider in self.api_costs:
                self.api_costs[provider] += cost
            self.last_request_cost = cost
        await self._broadcast({"type": "cost_update", "provider": provider, "cost": cost, "total": self.api_costs})

    async def reset_api_costs(self) -> None:
        """Reset all API costs to zero."""
        async with self._lock:
            self.api_costs = {"openrouter": 0.0, "google": 0.0, "lmstudio": 0.0}
            self.last_request_cost = None
        await self._broadcast({"type": "cost_reset", "total": self.api_costs})

    async def _broadcast(self, data: Dict[str, Any]) -> None:
        """Broadcast data to all connected WebSocket clients."""
        from src.dashboard.routers.ws_router import broadcast
        await broadcast(data)

    def get_countdown_data(self) -> Dict[str, Any]:
        """Get current countdown state for REST API."""
        if not self.next_check_utc:
            return {"next_check_utc": None, "seconds_remaining": None}
        now = datetime.now(timezone.utc)
        remaining = (self.next_check_utc.replace(tzinfo=timezone.utc) if self.next_check_utc.tzinfo is None else self.next_check_utc - now).total_seconds()
        return {
            "next_check_utc": self.next_check_utc.isoformat(),
            "seconds_remaining": max(0, int(remaining))
        }

    def get_cost_data(self) -> Dict[str, Any]:
        """Get current API cost data for REST API."""
        total = sum(self.api_costs.values())
        return {
            "costs_by_provider": self.api_costs.copy(),
            "total_session_cost": total,
            "last_request_cost": self.last_request_cost,
            "formatted_total": f"${total:.6f}" if total > 0 else "Free"
        }


dashboard_state = DashboardState()

