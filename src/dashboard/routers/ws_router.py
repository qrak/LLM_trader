"""WebSocket router for real-time dashboard updates."""

from typing import Set, Dict, Any
from collections import defaultdict
from urllib.parse import urlparse

from fastapi import APIRouter, WebSocket, WebSocketDisconnect


class ConnectionManager:
    """Manages WebSocket connections with rate limiting."""

    def __init__(self, max_connections: int = 1000, max_per_ip: int = 10):
        self.active_connections: Set[WebSocket] = set()
        self.ip_counts: Dict[str, int] = defaultdict(int)
        self.max_connections = max_connections
        self.max_per_ip = max_per_ip

    async def connect(self, websocket: WebSocket) -> bool:
        """Accept connection if limits allow."""
        # Check global limit
        if len(self.active_connections) >= self.max_connections:
            # 1013: Try Again Later
            await websocket.close(code=1013, reason="Server busy")
            return False

        # Check per-IP limit
        client_ip = websocket.client.host if websocket.client else "unknown"
        if self.ip_counts[client_ip] >= self.max_per_ip:
            # 1008: Policy Violation
            await websocket.close(code=1008, reason="Rate limit exceeded")
            return False

        # Pre-increment counters to prevent race conditions during await accept()
        self.active_connections.add(websocket)
        self.ip_counts[client_ip] += 1

        try:
            await websocket.accept()
            return True
        except Exception:
            # Rollback if accept fails
            self.disconnect(websocket)
            return False

    def disconnect(self, websocket: WebSocket):
        """Remove connection and update counts."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            client_ip = websocket.client.host if websocket.client else "unknown"
            self.ip_counts[client_ip] -= 1
            if self.ip_counts[client_ip] <= 0:
                del self.ip_counts[client_ip]

    async def broadcast(self, data: Dict[str, Any]):
        """Broadcast data to all active connections."""
        for connection in list(self.active_connections):
            try:
                await connection.send_json(data)
            except Exception:
                self.disconnect(connection)


manager = ConnectionManager()

# Legacy reference for  compatibility (read-only usage recommended)
connected_clients = manager.active_connections


async def broadcast(data: Dict[str, Any]) -> None:
    """Broadcast data to all connected WebSocket clients."""
    await manager.broadcast(data)


class WebSocketRouter:
    """Router for WebSocket connections."""
    def __init__(self, manager_instance, config, dashboard_state):
        self.router = APIRouter(tags=["websocket"])
        self.manager = manager_instance
        self.config = config
        self.dashboard_state = dashboard_state

        self.router.add_api_websocket_route("/ws", self.websocket_endpoint)
        self.router.add_api_route("/api/status/countdown", self.get_countdown, methods=["GET"])

    async def websocket_endpoint(self, websocket: WebSocket):
        """WebSocket endpoint for real-time dashboard updates."""
        # Security: Validate Origin to prevent CSWSH
        origin = websocket.headers.get("origin")
        if origin:
            allowed = False

            # 1. Allow if matches Host (Same Origin)
            host = websocket.headers.get("host")
            if host:
                # Robustly extract host from origin
                try:
                    parsed = urlparse(origin)
                    origin_host = parsed.netloc
                except Exception:
                    origin_host = None

                if origin_host == host:
                    allowed = True

            # 2. Allow if explicitly configured in CORS
            if not allowed and self.config and self.config.DASHBOARD_ENABLE_CORS:
                cors_origins = self.config.DASHBOARD_CORS_ORIGINS
                if "*" in cors_origins or origin in cors_origins:
                    allowed = True

            if not allowed:
                await websocket.close(code=1008, reason="Invalid Origin")
                return

        if await self.manager.connect(websocket):
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.manager.disconnect(websocket)
            except Exception:
                self.manager.disconnect(websocket)

    async def get_countdown(self) -> Dict[str, Any]:
        """Get countdown to next analysis (REST fallback for WebSocket)."""
        if self.dashboard_state:
            return self.dashboard_state.get_countdown_data()
        return {"next_check_utc": None, "seconds_remaining": None}
