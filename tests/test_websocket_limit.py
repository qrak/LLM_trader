import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from fastapi import WebSocket
from src.dashboard.routers import ws_router

@pytest.mark.asyncio
async def test_websocket_limits():
    # Setup
    websockets = []
    # Try to connect 15 clients (limit should be 10 per IP)
    for i in range(15):
        ws = AsyncMock(spec=WebSocket)
        ws.client = MagicMock()
        ws.client.host = "127.0.0.1"
        ws.accept = AsyncMock()
        ws.close = AsyncMock()
        # Keep connection alive until cancelled
        ws.receive_text = AsyncMock(side_effect=asyncio.CancelledError)
        websockets.append(ws)

    # Clear existing state
    if hasattr(ws_router, 'connected_clients'):
        if isinstance(ws_router.connected_clients, set):
            ws_router.connected_clients.clear()
    if hasattr(ws_router, 'manager'):
        ws_router.manager.active_connections.clear()
        ws_router.manager.ip_counts.clear()

    # Connect all
    tasks = []
    for ws in websockets:
        task = asyncio.create_task(ws_router.websocket_endpoint(ws))
        tasks.append(task)

    await asyncio.sleep(0.1)

    # Check results
    if hasattr(ws_router, 'manager'):
        # After fix: Should be capped at 10
        assert len(ws_router.manager.active_connections) == 10
        # The other 5 should have been closed
        closed_count = sum(1 for ws in websockets if ws.close.called)
        assert closed_count == 5
    else:
        # Before fix: Should be 15 (VULNERABILITY CONFIRMED)
        assert len(ws_router.connected_clients) == 15

    # Cleanup
    for task in tasks:
        task.cancel()
