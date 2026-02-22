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
        ws.headers = {}
        ws.accept = AsyncMock()
        ws.close = AsyncMock()
        # Keep connection alive until cancelled
        ws.receive_text = AsyncMock(side_effect=asyncio.CancelledError)
        websockets.append(ws)

    # Create router instance with mocked dependencies
    mock_config = MagicMock()
    mock_config.DASHBOARD_ENABLE_CORS = False
    mock_dashboard = MagicMock()
    router_instance = ws_router.WebSocketRouter(ws_router.manager, mock_config, mock_dashboard)

    # Clear existing state cleanly
    router_instance.manager.active_connections.clear()
    router_instance.manager.ip_counts.clear()

    # Connect all
    tasks = []
    for ws in websockets:
        task = asyncio.create_task(router_instance.websocket_endpoint(ws))
        tasks.append(task)

    await asyncio.sleep(0.1)

    # Check results
    assert len(router_instance.manager.active_connections) == 10
    # The other 5 should have been closed
    closed_count = sum(1 for ws in websockets if ws.close.called)
    assert closed_count == 5

    # Cleanup
    for task in tasks:
        task.cancel()
