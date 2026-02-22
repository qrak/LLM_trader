import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from fastapi import WebSocket
from src.dashboard.routers import ws_router

@pytest.mark.asyncio
async def test_websocket_origin_check():
    # Setup mock config
    mock_config = MagicMock()
    mock_config.DASHBOARD_ENABLE_CORS = False
    mock_config.DASHBOARD_CORS_ORIGINS = []

    # Mock app state
    mock_app = MagicMock()
    mock_app.state.config = mock_config

    # 1. Test Valid Origin (Same Origin)
    ws_valid = AsyncMock(spec=WebSocket)
    ws_valid.app = mock_app
    ws_valid.headers = {"origin": "http://localhost:8000", "host": "localhost:8000"}
    ws_valid.client = MagicMock()
    ws_valid.client.host = "127.0.0.1"
    ws_valid.accept = AsyncMock()
    ws_valid.close = AsyncMock()
    # Make receive_text wait forever so connection stays open
    ws_valid.receive_text = AsyncMock(side_effect=asyncio.CancelledError)

    # 2. Test Invalid Origin (CSWSH attempt)
    ws_invalid = AsyncMock(spec=WebSocket)
    ws_invalid.app = mock_app
    ws_invalid.headers = {"origin": "http://malicious.com", "host": "localhost:8000"}
    ws_invalid.client = MagicMock()
    ws_invalid.client.host = "127.0.0.1"
    ws_invalid.accept = AsyncMock()
    ws_invalid.close = AsyncMock()
    ws_invalid.receive_text = AsyncMock(side_effect=asyncio.CancelledError)

    # Create router instance
    router_instance = ws_router.WebSocketRouter(ws_router.manager, mock_config, MagicMock())

    # Clear existing state
    router_instance.manager.active_connections.clear()
    router_instance.manager.ip_counts.clear()

    # Run endpoint for valid origin
    task_valid = asyncio.create_task(router_instance.websocket_endpoint(ws_valid))
    await asyncio.sleep(0.05)

    # Run endpoint for invalid origin
    task_invalid = asyncio.create_task(router_instance.websocket_endpoint(ws_invalid))
    await asyncio.sleep(0.05)

    # Verify Valid Connection
    # It should have called accept()
    assert ws_valid.accept.called, "Valid origin should be accepted"

    # Verify Invalid Connection
    # It should NOT have called accept()
    assert not ws_invalid.accept.called, "Invalid origin should NOT be accepted"
    # It should have called close with 1008
    assert ws_invalid.close.called
    # Check close code if possible, typically close(code=1008, ...)
    # call_args is (args, kwargs)
    # ws_invalid.close.call_args[1].get('code') or ws_invalid.close.call_args[0][0]

    # Cleanup
    task_valid.cancel()
    task_invalid.cancel()

@pytest.mark.asyncio
async def test_websocket_cors_allowed():
    # Setup mock config with CORS enabled
    mock_config = MagicMock()
    mock_config.DASHBOARD_ENABLE_CORS = True
    mock_config.DASHBOARD_CORS_ORIGINS = ["http://trusted.com"]

    # Mock app state
    mock_app = MagicMock()
    mock_app.state.config = mock_config

    # Test Allowed CORS Origin
    ws_cors = AsyncMock(spec=WebSocket)
    ws_cors.app = mock_app
    ws_cors.headers = {"origin": "http://trusted.com", "host": "localhost:8000"}
    ws_cors.client = MagicMock()
    ws_cors.client.host = "127.0.0.1"
    ws_cors.accept = AsyncMock()
    ws_cors.close = AsyncMock()
    ws_cors.receive_text = AsyncMock(side_effect=asyncio.CancelledError)

    router_instance = ws_router.WebSocketRouter(ws_router.manager, mock_config, MagicMock())

    # Clear state
    router_instance.manager.active_connections.clear()
    router_instance.manager.ip_counts.clear()

    task_cors = asyncio.create_task(router_instance.websocket_endpoint(ws_cors))
    await asyncio.sleep(0.05)

    assert ws_cors.accept.called, "Trusted CORS origin should be accepted"

    task_cors.cancel()
