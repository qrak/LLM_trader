import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import time

# Adjust import path if necessary, but pytest usually handles it if run from root
from src.dashboard.server import DashboardServer

@pytest.fixture
def mock_dependencies():
    config = MagicMock()
    config.DASHBOARD_ENABLE_CORS = False
    config.DASHBOARD_CORS_ORIGINS = []

    logger = MagicMock()

    return {
        "brain_service": MagicMock(),
        "vector_memory": MagicMock(),
        "analysis_engine": MagicMock(),
        "config": config,
        "logger": logger,
        "unified_parser": MagicMock(),
        "persistence": MagicMock(),
        "exchange_manager": MagicMock(),
    }

def test_dos_prevention_inactive_ips(mock_dependencies):
    """Test that inactive IPs are cleaned up when limit is reached."""
    server = DashboardServer(**mock_dependencies)
    app = server.app
    client = TestClient(app)

    # Access exposed request_counts
    if not hasattr(app.state, 'request_counts'):
        pytest.fail("app.state.request_counts not exposed. Did you modify server.py?")

    request_counts = app.state.request_counts
    MAX_UNIQUE_IPS = 10000

    # Populate with > MAX inactive entries (empty lists)
    for i in range(MAX_UNIQUE_IPS + 100):
        request_counts[f"mock_ip_{i}"] = []

    assert len(request_counts) > MAX_UNIQUE_IPS

    # Trigger a request to an API endpoint
    # We use a non-static path to trigger middleware
    client.get("/api/status")

    # Expect cleanup to have removed the inactive IPs
    # The current request IP might be added
    assert len(request_counts) <= MAX_UNIQUE_IPS
    # Should be very low (just the test client IP)
    assert len(request_counts) < 50

def test_dos_prevention_emergency_clear(mock_dependencies):
    """Test that emergency clear works when too many active IPs exist."""
    server = DashboardServer(**mock_dependencies)
    app = server.app
    client = TestClient(app)
    request_counts = app.state.request_counts
    MAX_UNIQUE_IPS = 10000

    # Populate with > MAX ACTIVE entries (fresh timestamps)
    current_time = time.time()
    for i in range(MAX_UNIQUE_IPS + 100):
        request_counts[f"active_ip_{i}"] = [current_time]

    assert len(request_counts) > MAX_UNIQUE_IPS

    # Trigger request
    client.get("/api/status")

    # With the improved rate limiter, we use FIFO eviction (LRU-like) instead of Emergency Clear
    # so we maintain capacity at MAX_UNIQUE_IPS rather than wiping history.
    # The length will be MAX_UNIQUE_IPS (after eviction) + 1 (new request added)
    assert len(request_counts) >= MAX_UNIQUE_IPS
    assert len(request_counts) <= MAX_UNIQUE_IPS + 5  # Allow small buffer
