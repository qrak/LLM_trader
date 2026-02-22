import pytest
import json
from unittest.mock import MagicMock, patch, mock_open
from fastapi.testclient import TestClient

@pytest.fixture
def mock_app_state():
    mock_state = MagicMock()
    mock_state.config = MagicMock()
    mock_state.config.DATA_DIR = "test_data_dir"
    mock_state.logger = MagicMock()
    mock_state.dashboard_state = MagicMock()
    mock_state.dashboard_state.get_cached.return_value = None
    return mock_state

@pytest.fixture
def client(mock_app_state):
    from fastapi import FastAPI
    from src.dashboard.routers.performance import PerformanceRouter

    app = FastAPI()
    router_instance = PerformanceRouter(
        config=mock_app_state.config,
        logger=mock_app_state.logger,
        dashboard_state=mock_app_state.dashboard_state
    )
    app.include_router(router_instance.router)
    return TestClient(app)

def test_get_statistics(client, mock_app_state):
    stats_content = {"initial_capital": 10000.0, "current_capital": 15000.0}

    with patch("src.dashboard.routers.performance.Path") as mock_path:
        mock_path_instance = mock_path.return_value
        mock_path_instance.__truediv__.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True

        with patch("builtins.open", mock_open(read_data=json.dumps(stats_content))):
            response = client.get("/api/performance/stats")

            assert response.status_code == 200
            data = response.json()
            assert data["initial_capital"] == 10000.0
            assert data["current_capital"] == 15000.0
            mock_app_state.dashboard_state.set_cached.assert_called_with("statistics", stats_content)

def test_get_performance_history(client, mock_app_state):
    stats_content = {"initial_capital": 10000.0}
    trades_content = [
        {"timestamp": "2023-01-01T10:00:00Z", "action": "BUY", "price": 50000},
        {"timestamp": "2023-01-01T11:00:00Z", "action": "SELL", "price": 51000}
    ]

    with patch("src.dashboard.routers.performance.Path") as mock_path:
        mock_path_instance = mock_path.return_value
        mock_path_instance.__truediv__.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True

        # Need to mock two file reads sequentially: stats then trades
        mock_file = mock_open()
        mock_file.side_effect = [
            mock_open(read_data=json.dumps(stats_content)).return_value,
            mock_open(read_data=json.dumps(trades_content)).return_value
        ]

        with patch("builtins.open", mock_file):
            response = client.get("/api/performance/history")

            assert response.status_code == 200
            data = response.json()
            assert "history" in data
            assert len(data["history"]) == 2
            assert data["history"][0]["action"] == "BUY"
            assert data["history"][1]["action"] == "SELL"
