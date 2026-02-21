
import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import json

from fastapi.testclient import TestClient
from src.dashboard.server import DashboardServer

# Mock dependencies
@pytest.fixture
def mock_app_state():
    mock_state = MagicMock()
    mock_state.config = MagicMock()
    mock_state.config.DATA_DIR = "data"
    mock_state.logger = MagicMock()
    mock_state.rag_engine = None # Simplify for disk fallback test
    mock_state.dashboard_state = MagicMock()
    mock_state.dashboard_state.get_cached.return_value = None
    return mock_state

@pytest.fixture
def client(mock_app_state):
    # We need to patch the router into an app
    from fastapi import FastAPI
    from src.dashboard.routers.monitor import router

    app = FastAPI()
    app.include_router(router)
    app.state = mock_app_state
    return TestClient(app)

def test_get_news_from_disk(client, mock_app_state):
    # Setup mock file content
    news_content = {
        "articles": [
            {"title": "Bitcoin Hits $100k", "url": "http://example.com"}
        ]
    }

    # We need to mock Path.exists and open()
    # The code iterates: ["crypto_news.json", "news_cache/recent_news.json"]

    with patch("src.dashboard.routers.monitor.Path") as MockPath:
        # Mock file existence
        mock_path_instance = MockPath.return_value
        mock_path_instance.__truediv__.return_value = mock_path_instance

        # Scenario: first file exists
        mock_path_instance.exists.return_value = True

        with patch("builtins.open", mock_open(read_data=json.dumps(news_content))):
            response = client.get("/api/monitor/news")

            assert response.status_code == 200
            data = response.json()
            assert data["count"] == 1
            assert data["articles"][0]["title"] == "Bitcoin Hits $100k"

            # Verify caching was called
            mock_app_state.dashboard_state.set_cached.assert_called()

def test_get_news_fallback(client, mock_app_state):
    # Scenario: first file missing, second exists
    news_content = [{"title": "Second Source", "url": "http://test.com"}] # format can be list too based on code

    with patch("src.dashboard.routers.monitor.Path") as MockPath:
        mock_path_instance = MockPath.return_value
        # Use side_effect for exists to simulate first missing, second present
        # Note: logic: Path(data_dir) / news_path
        # The code creates a NEW Path object for each iteration

        # We need to be careful with how Path is mocked.
        # It's easier to mock os.path.exists if it was used, but here it's Path.exists.

        # Let's inspect the code again:
        # news_file = Path(data_dir) / news_path
        # if news_file.exists(): ...

        # If we mock Path, every instantiation returns the same mock object (usually).
        # We can control return values based on call args if we mock the __truediv__ properly.

        def truediv_side_effect(other):
            m = MagicMock()
            if str(other) == "crypto_news.json":
                m.exists.return_value = False
            elif str(other) == "news_cache/recent_news.json":
                m.exists.return_value = True
            return m

        mock_path_instance.__truediv__.side_effect = truediv_side_effect

        with patch("builtins.open", mock_open(read_data=json.dumps(news_content))):
            response = client.get("/api/monitor/news")

            assert response.status_code == 200
            data = response.json()
            assert data["count"] == 1
            assert data["articles"][0]["title"] == "Second Source"
