
import pytest
from unittest.mock import MagicMock, patch, mock_open
import json

from fastapi.testclient import TestClient

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
    mock_state.analysis_engine = MagicMock()
    return mock_state

@pytest.fixture
def client(mock_app_state):
    # We need to patch the router into an app
    from fastapi import FastAPI
    from src.dashboard.routers.monitor import MonitorRouter

    app = FastAPI()
    router_instance = MonitorRouter(
        config=mock_app_state.config,
        logger=mock_app_state.logger,
        dashboard_state=mock_app_state.dashboard_state,
        analysis_engine=mock_app_state.analysis_engine,
        rag_engine=mock_app_state.rag_engine
    )
    app.include_router(router_instance.router)
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

@pytest.mark.asyncio
async def test_get_api_costs(client, mock_app_state):
    # Setup CostStorage mock inside the route
    with patch("src.dashboard.routers.monitor.CostStorage") as mock_storage:
        mock_instance = mock_storage.return_value
        
        # Mock provider costs return
        mock_provider_costs = MagicMock()
        mock_provider_costs.total_cost = 1.50
        mock_instance.get_provider_costs.return_value = mock_provider_costs
        
        response = client.get("/api/monitor/costs")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_session_cost"] == 3.00 # 1.50 for openrouter + 1.50 for google
        assert data["costs_by_provider"]["openrouter"] == 1.50
        mock_app_state.dashboard_state.set_cached.assert_called()

@pytest.mark.asyncio
async def test_get_system_prompt(client, mock_app_state):
    # Mock analysis_engine having a last_system_prompt
    mock_app_state.analysis_engine.last_system_prompt = "You are a TRADING BRAIN"
    
    response = client.get("/api/monitor/system_prompt")
    
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "memory"
    assert data["has_brain_context"] == True
    assert data["system_prompt"] == "You are a TRADING BRAIN"

@pytest.mark.asyncio
async def test_get_last_prompt(client, mock_app_state):
    mock_app_state.analysis_engine.last_generated_prompt = "Analyze BTC"
    mock_app_state.analysis_engine.last_prompt_timestamp = "2023-01-01T12:00:00"
    
    response = client.get("/api/monitor/last_prompt")
    
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "memory"
    assert data["prompt"] == "Analyze BTC"
    assert data["timestamp"] == "2023-01-01T12:00:00"

@pytest.mark.asyncio
async def test_get_last_response(client, mock_app_state):
    mock_app_state.analysis_engine.last_llm_response = "Market is bullish"
    mock_app_state.analysis_engine.last_response_timestamp = "2023-01-01T12:01:00"
    
    response = client.get("/api/monitor/last_response")
    
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "memory"
    assert data["response"] == "Market is bullish"
    assert data["timestamp"] == "2023-01-01T12:01:00"
