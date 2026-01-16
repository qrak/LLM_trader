
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.loader import Config
from src.dashboard.server import DashboardServer

def test_config_loader_dashboard_defaults():
    # Mock configparser to return empty config
    with patch("configparser.ConfigParser") as MockParser:
        mock_parser_instance = MockParser.return_value
        mock_parser_instance.sections.return_value = []
        mock_parser_instance.items.return_value = []

        # Mock .env loading
        with patch("src.config.loader.dotenv_values", return_value={}):
             # Mock existence of files
            with patch("pathlib.Path.exists", return_value=True):
                # We need to suppress _build_model_configs validation or provide mock data
                with patch.object(Config, "_build_model_configs"):
                    config = Config()

                # Check defaults
                assert config.DASHBOARD_HOST == "0.0.0.0"
                assert config.DASHBOARD_PORT == 8000
                assert config.DASHBOARD_ENABLE_CORS == False
                assert config.DASHBOARD_CORS_ORIGINS == []

def test_config_loader_dashboard_custom():
    # Mock configparser to return custom config
    with patch("configparser.ConfigParser") as MockParser:
        mock_parser_instance = MockParser.return_value
        mock_parser_instance.read.return_value = None
        mock_parser_instance.sections.return_value = ["dashboard"]

        # Mock config data
        config_data = {
            "dashboard": {
                "host": "127.0.0.1",
                "port": "9000",
                "enable_cors": "true",
                "cors_origins": "http://localhost:3000, http://test.com"
            }
        }

        def mock_items(section):
            return config_data.get(section, {}).items()

        mock_parser_instance.items.side_effect = mock_items

        with patch("src.config.loader.dotenv_values", return_value={}):
            with patch("pathlib.Path.exists", return_value=True):
                with patch.object(Config, "_build_model_configs"):
                    config = Config()

                assert config.DASHBOARD_HOST == "127.0.0.1"
                assert config.DASHBOARD_PORT == 9000
                assert config.DASHBOARD_ENABLE_CORS == True
                assert config.DASHBOARD_CORS_ORIGINS == ["http://localhost:3000", "http://test.com"]

def test_config_loader_cors_star():
    # Mock configparser for "*"
    with patch("configparser.ConfigParser") as MockParser:
        mock_parser_instance = MockParser.return_value
        mock_parser_instance.sections.return_value = ["dashboard"]

        config_data = {
            "dashboard": {
                "cors_origins": "*"
            }
        }
        mock_parser_instance.items.side_effect = lambda s: config_data.get(s, {}).items()

        with patch("src.config.loader.dotenv_values", return_value={}):
            with patch("pathlib.Path.exists", return_value=True):
                with patch.object(Config, "_build_model_configs"):
                    config = Config()
                assert config.DASHBOARD_CORS_ORIGINS == ["*"]

def test_dashboard_server_cors_disabled():
    # Setup mock config
    mock_config = MagicMock()
    mock_config.DASHBOARD_ENABLE_CORS = False

    server = DashboardServer(
        brain_service=MagicMock(),
        vector_memory=MagicMock(),
        analysis_engine=MagicMock(),
        config=mock_config,
        logger=MagicMock()
    )

    # Check that CORSMiddleware is NOT added
    # This is tricky to check on private attribute _create_app result,
    # but we can check app.user_middleware or similar

    # Check middleware
    has_cors = False
    for middleware in server.app.user_middleware:
        if middleware.cls.__name__ == "CORSMiddleware":
            has_cors = True
    assert not has_cors

def test_dashboard_server_cors_enabled():
    # Setup mock config
    mock_config = MagicMock()
    mock_config.DASHBOARD_ENABLE_CORS = True
    mock_config.DASHBOARD_CORS_ORIGINS = ["http://localhost:3000"]

    server = DashboardServer(
        brain_service=MagicMock(),
        vector_memory=MagicMock(),
        analysis_engine=MagicMock(),
        config=mock_config,
        logger=MagicMock()
    )

    # Check middleware
    has_cors = False
    for middleware in server.app.user_middleware:
        if middleware.cls.__name__ == "CORSMiddleware":
            has_cors = True
            # In Starlette/FastAPI, options are in options dict usually, but difficult to access directly on Middleware object
            # middleware.options['allow_origins']
            # Inspect middleware object
            print(f"Middleware attributes: {dir(middleware)}")
            if hasattr(middleware, 'options'):
                print(f"Options: {middleware.options}")
                assert middleware.options['allow_origins'] == ["http://localhost:3000"]
            elif hasattr(middleware, 'kwargs'):
                 print(f"Kwargs: {middleware.kwargs}")
                 assert middleware.kwargs['allow_origins'] == ["http://localhost:3000"]

    assert has_cors

if __name__ == "__main__":
    # Simple manual run if pytest fails
    try:
        test_config_loader_dashboard_defaults()
        print("test_config_loader_dashboard_defaults PASSED")
        test_config_loader_dashboard_custom()
        print("test_config_loader_dashboard_custom PASSED")
        test_config_loader_cors_star()
        print("test_config_loader_cors_star PASSED")
        test_dashboard_server_cors_disabled()
        print("test_dashboard_server_cors_disabled PASSED")
        test_dashboard_server_cors_enabled()
        print("test_dashboard_server_cors_enabled PASSED")
        print("ALL TESTS PASSED")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
