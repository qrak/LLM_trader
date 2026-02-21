
import sys
import pytest
from unittest.mock import MagicMock

# Create the mock config object
mock_config = MagicMock()
mock_config.LOGGER_DEBUG = False
mock_config.get_config.return_value = {}
mock_config.get_env.return_value = None

# Create a mock module for src.config.loader
mock_loader_module = MagicMock()
mock_loader_module.config = mock_config
mock_loader_module.Config = MagicMock(return_value=mock_config)

# Patch sys.modules to return our mock
sys.modules['src.config.loader'] = mock_loader_module
