
import pytest
import sys
import os
from unittest.mock import MagicMock

# Ensure source is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_managers_imports():
    """Verify that new manager classes can be imported from their new locations."""
    try:
        from src.managers.model_manager import ModelManager  # noqa: F401
        from src.managers.persistence_manager import PersistenceManager  # noqa: F401
        from src.managers.risk_manager import RiskManager  # noqa: F401
    except ImportError as e:
        pytest.fail(f"Failed to import managers: {e}")

def test_contracts_imports():
    """Verify that new contracts can be imported."""
    try:
        from src.contracts.model_contract import ModelManagerProtocol  # noqa: F401
        from src.contracts.risk_contract import RiskManagerProtocol  # noqa: F401
    except ImportError as e:
        pytest.fail(f"Failed to import contracts: {e}")

def test_dataclasses_imports():
    """Verify that new dataclasses can be imported."""
    try:
        from src.trading.data_models import VectorSearchResult, RiskAssessment  # noqa: F401
    except ImportError as e:
        pytest.fail(f"Failed to import new dataclasses: {e}")

def test_risk_manager_instantiation():
    """Verify RiskManager can be instantiated."""
    from src.managers.risk_manager import RiskManager
    logger_mock = MagicMock()
    config_mock = MagicMock()
    rm = RiskManager(logger_mock, config_mock)
    assert rm is not None

def test_persistence_manager_instantiation():
    """Verify PersistenceManager can be instantiated."""
    from src.managers.persistence_manager import PersistenceManager
    logger_mock = MagicMock()
    # Use a temp dir or just check init
    pm = PersistenceManager(logger_mock, data_dir="data/test_persistence")
    assert pm is not None

def test_trading_strategy_injection():
    """Verify TradingStrategy accepts the new injections."""
    from src.trading.trading_strategy import TradingStrategy
    
    logger = MagicMock()
    persistence = MagicMock()
    # Mock load_position to return None (no existing position)
    persistence.load_position.return_value = None
    
    brain = MagicMock()
    stats = MagicMock()
    memory = MagicMock()
    risk = MagicMock()
    
    # Proper config mock with attributes that RiskManager might need
    config = MagicMock()
    config.TRANSACTION_FEE_PERCENT = 0.00075
    
    extractor = MagicMock()
    
    # It should accept risk_manager arg
    strategy = TradingStrategy(
        logger=logger,
        persistence=persistence,
        brain_service=brain,
        statistics_service=stats,
        memory_service=memory,
        risk_manager=risk,
        config=config,
        position_extractor=extractor
    )
    assert strategy.risk_manager == risk
