import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone, timedelta

from src.app import CryptoTradingBot
from src.logger.logger import Logger

@pytest.fixture
def mock_dependencies():
    """Provides a full mock dependency injection configuration for CryptoTradingBot."""
    return {
        'logger': MagicMock(spec=Logger),
        'config': MagicMock(),
        'shutdown_manager': MagicMock(),
        'exchange_manager': AsyncMock(),
        'market_analyzer': MagicMock(),
        'trading_strategy': MagicMock(),
        'discord_notifier': AsyncMock(),
        'keyboard_handler': AsyncMock(),
        'rag_engine': AsyncMock(),
        'coingecko_api': MagicMock(),
        'news_api': MagicMock(),
        'market_api': MagicMock(),
        'categories_api': MagicMock(),
        'alternative_me_api': MagicMock(),
        'cryptocompare_session': MagicMock(),
        'persistence': MagicMock(),
        'model_manager': MagicMock(),
        'brain_service': MagicMock(),
        'statistics_service': MagicMock(),
        'memory_service': MagicMock(),
        'dashboard_state': AsyncMock()
    }

@pytest.fixture
def bot(mock_dependencies):
    """Initializes the CryptoTradingBot with mocked dependencies."""
    return CryptoTradingBot(**mock_dependencies)

@pytest.mark.asyncio
async def test_bot_run_resumes_and_executes_one_loop(bot, mock_dependencies):
    """
    Tests that the bot accurately resumes from a wait and executes its core cyclic behavior
    up to the end of one loop interval, stopping properly without real waiting logic.
    """
    # 1. Setup specific mocks for the execution flow
    exchange_mock = AsyncMock()
    exchange_mock.fetch_ticker.return_value = {'last': 50000.0, 'close': 50000.0}
    mock_dependencies['exchange_manager'].find_symbol_exchange.return_value = (exchange_mock, "mock_exchange")
    
    mock_dependencies['config'].TIMEFRAME = "1h"
    
    market_analyzer_mock = mock_dependencies['market_analyzer']
    trading_strategy_mock = mock_dependencies['trading_strategy']

    # Assign AsyncMocks for asynchronous methods
    market_analyzer_mock.analyze_market = AsyncMock(return_value={"signal": "BUY", "confidence": 0.8})
    trading_strategy_mock.process_analysis = AsyncMock()
    trading_strategy_mock.check_position = AsyncMock()
    
    past_time = datetime.now(timezone.utc) - timedelta(minutes=30)
    mock_dependencies['persistence'].get_last_analysis_time.return_value = past_time
    mock_dependencies['persistence'].load_trade_history.return_value = []
    mock_dependencies['persistence'].load_previous_response.return_value = {}
    
    mock_dependencies['memory_service'].get_context_summary.return_value = "Mocked memory context"
    mock_dependencies['statistics_service'].get_context.return_value = "Mocked statistics context"
    mock_dependencies['brain_service'].get_dynamic_thresholds.return_value = {"threshold": 1.0}
    
    decision_mock = MagicMock()
    decision_mock.action = "BUY"
    trading_strategy_mock.process_analysis.return_value = decision_mock
    trading_strategy_mock.current_position = None
    trading_strategy_mock.get_position_context.return_value = "Mocked position context"
    
    # We patch _wait_for_next_timeframe to break out of the infinite `while self.running:` loop
    # by simulating an exit scenario as soon as it ends the first iteration cleanly.
    async def mock_wait_for_next_timeframe(*args, **kwargs):
        bot.running = False
        return False  # False means it was not a forced wait (normal execution)

    # We also eliminate the long delays caused by _interruptible_sleep.
    async def mock_interruptible_sleep(seconds, respect_force_analysis=True):
        return False
        
    # Execute the Bot Run Flow iteratively 
    with patch.object(bot, '_wait_for_next_timeframe', side_effect=mock_wait_for_next_timeframe), \
         patch.object(bot, '_interruptible_sleep', side_effect=mock_interruptible_sleep):

        await bot.run(symbol="BTC/USDT", timeframe="1h")

    # 1. Initialization and ticker fetching
    mock_dependencies['market_analyzer'].initialize_for_symbol.assert_called_once_with(
        symbol="BTC/USDT", exchange=exchange_mock, timeframe="1h"
    )
    exchange_mock.fetch_ticker.assert_called()
    
    # 2. Resume logic was evaluated
    assert mock_dependencies['persistence'].get_last_analysis_time.call_count >= 1
    
    # 3. Execution check methods were successively invoked
    mock_dependencies['rag_engine'].update_if_needed.assert_called_once_with(force_update=True)
    mock_dependencies['market_analyzer'].analyze_market.assert_called_once()
    mock_dependencies['trading_strategy'].process_analysis.assert_called_once()
    
    # 4. Persistence saving successfully finalized the pass
    mock_dependencies['persistence'].save_last_analysis_time.assert_called_once()

    # 5. Discord notifier correctly fired trading decision notification based on test input
    mock_dependencies['discord_notifier'].send_trading_decision.assert_called_once()
