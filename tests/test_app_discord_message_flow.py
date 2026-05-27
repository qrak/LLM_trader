from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.app import BotServices, CryptoTradingBot


def _make_bot_for_trading_check(**overrides):
    config = SimpleNamespace(
        MAIN_CHANNEL_ID=123,
        RAG_UPDATE_TIMEOUT=1,
    )

    discord_notifier = MagicMock()
    discord_notifier.send_trading_decision = AsyncMock()
    discord_notifier.send_analysis_notification = AsyncMock()

    market_analyzer = MagicMock()
    market_analyzer.analyze_market = AsyncMock(
        return_value={
            "analysis": {
                "signal": "SELL",
                "confidence": 82,
                "reasoning": "Bearish structure confirmed.",
            },
            "raw_response": "AI reasoning text {\"analysis\": {}}",
        }
    )
    market_analyzer.last_chart_buffer = None

    persistence = MagicMock()
    persistence.save_last_analysis_time = MagicMock()
    persistence.get_last_analysis_time = MagicMock(return_value=datetime.now(timezone.utc))
    persistence.async_load_previous_response = AsyncMock(return_value={})

    trading_strategy = MagicMock()
    trading_strategy.process_analysis = AsyncMock(return_value=SimpleNamespace(action="SELL"))
    trading_strategy.current_position = object()
    trading_strategy.get_position_context = MagicMock(return_value="position context")

    memory_service = MagicMock()
    memory_service.get_context_summary = MagicMock(return_value="memory")
    statistics_service = MagicMock()
    statistics_service.get_context = MagicMock(return_value="stats")
    brain_service = MagicMock()
    brain_service.get_dynamic_thresholds = MagicMock(return_value={})

    position_monitor = MagicMock()
    position_monitor.check_soft_exit_status = AsyncMock()
    position_monitor.handle_new_position = AsyncMock()

    services = {
        "logger": MagicMock(),
        "config": config,
        "shutdown_manager": None,
        "exchange_manager": MagicMock(),
        "market_analyzer": market_analyzer,
        "trading_strategy": trading_strategy,
        "discord_notifier": discord_notifier,
        "keyboard_handler": MagicMock(),
        "rag_engine": MagicMock(),
        "coingecko_api": MagicMock(),
        "market_api": MagicMock(),
        "alternative_me_api": MagicMock(),
        "http_session": MagicMock(),
        "persistence": persistence,
        "model_manager": MagicMock(),
        "brain_service": brain_service,
        "statistics_service": statistics_service,
        "memory_service": memory_service,
        "exit_monitor": MagicMock(),
        "dashboard_state": None,
        "discord_task": None,
        "position_monitor_factory": lambda _bot: position_monitor,
    }
    services.update(overrides)

    bot = CryptoTradingBot(BotServices(**services))
    bot.current_symbol = "BTC/USDC"
    bot.current_timeframe = "4h"
    bot._fetch_ticker_data = AsyncMock(return_value=({"last": 77163.94}, 77163.94))
    bot._execute_market_knowledge_update = AsyncMock()
    bot._build_analysis_context = AsyncMock(return_value={})
    bot._save_analysis_data = MagicMock()

    return SimpleNamespace(
        bot=bot,
        discord_notifier=discord_notifier,
        market_analyzer=market_analyzer,
        position_monitor=position_monitor,
        trading_strategy=trading_strategy,
    )


@pytest.mark.asyncio
async def test_execute_trading_check_uses_analysis_notification_without_decision_embed():
    fixture = _make_bot_for_trading_check()

    await fixture.bot._execute_trading_check(check_count=1, force_news_update=True, is_candle_close=True)

    fixture.discord_notifier.send_trading_decision.assert_not_awaited()
    fixture.discord_notifier.send_analysis_notification.assert_awaited_once()
    fixture.position_monitor.handle_new_position.assert_awaited_once()
