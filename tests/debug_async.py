import asyncio
from unittest.mock import MagicMock, AsyncMock

async def test():
    from src.trading.trading_strategy import TradingStrategy
    from src.trading.dataclasses import Position
    from datetime import datetime, timezone

    mock_logger = MagicMock()
    mock_persistence = MagicMock()
    mock_brain = MagicMock()
    mock_stats = MagicMock()
    mock_memory = MagicMock()
    mock_risk = MagicMock()
    mock_factory = MagicMock()
    
    mock_persistence.async_save_position = AsyncMock()
    mock_factory.create_updated_position = MagicMock(
        side_effect=lambda original_position, new_stop_loss, new_take_profit: Position(
            entry_price=original_position.entry_price,
            stop_loss=new_stop_loss,
            take_profit=new_take_profit,
            size=original_position.size,
            entry_time=original_position.entry_time,
            confidence=original_position.confidence,
            direction=original_position.direction,
            symbol=original_position.symbol,
        )
    )
    mock_persistence.load_position.return_value = None

    strategy = TradingStrategy(
        logger=mock_logger,
        persistence=mock_persistence,
        brain_service=mock_brain,
        statistics_service=mock_stats,
        memory_service=mock_memory,
        risk_manager=mock_risk,
        position_factory=mock_factory,
    )

    strategy.current_position = Position(
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        size=1.0,
        entry_time=datetime.now(timezone.utc),
        confidence="HIGH",
        direction="LONG",
        symbol="BTC/USDT",
    )

    try:
        await strategy._update_position_parameters(stop_loss=85.0, take_profit=125.0)
        print("Success")
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(test())
