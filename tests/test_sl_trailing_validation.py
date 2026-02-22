"""
Tests for SL trailing direction validation.
Verifies that TradingStrategy._update_position_parameters correctly handles
SL moves (including Full AI Autonomy widening).
"""
import unittest
from unittest.mock import MagicMock
from datetime import datetime, timezone
from src.trading.dataclasses import Position


class TestSLTrailingValidation(unittest.IsolatedAsyncioTestCase):
    """Tests for SL direction validation in _update_position_parameters."""

    def _create_strategy_with_position(self, direction: str, entry: float, sl: float, tp: float):
        """Helper: create a TradingStrategy with a mock position."""
        from src.trading.trading_strategy import TradingStrategy

        mock_logger = MagicMock()
        mock_persistence = MagicMock()
        mock_brain = MagicMock()
        mock_stats = MagicMock()
        mock_memory = MagicMock()
        mock_risk = MagicMock()
        mock_factory = MagicMock()
        from unittest.mock import AsyncMock
        mock_persistence.async_save_position = AsyncMock()

        # Make the factory return a new position when create_updated_position is called
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

        # Prevent loading from persistence
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

        # Manually set the position
        strategy.current_position = Position(
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            size=1.0,
            entry_time=datetime.now(timezone.utc),
            confidence="HIGH",
            direction=direction,
            symbol="BTC/USDT",
        )
        return strategy, mock_logger

    async def test_long_sl_upgrade_accepted(self):
        """LONG: Moving SL UP (tightening) should be accepted."""
        strategy, logger = self._create_strategy_with_position("LONG", 100.0, 90.0, 120.0)
        result = await strategy._update_position_parameters(stop_loss=95.0, take_profit=None)
        self.assertTrue(result)
        self.assertEqual(strategy.current_position.stop_loss, 95.0)
        logger.warning.assert_not_called()

    async def test_long_sl_downgrade_accepted(self):
        """LONG: Moving SL DOWN (widening risk) should be accepted due to AI Autonomy."""
        strategy, logger = self._create_strategy_with_position("LONG", 100.0, 90.0, 120.0)
        result = await strategy._update_position_parameters(stop_loss=85.0, take_profit=None)
        self.assertTrue(result)
        self.assertEqual(strategy.current_position.stop_loss, 85.0)  # Changed
        logger.info.assert_called()
        self.assertIn("AI Widening Stop Loss for LONG", str(logger.info.call_args_list))

    async def test_short_sl_downgrade_accepted(self):
        """SHORT: Moving SL DOWN (tightening) should be accepted."""
        strategy, logger = self._create_strategy_with_position("SHORT", 100.0, 110.0, 80.0)
        result = await strategy._update_position_parameters(stop_loss=105.0, take_profit=None)
        self.assertTrue(result)
        self.assertEqual(strategy.current_position.stop_loss, 105.0)
        logger.warning.assert_not_called()

    async def test_short_sl_upgrade_accepted(self):
        """SHORT: Moving SL UP (widening risk) should be accepted due to AI Autonomy."""
        strategy, logger = self._create_strategy_with_position("SHORT", 100.0, 110.0, 80.0)
        result = await strategy._update_position_parameters(stop_loss=115.0, take_profit=None)
        self.assertTrue(result)
        self.assertEqual(strategy.current_position.stop_loss, 115.0)  # Changed
        logger.info.assert_called()
        self.assertIn("AI Widening Stop Loss for SHORT", str(logger.info.call_args_list))

    async def test_tp_update_unaffected(self):
        """TP updates should still work independently of SL validation."""
        strategy, logger = self._create_strategy_with_position("LONG", 100.0, 90.0, 120.0)
        result = await strategy._update_position_parameters(stop_loss=None, take_profit=130.0)
        self.assertTrue(result)
        self.assertEqual(strategy.current_position.take_profit, 130.0)

    async def test_same_sl_no_update(self):
        """Passing the same SL should not trigger an update."""
        strategy, _ = self._create_strategy_with_position("LONG", 100.0, 90.0, 120.0)
        result = await strategy._update_position_parameters(stop_loss=90.0, take_profit=None)
        self.assertFalse(result)

    async def test_long_sl_combined_with_tp_update(self):
        """LONG: Widened SL + valid TP should update both."""
        strategy, logger = self._create_strategy_with_position("LONG", 100.0, 90.0, 120.0)
        result = await strategy._update_position_parameters(stop_loss=85.0, take_profit=125.0)
        self.assertTrue(result)  # Both succeeded
        self.assertEqual(strategy.current_position.stop_loss, 85.0)  # SL changed
        self.assertEqual(strategy.current_position.take_profit, 125.0)  # TP updated
        logger.info.assert_called()


if __name__ == "__main__":
    unittest.main()
