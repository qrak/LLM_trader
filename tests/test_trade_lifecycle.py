
import unittest
from datetime import datetime, timezone
import sys
import os

# Add src to path if running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading.data_models import Position
from src.trading.statistics_calculator import StatisticsCalculator

class TestTradeLifecycle(unittest.TestCase):
    """
    Comprehensive regression tests for trade lifecycle:
    - SHORT Position Closure & PnL
    - LONG Position Closure & PnL
    - SHORT Position Updates (Trailing SL)
    - LONG Position Updates (Trailing SL)
    """

    def test_short_position_lifecycle(self):
        """Test SHORT position PnL calculation and closure logic."""
        # Entry at 100, Current at 90. Profit should be 10%.
        pos = Position(
            entry_price=100.0,
            stop_loss=110.0,
            take_profit=80.0,
            size=1.0,
            entry_time=datetime.now(timezone.utc),
            confidence="HIGH",
            direction="SHORT",
            symbol="BTC/USDC"
        )
        
        # Scenario 1: Price drops (profit)
        pnl_pct = pos.calculate_pnl(90.0)
        self.assertAlmostEqual(pnl_pct, 10.0, msg="Short PnL should be positive when price drops")
        
        # Scenario 2: Price rises (loss)
        pnl_pct_loss = pos.calculate_pnl(110.0)
        self.assertAlmostEqual(pnl_pct_loss, -10.0, msg="Short PnL should be negative when price rises")

        # Test Statistics Update
        history = [
            {
                "action": "SELL",
                "price": 100.0,
                "quantity": 1.0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            {
                "action": "CLOSE_SHORT",
                "price": 90.0,
                "quantity": 1.0, 
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ]
        
        stats = StatisticsCalculator.calculate_from_history(history, initial_capital=10000.0)
        
        # Profit = (Entry - Exit) * Qty = (100 - 90) * 1 = 10.0
        # New Capital = 10000 + 10 = 10010
        self.assertEqual(stats.total_trades, 1)
        self.assertEqual(stats.winning_trades, 1)
        self.assertAlmostEqual(stats.total_pnl_quote, 10.0)
        self.assertAlmostEqual(stats.current_capital, 10010.0)

    def test_long_position_lifecycle(self):
        """Test LONG position PnL calculation and closure logic."""
        # Entry at 100, Current at 110. Profit should be 10%.
        pos = Position(
            entry_price=100.0,
            stop_loss=90.0,
            take_profit=120.0,
            size=1.0,
            entry_time=datetime.now(timezone.utc),
            confidence="HIGH",
            direction="LONG",
            symbol="BTC/USDC"
        )
        
        # Scenario 1: Price rises (profit)
        pnl_pct = pos.calculate_pnl(110.0)
        self.assertAlmostEqual(pnl_pct, 10.0, msg="Long PnL should be positive when price rises")
        
        # Scenario 2: Price drops (loss)
        pnl_pct_loss = pos.calculate_pnl(90.0)
        self.assertAlmostEqual(pnl_pct_loss, -10.0, msg="Long PnL should be negative when price drops")

        # Test Statistics Update
        history = [
            {
                "action": "BUY",
                "price": 100.0,
                "quantity": 1.0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            {
                "action": "CLOSE_LONG",
                "price": 110.0,
                "quantity": 1.0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ]
        
        stats = StatisticsCalculator.calculate_from_history(history, initial_capital=10000.0)
        
        # Profit = (Exit - Entry) * Qty = (110 - 100) * 1 = 10.0
        # New Capital = 10000 + 10 = 10010
        self.assertEqual(stats.total_trades, 1)
        self.assertEqual(stats.winning_trades, 1)
        self.assertAlmostEqual(stats.total_pnl_quote, 10.0)
        self.assertAlmostEqual(stats.current_capital, 10010.0)

    def test_short_position_update(self):
        """Test updates (trailing SL) for SHORT positions."""
        pos = Position(
            entry_price=100.0,
            stop_loss=110.0,
            take_profit=80.0,
            size=1.0,
            entry_time=datetime.now(timezone.utc),
            confidence="HIGH",
            direction="SHORT",
            symbol="BTC/USDC"
        )
        
        # Simulate Trailing Stop Logic found in TradingStrategy._update_position_parameters (conceptually)
        # 1. Price drops to 90 (Win).
        # 2. We want to move SL from 110 to 95 to lock in profit if reversal happens.
        
        new_sl = 95.0
        # In a real update, we recreate the Position with frozen fields
        updated_pos = Position(
            entry_price=pos.entry_price,
            stop_loss=new_sl,
            take_profit=pos.take_profit,
            size=pos.size,
            entry_time=pos.entry_time,
            confidence=pos.confidence,
            direction=pos.direction,
            symbol=pos.symbol
        )
        
        self.assertEqual(updated_pos.stop_loss, 95.0)
        self.assertEqual(updated_pos.entry_price, 100.0)
        
        # Ensure 'is_stop_hit' works with new SL
        # If price reverses to 96 (hit SL), we should close.
        self.assertTrue(updated_pos.is_stop_hit(96.0), "Short SL should be hit at 96 when SL is 95")
        self.assertFalse(updated_pos.is_stop_hit(94.0), "Short SL should NOT be hit at 94 when SL is 95")

    def test_long_position_update(self):
        """Test updates (trailing SL) for LONG positions."""
        pos = Position(
            entry_price=100.0,
            stop_loss=90.0,
            take_profit=120.0,
            size=1.0,
            entry_time=datetime.now(timezone.utc),
            confidence="HIGH",
            direction="LONG",
            symbol="BTC/USDC"
        )
        
        # Simulate Trailing Stop Logic
        # 1. Price rises to 110 (Win).
        # 2. Move SL from 90 to 105 to lock in profit.
        
        new_sl = 105.0
        updated_pos = Position(
            entry_price=pos.entry_price,
            stop_loss=new_sl,
            take_profit=pos.take_profit,
            size=pos.size,
            entry_time=pos.entry_time,
            confidence=pos.confidence,
            direction=pos.direction,
            symbol=pos.symbol
        )
        
        self.assertEqual(updated_pos.stop_loss, 105.0)
        
        # Ensure 'is_stop_hit' works with new SL
        # If price drops to 104 (hit SL), we should close.
        self.assertTrue(updated_pos.is_stop_hit(104.0), "Long SL should be hit at 104 when SL is 105")
        self.assertFalse(updated_pos.is_stop_hit(106.0), "Long SL should NOT be hit at 106 when SL is 105")

if __name__ == '__main__':
    unittest.main()
