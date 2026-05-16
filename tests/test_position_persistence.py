from datetime import datetime, timezone
import json
from unittest.mock import MagicMock
from unittest.mock import patch

from src.managers.persistence_manager import PersistenceManager
from src.trading.data_models import Position


def test_position_persistence_round_trips_exit_execution_snapshot(tmp_path):
    manager = PersistenceManager(MagicMock(), data_dir=str(tmp_path))
    position = Position(
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        size=1.0,
        entry_time=datetime(2026, 4, 30, tzinfo=timezone.utc),
        confidence="HIGH",
        direction="LONG",
        symbol="BTC/USDC",
        stop_loss_type_at_entry="hard",
        stop_loss_check_interval_at_entry="15m",
        take_profit_type_at_entry="soft",
        take_profit_check_interval_at_entry="4h",
        order_book_bias_at_entry="BUY_PRESSURE",
    )

    manager.save_position(position)
    loaded = PersistenceManager(MagicMock(), data_dir=str(tmp_path)).load_position()

    assert loaded is not None
    assert loaded.stop_loss_type_at_entry == "hard"
    assert loaded.stop_loss_check_interval_at_entry == "15m"
    assert loaded.take_profit_type_at_entry == "soft"
    assert loaded.take_profit_check_interval_at_entry == "4h"
    assert loaded.order_book_bias_at_entry == "BUY_PRESSURE"


def test_position_persistence_defaults_missing_exit_execution_snapshot_to_unknown(tmp_path):
    manager = PersistenceManager(MagicMock(), data_dir=str(tmp_path))
    manager.positions_file.write_text(
        json.dumps(
            {
                "entry_price": 100.0,
                "stop_loss": 95.0,
                "take_profit": 110.0,
                "size": 1.0,
                "entry_time": "2026-04-30T00:00:00+00:00",
                "confidence": "HIGH",
                "direction": "LONG",
                "symbol": "BTC/USDC",
            }
        ),
        encoding="utf-8",
    )

    loaded = manager.load_position()

    assert loaded is not None
    assert loaded.stop_loss_type_at_entry == "unknown"
    assert loaded.stop_loss_check_interval_at_entry == "unknown"
    assert loaded.take_profit_type_at_entry == "unknown"
    assert loaded.take_profit_check_interval_at_entry == "unknown"
    assert loaded.order_book_bias_at_entry == "BALANCED"


def test_entry_decision_matching_prefers_nearest_symbol_match(tmp_path):
    manager = PersistenceManager(MagicMock(), data_dir=str(tmp_path))
    entry_time = datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc)
    manager.history_file.write_text(
        json.dumps([
            {
                "timestamp": "2026-04-30T12:00:00.010000+00:00",
                "symbol": "ETH/USDC",
                "action": "BUY",
                "confidence": "HIGH",
                "price": 2000.0,
                "reasoning": "Wrong symbol",
            },
            {
                "timestamp": "2026-04-30T12:00:00.020000+00:00",
                "symbol": "BTC/USDC",
                "action": "BUY",
                "confidence": "MEDIUM",
                "price": 100.0,
                "reasoning": "Expected entry",
            },
            {
                "timestamp": "2026-04-30T12:00:00.400000+00:00",
                "symbol": "BTC/USDC",
                "action": "BUY",
                "confidence": "LOW",
                "price": 101.0,
                "reasoning": "Later entry",
            },
        ]),
        encoding="utf-8",
    )

    decision = manager.get_entry_decision_for_position(entry_time, symbol="BTC/USDC")

    assert decision is not None
    assert decision.reasoning == "Expected entry"
    assert decision.symbol == "BTC/USDC"


def test_failed_position_write_does_not_mark_cache_valid(tmp_path):
    manager = PersistenceManager(MagicMock(), data_dir=str(tmp_path))
    position = Position(
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        size=1.0,
        entry_time=datetime(2026, 4, 30, tzinfo=timezone.utc),
        confidence="HIGH",
        direction="LONG",
        symbol="BTC/USDC",
    )

    with patch("src.managers.persistence_manager.os.replace", side_effect=OSError("disk full")):
        manager.save_position(position)

    assert manager._position_cache_valid is False
