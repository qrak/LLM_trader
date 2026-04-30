from datetime import datetime, timezone
import json
from unittest.mock import MagicMock

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
    )

    manager.save_position(position)
    loaded = PersistenceManager(MagicMock(), data_dir=str(tmp_path)).load_position()

    assert loaded is not None
    assert loaded.stop_loss_type_at_entry == "hard"
    assert loaded.stop_loss_check_interval_at_entry == "15m"
    assert loaded.take_profit_type_at_entry == "soft"
    assert loaded.take_profit_check_interval_at_entry == "4h"


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