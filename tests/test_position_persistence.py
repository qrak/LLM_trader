"""Test Position Persistence unit tests.

Tests for test_position_persistence.py.
"""
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.managers.persistence_manager import PersistenceManager
from src.trading.data_models import Position, TradeDecision


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
    for record in [
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
    ]:
        manager.sqlite_history.insert(record)

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


def test_last_execution_timestamp_propagates_sqlite_failure(tmp_path):
    manager = PersistenceManager(MagicMock(), data_dir=str(tmp_path))
    manager.sqlite_history.get_last_execution_timestamp = MagicMock(side_effect=RuntimeError("db down"))

    with pytest.raises(RuntimeError, match="db down"):
        manager.get_last_execution_timestamp()


def test_save_trade_decision_returns_row_id(tmp_path):
    """save_trade_decision must return the SQLite rowid of the inserted row.

    Regression guard: the rowid is what links the post-mortem journal
    (trade_post_mortem.trade_id) back to trade_history.id.
    """
    manager = PersistenceManager(MagicMock(), data_dir=str(tmp_path))
    decision = TradeDecision(
        timestamp=datetime(2026, 8, 2, 23, 46, 36, tzinfo=timezone.utc),
        symbol="BTC/USDC",
        action="CLOSE_SHORT",
        confidence="HIGH",
        price=63671.19,
        stop_loss=None,
        take_profit=None,
        position_size=0.0,
        quote_amount=0.0,
        quantity=0.0,
        fee=0.3702,
        reasoning="Position closed: stop_loss",
    )

    row_id = manager.save_trade_decision(decision)
    assert isinstance(row_id, int)
    assert row_id > 0

    # Second insert must get a strictly greater id (append-only)
    decision2 = TradeDecision(
        timestamp=datetime(2026, 8, 3, 20, 1, 7, tzinfo=timezone.utc),
        symbol="BTC/USDC",
        action="BUY",
        confidence="HIGH",
        price=63826.23,
    )
    row_id2 = manager.save_trade_decision(decision2)
    assert row_id2 > row_id

    # And the ids must resolve to the right rows in trade_history
    rows = manager.sqlite_history.query()
    by_id = {r["id"]: r for r in rows}
    assert by_id[row_id]["action"] == "CLOSE_SHORT"
    assert by_id[row_id2]["action"] == "BUY"


def test_clear_latest_decision_removes_stale_fallback_file(tmp_path):
    """clear_latest_decision must delete the fallback file if present."""
    manager = PersistenceManager(MagicMock(), data_dir=str(tmp_path))
    # Simulate a stale decision file left from a past HTTP outage
    manager.save_latest_decision({"signal": "UPDATE", "symbol": "BTC/USDC"})
    decision_path = tmp_path / "latest_decision.json"
    assert decision_path.exists()

    manager.clear_latest_decision()

    assert not decision_path.exists()


def test_clear_latest_decision_noop_when_file_absent(tmp_path):
    """clear_latest_decision with no fallback file must not raise."""
    manager = PersistenceManager(MagicMock(), data_dir=str(tmp_path))
    manager.clear_latest_decision()  # must not raise
