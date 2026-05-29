from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.managers.sqlite_trade_history import SQLiteTradeHistory


def _seed_decision(ts: str, action: str = "BUY", symbol: str = "BTC/USDC") -> dict:
    return {
        "timestamp": ts,
        "symbol": symbol,
        "action": action,
        "confidence": "HIGH",
        "price": 100.0,
        "stop_loss": 95.0,
        "take_profit": 110.0,
        "position_size": 0.05,
        "quote_amount": 500.0,
        "quantity": 5.0,
        "fee": 0.4,
        "reasoning": "test",
    }


def test_initializes_empty_sqlite_store(tmp_path: Path):
    store = SQLiteTradeHistory(logger=MagicMock(), db_path=str(tmp_path / "trade_history.db"))
    assert store.count() == 0


def test_query_rejects_invalid_order(tmp_path: Path):
    store = SQLiteTradeHistory(
        logger=MagicMock(),
        db_path=str(tmp_path / "trade_history.db"),
    )
    with pytest.raises(ValueError, match="Invalid order"):
        store.query(order="DROP TABLE")


def test_query_clamps_negative_limit_and_offset(tmp_path: Path):
    store = SQLiteTradeHistory(
        logger=MagicMock(),
        db_path=str(tmp_path / "trade_history.db"),
    )
    store.insert(_seed_decision("2026-05-21T00:00:00+00:00"))
    rows = store.query(limit=-100, offset=-5)
    assert len(rows) == 1


def test_get_last_execution_timestamp_returns_latest_buy_sell(tmp_path: Path):
    store = SQLiteTradeHistory(
        logger=MagicMock(),
        db_path=str(tmp_path / "trade_history.db"),
    )
    now = datetime.now(timezone.utc)
    store.insert(_seed_decision((now - timedelta(hours=2)).isoformat(), "BUY"))
    store.insert(_seed_decision((now - timedelta(hours=1)).isoformat(), "CLOSE_LONG"))
    store.insert(_seed_decision(now.isoformat(), "SELL"))

    latest = store.get_last_execution_timestamp(actions=("BUY", "SELL"))

    assert latest is not None
    assert latest == now.isoformat()
