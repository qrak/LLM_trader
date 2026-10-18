"""Read-only trade-history tools: filtering, coverage stats, and a hard no-write guarantee."""

import sqlite3

import pytest

from src.mcp_tools.history_server import decision_stats, recent_trades

SCHEMA = """
CREATE TABLE trade_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    confidence TEXT,
    price REAL,
    stop_loss REAL,
    take_profit REAL,
    position_size REAL,
    quote_amount REAL,
    quantity REAL,
    fee REAL,
    reasoning TEXT,
    indicators_json TEXT,
    execution_quality TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

ROWS = [
    ("2026-10-18 04:00:00", "BTC/USDC", "BUY", "HIGH", 82412.23, 83500.0, 80302.84, 0.048, 50.0, 0.0006, 0.03),
    ("2026-10-18 08:00:00", "BTC/USDC", "HOLD", "MEDIUM", 82000.0, None, None, None, None, None, None),
    ("2026-10-17 12:00:00", "ETH/USDC", "SELL", "MEDIUM", 3000.0, 3100.0, 2800.0, 0.03, 30.0, 0.01, 0.02),
]


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "trade_history.db"
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    connection.executemany(
        "INSERT INTO trade_history (timestamp, symbol, action, confidence, price, stop_loss, "
        "take_profit, position_size, quote_amount, quantity, fee) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ROWS,
    )
    connection.commit()
    connection.close()
    return path


def test_recent_trades_are_newest_first_and_limited(db):
    rows = recent_trades(db, limit=2)

    assert [row["timestamp"] for row in rows] == ["2026-10-18 08:00:00", "2026-10-18 04:00:00"]


def test_filters_by_symbol_and_action(db):
    assert len(recent_trades(db, symbol="ETH/USDC")) == 1
    assert len(recent_trades(db, action="hold")) == 1
    assert recent_trades(db, symbol="SOL/USDC") == []


def test_limit_is_clamped(db):
    assert len(recent_trades(db, limit=10_000)) == 3
    assert len(recent_trades(db, limit=0)) == 1
    assert len(recent_trades(db, limit="nonsense")) == 3


def test_decision_stats_counts_actions(db):
    stats = decision_stats(db, days=365)

    assert stats["rows"] == 3
    assert stats["by_action"] == {"BUY": 1, "HOLD": 1, "SELL": 1}
    assert stats["last"] == "2026-10-18 08:00:00"


def test_symbol_stats_are_scoped(db):
    stats = decision_stats(db, symbol="BTC/USDC", days=365)

    assert stats["rows"] == 2
    assert stats["by_action"] == {"BUY": 1, "HOLD": 1}


def test_missing_database_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        recent_trades(tmp_path / "missing.db")


def test_connection_refuses_writes(db):
    from src.mcp_tools.history_server import _connect_ro

    with _connect_ro(db) as connection, pytest.raises(sqlite3.OperationalError):
        connection.execute("DELETE FROM trade_history")
