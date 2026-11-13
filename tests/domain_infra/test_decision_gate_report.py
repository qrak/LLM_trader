

import logging
import sqlite3

import pytest

from scripts.decision_gate_report import (
    GateReport,
    bars_to_array,
    ranks,
    spearman,
)


def make_report(tmp_path):
    return GateReport(logging.getLogger("test"), None, tmp_path)


def seed_database(tmp_path):
    con = sqlite3.connect(tmp_path / "trade_history.db")
    con.executescript(
        """
        create table trade_history (
            id integer primary key autoincrement,
            timestamp text not null,
            symbol text not null,
            action text not null,
            confidence text,
            price real,
            stop_loss real,
            take_profit real,
            position_size real,
            reasoning text,
            indicators_json text
        );
        create table trade_post_mortem (
            id integer primary key autoincrement,
            trade_id integer,
            direction text,
            pnl_pct real,
            created_at text not null
        );
        """
    )
    rows = [
        ("2026-10-01T00:05:00+00:00", "BUY", "MEDIUM", 70000.0),
        ("2026-10-01T00:05:30+00:00", "UPDATE", "MEDIUM", 70000.0),
        ("2026-10-02T00:05:00+00:00", "CLOSE_LONG", "MEDIUM", 69000.0),
        ("2026-10-03T00:05:00+00:00", "SELL", "HIGH", 68000.0),
        ("2026-10-04T00:05:00+00:00", "CLOSE_SHORT", "HIGH", 66000.0),
    ]
    for index, (stamp, action, confidence, price) in enumerate(rows, start=1):
        con.execute(
            "insert into trade_history (id, timestamp, symbol, action, confidence, price, "
            "position_size, reasoning, indicators_json) values (?,?,?,?,?,?,?,?,?)",
            (index, stamp, "BTC/USDC", action, confidence, price, 0.04, "reasoning", "{}"),
        )
    con.execute(
        "insert into trade_post_mortem (trade_id, direction, pnl_pct, created_at) "
        "values (3, 'LONG', -1.5, '2026-10-02T00:06:00+00:00')"
    )
    con.execute(
        "insert into trade_post_mortem (trade_id, direction, pnl_pct, created_at) "
        "values (5, 'SHORT', 2.5, '2026-10-04T00:06:00+00:00')"
    )
    con.commit()
    con.close()


def test_closed_pairs_uses_the_entry_decision_that_precedes_the_close(tmp_path):

    seed_database(tmp_path)

    pairs = make_report(tmp_path).closed_pairs()

    assert [pair["entry"]["action"] for pair in pairs] == ["BUY", "SELL"]
    assert [pair["entry"]["id"] for pair in pairs] == [1, 4]
    assert [pair["close"]["pnl_pct"] for pair in pairs] == [-1.5, 2.5]


def test_forward_returns_are_signed_by_the_trade_direction(tmp_path):

    report = make_report(tmp_path)
    bars = [
        [0, 0, 0, 0, 69000.0, 0],
        [0, 0, 0, 0, 68000.0, 0],
        [0, 0, 0, 0, 67000.0, 0],
        [0, 0, 0, 0, 66000.0, 0],
        [0, 0, 0, 0, 65000.0, 0],
        [0, 0, 0, 0, 64000.0, 0],
    ]

    short = report.forward_returns({"action": "SELL", "price": 70000.0}, bars)
    long = report.forward_returns({"action": "BUY", "price": 70000.0}, bars)

    assert short[3] == pytest.approx(4.286, abs=0.01)
    assert long[3] == pytest.approx(-4.286, abs=0.01)
    assert set(short) == {1, 3, 6}


def test_forward_returns_need_candles_and_a_price(tmp_path):

    report = make_report(tmp_path)

    assert report.forward_returns({"action": "BUY", "price": 70000.0}, []) == {}
    assert report.forward_returns({"action": "BUY", "price": None}, [[0, 0, 0, 0, 1, 0]]) == {}


def test_ranks_average_ties():

    assert ranks([10.0, 20.0, 20.0, 30.0]) == [1.0, 2.5, 2.5, 4.0]


def test_spearman_reads_a_monotonic_pair_and_refuses_a_tiny_sample():

    assert spearman([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]) == 1.0
    assert spearman([(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]) == -1.0
    assert spearman([(1.0, 1.0), (2.0, 2.0)]) is None


def test_histogram_buckets_the_score(tmp_path):

    report = make_report(tmp_path)

    assert report._histogram([]) == "none"
    assert report._histogram([0.2, 1.6, 2.4, 3.0]) == "0:1 1:0 2:2 3:1"


def test_bars_to_array_keeps_the_six_columns():

    array = bars_to_array([["1", "2", "3", "4", "5", "6"]])

    assert array.shape == (1, 6)
    assert array[0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
