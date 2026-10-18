"""Cycle checkpoint: stable intent identity, atomic file, duplicate suppression."""

import json

from src.trading.cycle_state import (
    CycleCheckpoint,
    CycleCheckpointStore,
    CycleState,
    hash_decision,
    is_duplicate_intent,
)

ANALYSIS = {"signal": "SELL", "entry_price": 82412.23, "stop_loss": 83500.0,
            "take_profit": 80302.84, "quantity": 0.00581}


def test_hash_is_stable_across_key_order_and_changes_with_price():
    same = {"quantity": 0.00581, "take_profit": 80302.84, "stop_loss": 83500.0,
            "entry_price": 82412.23, "signal": "SELL"}
    moved = {**ANALYSIS, "entry_price": 82413.0}

    first = hash_decision("SELL", "BTC/USDC", ANALYSIS)

    assert first == hash_decision("SELL", "BTC/USDC", same)
    assert first != hash_decision("SELL", "BTC/USDC", moved)
    assert first != hash_decision("BUY", "BTC/USDC", ANALYSIS)
    assert len(first) == 16


def test_save_load_roundtrip_is_atomic_and_leaves_no_temp_file(tmp_path):
    store = CycleCheckpointStore(tmp_path / "cycle_state.json")

    store.record(CycleState.INTENT_SENT, symbol="BTC/USDC", decision_hash="abc123")
    loaded = store.load()

    assert loaded is not None
    assert loaded.state == "intent_sent"
    assert loaded.decision_hash == "abc123"
    assert not list(tmp_path.glob(".cycle_state-*.tmp"))


def test_corrupt_file_reads_as_none(tmp_path):
    path = tmp_path / "cycle_state.json"
    path.write_text("{not json", encoding="utf-8")

    assert CycleCheckpointStore(path).load() is None


def test_unknown_state_still_round_trips(tmp_path):
    path = tmp_path / "cycle_state.json"
    path.write_text(json.dumps({"cycle_id": "x", "state": "future_state", "ts": "t"}), encoding="utf-8")

    loaded = CycleCheckpointStore(path).load()

    assert loaded is not None and loaded.state == "future_state"


def test_duplicate_requires_same_hash_and_unconfirmed_intent():
    checkpoint = CycleCheckpoint(cycle_id="c1", state=CycleState.INTENT_SENT.value, ts="t",
                                 symbol="BTC/USDC", decision_hash="abc123")

    assert is_duplicate_intent(checkpoint, "abc123") is True
    assert is_duplicate_intent(checkpoint, "other") is False
    assert is_duplicate_intent(None, "abc123") is False
    assert is_duplicate_intent(
        CycleCheckpoint(cycle_id="c2", state=CycleState.CONFIRMED.value, ts="t", decision_hash="abc123"),
        "abc123",
    ) is False
