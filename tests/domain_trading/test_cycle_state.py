

import json
from pathlib import Path

from src.trading.cycle_state import (
    CycleCheckpoint,
    CycleCheckpointStore,
    hash_decision,
    is_duplicate_intent,
)
from src.trading.executor_reconciliation import PositionIntentLedger
from src.trading.lifecycle import (
    BRAIN_LIFECYCLE_LABELS,
    BRAIN_LIFECYCLE_SEVERITIES,
    CYCLE_LABELS,
    CYCLE_SEVERITIES,
    CYCLE_TRANSITIONS,
    EXECUTOR_STATE_LABELS,
    INTENT_ACTION_ENTRY,
    INTENT_CONFIRMED,
    INTENT_LABELS,
    INTENT_LOCAL_ONLY,
    INTENT_PENDING,
    INTENT_REFUSED,
    INTENT_TRANSITIONS,
    INTENT_UNKNOWN,
    POSITION_PHASE_LABELS,
    POSITION_PHASE_SEVERITIES,
    SKIP_REASON_LABELS,
    BrainLifecycle,
    CycleState,
    ExecutorQueryState,
    LifecycleProjection,
    PositionPhase,
)
from src.trading.order_lifecycle import OrderLifecycle

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


def test_every_cycle_state_is_reachable_and_only_analyzing_is_initial():
    targets = set().union(*CYCLE_TRANSITIONS.values())

    assert set(CYCLE_TRANSITIONS) - targets == {CycleState.ANALYZING}


def test_terminal_cycle_states_do_not_transition():
    assert CycleState.INTENT_SENT.can_transition_to(CycleState.CONFIRMED) is True
    assert CycleState.SKIPPED.can_transition_to(CycleState.DECIDED) is False
    assert CycleState.CONFIRMED.can_transition_to(CycleState.ANALYZING) is False
    assert CycleState.FAILED.can_transition_to(CycleState.DECIDED) is False


def test_skip_label_explains_the_reason_code():
    label = LifecycleProjection.cycle_label(CycleState.SKIPPED, "duplicate_intent")

    assert "duplicate intent" in label


def test_order_lifecycle_has_no_execution_terminal():
    assert {state.value for state in OrderLifecycle} == {
        "INTENT",
        "READY_FOR_REVIEW",
        "REJECTED",
    }


def test_pending_entry_survives_a_restart_and_clears_when_confirmed(tmp_path):
    path = tmp_path / "intents.jsonl"
    ledger = PositionIntentLedger(path)
    ledger.record(
        INTENT_ACTION_ENTRY,
        "BTC/USDC",
        key="entry:BTC/USDC:order-1",
        order_id="order-1",
        decision_hash="hash-1",
        state=INTENT_PENDING,
    )

    assert "hash-1" in ledger.pending_entry_hashes()
    assert "other" not in ledger.pending_entry_hashes()
    assert ledger.pending_entry_hashes() == frozenset({"hash-1"})
    assert PositionIntentLedger(path).pending_entry_hashes() == frozenset({"hash-1"})

    ledger.record(
        INTENT_ACTION_ENTRY,
        "BTC/USDC",
        key="entry:BTC/USDC:order-1",
        order_id="order-1",
        state=INTENT_CONFIRMED,
        evidence="executor_verdict_journal",
    )

    assert "hash-1" not in ledger.pending_entry_hashes()


def test_transition_tables_cover_every_declared_state():
    cycle_states = set(CycleState)
    assert set(CYCLE_TRANSITIONS) == cycle_states
    assert set().union(*CYCLE_TRANSITIONS.values()) <= cycle_states
    assert all(state not in targets for state, targets in CYCLE_TRANSITIONS.items())

    intent_states = {
        INTENT_PENDING, INTENT_UNKNOWN, INTENT_CONFIRMED, INTENT_REFUSED, INTENT_LOCAL_ONLY,
    }
    assert set(INTENT_TRANSITIONS) == intent_states
    assert set().union(*INTENT_TRANSITIONS.values()) <= intent_states
    assert all(state not in targets for state, targets in INTENT_TRANSITIONS.items())


def test_every_state_has_a_label_and_severity():
    assert set(CYCLE_LABELS) == set(CycleState)
    assert set(CYCLE_SEVERITIES) == set(CycleState)
    assert set(POSITION_PHASE_LABELS) == set(PositionPhase)
    assert set(POSITION_PHASE_SEVERITIES) == set(PositionPhase)
    assert set(BRAIN_LIFECYCLE_LABELS) == set(BrainLifecycle)
    assert set(BRAIN_LIFECYCLE_SEVERITIES) == set(BrainLifecycle)
    assert set(EXECUTOR_STATE_LABELS) == set(ExecutorQueryState)
    assert set(INTENT_LABELS) == set(INTENT_TRANSITIONS)


def test_every_cycle_detail_has_an_operator_label():
    for detail in (
        "analysis_error",
        "stale_position",
        "protection_changed",
        "duplicate_intent",
        "approval_not_granted",
        "forward_failed",
        "no_decision",
        "no_analysis",
        "executor_disabled",
        "hold",
        "no_verdict",
        "forwarded",
        "confirmed",
        "unknown",
        "refused",
        "local_only",
    ):
        assert detail in SKIP_REASON_LABELS, detail


def test_state_machine_document_covers_the_vocabulary():
    document = (
        Path(__file__).resolve().parents[2] / "docs" / "STATE_MACHINES.md"
    ).read_text(encoding="utf-8")

    names = {
        "CycleState",
        "INTENT_PENDING",
        "INTENT_UNKNOWN",
        "INTENT_CONFIRMED",
        "INTENT_REFUSED",
        "INTENT_LOCAL_ONLY",
        "RECONCILE_NO_POSITION",
        "RECONCILE_OPEN_EXECUTOR_REPORTED",
        "RECONCILE_EXIT_BOOKED",
        "RECONCILE_DIVERGENCE",
        "RECONCILE_UNVERIFIED",
        "VERIFICATION_EXCHANGE_VERIFIED",
        "VERIFICATION_EXECUTOR_REPORTED",
        "VERIFICATION_UNVERIFIED",
        "PositionPhase",
        "ExecutorQueryState",
        "BrainLifecycle",
    }
    for state in (*CycleState, *PositionPhase, *ExecutorQueryState, *BrainLifecycle):
        names.add(state.value)

    for name in names:
        assert name in document, name
