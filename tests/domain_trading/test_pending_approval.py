"""Approval gate: pending → approve/reject, expiry is a refusal, state survives restart."""

from datetime import datetime, timedelta, timezone

from src.trading.pending_approval import (
    APPROVED,
    EXPIRED,
    PENDING,
    REJECTED,
    PendingApprovalStore,
    requires_approval,
)

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)


def _store(tmp_path) -> PendingApprovalStore:
    return PendingApprovalStore(tmp_path / "approvals.json")


def _submit(store, *, ttl=15, now=NOW):
    return store.submit(
        action="BUY", symbol="BTC/USDC", decision_hash="abc123",
        payload={"analysis": {"signal": "BUY"}, "action": "BUY"},
        quote_amount=60.0, ttl_minutes=ttl, now=now,
    )


def test_submit_is_pending_and_approvable(tmp_path):
    store = _store(tmp_path)
    approval = _submit(store)

    assert approval.status == PENDING
    assert [item.approval_id for item in store.pending(now=NOW)] == [approval.approval_id]

    decided = store.approve(approval.approval_id, now=NOW)

    assert decided is not None and decided.status == APPROVED
    assert store.pending(now=NOW) == []


def test_expired_entry_cannot_be_approved(tmp_path):
    store = _store(tmp_path)
    approval = _submit(store, ttl=15)

    decided = store.approve(approval.approval_id, now=NOW + timedelta(minutes=16))

    assert decided is not None
    assert decided.status == EXPIRED


def test_reject_is_terminal(tmp_path):
    store = _store(tmp_path)
    approval = _submit(store)

    rejected = store.reject(approval.approval_id, now=NOW)
    assert rejected is not None and rejected.status == REJECTED
    late = store.approve(approval.approval_id, now=NOW)
    assert late is not None and late.status == REJECTED


def test_unknown_id_returns_none(tmp_path):
    assert _store(tmp_path).approve("missing", now=NOW) is None


def test_state_survives_a_new_store_instance(tmp_path):
    approval = _submit(_store(tmp_path))

    reloaded = PendingApprovalStore(tmp_path / "approvals.json").get(approval.approval_id, now=NOW)

    assert reloaded is not None
    assert reloaded.payload["action"] == "BUY"
    assert reloaded.decision_hash == "abc123"


def test_corrupt_file_does_not_raise(tmp_path):
    path = tmp_path / "approvals.json"
    path.write_text("{broken", encoding="utf-8")

    assert PendingApprovalStore(path).pending(now=NOW) == []


def test_requires_approval_threshold_semantics():
    assert requires_approval(60.0, 50) is True
    assert requires_approval(49.0, 50) is False
    assert requires_approval(60.0, 0) is False, "zero threshold disables the gate"
    assert requires_approval(None, 50) is False
    assert requires_approval(60.0, None) is False

def test_store_rejects_non_path_input():
    """A MagicMock path used to create a stray MagicMock/ directory in the repo."""
    import pytest as _pytest

    with _pytest.raises(TypeError):
        PendingApprovalStore(object())
