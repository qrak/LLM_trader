"""Tests for GET /api/brain/decision-summary aggregation and graph topology."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.dashboard.dashboard_state import DashboardState
from src.dashboard.routers.brain import (
    BrainRouter,
    _build_decision_graph,
    _build_decision_synopsis,
)
from src.trading.data_models import Position


@pytest.fixture
def config(tmp_path):
    trading = tmp_path / "trading"
    trading.mkdir()
    (trading / "previous_response.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-07-09T12:00:00+00:00",
                "technical_data": {
                    "adx": 28.0,
                    "rsi": 55.0,
                    "atr_percent": 2.0,
                    "plus_di": 30.0,
                    "minus_di": 12.0,
                },
                "response": {
                    "text_analysis": (
                        "SIGNAL: HOLD\nConfidence: 75%\n"
                        "Thesis: Range-bound under resistance; wait for break.\n"
                        "Invalidation: daily close above 71000."
                    ),
                    "current_price": 70000.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        DATA_DIR=str(tmp_path),
        TIMEFRAME="4h",
        STOP_LOSS_TYPE="hard",
        STOP_LOSS_CHECK_INTERVAL="15m",
        TAKE_PROFIT_TYPE="hard",
        TAKE_PROFIT_CHECK_INTERVAL="15m",
        CRYPTO_PAIR="BTC/USDC",
    )


def _make_router(config, **kwargs):
    logger = MagicMock()
    state = kwargs.get("dashboard_state") or DashboardState()
    vector_memory = kwargs.get("vector_memory", "__default__")
    if vector_memory == "__default__":
        vector_memory = MagicMock()
        vector_memory.trade_count = 2
        vector_memory.semantic_rule_count = 1
        vector_memory.experience_count = 2
        vector_memory.compute_confidence_stats.return_value = {"HIGH": 1, "LOW": 1}
        vector_memory.compute_adx_performance.return_value = {}
        vector_memory.compute_factor_performance.return_value = {}
        vector_memory.get_active_rules.return_value = [
            {
                "text": "Avoid longs into low ADX bear flags",
                "metadata": {
                    "rule_type": "anti_pattern",
                    "win_rate": 0.2,
                    "final_score": 2.0,
                    "recommended_adjustment": "Wait for ADX>25",
                },
                "final_score": 2.0,
            }
        ]
        vector_memory.retrieve_similar_experiences.return_value = []
        vector_memory.get_all_experiences.return_value = []
        vector_memory.get_recent_blocked_trades.return_value = []
        vector_memory.get_blocked_trade_count.return_value = 0

    persistence = kwargs.get("persistence", "__default__")
    if persistence == "__default__":
        persistence = MagicMock()
        persistence.load_position.return_value = None
        persistence.load_trade_history.return_value = []

    post_mortem_repo = kwargs.get("post_mortem_repo", "__default__")
    if post_mortem_repo == "__default__":
        post_mortem_repo = MagicMock()
        post_mortem_repo.get_recent_post_mortems.return_value = [
            {
                "verdict": "premature_entry",
                "symbol": "BTCUSDC",
                "direction": "LONG",
                "pnl_pct": -1.2,
                "close_reason": "stop_loss",
                "lesson_learned": "Do not long into falling volume.",
                "created_at": "2026-07-01T00:00:00+00:00",
            }
        ]
        post_mortem_repo.search_post_mortems.return_value = []

    return BrainRouter(
        config=config,
        logger=logger,
        dashboard_state=state,
        vector_memory=vector_memory,
        unified_parser=None,
        persistence=persistence,
        exchange_manager=None,
        post_mortem_repo=post_mortem_repo,
    )


@pytest.mark.asyncio
async def test_decision_summary_has_required_sections(config):
    router = _make_router(config)
    result = await router.get_decision_summary()
    for key in (
        "generated_at",
        "synopsis",
        "now",
        "last_decision",
        "position",
        "memory",
        "journal",
        "counts",
        "graph",
    ):
        assert key in result
    assert result["position"]["has_position"] is False
    assert result["journal"]["count"] >= 1
    assert isinstance(result["synopsis"], str) and len(result["synopsis"]) > 20
    assert "nodes" in result["graph"] and "edges" in result["graph"]


@pytest.mark.asyncio
async def test_decision_summary_synopsis_mentions_flat_when_no_position(config):
    router = _make_router(config)
    result = await router.get_decision_summary()
    low = result["synopsis"].lower()
    assert "no open position" in low or "flat" in low or "no active position" in low


@pytest.mark.asyncio
async def test_decision_summary_tolerates_missing_deps(config):
    router = _make_router(config, vector_memory=None, post_mortem_repo=None)
    result = await router.get_decision_summary()
    assert result["memory"]["top_experiences"] == []
    assert result["journal"]["items"] == []
    assert "synopsis" in result
    node_ids = {n["id"] for n in result["graph"]["nodes"]}
    assert "hub_now" in node_ids
    assert "hub_position" in node_ids


@pytest.mark.asyncio
async def test_decision_summary_graph_topology_labels(config):
    router = _make_router(config)
    result = await router.get_decision_summary()
    nodes = result["graph"]["nodes"]
    edges = result["graph"]["edges"]
    by_id = {n["id"]: n for n in nodes}

    for required in ("hub_now", "hub_position", "hub_memory", "hub_rules", "hub_journal", "hub_context"):
        assert required in by_id, f"missing {required}"
        assert by_id[required]["label"], f"empty label on {required}"
        assert by_id[required]["type"], f"empty type on {required}"

    assert by_id["hub_position"]["label"] == "FLAT"
    assert any(e["from"] == "hub_now" and e["to"] == "hub_journal" for e in edges)
    assert any(n["id"].startswith("pm_") for n in nodes)
    assert any(n["id"].startswith("rule_") for n in nodes)
    for n in nodes:
        assert n.get("label"), f"node {n.get('id')} missing label"
        assert n.get("type"), f"node {n.get('id')} missing type"


@pytest.mark.asyncio
async def test_decision_summary_includes_open_position_fields(config):
    position = Position(
        entry_price=68000.0,
        stop_loss=67000.0,
        take_profit=71000.0,
        size=0.01,
        entry_time=datetime.now(timezone.utc),
        confidence="HIGH",
        direction="LONG",
        symbol="BTC/USDC",
        sl_distance_pct=0.0147,
        tp_distance_pct=0.0441,
        rr_ratio_at_entry=2.0,
        stop_loss_type_at_entry="hard",
        stop_loss_check_interval_at_entry="15m",
        take_profit_type_at_entry="hard",
        take_profit_check_interval_at_entry="15m",
    )
    persistence = MagicMock()
    persistence.load_position.return_value = position
    persistence.load_trade_history.return_value = []
    router = _make_router(config, persistence=persistence)
    result = await router.get_decision_summary()
    assert result["position"]["has_position"] is True
    assert "LONG" in result["synopsis"] or "long" in result["synopsis"].lower()
    pos_node = next(n for n in result["graph"]["nodes"] if n["id"] == "hub_position")
    assert "LONG" in pos_node["label"]


@pytest.mark.asyncio
async def test_decision_summary_cache_invalidation(config):
    state = DashboardState()
    router = _make_router(config, dashboard_state=state)
    first = await router.get_decision_summary()
    state.invalidate_brain_caches()
    second = await router.get_decision_summary()
    assert first["generated_at"] and second["generated_at"]
    assert "decision_summary" not in state._cache or state.get_cached("decision_summary", ttl_seconds=15.0)


def test_build_decision_graph_pure():
    graph = _build_decision_graph(
        now={"action": "HOLD", "confidence": 80, "trend": "BEARISH", "adx": 20, "rsi": 40},
        last_decision={"signal": "HOLD", "confidence": 80, "reasoning_excerpt": "Wait"},
        position={"has_position": False},
        memory={
            "current_context": "BEARISH + Low ADX",
            "experience_count": 2,
            "rule_count": 1,
            "top_experiences": [
                {
                    "outcome": "WIN",
                    "pnl_pct": 2.1,
                    "direction": "LONG",
                    "similarity": 0.9,
                    "document_excerpt": "good long",
                }
            ],
            "top_rules": [
                {"rule_type": "anti_pattern", "rule_text": "Avoid chop longs", "final_score": 1}
            ],
            "blocked": {"blocked_count": 0, "items": []},
        },
        journal={
            "count": 1,
            "items": [
                {
                    "verdict": "premature_entry",
                    "symbol": "BTC",
                    "lesson_learned": "wait",
                }
            ],
        },
    )
    ids = {n["id"] for n in graph["nodes"]}
    assert "hub_now" in ids and "exp_0" in ids and "rule_0" in ids and "pm_0" in ids
    assert all(n["label"] for n in graph["nodes"])


def test_build_decision_synopsis_flat():
    text = _build_decision_synopsis(
        now={"action": "HOLD", "confidence": 75, "trend": "BEARISH"},
        position={"has_position": False},
        memory={"current_context": "BEARISH + Low ADX", "top_rules": [], "blocked": {}},
        journal={"items": []},
        last_decision={"signal": "HOLD", "confidence": 75},
    )
    assert "flat" in text.lower() or "no open position" in text.lower()
    assert "HOLD" in text
