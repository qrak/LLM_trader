"""Brain, performance and serving-layer dashboard tests.

Covers the brain and performance routers, the cache and security middleware of
src/dashboard/server.py, the decision presenters and the static dashboard bindings.
"""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

from src.dashboard.dashboard_state import DashboardState
from src.dashboard.decision_presenter import (
    build_current_market_context,
    distance_pct_or_fallback,
    extract_market_status,
)
from src.dashboard.routers.brain import BrainRouter
from src.dashboard.routers.performance import PerformanceRouter
from src.dashboard.server import DashboardServer, _api_cache_policies
from src.trading.data_models import VectorSearchResult
from tests.conftest import make_config, make_position

STATIC_DIR = Path(__file__).resolve().parents[2] / "src" / "dashboard" / "static"

HARD_EXIT_CONFIG = {
    "TIMEFRAME": "4h",
    "STOP_LOSS_TYPE": "hard",
    "STOP_LOSS_CHECK_INTERVAL": "15m",
    "TAKE_PROFIT_TYPE": "hard",
    "TAKE_PROFIT_CHECK_INTERVAL": "15m",
}

EXIT_MANAGEMENT_HARD_15M = {
    "stop_loss_type": "hard",
    "stop_loss_check_interval": "15m",
    "take_profit_type": "hard",
    "take_profit_check_interval": "15m",
}

RISK_MANAGEMENT_HARD_15M = {
    "current": EXIT_MANAGEMENT_HARD_15M,
    "at_entry": EXIT_MANAGEMENT_HARD_15M,
    "current_labels": {"stop_loss": "hard / 15m", "take_profit": "hard / 15m"},
    "at_entry_labels": {"stop_loss": "hard / 15m", "take_profit": "hard / 15m"},
    "policy_changed": False,
}

LIFECYCLE_IDLE = {
    "status": "idle",
    "started_at": None,
    "completed_at": None,
    "message": "",
    "sequence": 0,
}

DEFAULT_STATISTICS = {
    "total_trades": 0,
    "winning_trades": 0,
    "losing_trades": 0,
    "win_rate": 0.0,
    "total_pnl_pct": 0.0,
    "total_pnl_quote": 0.0,
    "initial_capital": 10000.0,
    "current_capital": 10000.0,
    "avg_trade_pct": 0.0,
    "best_trade_pct": 0.0,
    "worst_trade_pct": 0.0,
    "max_drawdown_pct": 0.0,
    "avg_drawdown_pct": 0.0,
    "sharpe_ratio": 0.0,
    "sortino_ratio": 0.0,
    "profit_factor": 0.0,
}

NO_STORE_BROWSER = "no-store, no-cache, must-revalidate, proxy-revalidate"
API_BROWSER = "public, max-age=15"
API_EDGE = "public, max-age=60, stale-while-revalidate=30, stale-if-error=300"
ASSET_BROWSER = "public, max-age=3600"
ASSET_EDGE = "public, max-age=86400, stale-while-revalidate=3600, stale-if-error=86400"
SHELL_BROWSER = "public, max-age=30, must-revalidate"
SHELL_EDGE = "public, max-age=300, stale-while-revalidate=60, stale-if-error=600"
IMMUTABLE_BROWSER = "public, max-age=31536000, immutable"
IMMUTABLE_EDGE = (
    "public, max-age=31536000, stale-while-revalidate=86400, stale-if-error=604800"
)

POST_MORTEM_RECENT = {
    "id": 1,
    "symbol": "BTC/USDC",
    "verdict": "good_exit",
    "lesson_learned": "Follow the plan",
    "pnl_pct": 2.5,
}

POST_MORTEM_SEARCHED = {
    "id": 2,
    "symbol": "BTC/USDC",
    "verdict": "held_too_long",
    "lesson_learned": "Take profit earlier",
    "llm_analysis": "Breakout failed",
}


def hard_exit_config(data_dir) -> object:
    """Config double pinned to the hard/15m exit policy used across these tests."""
    return make_config(DATA_DIR=str(data_dir), **HARD_EXIT_CONFIG)


def persist_previous_response(root: Path, raw: str) -> Path:
    """Write root/trading/previous_response.json with the given raw text."""
    path = root / "trading" / "previous_response.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw, encoding="utf-8")
    return path


def flat_persistence() -> MagicMock:
    """Persistence double that reports no open position."""
    persistence = MagicMock()
    persistence.load_position.return_value = None
    return persistence


def brain_router(
    config,
    *,
    state=None,
    vector_memory=None,
    persistence=None,
    exchange_manager=None,
    post_mortem_repo=None,
) -> BrainRouter:
    """BrainRouter wired to silent doubles for the dependencies the test does not set."""
    return BrainRouter(
        config=config,
        logger=MagicMock(),
        dashboard_state=DashboardState() if state is None else state,
        vector_memory=vector_memory,
        unified_parser=None,
        persistence=persistence,
        exchange_manager=exchange_manager,
        post_mortem_repo=post_mortem_repo,
    )


def performance_router(config, state, persistence=None) -> PerformanceRouter:
    """PerformanceRouter with a silent logger."""
    return PerformanceRouter(
        config=config, logger=MagicMock(), dashboard_state=state, persistence=persistence
    )


def vector_memory_double(experiences, *, trade_count, min_trades=3) -> MagicMock:
    """Vector-memory double with fixed counters and a fixed retrieval payload."""
    memory = MagicMock()
    memory.trade_count = trade_count
    memory.MIN_EVIDENCE_TRADES = min_trades
    memory.semantic_rule_count = 0
    memory.compute_confidence_stats.return_value = {}
    memory.compute_adx_performance.return_value = {}
    memory.compute_factor_performance.return_value = {}
    memory.retrieve_similar_experiences.return_value = experiences
    memory.get_all_experiences.return_value = experiences
    return memory


def confidence_experiences() -> list:
    """Fresh experience list whose confidences are HIGH, unknown and missing."""
    return [
        VectorSearchResult("a", "doc", 1.0, 0.0, 1.0, {"confidence": "HIGH"}),
        VectorSearchResult("b", "doc", 2.0, 0.0, 2.0, {"confidence": "weird"}),
        VectorSearchResult("c", "doc", 3.0, 0.0, 3.0, {}),
    ]


def exchange_double(ticker=None, error=None) -> MagicMock:
    """Exchange double whose fetch_ticker returns ticker or raises error."""
    exchange = MagicMock()
    exchange.fetch_ticker = (
        AsyncMock(side_effect=error) if error else AsyncMock(return_value=ticker)
    )
    return exchange


def exchange_manager_for(exchange) -> MagicMock:
    """Exchange-manager double resolving every symbol to the given exchange."""
    manager = MagicMock()
    manager.find_symbol_exchange = AsyncMock(return_value=(exchange, "binance"))
    return manager


def without_last_updated(payload: dict) -> dict:
    """Return a statistics payload without its generated timestamp."""
    return {key: value for key, value in payload.items() if key != "last_updated"}


def assert_cache_policy(response, expected_status, browser_policy, edge_policy) -> None:
    """Assert status plus the browser, edge and Cloudflare cache policies."""
    assert response.status_code == expected_status
    assert response.headers["Cache-Control"] == browser_policy
    assert response.headers["CDN-Cache-Control"] == edge_policy
    assert response.headers["Cloudflare-CDN-Cache-Control"] == edge_policy


@pytest.fixture
def dashboard_server(tmp_path) -> DashboardServer:
    """Full DashboardServer with silent services and an empty data directory."""
    return DashboardServer(
        brain_service=MagicMock(),
        vector_memory=MagicMock(),
        analysis_engine=MagicMock(),
        config=make_config(DATA_DIR=str(tmp_path), DEMO_QUOTE_CAPITAL=10000.0),
        logger=MagicMock(),
        unified_parser=MagicMock(),
        persistence=flat_persistence(),
        exchange_manager=MagicMock(),
    )


@pytest.fixture
def client(dashboard_server) -> TestClient:
    """TestClient driving the real middleware stack and lifespan."""
    with TestClient(dashboard_server.app) as test_client:
        yield test_client


def test_build_current_market_context_renders_indicators_and_exit_execution(tmp_path):
    persist_previous_response(
        tmp_path,
        json.dumps(
            {
                "response": {
                    "text_analysis": "SIGNAL: UPDATE\nConfidence: 82%\nTrend remains bearish.",
                    "adx": 18.1,
                    "rsi": 40.5,
                    "atr_percent": 1.2,
                    "plus_di": 14.8,
                    "minus_di": 24.2,
                    "macd_line": -268.6,
                    "macd_signal": -28.0,
                    "obv_slope": -0.7,
                    "bb_upper": 72140.5,
                    "bb_lower": 68571.7,
                    "current_price": 68795.96,
                },
                "timestamp": "2026-03-27T00:00:50.843610+00:00",
            }
        ),
    )

    display_context, query_document = build_current_market_context(
        hard_exit_config(tmp_path), MagicMock()
    )

    assert display_context == (
        "BEARISH + Low ADX + LOW Volatility + MACD BEARISH + Volume DISTRIBUTION"
        " + Price at BB LOWER + Exit Execution: SL hard/15m | TP hard/15m"
    )
    assert query_document.startswith(
        f"{display_context} Indicators: ADX=18.1 (Low ADX) | RSI=40.5 (NEUTRAL)"
        " | Vol=LOW | MACD=BEARISH | BB=LOWER | VolState=DISTRIBUTION"
    )

    persist_previous_response(
        tmp_path,
        json.dumps(
            {
                "technical_data": {
                    "adx": 28.0,
                    "rsi": 61.0,
                    "atr_percent": 2.1,
                    "plus_di": 30.0,
                    "minus_di": 10.0,
                },
                "response": {"current_price": 70000.0},
            }
        ),
    )
    soft_tp_config = make_config(
        DATA_DIR=str(tmp_path),
        **{
            **HARD_EXIT_CONFIG,
            "TAKE_PROFIT_TYPE": "soft",
            "TAKE_PROFIT_CHECK_INTERVAL": "4h",
        },
    )

    display_context, query_document = build_current_market_context(soft_tp_config, MagicMock())

    assert display_context == (
        "BULLISH + High ADX + MEDIUM Volatility + RSI STRONG"
        " + Exit Execution: SL hard/15m | TP soft/4h"
    )
    assert "Indicators: ADX=28.0 (High ADX)" in query_document
    assert "Exit Execution: SL hard/15m | TP soft/4h" in query_document


def test_build_current_market_context_returns_empty_pair_for_unusable_payloads(tmp_path):
    persist_previous_response(tmp_path, json.dumps([]))

    assert build_current_market_context(hard_exit_config(tmp_path), MagicMock()) == ("", "")

    persist_previous_response(tmp_path, '{"response": ')
    logger = MagicMock()

    assert build_current_market_context(hard_exit_config(tmp_path), logger) == ("", "")
    logger.error.assert_called_once_with("Failed to build market context", exc_info=True)


def test_build_current_market_context_falls_back_to_text_status(tmp_path):
    persist_previous_response(
        tmp_path,
        json.dumps(
            {
                "response": {
                    "text_analysis": "The structure remains bearish.\nSIGNAL: UPDATE\n"
                    "Confidence: 82%"
                }
            }
        ),
    )

    display_context, query_document = build_current_market_context(
        hard_exit_config(tmp_path), MagicMock()
    )

    assert display_context == (
        "BEARISH + Low ADX + MEDIUM Volatility"
        " + Exit Execution: SL hard/15m | TP hard/15m"
    )
    assert query_document == display_context


@pytest.mark.parametrize(
    ("parser_block", "payload", "expected"),
    [
        (
            None,
            {
                "response": {
                    "text_analysis": "SIGNAL: UPDATE\nConfidence: 82%\n"
                    "The structure remains bearish.\n```json\n{bad json}\n```",
                    "adx": 18.1,
                    "rsi": 40.5,
                    "plus_di": 14.8,
                    "minus_di": 24.2,
                }
            },
            {"trend": "BEARISH", "action": "UPDATE", "confidence": 82, "adx": 18.1, "rsi": 40.5},
        ),
        (
            {"signal": "UPDATE", "confidence": 75},
            {
                "response": {
                    "text_analysis": '```json\n{"analysis": {"signal": "UPDATE", '
                    '"confidence": 75}}\n```',
                    "adx": 19.9,
                    "rsi": 61.0,
                    "plus_di": 30.0,
                    "minus_di": 16.8,
                }
            },
            {"trend": "BULLISH", "action": "UPDATE", "confidence": 75, "adx": 19.9, "rsi": 61.0},
        ),
        (
            None,
            {"response": {"text_analysis": "Momentum stays BULLISH"}},
            {"trend": "BULLISH", "action": "--", "confidence": "--", "adx": None, "rsi": None},
        ),
        (
            None,
            {"response": {"text_analysis": "SIGNAL: BUY\nConfidence: 82.5%"}},
            {"trend": "NEUTRAL", "action": "BUY", "confidence": 82.5, "adx": None, "rsi": None},
        ),
    ],
    ids=["text_fallback", "parsed_json_block", "text_without_signal", "fractional_confidence"],
)
def test_extract_market_status_merges_text_and_technical_data(parser_block, payload, expected):
    parser = MagicMock()
    parser.extract_json_block.return_value = parser_block

    status = extract_market_status(payload, parser)

    assert status == expected
    parser.extract_json_block.assert_called_once_with(
        payload["response"]["text_analysis"], unwrap_key="analysis"
    )


async def test_get_active_rules_maps_metadata_and_rewrites_unknown_exit_profile(tmp_path):
    vector_memory = MagicMock()
    vector_memory.get_active_rules.return_value = [
        {
            "rule_id": "rule-ai-1",
            "text": "AI MISTAKE: high confidence breakout failed in chop",
            "metadata": {
                "rule_type": "ai_mistake",
                "source_trades": 3,
                "wins": 0,
                "losses": 3,
                "win_rate": 0.0,
                "avg_pnl_pct": -0.8,
                "mistake_type": "sideways_overconfidence",
                "entry_confidence": "HIGH",
                "failed_assumption": "expected breakout continuation",
                "failure_reason": "AI used HIGH confidence in chop",
                "recommended_adjustment": "downgrade confidence until ADX confirms expansion",
                "dominant_exit_profile": "SL hard/1m | TP soft/15m",
                "dominant_stop_loss_type": "hard",
                "dominant_stop_loss_interval": "1m",
                "dominant_take_profit_type": "soft",
                "dominant_take_profit_interval": "15m",
                "freshness_label": "fresh",
                "freshness_score": 97.5,
                "evidence_score": 70.0,
                "final_score": 82.0,
                "support_count": 3,
                "validation_hit_count": 2,
                "contradiction_count": 1,
                "source_timeframe_minutes": 240,
                "source_timeframe_bucket": "swing",
            },
            "final_score": 82.0,
        }
    ]
    router = brain_router(hard_exit_config(tmp_path), vector_memory=vector_memory)

    rules = await router.get_active_rules()

    assert len(rules) == 1
    expected_mapping = {
        "rule_text": "AI MISTAKE: high confidence breakout failed in chop",
        "rule_type": "ai_mistake",
        "source_trades": 3,
        "mistake_type": "sideways_overconfidence",
        "entry_confidence": "HIGH",
        "failed_assumption": "expected breakout continuation",
        "dominant_exit_profile": "SL hard/1m | TP soft/15m",
        "dominant_stop_loss_interval": "1m",
        "dominant_take_profit_interval": "15m",
        "freshness_label": "fresh",
        "freshness_score": 97.5,
        "final_score": 82.0,
        "support_count": 3,
        "validation_hit_count": 2,
        "contradiction_count": 1,
        "source_timeframe_bucket": "swing",
    }
    assert {field: rules[0][field] for field in expected_mapping} == expected_mapping

    legacy_memory = MagicMock()
    legacy_memory.get_active_rules.return_value = [
        {
            "rule_id": "rule_best_long_bullish_high_adx_sl_unknown_unknown_tp_unknown_unknown",
            "text": "LONG trades perform well. Exit profile: SL unknown/unknown"
            " | TP unknown/unknown. (3 wins)",
            "metadata": {
                "rule_type": "best_practice",
                "source_trades": 3,
                "dominant_exit_profile": "SL unknown/unknown | TP unknown/unknown",
                "dominant_stop_loss_type": "unknown",
                "dominant_take_profit_type": "unknown",
            },
        }
    ]
    rewritten = await brain_router(
        hard_exit_config(tmp_path), vector_memory=legacy_memory
    ).get_active_rules()

    assert rewritten[0]["rule_text"] == (
        "LONG trades perform well. Exit profile: SL hard/15m | TP hard/15m. (3 wins)"
    )
    assert rewritten[0]["dominant_exit_profile"] == "SL hard/15m | TP hard/15m"


async def test_get_active_rules_returns_empty_without_memory_or_when_retrieval_fails(tmp_path):
    config = hard_exit_config(tmp_path)

    assert await brain_router(config).get_active_rules() == []

    broken_memory = MagicMock()
    broken_memory.get_active_rules.side_effect = RuntimeError("chroma unreachable")

    assert await brain_router(config, vector_memory=broken_memory).get_active_rules() == []


async def test_get_current_position_recomputes_missing_distance_percentages(tmp_path):
    entry_price, stop_loss, take_profit = 69009.78, 69350.00, 65612.00
    position = make_position(
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        size=0.1,
        direction="SHORT",
        symbol="BTC/USDC",
        confidence="HIGH",
        sl_distance_pct=0.0,
        tp_distance_pct=0.0,
        rr_ratio_at_entry=2.2,
        stop_loss_type_at_entry="hard",
        stop_loss_check_interval_at_entry="15m",
        take_profit_type_at_entry="hard",
        take_profit_check_interval_at_entry="15m",
    )
    persistence = MagicMock()
    persistence.load_position.return_value = position
    router = brain_router(
        hard_exit_config(tmp_path),
        state=DashboardState(current_price=68366.03),
        persistence=persistence,
    )

    result = await router.get_current_position()

    assert result["has_position"] is True
    assert result["sl_distance_pct"] == pytest.approx(abs(stop_loss - entry_price) / entry_price)
    assert result["tp_distance_pct"] == pytest.approx(abs(take_profit - entry_price) / entry_price)
    assert result["exit_management"] == EXIT_MANAGEMENT_HARD_15M
    assert result["exit_management_at_entry"] == result["exit_management"]
    assert result["risk_management"] == RISK_MANAGEMENT_HARD_15M
    assert {
        "direction": result["direction"],
        "symbol": result["symbol"],
        "entry_price": result["entry_price"],
        "stop_loss": result["stop_loss"],
        "take_profit": result["take_profit"],
        "rr_ratio": result["rr_ratio"],
        "size": result["size"],
        "confidence": result["confidence"],
        "current_price": result["current_price"],
        "entry_time": result["entry_time"],
    } == {
        "direction": "SHORT",
        "symbol": "BTC/USDC",
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "rr_ratio": 2.2,
        "size": 0.1,
        "confidence": "HIGH",
        "current_price": 68366.03,
        "entry_time": "2026-09-17T00:00:00+00:00",
    }


def test_distance_pct_or_fallback_prefers_stored_positive_percentage():
    assert distance_pct_or_fallback(2.5, 100.0, 90.0) == 2.5
    assert distance_pct_or_fallback(0.0, 100.0, 90.0) == pytest.approx(0.1)
    assert distance_pct_or_fallback(-3.0, 100.0, 90.0) == pytest.approx(0.1)
    assert distance_pct_or_fallback(None, 100.0, 110.0) == pytest.approx(0.1)
    assert distance_pct_or_fallback(None, 0.0, 90.0) == 0.0


async def test_get_current_position_reports_missing_persistence_and_flat_payloads(tmp_path):
    config = hard_exit_config(tmp_path)

    without_persistence = await brain_router(config).get_current_position()

    assert without_persistence == {"has_position": False, "error": "Persistence not available"}

    priced = brain_router(
        config, state=DashboardState(current_price=123.0), persistence=flat_persistence()
    )
    assert await priced.get_current_position() == {
        "has_position": False,
        "current_price": 123.0,
        "exit_management": EXIT_MANAGEMENT_HARD_15M,
        "risk_management": RISK_MANAGEMENT_HARD_15M,
    }

    persist_previous_response(tmp_path, json.dumps({"prompt": "Current Price: $68,366.03 USD"}))
    parsed = await brain_router(config, persistence=flat_persistence()).get_current_position()

    assert parsed == {
        "has_position": False,
        "current_price": 68366.03,
        "exit_management": EXIT_MANAGEMENT_HARD_15M,
        "risk_management": RISK_MANAGEMENT_HARD_15M,
    }


async def test_get_brain_status_merges_defaults_config_and_statistics(tmp_path):
    config = hard_exit_config(tmp_path)
    trading_dir = tmp_path / "trading"
    trading_dir.mkdir()
    (trading_dir / "statistics.json").write_text(
        json.dumps({"total_trades": 12, "win_rate": 58.3, "current_capital": 11234.5}),
        encoding="utf-8",
    )

    merged = await brain_router(config).get_brain_status()

    assert merged == {
        "status": "active",
        "trend": "--",
        "confidence": "--",
        "action": "--",
        "adx": None,
        "rsi": None,
        "exit_management": EXIT_MANAGEMENT_HARD_15M,
        "brain_lifecycle": LIFECYCLE_IDLE,
        "total_trades": 12,
        "win_rate": 58.3,
        "current_capital": 11234.5,
    }

    (trading_dir / "statistics.json").unlink()
    defaults = await brain_router(config).get_brain_status()

    assert defaults == {
        "status": "active",
        "trend": "--",
        "confidence": "--",
        "action": "--",
        "adx": None,
        "rsi": None,
        "exit_management": EXIT_MANAGEMENT_HARD_15M,
        "brain_lifecycle": LIFECYCLE_IDLE,
    }


async def test_get_brain_status_serves_cache_then_recomputes_after_ttl(tmp_path):
    config = hard_exit_config(tmp_path)
    trading_dir = tmp_path / "trading"
    trading_dir.mkdir()
    (trading_dir / "statistics.json").write_text(
        json.dumps({"total_trades": 7}), encoding="utf-8"
    )
    state = DashboardState()
    state.brain_rebuild_status = "updating"
    state.set_cached("brain_status", {"status": "active", "trend": "BULLISH"})

    cached = await brain_router(config, state=state).get_brain_status()

    assert cached == {
        "status": "active",
        "trend": "BULLISH",
        "brain_lifecycle": {**LIFECYCLE_IDLE, "status": "updating"},
    }

    state.cache_timestamps["brain_status"] = time.time() - 31
    expired = await brain_router(config, state=state).get_brain_status()

    assert expired == {
        "status": "active",
        "trend": "--",
        "confidence": "--",
        "action": "--",
        "adx": None,
        "rsi": None,
        "exit_management": EXIT_MANAGEMENT_HARD_15M,
        "brain_lifecycle": {**LIFECYCLE_IDLE, "status": "updating"},
        "total_trades": 7,
        "win_rate": 0,
        "current_capital": 0,
    }


async def test_refresh_brain_state_invalidates_brain_bound_caches_only(tmp_path):
    state = DashboardState()
    for key in (
        "brain_status",
        "position",
        "rules",
        "decision_summary",
        "vectors_50_date_desc",
        "statistics",
        "costs",
    ):
        state.set_cached(key, {"cached": True})
    router = brain_router(hard_exit_config(tmp_path), state=state)

    result = await router.refresh_brain_state()

    assert result["success"] is True
    assert datetime.fromisoformat(result["refreshed_at"]).tzinfo is timezone.utc
    assert result["lifecycle"] == LIFECYCLE_IDLE
    for key in (
        "brain_status",
        "position",
        "rules",
        "decision_summary",
        "vectors_50_date_desc",
        "statistics",
    ):
        assert state.get_cached(key) is None
    assert state.get_cached("costs") == {"cached": True}


async def test_get_blocked_trades_returns_friction_report_and_query_bounds(tmp_path):
    vector_memory = MagicMock()
    vector_memory.get_blocked_trade_count.return_value = 3
    blocked_trades = [
        {
            "id": "blocked-1",
            "guard_type": "rr_minimum",
            "suggested_rr": 1.2,
            "required_rr": 1.5,
        }
    ]
    vector_memory.get_recent_blocked_trades.return_value = blocked_trades
    router = brain_router(hard_exit_config(tmp_path), vector_memory=vector_memory)

    result = await router.get_blocked_trades(limit=10, guard_type="rr_minimum")

    assert result == {"blocked_count": 3, "blocked_trades": blocked_trades}
    vector_memory.get_recent_blocked_trades.assert_called_once_with(
        n=10, guard_type="rr_minimum", max_age_hours=168
    )


async def test_get_blocked_trades_without_memory_or_on_outage(tmp_path):
    config = hard_exit_config(tmp_path)

    assert await brain_router(config).get_blocked_trades() == {
        "blocked_count": 0,
        "blocked_trades": [],
    }

    broken_memory = MagicMock()
    broken_memory.get_recent_blocked_trades.side_effect = RuntimeError("chroma down")

    assert await brain_router(config, vector_memory=broken_memory).get_blocked_trades() == {
        "blocked_count": 0,
        "blocked_trades": [],
        "error": "Internal error",
    }


async def test_get_vector_details_sorts_by_pnl_confidence_and_falls_back_on_unknown_sort(tmp_path):
    legacy_pnl = [
        VectorSearchResult("low", "doc", 10.0, 0.0, 10.0, {"pnl_pct": "-1.0"}),
        VectorSearchResult("missing", "doc", 20.0, 0.0, 20.0, {"pnl_pct": None}),
        VectorSearchResult("high", "doc", 30.0, 0.0, 30.0, {"pnl_pct": "2.5"}),
    ]
    request = MagicMock()
    request.query_params = {"sort_by": "pnl", "order": "desc"}
    router = brain_router(
        hard_exit_config(tmp_path),
        vector_memory=vector_memory_double(legacy_pnl, trade_count=3),
    )

    result = await router.get_vector_details(request, query="BULLISH", limit=3)

    assert [item["id"] for item in result["experiences"]] == ["high", "missing", "low"]
    assert [item["similarity"] for item in result["experiences"]] == [30.0, 20.0, 10.0]
    assert [item.id for item in legacy_pnl] == ["low", "missing", "high"]

    confidences = confidence_experiences()
    request = MagicMock()
    request.query_params = {"sort_by": "confidence", "order": "asc"}
    router = brain_router(
        hard_exit_config(tmp_path),
        vector_memory=vector_memory_double(confidences, trade_count=3),
    )

    sorted_asc = await router.get_vector_details(request, query="BULLISH", limit=3)

    assert [item["id"] for item in sorted_asc["experiences"]] == ["b", "c", "a"]

    request = MagicMock()
    request.query_params = {"sort_by": "bogus", "order": "sideways"}
    router = brain_router(
        hard_exit_config(tmp_path),
        vector_memory=vector_memory_double(confidence_experiences(), trade_count=3),
    )

    fallback = await router.get_vector_details(request, query="BULLISH", limit=3)

    assert [item["id"] for item in fallback["experiences"]] == ["a", "b", "c"]


async def test_get_vector_details_reports_evidence_gate_and_match_factors(tmp_path):
    vector_memory = vector_memory_double(
        [VectorSearchResult("x1", "doc", 90.0, 0.0, 90.0, {"outcome": "LOSS"})],
        trade_count=2,
    )
    vector_memory._build_match_factors.return_value = "ADX=26 | BB%B=0.40"
    request = MagicMock()
    request.query_params = {}
    router = brain_router(hard_exit_config(tmp_path), vector_memory=vector_memory)

    result = await router.get_vector_details(request, query="BULLISH", limit=5)

    assert result["evidence_gate"] == {"trade_count": 2, "min_trades": 3, "limited": True}
    assert result["experiences"][0]["match_factors"] == "ADX=26 | BB%B=0.40"
    vector_memory._build_match_factors.assert_called_once_with({"outcome": "LOSS"}, "BULLISH", None)


async def test_get_vector_details_without_query_context_skips_match_factors(tmp_path):
    vector_memory = vector_memory_double(
        [VectorSearchResult("x1", "doc", 0.0, 0.0, 0.0, {"outcome": "LOSS"})], trade_count=0
    )
    request = MagicMock()
    request.query_params = {}
    router = brain_router(hard_exit_config(tmp_path), vector_memory=vector_memory)

    result = await router.get_vector_details(request, query="", limit=5)

    assert result["current_context"] is None
    assert result["evidence_gate"] == {"trade_count": 0, "min_trades": 3, "limited": True}
    assert result["experiences"][0]["id"] == "x1"
    assert result["experiences"][0]["match_factors"] is None
    vector_memory._build_match_factors.assert_not_called()
    vector_memory.get_all_experiences.assert_called_once_with(
        limit=5, where={"outcome": {"$ne": "UPDATE"}}
    )


async def test_get_vector_details_without_vector_memory_returns_empty_payload(tmp_path):
    request = MagicMock()
    request.query_params = {}
    router = brain_router(hard_exit_config(tmp_path))

    result = await router.get_vector_details(request, query="BULLISH", limit=5)

    assert result == {
        "experience_count": 0,
        "experiences": [],
        "confidence_stats": {},
        "adx_stats": {},
        "factor_stats": {},
        "rule_count": 0,
        "current_context": None,
        "evidence_gate": None,
    }


async def test_evidence_gate_opens_at_exactly_the_minimum_trade_count(tmp_path):
    vector_memory = vector_memory_double([], trade_count=3, min_trades=3)
    request = MagicMock()
    request.query_params = {}
    router = brain_router(hard_exit_config(tmp_path), vector_memory=vector_memory)

    open_gate = (await router.get_vector_details(request, query="BULLISH", limit=3))[
        "evidence_gate"
    ]
    vector_memory.trade_count = 2
    closed_gate = (await router.get_vector_details(request, query="BULLISH", limit=3))[
        "evidence_gate"
    ]

    assert open_gate == {"trade_count": 3, "min_trades": 3, "limited": False}
    assert closed_gate == {"trade_count": 2, "min_trades": 3, "limited": True}


@pytest.mark.parametrize(
    ("query", "limit", "expected_method", "expected_limit", "expected_entry"),
    [
        (None, 20, "recent", 20, POST_MORTEM_RECENT),
        (None, 500, "recent", 100, POST_MORTEM_RECENT),
        (None, 0, "recent", 1, POST_MORTEM_RECENT),
        (None, -5, "recent", 1, POST_MORTEM_RECENT),
        ("breakout", 20, "search", 20, POST_MORTEM_SEARCHED),
        ("  breakout ", 20, "search", 20, POST_MORTEM_SEARCHED),
        ("   ", 20, "recent", 20, POST_MORTEM_RECENT),
    ],
    ids=["recent", "clamped_high", "zero_limit_normalised", "negative_limit_normalised", "search", "search_stripped", "blank_query"],
)
async def test_get_post_mortems_dispatches_on_query_and_clamps_limit(
    tmp_path, query, limit, expected_method, expected_limit, expected_entry
):
    repository = MagicMock()
    repository.get_recent_post_mortems.return_value = [POST_MORTEM_RECENT]
    repository.search_post_mortems.return_value = [POST_MORTEM_SEARCHED]
    router = brain_router(make_config(DATA_DIR=str(tmp_path)), post_mortem_repo=repository)

    result = await router.get_post_mortems(q=query, limit=limit)

    assert result == {"count": 1, "post_mortems": [expected_entry]}
    if expected_method == "search":
        repository.search_post_mortems.assert_called_once_with("breakout", limit=expected_limit)
        repository.get_recent_post_mortems.assert_not_called()
    else:
        repository.get_recent_post_mortems.assert_called_once_with(limit=expected_limit)
        repository.search_post_mortems.assert_not_called()


async def test_get_post_mortems_reports_absent_repository_or_failure(tmp_path):
    config = hard_exit_config(tmp_path)

    assert await brain_router(config).get_post_mortems() == {"count": 0, "post_mortems": []}

    broken_repository = MagicMock()
    broken_repository.get_recent_post_mortems.side_effect = RuntimeError("DB locked")

    assert await brain_router(config, post_mortem_repo=broken_repository).get_post_mortems() == {
        "count": 0,
        "post_mortems": [],
        "error": "DB locked",
    }


async def test_refresh_current_price_reports_missing_or_failing_exchange(tmp_path):
    config = make_config(DATA_DIR=str(tmp_path), CRYPTO_PAIR="BTC/USDT")

    assert await brain_router(config).refresh_current_price() == {
        "success": False,
        "error": "Exchange manager not available",
    }

    unresolved = brain_router(config, exchange_manager=exchange_manager_for(None))

    assert await unresolved.refresh_current_price() == {
        "success": False,
        "error": "No exchange found for BTC/USDT",
    }

    broken_manager = exchange_manager_for(exchange_double(error=RuntimeError("exploded")))
    failing = brain_router(config, exchange_manager=broken_manager)

    assert await failing.refresh_current_price() == {
        "success": False,
        "error": "Internal error during price refresh",
    }


async def test_refresh_current_price_updates_state_only_for_positive_price(tmp_path):
    config = make_config(DATA_DIR=str(tmp_path), CRYPTO_PAIR="BTC/USDT")
    state = DashboardState()
    zero_ticker_manager = exchange_manager_for(exchange_double({"last": 0}))
    zero = brain_router(config, state=state, exchange_manager=zero_ticker_manager)

    assert await zero.refresh_current_price() == {
        "success": True,
        "current_price": 0.0,
        "symbol": "BTC/USDT",
    }
    assert state.current_price is None

    close_ticker_manager = exchange_manager_for(exchange_double({"close": 71000.5}))
    positive = brain_router(config, state=state, exchange_manager=close_ticker_manager)

    assert await positive.refresh_current_price() == {
        "success": True,
        "current_price": 71000.5,
        "symbol": "BTC/USDT",
    }
    assert state.current_price == 71000.5


async def test_get_statistics_defaults_reads_file_and_serves_cache(tmp_path):
    config = make_config(DATA_DIR=str(tmp_path), DEMO_QUOTE_CAPITAL=10000.0)
    state = DashboardState()
    router = performance_router(config, state)

    defaults = await router.get_statistics()

    assert without_last_updated(defaults) == DEFAULT_STATISTICS
    assert set(defaults) == set(DEFAULT_STATISTICS) | {"last_updated"}

    stats_path = tmp_path / "trading" / "statistics.json"
    stats_path.parent.mkdir()
    stats_path.write_text(
        json.dumps({"total_trades": 4, "win_rate": 25.0}), encoding="utf-8"
    )
    state.invalidate_cache("statistics")

    from_file = await router.get_statistics()

    assert from_file == {"total_trades": 4, "win_rate": 25.0}

    stats_path.unlink()

    assert await router.get_statistics() == from_file


async def test_get_statistics_reports_unreadable_file(tmp_path):
    stats_path = tmp_path / "trading" / "statistics.json"
    stats_path.parent.mkdir()
    stats_path.write_text("{not json", encoding="utf-8")
    router = performance_router(make_config(DATA_DIR=str(tmp_path)), DashboardState())

    assert await router.get_statistics() == {"error": "Failed to load stats"}


async def test_get_performance_history_builds_equity_curve_from_persistence(tmp_path):
    config = make_config(DATA_DIR=str(tmp_path), DEMO_QUOTE_CAPITAL=10000.0)
    state = DashboardState()
    persistence = MagicMock()
    persistence.load_trade_history.return_value = [
        {"timestamp": "2026-05-28T10:00:00+00:00", "action": "BUY", "price": 100.0},
        {
            "timestamp": "2026-05-28T12:00:00+00:00",
            "action": "CLOSE_LONG",
            "price": 103.0,
            "reasoning": "P&L: +3.0%",
        },
    ]

    result = await performance_router(config, state, persistence).get_performance_history()

    assert result["history"] == [
        {"time": "2026-05-28T10:00:00+00:00", "value": 10000.0, "action": "BUY", "price": 100.0},
        {
            "time": "2026-05-28T12:00:00+00:00",
            "value": 10300.0,
            "action": "CLOSE_LONG",
            "price": 103.0,
        },
    ]
    assert without_last_updated(result["stats"]) == DEFAULT_STATISTICS
    persistence.load_trade_history.assert_called_once_with()

    persistence.load_trade_history.return_value = [
        {"action": "CLOSE_LONG", "price": 50.0},
        {"action": "BUY", "price": 100.0},
        {"action": "HOLD", "price": 101.0},
        {"timestamp": "t2", "action": "CLOSE_LONG", "price": 103.0},
    ]
    state.invalidate_cache("performance_history")
    rerun_router = performance_router(config, state, persistence)
    skipped_actions = await rerun_router.get_performance_history()

    assert skipped_actions["history"] == [
        {"time": None, "value": 10000.0, "action": "BUY", "price": 100.0},
        {"time": "t2", "value": 10300.0, "action": "CLOSE_LONG", "price": 103.0},
    ]


async def test_get_performance_history_reports_persistence_failure(tmp_path):
    persistence = MagicMock()
    persistence.load_trade_history.side_effect = RuntimeError("disk gone")
    router = performance_router(
        make_config(DATA_DIR=str(tmp_path), DEMO_QUOTE_CAPITAL=10000.0),
        DashboardState(),
        persistence,
    )

    assert await router.get_performance_history() == {"error": "Failed to load trade history"}


def test_cache_policy_matrix_per_path_family(client):
    cases = (
        ("GET", "/main.js?v=5.0", 200, IMMUTABLE_BROWSER, IMMUTABLE_EDGE),
        ("GET", "/main.js", 200, ASSET_BROWSER, ASSET_EDGE),
        ("GET", "/main.js?v=", 200, ASSET_BROWSER, ASSET_EDGE),
        ("GET", "/css/does-not-exist.css", 404, ASSET_BROWSER, ASSET_EDGE),
        ("GET", "/", 200, SHELL_BROWSER, SHELL_EDGE),
        ("GET", "/index.html", 200, SHELL_BROWSER, SHELL_EDGE),
        ("GET", "/api/does-not-exist", 404, API_BROWSER, API_EDGE),
        ("GET", "/api/brain/vectors", 200, API_BROWSER, API_EDGE),
        ("HEAD", "/api/brain/status", 404, API_BROWSER, API_EDGE),
        ("GET", "/api/monitor/health", 200, API_BROWSER, API_EDGE),
    )

    for method, path, status, browser_policy, edge_policy in cases:
        assert_cache_policy(
            client.request(method, path), status, browser_policy, edge_policy
        )


def test_no_store_endpoints_bypass_cache_and_omit_etag(client):
    cases = (
        ("GET", "/api/brain/refresh-price", 200),
        ("GET", "/api/brain/vectors?query=btc&limit=50", 200),
        ("GET", "/api/brain/lifecycle", 200),
        ("POST", "/api/brain/refresh", 200),
        ("POST", "/api/does-not-exist", 405),
    )

    for method, path, status in cases:
        response = client.request(method, path)
        assert_cache_policy(response, status, NO_STORE_BROWSER, "no-store")
        assert response.headers.get("ETag") is None


def test_cacheable_api_emits_etag_and_honours_conditionals(client):
    first = client.get("/api/brain/status")
    etag = first.headers["ETag"]

    assert first.status_code == 200
    assert etag.startswith('W/"')
    assert client.get("/api/brain/status").headers["ETag"] == etag
    assert client.get("/api/brain/status", headers={"If-None-Match": etag}).status_code == 304

    listed = client.get("/api/brain/status", headers={"If-None-Match": f'W/"other", {etag}'})

    assert listed.status_code == 304
    assert listed.headers["ETag"] == etag
    assert listed.headers.get("Content-Length") is None
    assert client.get("/api/brain/status", headers={"If-None-Match": "*"}).status_code == 304
    assert (
        client.get("/api/brain/status", headers={"If-None-Match": 'W/"other"'}).status_code == 200
    )


def test_security_headers_are_set_and_hsts_only_when_tls_is_forwarded(client):
    plain = client.get("/api/does-not-exist")

    assert plain.headers["X-Content-Type-Options"] == "nosniff"
    assert plain.headers["X-Frame-Options"] == "DENY"
    assert plain.headers["X-XSS-Protection"] == "1; mode=block"
    assert plain.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert plain.headers["Permissions-Policy"] == "geolocation=(), microphone=(), camera=()"
    assert plain.headers["Content-Security-Policy"].startswith(
        "default-src 'self'; frame-ancestors 'none';"
    )
    assert plain.headers.get("Strict-Transport-Security") is None

    proxied = client.get("/api/does-not-exist", headers={"x-forwarded-proto": "https"})

    assert proxied.headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"


def test_rate_limit_window_blocks_api_clients_and_exempts_static(client, dashboard_server):
    request_counts = dashboard_server.app.state.request_counts
    request_counts["testclient"] = [time.monotonic()] * 300

    blocked = client.get("/api/does-not-exist")

    assert blocked.status_code == 429
    assert blocked.json() == {"error": "Rate limit exceeded. Try again later."}
    assert client.get("/main.js").status_code == 200

    request_counts["testclient"] = [time.monotonic() - 61] * 300

    assert client.get("/api/does-not-exist").status_code == 404


def test_api_cache_policy_matrix_maps_each_endpoint_family():
    assert _api_cache_policies("/api/brain/refresh-price", {}) == (NO_STORE_BROWSER, "no-store")
    assert _api_cache_policies("/api/brain/refresh", {}) == (NO_STORE_BROWSER, "no-store")
    assert _api_cache_policies("/api/brain/lifecycle", {}) == (NO_STORE_BROWSER, "no-store")
    assert _api_cache_policies("/api/brain/vectors", {"query": "btc"}) == (
        NO_STORE_BROWSER,
        "no-store",
    )
    assert _api_cache_policies("/api/brain/vectors", {}) == (API_BROWSER, API_EDGE)
    assert _api_cache_policies("/api/monitor/health", {}) == (API_BROWSER, API_EDGE)
    assert _api_cache_policies("/api/status/countdown", {}) == (
        "public, max-age=5",
        "public, max-age=15, stale-while-revalidate=10, stale-if-error=60",
    )


def test_index_html_exposes_lifecycle_risk_and_decision_bindings():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    for element_id in (
        "brain-lifecycle-badge",
        "risk-policy-strip",
        "overview-sl-policy",
        "overview-tp-policy",
        "experience-count",
        "panel-decision-pathways",
        "decision-synopsis",
        "decision-graph",
        "decision-detail",
        "decision-legend",
    ):
        assert f'id="{element_id}"' in html
    assert "Decision Pathways" in html
    assert "vis-network" in html
    assert "synapse-network" not in html
    assert "main.js?v=" in html


def test_dashboard_scripts_reference_the_new_bindings():
    main_js = (STATIC_DIR / "main.js").read_text(encoding="utf-8")
    websocket_js = (STATIC_DIR / "modules" / "websocket.js").read_text(encoding="utf-8")
    position_js = (STATIC_DIR / "modules" / "position_panel.js").read_text(encoding="utf-8")

    for reference in (
        "overview-sl-policy",
        "overview-tp-policy",
        "brain-lifecycle-badge",
        "decision_pathways_panel.js",
        "updateDecisionPathways",
    ):
        assert reference in main_js
    assert "brain_rebuild_completed" in websocket_js
    assert "trade-closed-detected" in position_js
    assert "initSynapseNetwork" not in main_js
    assert "synapse_viewer.js" not in main_js

    for asset in STATIC_DIR.rglob("*"):
        if asset.suffix in (".js", ".html", ".css"):
            text = asset.read_text(encoding="utf-8")
            assert "initSynapseNetwork" not in text
            assert "synapse_viewer.js" not in text


def test_main_js_binds_only_element_ids_present_in_the_html_shell():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    main_js = (STATIC_DIR / "main.js").read_text(encoding="utf-8")

    bound_ids = set(re.findall(r"getElementById\(\s*['\"]([\w-]+)['\"]\s*\)", main_js))
    unbound_ids = [
        element_id for element_id in sorted(bound_ids) if f'id="{element_id}"' not in html
    ]

    assert bound_ids
    assert unbound_ids == []
