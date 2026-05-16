import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.dashboard.dashboard_state import DashboardState
from src.dashboard.routers.brain import BrainRouter
from src.dashboard.routers.brain import _build_current_market_context, _extract_market_status
from src.trading.data_models import Position, VectorSearchResult


def test_build_current_market_context_uses_legacy_response_indicators(tmp_path):
    trading_dir = tmp_path / "trading"
    trading_dir.mkdir()
    previous_response = trading_dir / "previous_response.json"
    previous_response.write_text(
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
        encoding="utf-8",
    )

    config = SimpleNamespace(
        DATA_DIR=str(tmp_path),
        TIMEFRAME="4h",
        STOP_LOSS_TYPE="hard",
        STOP_LOSS_CHECK_INTERVAL="15m",
        TAKE_PROFIT_TYPE="hard",
        TAKE_PROFIT_CHECK_INTERVAL="15m",
    )
    logger = MagicMock()

    display_context, query_document = _build_current_market_context(config, logger)

    assert display_context.startswith("BEARISH + Low ADX + LOW Volatility")
    assert "MACD BEARISH" in display_context
    assert "Indicators: ADX=18.1 (Low ADX)" in query_document


def test_build_current_market_context_includes_exit_execution_settings(tmp_path):
    trading_dir = tmp_path / "trading"
    trading_dir.mkdir()
    previous_response = trading_dir / "previous_response.json"
    previous_response.write_text(
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
        encoding="utf-8",
    )

    config = SimpleNamespace(
        DATA_DIR=str(tmp_path),
        TIMEFRAME="4h",
        STOP_LOSS_TYPE="hard",
        STOP_LOSS_CHECK_INTERVAL="15m",
        TAKE_PROFIT_TYPE="soft",
        TAKE_PROFIT_CHECK_INTERVAL="4h",
    )

    display_context, query_document = _build_current_market_context(config, MagicMock())

    assert "Exit Execution: SL hard/15m | TP soft/4h" in display_context
    assert "Exit Execution: SL hard/15m | TP soft/4h" in query_document


def test_build_current_market_context_ignores_non_object_previous_response(tmp_path):
    trading_dir = tmp_path / "trading"
    trading_dir.mkdir()
    previous_response = trading_dir / "previous_response.json"
    previous_response.write_text(json.dumps([]), encoding="utf-8")

    config = SimpleNamespace(
        DATA_DIR=str(tmp_path),
        TIMEFRAME="4h",
        STOP_LOSS_TYPE="hard",
        STOP_LOSS_CHECK_INTERVAL="15m",
        TAKE_PROFIT_TYPE="hard",
        TAKE_PROFIT_CHECK_INTERVAL="15m",
    )

    display_context, query_document = _build_current_market_context(config, MagicMock())

    assert display_context == ""
    assert query_document == ""


def test_extract_market_status_uses_text_and_indicators_without_json_parser():
    parser = MagicMock()
    parser.extract_json_block.return_value = None
    data = {
        "response": {
            "text_analysis": "SIGNAL: UPDATE\nConfidence: 82%\nThe structure remains bearish.\n```json\n{bad json}\n```",
            "adx": 18.1,
            "rsi": 40.5,
            "plus_di": 14.8,
            "minus_di": 24.2,
        }
    }

    status = _extract_market_status(data, parser)

    parser.extract_json_block.assert_called_once()
    assert status["action"] == "UPDATE"
    assert status["confidence"] == 82
    assert status["trend"] == "BEARISH"
    assert status["adx"] == 18.1
    assert status["rsi"] == 40.5


def test_extract_market_status_parses_json_confidence_with_technical_data_present():
    parser = MagicMock()
    parser.extract_json_block.return_value = {
        "signal": "UPDATE",
        "confidence": 75,
    }
    data = {
        "response": {
            "text_analysis": "```json\n{\"analysis\": {\"signal\": \"UPDATE\", \"confidence\": 75}}\n```",
            "adx": 19.9,
            "rsi": 61.0,
            "plus_di": 30.0,
            "minus_di": 16.8,
        }
    }

    status = _extract_market_status(data, parser)

    parser.extract_json_block.assert_called_once()
    assert status["action"] == "UPDATE"
    assert status["confidence"] == 75
    assert status["adx"] == 19.9
    assert status["rsi"] == 61.0


@pytest.mark.asyncio
async def test_get_active_rules_maps_ai_mistake_and_exit_metadata():
    dashboard_state = DashboardState()
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
            },
        }
    ]
    router = BrainRouter(
        config=SimpleNamespace(DATA_DIR="unused"),
        logger=MagicMock(),
        dashboard_state=dashboard_state,
        vector_memory=vector_memory,
        unified_parser=None,
        persistence=MagicMock(),
        exchange_manager=MagicMock(),
    )

    rules = await router.get_active_rules()

    assert rules[0]["rule_text"] == "AI MISTAKE: high confidence breakout failed in chop"
    assert rules[0]["rule_type"] == "ai_mistake"
    assert rules[0]["source_trades"] == 3
    assert rules[0]["mistake_type"] == "sideways_overconfidence"
    assert rules[0]["entry_confidence"] == "HIGH"
    assert rules[0]["failed_assumption"] == "expected breakout continuation"
    assert rules[0]["dominant_exit_profile"] == "SL hard/1m | TP soft/15m"
    assert rules[0]["dominant_stop_loss_interval"] == "1m"
    assert rules[0]["dominant_take_profit_interval"] == "15m"


@pytest.mark.asyncio
async def test_get_active_rules_rewrites_legacy_unknown_exit_profile_from_config():
    dashboard_state = DashboardState()
    vector_memory = MagicMock()
    vector_memory.get_active_rules.return_value = [
        {
            "rule_id": "rule_best_long_bullish_high_adx_sl_unknown_unknown_tp_unknown_unknown",
            "text": "LONG trades perform well. Exit profile: SL unknown/unknown | TP unknown/unknown. (3 wins)",
            "metadata": {
                "rule_type": "best_practice",
                "source_trades": 3,
                "dominant_exit_profile": "SL unknown/unknown | TP unknown/unknown",
                "dominant_stop_loss_type": "unknown",
                "dominant_take_profit_type": "unknown",
            },
        }
    ]
    router = BrainRouter(
        config=SimpleNamespace(
            DATA_DIR="unused",
            TIMEFRAME="4h",
            STOP_LOSS_TYPE="hard",
            STOP_LOSS_CHECK_INTERVAL="15m",
            TAKE_PROFIT_TYPE="hard",
            TAKE_PROFIT_CHECK_INTERVAL="15m",
        ),
        logger=MagicMock(),
        dashboard_state=dashboard_state,
        vector_memory=vector_memory,
        unified_parser=None,
        persistence=MagicMock(),
        exchange_manager=MagicMock(),
    )

    rules = await router.get_active_rules()

    assert "Exit profile: SL hard/15m | TP hard/15m" in rules[0]["rule_text"]
    assert rules[0]["dominant_exit_profile"] == "SL hard/15m | TP hard/15m"


@pytest.mark.asyncio
async def test_get_current_position_recomputes_missing_distance_percentages():
    position = Position(
        entry_price=69009.78,
        stop_loss=69350.00,
        take_profit=65612.00,
        size=0.1,
        entry_time=MagicMock(isoformat=MagicMock(return_value="2026-03-27T00:00:00+00:00")),
        confidence="HIGH",
        direction="SHORT",
        symbol="BTC/USDC",
        sl_distance_pct=0.0,
        tp_distance_pct=0.0,
        rr_ratio_at_entry=2.2,
        stop_loss_type_at_entry="hard",
        stop_loss_check_interval_at_entry="15m",
        take_profit_type_at_entry="hard",
        take_profit_check_interval_at_entry="15m",
    )
    dashboard_state = DashboardState(current_price=68366.03)
    persistence = MagicMock()
    persistence.load_position.return_value = position
    router = BrainRouter(
        config=SimpleNamespace(
            DATA_DIR="unused",
            TIMEFRAME="4h",
            STOP_LOSS_TYPE="hard",
            STOP_LOSS_CHECK_INTERVAL="15m",
            TAKE_PROFIT_TYPE="hard",
            TAKE_PROFIT_CHECK_INTERVAL="15m",
        ),
        logger=MagicMock(),
        dashboard_state=dashboard_state,
        vector_memory=None,
        unified_parser=None,
        persistence=persistence,
        exchange_manager=None,
    )

    result = await router.get_current_position()

    assert result["has_position"] is True
    assert result["sl_distance_pct"] == pytest.approx(abs(69350.00 - 69009.78) / 69009.78)
    assert result["tp_distance_pct"] == pytest.approx(abs(65612.00 - 69009.78) / 69009.78)
    assert result["exit_management"] == {
        "stop_loss_type": "hard",
        "stop_loss_check_interval": "15m",
        "take_profit_type": "hard",
        "take_profit_check_interval": "15m",
    }
    assert result["exit_management_at_entry"] == result["exit_management"]
    assert result["risk_management"] == {
        "current": result["exit_management"],
        "at_entry": result["exit_management_at_entry"],
        "current_labels": {"stop_loss": "hard / 15m", "take_profit": "hard / 15m"},
        "at_entry_labels": {"stop_loss": "hard / 15m", "take_profit": "hard / 15m"},
        "policy_changed": False,
    }


@pytest.mark.asyncio
async def test_get_brain_status_includes_lifecycle_from_cache(tmp_path):
    dashboard_state = DashboardState()
    dashboard_state.set_cached("brain_status", {"status": "active", "trend": "BULLISH"})
    dashboard_state.brain_rebuild_status = "updating"
    router = BrainRouter(
        config=SimpleNamespace(DATA_DIR=str(tmp_path), TIMEFRAME="4h"),
        logger=MagicMock(),
        dashboard_state=dashboard_state,
        vector_memory=None,
        unified_parser=None,
        persistence=MagicMock(),
        exchange_manager=None,
    )

    result = await router.get_brain_status()

    assert result["trend"] == "BULLISH"
    assert result["brain_lifecycle"]["status"] == "updating"


@pytest.mark.asyncio
async def test_refresh_brain_state_invalidates_related_caches():
    dashboard_state = DashboardState()
    for key in ("brain_status", "position", "rules", "memory_50", "vectors_50_date_desc", "statistics"):
        dashboard_state.set_cached(key, {"cached": True})
    router = BrainRouter(
        config=SimpleNamespace(DATA_DIR="unused", TIMEFRAME="4h"),
        logger=MagicMock(),
        dashboard_state=dashboard_state,
        vector_memory=None,
        unified_parser=None,
        persistence=MagicMock(),
        exchange_manager=None,
    )

    result = await router.refresh_brain_state()

    assert result["success"] is True
    assert dashboard_state.get_cached("brain_status") is None
    assert dashboard_state.get_cached("position") is None
    assert dashboard_state.get_cached("rules") is None
    assert dashboard_state.get_cached("memory_50") is None
    assert dashboard_state.get_cached("vectors_50_date_desc") is None
    assert dashboard_state.get_cached("statistics") is None


@pytest.mark.asyncio
async def test_get_blocked_trades_returns_vector_memory_payload():
    dashboard_state = DashboardState()
    vector_memory = MagicMock()
    vector_memory.get_blocked_trade_count.return_value = 3
    vector_memory.get_recent_blocked_trades.return_value = [
        {"id": "blocked-1", "guard_type": "rr_minimum", "suggested_rr": 1.2, "required_rr": 1.5}
    ]
    router = BrainRouter(
        config=SimpleNamespace(DATA_DIR="unused", TIMEFRAME="4h"),
        logger=MagicMock(),
        dashboard_state=dashboard_state,
        vector_memory=vector_memory,
        unified_parser=None,
        persistence=MagicMock(),
        exchange_manager=None,
    )

    result = await router.get_blocked_trades(limit=10, guard_type="rr_minimum")

    assert result["blocked_count"] == 3
    assert result["blocked_trades"][0]["guard_type"] == "rr_minimum"
    vector_memory.get_recent_blocked_trades.assert_called_once_with(
        n=10,
        guard_type="rr_minimum",
        max_age_hours=168,
    )


@pytest.mark.asyncio
async def test_get_vector_details_sorts_legacy_pnl_metadata_safely():
    dashboard_state = DashboardState()
    vector_memory = MagicMock()
    vector_memory.trade_count = 3
    vector_memory.semantic_rule_count = 0
    vector_memory.compute_confidence_stats.return_value = {}
    vector_memory.compute_adx_performance.return_value = {}
    vector_memory.compute_factor_performance.return_value = {}
    vector_memory.retrieve_similar_experiences.return_value = [
        VectorSearchResult("low", "doc", 10.0, 0.0, 10.0, {"pnl_pct": "-1.0"}),
        VectorSearchResult("missing", "doc", 20.0, 0.0, 20.0, {"pnl_pct": None}),
        VectorSearchResult("high", "doc", 30.0, 0.0, 30.0, {"pnl_pct": "2.5"}),
    ]
    router = BrainRouter(
        config=SimpleNamespace(DATA_DIR="unused"),
        logger=MagicMock(),
        dashboard_state=dashboard_state,
        vector_memory=vector_memory,
        unified_parser=None,
        persistence=MagicMock(),
        exchange_manager=None,
    )
    request = MagicMock()
    request.query_params = {"sort_by": "pnl", "order": "desc"}

    result = await router.get_vector_details(request, query="BULLISH", limit=3)

    assert [item["id"] for item in result["experiences"]] == ["high", "missing", "low"]