from types import SimpleNamespace
from unittest.mock import MagicMock

from src.dashboard.routers.performance import PerformanceRouter


async def test_get_statistics_returns_initial_capital_when_stats_file_missing(tmp_path):
    config = SimpleNamespace(DATA_DIR=str(tmp_path), DEMO_QUOTE_CAPITAL=10000.0)
    logger = MagicMock()
    dashboard_state = MagicMock()
    dashboard_state.get_cached.return_value = None
    persistence = MagicMock()

    router = PerformanceRouter(
        config=config,
        logger=logger,
        dashboard_state=dashboard_state,
        persistence=persistence,
    )

    result = await router.get_statistics()

    assert result["total_trades"] == 0
    assert result["win_rate"] == 0.0
    assert result["total_pnl_pct"] == 0.0
    assert result["initial_capital"] == 10000.0
    assert result["current_capital"] == 10000.0


async def test_get_performance_history_uses_persistence_trades(tmp_path):
    config = SimpleNamespace(DATA_DIR=str(tmp_path), DEMO_QUOTE_CAPITAL=10000.0)
    logger = MagicMock()
    dashboard_state = MagicMock()
    dashboard_state.get_cached.return_value = None
    dashboard_state.set_cached = MagicMock()

    persistence = MagicMock()
    persistence.load_trade_history.return_value = [
        {
            "timestamp": "2026-05-28T10:00:00+00:00",
            "action": "BUY",
            "price": 100.0,
        },
        {
            "timestamp": "2026-05-28T12:00:00+00:00",
            "action": "CLOSE_LONG",
            "price": 103.0,
            "reasoning": "P&L: +3.0%",
        },
    ]

    router = PerformanceRouter(
        config=config,
        logger=logger,
        dashboard_state=dashboard_state,
        persistence=persistence,
    )

    result = await router.get_performance_history()

    persistence.load_trade_history.assert_called_once_with()
    assert "history" in result
    assert len(result["history"]) == 2
    assert result["history"][0]["action"] == "BUY"