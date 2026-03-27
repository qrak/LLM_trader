from types import SimpleNamespace
from unittest.mock import MagicMock

from src.dashboard.routers.performance import PerformanceRouter


async def test_get_statistics_returns_initial_capital_when_stats_file_missing(tmp_path):
    config = SimpleNamespace(DATA_DIR=str(tmp_path), DEMO_QUOTE_CAPITAL=10000.0)
    logger = MagicMock()
    dashboard_state = MagicMock()
    dashboard_state.get_cached.return_value = None

    router = PerformanceRouter(config=config, logger=logger, dashboard_state=dashboard_state)

    result = await router.get_statistics()

    assert result["total_trades"] == 0
    assert result["win_rate"] == 0.0
    assert result["total_pnl_pct"] == 0.0
    assert result["initial_capital"] == 10000.0
    assert result["current_capital"] == 10000.0