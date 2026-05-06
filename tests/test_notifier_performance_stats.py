from types import SimpleNamespace
from typing import Any

import pytest

from src.notifiers.base_notifier import BaseNotifier


class _PerformanceStatsNotifier(BaseNotifier):
    async def start(self) -> None:
        pass

    async def wait_until_ready(self) -> None:
        pass

    async def send_message(
        self,
        message: str,
        channel_id: int,
        expire_after: int | None = None,
    ) -> Any:
        pass

    async def send_trading_decision(self, decision: Any, channel_id: int) -> None:
        pass

    async def send_analysis_notification(
        self,
        result: dict[str, Any],
        symbol: str,
        timeframe: str,
        channel_id: int,
        chart_image: Any = None,
    ) -> None:
        pass

    async def send_position_status(
        self,
        position: Any,
        current_price: float,
        channel_id: int,
    ) -> None:
        pass

    async def send_performance_stats(
        self,
        trade_history: list[dict[str, Any]],
        symbol: str,
        channel_id: int,
    ) -> None:
        pass


def test_performance_total_percent_uses_capital_not_sum_of_trade_returns():
    config = SimpleNamespace(
        DEMO_QUOTE_CAPITAL=10000.0,
        TRANSACTION_FEE_PERCENT=0.0,
    )
    notifier = _PerformanceStatsNotifier(None, config, None, None)
    trade_history = [
        {"action": "BUY", "price": 100.0, "quantity": 1.0, "fee": 0.0},
        {"action": "CLOSE_LONG", "price": 110.0, "quantity": 1.0, "fee": 0.0},
        {"action": "BUY", "price": 100.0, "quantity": 20.0, "fee": 0.0},
        {"action": "CLOSE_LONG", "price": 99.0, "quantity": 20.0, "fee": 0.0},
    ]

    stats = notifier.calculate_performance_stats(trade_history)

    assert stats is not None
    assert stats["total_pnl_quote"] == pytest.approx(-10.0)
    assert stats["total_pnl_pct"] == pytest.approx(-0.1)
    assert stats["avg_pnl_pct"] == pytest.approx(4.5)
    assert stats["winning_trades"] == 1
    assert stats["closed_trades"] == 2
