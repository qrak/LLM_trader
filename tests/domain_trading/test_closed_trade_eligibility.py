from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.trading.data_models import (
    FILL_DIVERGENCE_MARKER,
    PHANTOM_CLOSE_MARKER,
    TradeDecision,
    TradingMemory,
)
from src.trading.statistics_calculator import StatisticsCalculator
from tests.domain_infra.test_notifiers import StatsNotifier

START = datetime(2026, 11, 1, tzinfo=timezone.utc)
INELIGIBLE = {"eligible": False, "reason": "entry never executed"}
ELIGIBLE = {"eligible": True, "reason": ""}


def row(index: int, action: str, price: float, quality, reasoning: str) -> dict:
    return {
        "timestamp": (START + timedelta(hours=index)).isoformat(),
        "action": action,
        "price": price,
        "quantity": 1.0,
        "quote_amount": price,
        "fee": 0.0,
        "execution_quality": quality,
        "reasoning": reasoning,
    }


def mixed_history() -> list[dict]:
    return [
        row(0, "BUY", 100.0, ELIGIBLE, "entry"),
        row(1, "CLOSE", 110.0, ELIGIBLE, "Position closed: take_profit. P&L: +10.00%."),
        row(2, "SELL", 100.0, None, "entry"),
        row(3, "CLOSE", 100.0, None, f"{PHANTOM_CLOSE_MARKER} — the entry never reached the exchange"),
        row(4, "BUY", 100.0, None, "entry"),
        row(5, "CLOSE", 100.0, INELIGIBLE, "Position closed: executor blocked the entry."),
        row(6, "BUY", 100.0, None, "entry"),
        row(7, "CLOSE", 105.0, INELIGIBLE, f"{FILL_DIVERGENCE_MARKER}: the fill diverged. Position closed."),
        row(8, "BUY", 100.0, ELIGIBLE, "entry"),
        row(9, "CLOSE", 100.0, ELIGIBLE, "Position closed: stop_loss. P&L: +0.00%."),
    ]


def decisions(history: list[dict]) -> list[TradeDecision]:
    return [
        TradeDecision(
            timestamp=datetime.fromisoformat(item["timestamp"]),
            symbol="BTC/USDC",
            action=item["action"],
            confidence="HIGH",
            price=item["price"],
            quantity=item["quantity"],
            quote_amount=item["quote_amount"],
            fee=item["fee"],
            reasoning=item["reasoning"],
            execution_quality=item["execution_quality"],
        )
        for item in history
    ]


def notifier_stats(history: list[dict]) -> dict:
    notifier = StatsNotifier(
        None, SimpleNamespace(DEMO_QUOTE_CAPITAL=10000.0, TRANSACTION_FEE_PERCENT=0.0), None, None
    )
    return notifier.calculate_performance_stats(history)


def test_a_marker_only_phantom_is_dropped_by_the_calculator_and_the_prompt() -> None:
    history = [
        row(0, "SELL", 77163.94, None, "entry"),
        row(1, "CLOSE", 77163.94, None, f"{PHANTOM_CLOSE_MARKER} — the entry never reached the exchange"),
    ]

    stats = StatisticsCalculator.calculate_from_history(history, 10000.0)
    summary = TradingMemory(decisions=decisions(history)).get_context_summary()

    assert stats.total_trades == 0
    assert "Total Closed Trades" not in summary


def test_every_consumer_agrees_on_which_pairs_are_closed_trades() -> None:
    history = mixed_history()

    stats = StatisticsCalculator.calculate_from_history(history, 10000.0)
    summary = TradingMemory(decisions=decisions(history)).get_context_summary()
    notified = notifier_stats(history)

    assert stats.total_trades == 2
    assert stats.winning_trades == 1
    assert "2 Total Closed Trades" in summary
    assert "(1/2 trades)" in summary
    assert notified["closed_trades"] == 2
    assert notified["winning_trades"] == 1
    assert notified["last_closed_trade"]["outcome"] == "BREAKEVEN"
    assert notified["last_closed_trade"]["pnl_pct"] == 0.0
