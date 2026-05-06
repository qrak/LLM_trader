"""Regression tests for trading history context formatting."""

from datetime import datetime, timezone

from src.trading.data_models import TradeDecision, TradingMemory


def _decision(**overrides):
    base = {
        "timestamp": datetime(2026, 4, 14, 0, 0, tzinfo=timezone.utc),
        "symbol": "BTC/USDC",
        "action": "HOLD",
        "confidence": "HIGH",
        "price": 74000.0,
        "quantity": 0.0,
        "reasoning": "",
    }
    base.update(overrides)
    return TradeDecision(**base)


def test_profitable_stop_loss_is_labeled_as_profit_protection():
    history = [
        _decision(
            timestamp=datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc),
            action="BUY",
            price=72000.0,
            quantity=0.001,
            reasoning="Entered long on breakout.",
        ),
        _decision(
            timestamp=datetime(2026, 4, 14, 20, 0, tzinfo=timezone.utc),
            action="CLOSE_LONG",
            price=74286.53,
            quantity=0.001,
            reasoning="Position closed: stop_loss. P&L: +3.30%. Fee: $0.0426",
        ),
    ]

    summary = TradingMemory(decisions=history).get_context_summary(full_history=history)

    assert "profit-protecting stop" in summary
    assert "+3.18%" in summary


def test_losing_stop_loss_is_labeled_as_loss_cut():
    history = [
        _decision(
            timestamp=datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc),
            action="SELL",
            price=72000.0,
            quantity=0.001,
            reasoning="Entered short on breakdown.",
        ),
        _decision(
            timestamp=datetime(2026, 4, 14, 20, 0, tzinfo=timezone.utc),
            action="CLOSE_SHORT",
            price=72720.0,
            quantity=0.001,
            reasoning="Position closed: stop_loss. P&L: -1.00%. Fee: $0.0426",
        ),
    ]

    summary = TradingMemory(decisions=history).get_context_summary(full_history=history)

    assert "loss-cutting stop" in summary
    assert "-1.00%" in summary


def test_total_pnl_percent_uses_capital_not_sum_of_trade_returns():
    history = [
        _decision(
            timestamp=datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc),
            action="BUY",
            price=100.0,
            quantity=1.0,
            quote_amount=100.0,
            reasoning="Entered long.",
        ),
        _decision(
            timestamp=datetime(2026, 4, 14, 13, 0, tzinfo=timezone.utc),
            action="CLOSE_LONG",
            price=110.0,
            quantity=1.0,
            quote_amount=100.0,
            reasoning="Position closed: take_profit. P&L: +10.00%. Fee: $0.0000",
        ),
        _decision(
            timestamp=datetime(2026, 4, 14, 14, 0, tzinfo=timezone.utc),
            action="BUY",
            price=100.0,
            quantity=20.0,
            quote_amount=2000.0,
            reasoning="Entered long.",
        ),
        _decision(
            timestamp=datetime(2026, 4, 14, 15, 0, tzinfo=timezone.utc),
            action="CLOSE_LONG",
            price=99.0,
            quantity=20.0,
            quote_amount=2000.0,
            reasoning="Position closed: stop_loss. P&L: -1.00%. Fee: $0.0000",
        ),
    ]

    summary = TradingMemory(decisions=history).get_context_summary(
        full_history=history,
        initial_capital=10000.0,
    )

    assert "Total P&L: $-10.00 (-0.10%)" in summary
    assert "Average P&L per Trade: +4.50%" in summary
