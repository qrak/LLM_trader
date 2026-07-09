"""Tests for CCXT execution decision payload + executor HTTP forward path."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.app import (
    ACTIONABLE_EXECUTION_SIGNALS,
    BotServices,
    CryptoTradingBot,
    _build_execution_decision,
)
from src.managers.persistence_manager import PersistenceManager


def test_build_execution_decision_skips_hold_and_missing():
    assert _build_execution_decision(None, symbol="BTC/USDC") is None
    assert _build_execution_decision({"signal": "HOLD"}, symbol="BTC/USDC") is None
    assert _build_execution_decision({"signal": "BUY"}, symbol="") is None
    assert _build_execution_decision({"signal": "BUY"}, symbol=None) is None


def test_build_execution_decision_maps_ccxt_fields():
    ts = "2026-07-09T12:00:00+00:00"
    decision = _build_execution_decision(
        {
            "signal": "BUY",
            "order_type": "market",
            "quantity": 0.15,
            "entry_price": 70000.0,
            "stop_loss": 68000.0,
            "take_profit": 74000.0,
            "reduce_only": False,
            "leverage": 3,
            "confidence": 88,
            "reasoning": "Breakout confirmation",
        },
        symbol="BTC/USDC",
        timestamp=ts,
    )
    assert decision == {
        "timestamp": ts,
        "symbol": "BTC/USDC",
        "signal": "BUY",
        "order_type": "market",
        "quantity": 0.15,
        "entry_price": 70000.0,
        "stop_loss": 68000.0,
        "take_profit": 74000.0,
        "reduce_only": False,
        "leverage": 3,
        "confidence": 88,
        "reasoning": "Breakout confirmation",
    }


def test_build_execution_decision_defaults_and_signal_set():
    assert ACTIONABLE_EXECUTION_SIGNALS == frozenset({"BUY", "SELL", "CLOSE", "UPDATE"})
    decision = _build_execution_decision({"signal": "CLOSE"}, symbol="ETH/USDC", timestamp="t")
    assert decision is not None
    assert decision["order_type"] == "limit"
    assert decision["quantity"] == 0.0
    assert decision["reduce_only"] is False
    assert decision["leverage"] == 1
    assert decision["reasoning"] == ""


def _make_bot(**overrides):
    config = SimpleNamespace(
        MAIN_CHANNEL_ID=123,
        RAG_UPDATE_TIMEOUT=1,
        EXECUTOR_API_ENABLED=overrides.pop("executor_enabled", True),
        EXECUTOR_API_URL=overrides.pop("executor_url", "http://127.0.0.1:9199/decision"),
    )
    persistence = MagicMock()
    services = {
        "logger": MagicMock(),
        "config": config,
        "shutdown_manager": None,
        "exchange_manager": MagicMock(),
        "market_analyzer": MagicMock(),
        "trading_strategy": MagicMock(),
        "discord_notifier": MagicMock(),
        "keyboard_handler": MagicMock(),
        "rag_engine": MagicMock(),
        "coingecko_api": MagicMock(),
        "market_api": MagicMock(),
        "alternative_me_api": MagicMock(),
        "http_session": MagicMock(),
        "persistence": persistence,
        "model_manager": MagicMock(),
        "brain_service": MagicMock(),
        "statistics_service": MagicMock(),
        "memory_service": MagicMock(),
        "exit_monitor": MagicMock(),
        "dashboard_state": None,
        "discord_task": None,
        "position_monitor_factory": lambda _bot: MagicMock(),
    }
    services.update(overrides.get("services", {}))
    bot = CryptoTradingBot(BotServices(**services))
    bot.current_symbol = "BTC/USDC"
    return bot, persistence


def test_save_execution_decision_writes_actionable_and_skips_hold():
    bot, persistence = _make_bot()
    result = {
        "analysis": {
            "signal": "SELL",
            "order_type": "limit",
            "quantity": 0.2,
            "entry_price": 71_000,
            "stop_loss": 72_000,
            "take_profit": 68_000,
            "reduce_only": False,
            "leverage": 1,
            "confidence": 70,
            "reasoning": "rejection",
        }
    }
    written = bot._save_execution_decision(result)
    assert written is not None
    assert written["signal"] == "SELL"
    assert written["symbol"] == "BTC/USDC"
    persistence.save_latest_decision.assert_called_once_with(written)

    persistence.reset_mock()
    assert bot._save_execution_decision({"analysis": {"signal": "HOLD"}}) is None
    persistence.save_latest_decision.assert_not_called()


@pytest.mark.asyncio
async def test_forward_decision_posts_when_enabled():
    bot, _ = _make_bot(executor_enabled=True, executor_url="http://executor.local/decision")
    decision = {
        "timestamp": "t",
        "symbol": "BTC/USDC",
        "signal": "BUY",
        "order_type": "limit",
        "quantity": 0.1,
        "entry_price": 1.0,
        "stop_loss": 0.9,
        "take_profit": 1.2,
        "reduce_only": False,
        "leverage": 1,
        "confidence": 80,
        "reasoning": "x",
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "ok"

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("src.app.httpx.AsyncClient", return_value=mock_client) as client_cls:
        await bot._forward_decision_to_executor(decision)

    client_cls.assert_called_once()
    mock_client.post.assert_awaited_once_with(
        "http://executor.local/decision",
        json=decision,
    )
    bot.logger.info.assert_called()


@pytest.mark.asyncio
async def test_forward_decision_skips_when_disabled_or_empty():
    bot, _ = _make_bot(executor_enabled=False)
    with patch("src.app.httpx.AsyncClient") as client_cls:
        await bot._forward_decision_to_executor({"signal": "BUY"})
        await bot._forward_decision_to_executor(None)
    client_cls.assert_not_called()


@pytest.mark.asyncio
async def test_forward_decision_logs_error_on_network_failure():
    bot, _ = _make_bot(executor_enabled=True)
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=ConnectionError("offline"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("src.app.httpx.AsyncClient", return_value=mock_client):
        await bot._forward_decision_to_executor(
            {
                "timestamp": "t",
                "symbol": "BTC/USDC",
                "signal": "BUY",
                "order_type": "limit",
                "quantity": 0.1,
                "entry_price": 1.0,
                "stop_loss": 0.9,
                "take_profit": 1.2,
                "reduce_only": False,
                "leverage": 1,
                "confidence": 80,
                "reasoning": "x",
            }
        )

    bot.logger.error.assert_called()
    # Must include exc_info so square-tracebacks are available
    assert bot.logger.error.call_args.kwargs.get("exc_info") is True


def test_save_latest_decision_atomic_write(tmp_path):
    logger = MagicMock()
    pm = PersistenceManager(logger=logger, data_dir=str(tmp_path))
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": "BTC/USDC",
        "signal": "BUY",
        "order_type": "limit",
        "quantity": 0.01,
        "entry_price": 10.0,
        "stop_loss": 9.0,
        "take_profit": 12.0,
        "reduce_only": False,
        "leverage": 1,
        "confidence": 50,
        "reasoning": "unit",
    }
    pm.save_latest_decision(payload)
    path = Path(tmp_path) / "latest_decision.json"
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["signal"] == "BUY"
    assert loaded["symbol"] == "BTC/USDC"
    # no temp leftovers
    leftovers = list(Path(tmp_path).glob("*.json"))
    assert leftovers == [path]
