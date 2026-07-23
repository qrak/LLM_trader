"""Tests for ExecutorHandler — payload building, persistence, forwarding, dead-letter."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.managers.persistence_manager import PersistenceManager
from src.trading.data_models import TradeDecision
from src.trading.executor_handler import (
    ACTIONABLE_SIGNALS,
    ExecutorHandler,
)


def _make_handler(**overrides):
    config = SimpleNamespace(
        ENTRY_ORDER_TYPE=overrides.pop("entry_order_type", "market"),
        EXECUTOR_API_ENABLED=overrides.pop("executor_enabled", True),
        EXECUTOR_API_URL=overrides.pop("executor_url", "http://127.0.0.1:9199/decision"),
    )
    return ExecutorHandler(
        persistence=MagicMock(),
        config=config,
        logger=MagicMock(),
    )


def _make_decision(**overrides):
    defaults = {
        "timestamp": datetime.now(timezone.utc),
        "symbol": "BTC/USDC",
        "action": "BUY",
        "confidence": "HIGH",
        "price": 70000.0,
        "stop_loss": 68000.0,
        "take_profit": 74000.0,
        "quantity": 0.15,
        "reasoning": "Breakout confirmation",
    }
    defaults.update(overrides)
    return TradeDecision(**defaults)


class TestBuildPayload:
    def test_skips_none_analysis_or_symbol(self):
        handler = _make_handler()
        assert handler._build(None, _make_decision(), symbol="BTC/USDC") is None
        assert handler._build({}, _make_decision(), symbol="") is None

    def test_skips_when_strategy_decision_is_none(self):
        handler = _make_handler()
        assert handler._build({"signal": "BUY"}, None, symbol="BTC/USDC") is None

    def test_skips_non_actionable_signals(self):
        handler = _make_handler()
        assert handler._build({"signal": "HOLD"}, _make_decision(action="HOLD"), symbol="BTC/USDC") is None

    def test_skips_hold_decision(self):
        handler = _make_handler()
        result = handler._build({"signal": "BUY"}, _make_decision(action="HOLD"), symbol="BTC/USDC")
        assert result is None

    def test_maps_strategy_decision_fields(self):
        handler = _make_handler()
        decision = _make_decision()
        payload = handler._build(
            {"signal": "BUY", "reduce_only": False, "leverage": 1},
            decision,
            symbol="BTC/USDC",
        )
        assert payload is not None
        assert payload["symbol"] == "BTC/USDC"
        assert payload["signal"] == "BUY"
        assert payload["quantity"] == 0.15
        assert payload["entry_price"] == 70000.0
        assert payload["stop_loss"] == 68000.0
        assert payload["take_profit"] == 74000.0
        assert payload["confidence"] == "HIGH"
        assert payload["reasoning"] == "Breakout confirmation"

    def test_uses_config_entry_order_type(self):
        handler = _make_handler(entry_order_type="limit")
        decision = _make_decision()
        payload = handler._build({"signal": "BUY"}, decision, symbol="BTC/USDC")
        assert payload["order_type"] == "limit"

    def test_falls_back_to_analysis(self):
        handler = _make_handler()
        decision = _make_decision(quantity=None, price=None, stop_loss=None,
                                   take_profit=None, reasoning=None)
        payload = handler._build(
            {"signal": "BUY", "quantity": 0.5, "entry_price": 50000.0,
             "stop_loss": 49000.0, "take_profit": 52000.0},
            decision, symbol="ETH/USDC",
        )
        assert payload["quantity"] == 0.5
        assert payload["entry_price"] == 50000.0
        assert payload["stop_loss"] == 49000.0
        assert payload["take_profit"] == 52000.0
        assert payload["reasoning"] == ""

    def test_defaults(self):
        handler = _make_handler()
        decision = _make_decision(quantity=0.0)
        payload = handler._build({"signal": "SELL"}, decision, symbol="BTC/USDC")
        assert payload["quantity"] == 0.0
        assert payload["reduce_only"] is False
        assert payload["leverage"] == 1

    def test_actionable_signals_includes_futures(self):
        assert ACTIONABLE_SIGNALS == frozenset({"BUY", "SELL", "CLOSE", "UPDATE", "LONG", "SHORT"})


class TestHandle:
    @pytest.mark.asyncio
    async def test_persists_and_forwards(self):
        handler = _make_handler()
        handler._persist = MagicMock()
        handler._forward = AsyncMock()

        decision = _make_decision()
        await handler.handle({"signal": "BUY"}, decision, "BTC/USDC")

        handler._persist.assert_called_once()
        handler._forward.assert_awaited_once()
        payload = handler._forward.call_args[0][0]
        assert payload["signal"] == "BUY"

    @pytest.mark.asyncio
    async def test_skips_none_analysis(self):
        handler = _make_handler()
        handler._persist = MagicMock()
        handler._forward = AsyncMock()

        await handler.handle(None, _make_decision(), "BTC/USDC")
        handler._persist.assert_not_called()
        handler._forward.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_null_strategy_decision(self):
        handler = _make_handler()
        handler._persist = MagicMock()
        handler._forward = AsyncMock()

        await handler.handle({"signal": "BUY"}, None, "BTC/USDC")
        handler._persist.assert_not_called()
        handler._forward.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dead_letter_on_forward_failure(self):
        handler = _make_handler()
        handler._persist = MagicMock()
        handler._forward = AsyncMock(side_effect=RuntimeError("network down"))
        handler._write_dead_letter = MagicMock()

        decision = _make_decision()
        await handler.handle({"signal": "SELL"}, decision, "ETH/USDC")

        handler._persist.assert_called_once()
        handler._write_dead_letter.assert_called_once()
        assert handler._write_dead_letter.call_args[0][0]["signal"] == "SELL"


class TestPersist:
    def test_persist_saves_latest_decision(self):
        persistence = MagicMock()
        handler = ExecutorHandler(
            persistence=persistence,
            config=SimpleNamespace(ENTRY_ORDER_TYPE="market"),
            logger=MagicMock(),
        )
        payload = {"signal": "BUY", "symbol": "BTC/USDC"}
        handler._persist(payload)
        persistence.save_latest_decision.assert_called_once_with(payload)

    def test_persist_logs_error_on_failure(self):
        persistence = MagicMock()
        persistence.save_latest_decision.side_effect = RuntimeError("disk full")
        logger = MagicMock()
        handler = ExecutorHandler(
            persistence=persistence,
            config=SimpleNamespace(ENTRY_ORDER_TYPE="market"),
            logger=logger,
        )
        handler._persist({"signal": "BUY"})
        logger.error.assert_called_once()


class TestDeadLetter:
    def test_write_dead_letter_appends_jsonl(self, tmp_path):
        import src.trading.executor_handler as mod
        orig_path = mod.DEAD_LETTER_PATH
        dl_path = tmp_path / "failed_forwards.jsonl"
        mod.DEAD_LETTER_PATH = dl_path

        try:
            handler = _make_handler()
            payload = {"timestamp": "2026-07-23T12:00:00.000000", "symbol": "BTC/USDC", "signal": "BUY"}
            handler._write_dead_letter(payload)

            assert dl_path.exists()
            lines = dl_path.read_text().strip().split("\n")
            assert len(lines) == 1
            loaded = json.loads(lines[0])
            assert loaded["signal"] == "BUY"
        finally:
            mod.DEAD_LETTER_PATH = orig_path

    def test_write_dead_letter_logs_error_on_write_failure(self):
        handler = _make_handler()
        handler.logger = MagicMock()

        # Cause open() to fail: path where parent is a file, not a directory
        import src.trading.executor_handler as mod
        import tempfile
        orig_path = mod.DEAD_LETTER_PATH
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(b"block")
                block_path = tf.name
            mod.DEAD_LETTER_PATH = Path(block_path) / "sub" / "nope.jsonl"

            handler._write_dead_letter({"signal": "SELL"})
            handler.logger.error.assert_called_once()
        finally:
            mod.DEAD_LETTER_PATH = orig_path
            try:
                Path(block_path).unlink(missing_ok=True)
            except Exception:
                pass


class TestForward:
    @pytest.mark.asyncio
    async def test_forward_posts_to_executor(self):
        handler = _make_handler()
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        payload = {"timestamp": "t", "symbol": "BTC/USDC", "signal": "BUY"}

        # Patch httpx at module level so @retry_async sees the mock too
        with patch("src.trading.executor_handler.httpx.AsyncClient", return_value=mock_client):
            await handler._forward(payload)

        mock_client.post.assert_awaited()
        handler.logger.info.assert_called()


class TestAtomicPersistence:
    def test_save_latest_decision_atomic_write(self, tmp_path):
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
        leftovers = list(Path(tmp_path).glob("*.json"))
        assert leftovers == [path]
