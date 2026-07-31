"""Test suite for ExecutorHandler wire contract compliance and type safety."""

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.logger.logger import Logger
from src.trading.data_models import TradeDecision
from src.trading.executor_handler import ExecutorHandler


class TestExecutorWireContract(unittest.TestCase):
    """Verifies that ExecutorHandler._build output strictly matches expected schemas."""

    def setUp(self):
        self.logger = MagicMock(spec=Logger)
        self.config = MagicMock()
        self.config.ENTRY_ORDER_TYPE = "limit"
        self.config.EXECUTOR_API_ENABLED = True
        self.config.EXECUTOR_API_URL = "http://localhost:8080/trade"
        self.handler = ExecutorHandler(
            logger=self.logger,
            config=self.config,
            persistence=MagicMock(),
        )

    def test_build_returns_none_for_hold_decision(self):
        decision = TradeDecision(
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDC",
            action="HOLD",
            confidence="HIGH",
            price=68000.0,
            reasoning="Neutral",
        )
        result = self.handler._build({"signal": "HOLD"}, decision, "BTC/USDC")
        self.assertIsNone(result)

    def test_build_returns_none_for_none_decision(self):
        result = self.handler._build({"signal": "BUY"}, None, "BTC/USDC")
        self.assertIsNone(result)

    def test_build_returns_none_for_empty_symbol(self):
        decision = TradeDecision(
            timestamp=datetime.now(timezone.utc),
            symbol="",
            action="BUY",
            confidence="HIGH",
            price=68000.0,
            reasoning="Bullish",
        )
        result = self.handler._build({"signal": "BUY"}, decision, "")
        self.assertIsNone(result)

    def test_build_strict_type_coercion_and_finite_floats(self):
        decision = TradeDecision(
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDC",
            action="BUY",
            confidence="HIGH",
            price=68000.0,
            stop_loss=66000.0,
            take_profit=72000.0,
            quantity=0.05,
            reasoning="Strong technical setup",
        )
        analysis = {
            "signal": "BUY",
            "quantity": "0.05",
            "entry_price": "68000.0",
            "stop_loss": "66000.0",
            "take_profit": "72000.0",
            "reduce_only": False,
            "leverage": "1",
        }
        payload = self.handler._build(analysis, decision, "BTC/USDC")
        self.assertIsNotNone(payload)
        self.assertEqual(payload["symbol"], "BTC/USDC")
        self.assertEqual(payload["signal"], "BUY")
        self.assertEqual(payload["order_type"], "limit")
        self.assertIsInstance(payload["quantity"], float)
        self.assertIsInstance(payload["entry_price"], float)
        self.assertIsInstance(payload["stop_loss"], float)
        self.assertIsInstance(payload["take_profit"], float)
        self.assertIsInstance(payload["reduce_only"], bool)
        self.assertIsInstance(payload["leverage"], int)
        self.assertIsInstance(payload["confidence"], str)
        self.assertIsInstance(payload["reasoning"], str)
        self.assertEqual(payload["quantity"], 0.05)
        self.assertEqual(payload["entry_price"], 68000.0)

    def test_build_handles_nan_and_inf_gracefully(self):
        decision = TradeDecision(
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDC",
            action="BUY",
            confidence="HIGH",
            price=float("inf"),
            stop_loss=float("-inf"),
            take_profit=None,
            quantity=float("nan"),
            reasoning="Setup",
        )
        analysis = {"signal": "BUY"}
        payload = self.handler._build(analysis, decision, "BTC/USDC")
        self.assertIsNotNone(payload)
        # NaN quantity defaults to 0.0, Inf floats default to None
        self.assertEqual(payload["quantity"], 0.0)
        self.assertIsNone(payload["entry_price"])
        self.assertIsNone(payload["stop_loss"])
        self.assertIsNone(payload["take_profit"])


if __name__ == "__main__":
    unittest.main()
