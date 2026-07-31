"""Fault Injection and Chaos Test Suite.

Tests system resilience against unhandled exceptions, malformed payloads,
corrupted numbers (NaN/Inf), and fail-closed security properties.
"""

import unittest
from unittest.mock import MagicMock

from src.logger.logger import Logger
from src.trading.guards import GuardProtocol, GuardResult
from src.trading.guards.pipeline import GuardPipeline
from src.trading.position_extractor import PositionExtractor


class FaultyGuard(GuardProtocol):
    """A guard that deliberately raises an unexpected RuntimeError."""

    @property
    def name(self) -> str:
        return "faulty_guard"

    def check(self, intent, /, *, capital: float, config: float) -> GuardResult:
        raise RuntimeError("Simulated internal guard crash!")


class PassingGuard(GuardProtocol):
    """A guard that passes clean."""

    @property
    def name(self) -> str:
        return "passing_guard"

    def check(self, intent, /, *, capital: float, config: float) -> GuardResult:
        return GuardResult(guard_name="passing_guard", passed=True, reason="ok")


class TestFaultInjectionChaos(unittest.TestCase):
    """Chaos and fault injection test cases."""

    def test_guard_pipeline_fails_closed_on_guard_exception(self):
        """Verify that an unhandled exception inside any guard rejects the order intent."""
        pipeline = GuardPipeline(guards=[PassingGuard(), FaultyGuard()])
        intent = MagicMock()
        results = pipeline.evaluate(intent, capital=10000.0, config=MagicMock())

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].passed)
        self.assertFalse(results[1].passed)
        self.assertIn("failed closed due to error", results[1].reason)

    def test_position_extractor_handles_corrupted_floats_and_nan(self):
        """Verify PositionExtractor handles NaN, Inf, and malformed types safely."""
        extractor = PositionExtractor(logger=MagicMock(spec=Logger))
        analysis = {
            "signal": "BUY",
            "confidence": "HIGH",
            "entry_price": float("nan"),
            "stop_loss": float("inf"),
            "take_profit": "not_a_number",
            "position_size": float("-inf"),
            "reasoning": "Corrupted text test",
        }
        signal, confidence, sl, tp, pos_size, reasoning = extractor._extract_from_dict(analysis)

        self.assertEqual(signal, "BUY")
        self.assertEqual(confidence, "HIGH")
        self.assertIsNone(sl)
        self.assertIsNone(tp)
        self.assertIsNone(pos_size)
        self.assertEqual(reasoning, "Corrupted text test")


if __name__ == "__main__":
    unittest.main()
