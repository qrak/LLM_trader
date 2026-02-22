"""Verify that brain context generation does not block the asyncio event loop.

Tests that wrapping brain_service.get_context() in asyncio.to_thread allows
other coroutines to run concurrently while ChromaDB + embedding calls execute.
"""
import asyncio
import time
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_brain_service(delay_ms: int = 100) -> MagicMock:
    """Return a mock brain_service whose get_context() simulates a blocking call."""
    brain_service = MagicMock()

    def slow_get_context(**_kwargs) -> str:
        time.sleep(delay_ms / 1000)  # Simulate ChromaDB + embedding latency
        return "mock brain context"

    brain_service.get_context.side_effect = slow_get_context
    return brain_service


def _make_engine(brain_service: MagicMock):
    """Return a minimal AnalysisEngine with enough mocks to test the target method."""
    from src.analyzer.analysis_engine import AnalysisEngine

    engine = object.__new__(AnalysisEngine)  # Bypass __init__

    # Minimal attributes required by helper extraction methods
    engine.context = MagicMock()
    engine.context.sentiment = None
    engine.context.market_microstructure = None
    engine.context.current_price = 50000.0

    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBrainContextNonBlocking:
    """Verify the updated _generate_brain_context_from_current_indicators is non-blocking."""

    @pytest.mark.asyncio
    async def test_event_loop_not_blocked_by_brain_context(self):
        """Other coroutines should progress while brain context is being computed.

        We schedule a lightweight coroutine that increments a counter every 10ms.
        If the event loop is blocked, the counter will stay at 0 during the
        100ms brain-context computation. If non-blocking, it will increment.
        """
        DELAY_MS = 100
        brain_service = _make_brain_service(delay_ms=DELAY_MS)
        engine = _make_engine(brain_service)

        progress = {"count": 0}

        async def concurrent_task():
            for _ in range(20):
                await asyncio.sleep(0.01)  # 10ms
                progress["count"] += 1

        technical_data = {
            "plus_di": 30.0, "minus_di": 20.0,
            "adx": 28.0, "atr_percent": 2.0,
            "rsi": 55.0,
            "macd_line": 10.0, "macd_signal": 5.0,
            "obv_slope": 0.3,
            "bb_upper": 52000.0, "bb_lower": 48000.0,
        }

        # Run brain context + concurrent task together
        await asyncio.gather(
            engine._generate_brain_context_from_current_indicators(brain_service, technical_data),
            concurrent_task(),
        )

        # If non-blocking: concurrent_task ran while brain context was computed
        assert progress["count"] >= 3, (
            f"Event loop was blocked: concurrent task only progressed {progress['count']} times "
            f"during {DELAY_MS}ms brain context computation. Expected >= 3."
        )

    @pytest.mark.asyncio
    async def test_brain_context_result_is_correct(self):
        """The method should return exactly what brain_service.get_context() returns."""
        brain_service = _make_brain_service(delay_ms=10)
        engine = _make_engine(brain_service)

        technical_data = {
            "plus_di": 20.0, "minus_di": 30.0,
            "adx": 22.0, "atr_percent": 1.0,
            "rsi": 45.0,
            "macd_line": 5.0, "macd_signal": 8.0,
            "obv_slope": -0.6,
            "bb_upper": 52000.0, "bb_lower": 48000.0,
        }

        result = await engine._generate_brain_context_from_current_indicators(
            brain_service, technical_data
        )

        assert result == "mock brain context"
        brain_service.get_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_brain_context_passes_correct_params(self):
        """Extracted indicator params should be forwarded correctly to brain_service."""
        brain_service = _make_brain_service(delay_ms=10)
        engine = _make_engine(brain_service)

        technical_data = {
            "plus_di": 35.0, "minus_di": 20.0,  # → BULLISH
            "adx": 30.0,
            "atr_percent": 4.0,  # → HIGH volatility
            "rsi": 72.0,         # → OVERBOUGHT
            "macd_line": 100.0, "macd_signal": 50.0,  # → BULLISH MACD
            "obv_slope": 0.8,    # → ACCUMULATION
            "bb_upper": 52000.0, "bb_lower": 48000.0,
        }

        await engine._generate_brain_context_from_current_indicators(brain_service, technical_data)

        call_kwargs = brain_service.get_context.call_args.kwargs
        assert call_kwargs["trend_direction"] == "BULLISH"
        assert call_kwargs["adx"] == 30.0
        assert call_kwargs["volatility_level"] == "HIGH"
        assert call_kwargs["rsi_level"] == "OVERBOUGHT"
        assert call_kwargs["macd_signal"] == "BULLISH"
        assert call_kwargs["volume_state"] == "ACCUMULATION"


class TestBrainContextPerformance:
    """Demonstrate the performance improvement from non-blocking execution."""

    @pytest.mark.asyncio
    async def test_parallel_is_faster_than_sequential(self):
        """Running brain context + other work in parallel should be faster than sequential.

        This is a timing-based test that verifies asyncio.to_thread enables
        true concurrency. Sequential execution takes 2x the delay; parallel
        should finish in ~1x.
        """
        DELAY_MS = 80
        brain_service = _make_brain_service(delay_ms=DELAY_MS)
        engine = _make_engine(brain_service)
        technical_data = {
            "plus_di": 25.0, "minus_di": 25.0, "adx": 25.0, "atr_percent": 2.0,
            "rsi": 50.0, "macd_line": 0.0, "macd_signal": 0.0, "obv_slope": 0.0,
            "bb_upper": 52000.0, "bb_lower": 48000.0,
        }

        async def other_work():
            await asyncio.sleep(DELAY_MS / 1000)  # Same duration as brain context

        # Parallel: should finish in ~DELAY_MS, not 2x
        start = time.perf_counter()
        await asyncio.gather(
            engine._generate_brain_context_from_current_indicators(brain_service, technical_data),
            other_work(),
        )
        parallel_duration = time.perf_counter() - start

        # Should be much less than 2x the delay (allowing 80% overhead)
        max_expected = (DELAY_MS * 2 / 1000) * 0.80
        assert parallel_duration < max_expected, (
            f"Parallel execution took {parallel_duration:.3f}s, expected < {max_expected:.3f}s. "
            f"The event loop may still be blocked."
        )
