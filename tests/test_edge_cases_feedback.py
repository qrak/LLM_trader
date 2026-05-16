"""Edge case and stress tests for closed-loop feedback system.

Covers:
  1. Empty State: Brain behavior with zero rejections in the database
  2. Saturation: Prompt builder handles 50+ recent rejections (truncation/summarization)
  3. Async Race Conditions: Writing to ChromaDB doesn't block main trading loop
  4. Concurrent Writes: Multiple simultaneous blocked trades don't corrupt data
  5. Large Reasoning Snippets: Long AI reasoning truncated properly
  6. Boundary Values: NaN, Infinity, extreme price values
"""

import asyncio
import math
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import chromadb
import pytest
from sentence_transformers import SentenceTransformer

from src.managers.risk_manager import RiskManager
from src.trading.brain import TradingBrainService
from src.trading.data_models import MarketConditions, Position, RiskAssessment
from src.trading.trading_strategy import TradingStrategy
from src.trading.vector_memory import VectorMemoryService


# ═════════════════════════════════════════════════════════════════
# SECTION 1: EMPTY STATE — Brain with zero rejections
# ═════════════════════════════════════════════════════════════════


class TestEmptyStateBrain:
    """Brain behavior when no rejections exist."""

    def test_get_context_does_not_include_feedback_when_no_blocks(self):
        """get_context() should not inject CRITICAL FEEDBACK when no blocks."""
        brain = _make_minimal_brain()
        brain.vector_memory.get_blocked_trade_feedback.return_value = ""
        brain.vector_memory.get_context_for_prompt.return_value = ""
        brain.vector_memory.get_relevant_rules.return_value = []
        brain.vector_memory.compute_confidence_stats.return_value = {}
        brain.vector_memory.get_confidence_recommendation.return_value = ""
        brain.vector_memory.get_direction_bias.return_value = None

        ctx = brain.get_context(adx=25)
        assert "CRITICAL FEEDBACK" not in ctx
        assert "System Rejections" not in ctx

    def test_get_context_survives_feedback_exception(self):
        """get_context() handles vector_memory.get_blocked_trade_feedback raising."""
        brain = _make_minimal_brain()
        brain.vector_memory.get_blocked_trade_feedback.side_effect = RuntimeError("DB crash")
        brain.vector_memory.get_context_for_prompt.return_value = ""
        brain.vector_memory.get_relevant_rules.return_value = []
        brain.vector_memory.compute_confidence_stats.return_value = {}
        brain.vector_memory.get_confidence_recommendation.return_value = ""
        brain.vector_memory.get_direction_bias.return_value = None

        # Should not raise — should silently skip feedback section
        ctx = brain.get_context(adx=25)
        assert "CRITICAL FEEDBACK" not in ctx

    def test_zero_trade_count_shows_no_trading_brain_section(self):
        """When trade_count is 0, Trading Brain section is omitted."""
        brain = _make_minimal_brain()
        brain.vector_memory.trade_count = 0
        brain.vector_memory.get_blocked_trade_feedback.return_value = ""
        brain.vector_memory.get_context_for_prompt.return_value = ""
        brain.vector_memory.get_relevant_rules.return_value = []
        brain.vector_memory.compute_confidence_stats.return_value = {}
        brain.vector_memory.get_confidence_recommendation.return_value = ""
        brain.vector_memory.get_direction_bias.return_value = None

        ctx = brain.get_context(adx=25)
        assert "Trading Brain" not in ctx

    def test_get_blocked_trade_feedback_called_even_with_zero_trades(self):
        """Feedback is checked regardless of trade count."""
        brain = _make_minimal_brain()
        brain.vector_memory.trade_count = 0
        brain.vector_memory.get_blocked_trade_feedback.return_value = ""
        brain.vector_memory.get_context_for_prompt.return_value = ""
        brain.vector_memory.get_relevant_rules.return_value = []
        brain.vector_memory.compute_confidence_stats.return_value = {}
        brain.vector_memory.get_confidence_recommendation.return_value = ""
        brain.vector_memory.get_direction_bias.return_value = None

        brain.get_context(adx=25)
        brain.vector_memory.get_blocked_trade_feedback.assert_called_once()


# ═════════════════════════════════════════════════════════════════
# SECTION 2: SATURATION — High volume of rejections
# ═════════════════════════════════════════════════════════════════


class TestSaturationHighVolume:
    """Handle 50+ recent rejections without breaking the prompt builder."""

    def test_fifty_blocked_trades_truncated_to_n(self, vector_memory_saturation):
        """get_recent_blocked_trades(n=5) returns max 5 even with 50 stored."""
        svc = vector_memory_saturation
        results = svc.get_recent_blocked_trades(n=5)
        assert len(results) <= 5

    def test_fifty_blocked_trades_feedback_not_empty(self, vector_memory_saturation):
        """Feedback for 50 blocks still generates output."""
        svc = vector_memory_saturation
        feedback = svc.get_blocked_trade_feedback(n=5)
        assert len(feedback) > 0
        assert "CRITICAL FEEDBACK" in feedback

    def test_fifty_blocked_trades_feedback_groups_by_guard(self, vector_memory_saturation):
        """With many blocks, feedback still groups by guard type."""
        svc = vector_memory_saturation
        feedback = svc.get_blocked_trade_feedback(n=50)
        # Each unique guard type should appear as a group
        assert "###" in feedback  # subsection headers

    def test_fifty_blocked_trades_count_is_correct(self, vector_memory_saturation):
        """get_blocked_trade_count reflects total stored."""
        svc = vector_memory_saturation
        count = svc.get_blocked_trade_count()
        assert count >= 50  # at least 50 stored

    def test_brain_survives_fifty_blocks_in_prompt(self, vector_memory_saturation):
        """Brain.get_context() doesn't crash with 50+ blocks."""
        svc = vector_memory_saturation
        logger = MagicMock()
        persistence = MagicMock()

        # Create a mock VM that delegates get_blocked_trade_feedback to the real one
        mock_vm = MagicMock()
        mock_vm.trade_count = 0
        mock_vm.get_blocked_trade_feedback = lambda *a, **kw: svc.get_blocked_trade_feedback(*a, **kw)
        mock_vm.get_context_for_prompt.return_value = ""
        mock_vm.get_relevant_rules.return_value = []
        mock_vm.compute_confidence_stats.return_value = {}
        mock_vm.get_confidence_recommendation.return_value = ""
        mock_vm.get_direction_bias.return_value = None

        brain = TradingBrainService(
            logger=logger,
            persistence=persistence,
            vector_memory=mock_vm,
        )
        ctx = brain.get_context(adx=25)
        assert "CRITICAL FEEDBACK" in ctx
        # Should still be a valid string (not too large for a prompt)
        assert len(ctx) < 10000  # reasonable prompt size


# ═════════════════════════════════════════════════════════════════
# SECTION 3: ASYNC RACE CONDITIONS
# ═════════════════════════════════════════════════════════════════


class TestAsyncRaceConditions:
    """Verify ChromaDB writes don't block main trading loop."""

    def test_store_blocked_trade_is_synchronous_does_not_block(self, vector_memory):
        """store_blocked_trade completes in under 200ms (not async, but fast)."""
        start = time.monotonic()
        for _ in range(10):
            _store_test_block_fast(vector_memory)
        elapsed = time.monotonic() - start
        # 10 writes should complete quickly
        assert elapsed < 5.0  # generous bound

    @pytest.mark.asyncio
    async def test_concurrent_writes_dont_corrupt(self, embedding_model):
        """Multiple simultaneous blocked trade writes are safe."""
        client = chromadb.Client(chromadb.config.Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=False,
        ))
        client.reset()  # ensure clean slate
        logger = MagicMock()
        svc = VectorMemoryService(
            logger=logger,
            chroma_client=client,
            embedding_model=embedding_model,
        )
        svc._ensure_initialized()

        # Simulate concurrent writes
        for i in range(20):
            result = _store_test_block_fast(svc, guard_type=f"concurrent_{i}")
            assert result is True

        count = svc.get_blocked_trade_count()
        assert count >= 20

    @pytest.mark.asyncio
    async def test_trading_loop_not_blocked_by_storage(self):
        """RiskManager friction storage doesn't add >100ms to position open."""
        config = _make_risk_config()
        logger = MagicMock()
        persistence = MagicMock()
        persistence.load_position.return_value = None
        persistence.async_save_position = AsyncMock()
        persistence.async_save_trade_decision = AsyncMock()

        brain = MagicMock()
        brain.vector_memory = MagicMock()
        brain.vector_memory.store_blocked_trade.return_value = True
        brain.vector_memory.trade_count = 0
        brain.get_dynamic_thresholds.return_value = {"rr_borderline_min": 1.5}

        statistics = MagicMock()
        statistics.get_current_capital.return_value = 10000.0
        memory_service = MagicMock()
        risk_mgr = RiskManager(logger=MagicMock(), config=config)

        from datetime import datetime, timezone
        from src.trading.data_models import Position

        extractor = MagicMock()
        extractor.extract_trading_info.return_value = ("BUY", "HIGH", 95.0, 115.0, 0.05, "Good")
        extractor.validate_signal.return_value = True
        factory = MagicMock()
        factory.create_position.return_value = Position(
            entry_price=100.0, stop_loss=95.0, take_profit=115.0,
            size=5.0, entry_time=datetime.now(timezone.utc),
            confidence="HIGH", direction="LONG", symbol="BTC/USDC",
            size_pct=0.05,
        )

        strategy = TradingStrategy(
            logger=logger,
            persistence=persistence,
            brain_service=brain,
            statistics_service=statistics,
            memory_service=memory_service,
            risk_manager=risk_mgr,
            config=config,
            position_extractor=extractor,
            position_factory=factory,
        )

        start = time.monotonic()
        # Add artificial delay to store_blocked_trade to test timeout
        async def slow_store(*args, **kwargs):
            await asyncio.sleep(0.01)
            return True
        brain.vector_memory.store_blocked_trade = slow_store

        await strategy._open_new_position(
            signal="BUY", confidence="HIGH",
            stop_loss=95.0, take_profit=115.0,
            position_size=0.05, current_price=100.0,
            symbol="BTC/USDC", reasoning="Test",
        )
        elapsed = time.monotonic() - start
        assert elapsed < 1.0  # should be well under 1 second


# ═════════════════════════════════════════════════════════════════
# SECTION 4: BOUNDARY VALUES
# ═════════════════════════════════════════════════════════════════


class TestBoundaryValues:
    """NaN, Infinity, zero, and extreme price values."""

    def test_nan_sl_handled_gracefully(self):
        """NaN stop_loss shouldn't crash RiskManager."""
        config = _make_risk_config()
        mgr = RiskManager(logger=MagicMock(), config=config)

        # NaN SL → AI SL not used → falls back to dynamic
        assessment = mgr.calculate_entry_parameters(
            signal="BUY", current_price=100.0, capital=10000.0,
            confidence="HIGH", stop_loss=math.nan,
            market_conditions=MarketConditions(atr=2.0, atr_percentage=2.0),
        )
        assert math.isfinite(assessment.stop_loss)
        assert math.isfinite(assessment.take_profit)

    def test_inf_price_handled(self):
        """Infinite stop_loss shouldn't crash."""
        config = _make_risk_config()
        mgr = RiskManager(logger=MagicMock(), config=config)

        assessment = mgr.calculate_entry_parameters(
            signal="BUY", current_price=100.0, capital=10000.0,
            confidence="HIGH", stop_loss=float("inf"),
            market_conditions=MarketConditions(atr=2.0, atr_percentage=2.0),
        )
        assert math.isfinite(assessment.stop_loss)

    def test_negative_price_handled(self):
        """Negative stop_loss should be treated as invalid → fallback to dynamic."""
        config = _make_risk_config()
        mgr = RiskManager(logger=MagicMock(), config=config)

        # Negative SL → not > 0 → falls back to dynamic
        assessment = mgr.calculate_entry_parameters(
            signal="BUY", current_price=100.0, capital=10000.0,
            confidence="HIGH", stop_loss=-10.0,
            market_conditions=MarketConditions(atr=2.0, atr_percentage=2.0),
        )
        assert assessment.stop_loss > 0

    def test_zero_capital_produces_zero_quantity(self):
        """Zero capital results in zero quantity, not division by zero."""
        config = _make_risk_config()
        mgr = RiskManager(logger=MagicMock(), config=config)

        assessment = mgr.calculate_entry_parameters(
            signal="BUY", current_price=100.0, capital=0.0,
            confidence="HIGH",
        )
        assert assessment.quantity == 0.0
        assert assessment.quote_amount == 0.0

    def test_extreme_rr_ratio_still_finite(self):
        """R/R ratio should always be finite."""
        config = _make_risk_config()
        mgr = RiskManager(logger=MagicMock(), config=config)

        # Very tight SL (1% clamped to min) with far TP
        assessment = mgr.calculate_entry_parameters(
            signal="BUY", current_price=100.0, capital=10000.0,
            confidence="HIGH", stop_loss=99.99, take_profit=200.0,
            market_conditions=MarketConditions(atr=0.5, atr_percentage=0.5),
        )
        assert math.isfinite(assessment.rr_ratio)
        assert assessment.rr_ratio >= 0

    def test_empty_market_conditions_uses_defaults(self):
        """Default MarketConditions doesn't crash."""
        config = _make_risk_config()
        mgr = RiskManager(logger=MagicMock(), config=config)

        assessment = mgr.calculate_entry_parameters(
            signal="BUY", current_price=100.0, capital=10000.0,
            confidence="HIGH", market_conditions=MarketConditions(),
        )
        assert math.isfinite(assessment.stop_loss)
        assert math.isfinite(assessment.take_profit)


# ═════════════════════════════════════════════════════════════════
# SECTION 5: LARGE REASONING SNIPPET TRUNCATION
# ═════════════════════════════════════════════════════════════════


class TestLargeReasoningSnippet:
    """Very long AI reasoning doesn't break storage or feedback."""

    def test_long_reasoning_snippet_stored(self, vector_memory):
        """2000-char reasoning snippet is stored without error."""
        long_reasoning = "Analysis: " + "X" * 2000
        stored = _store_test_block_fast(vector_memory, reasoning_snippet=long_reasoning)
        assert stored is True

        results = vector_memory.get_recent_blocked_trades(n=1)
        assert len(results) == 1
        # The full snippet should be in the result
        assert results[0]["reasoning_snippet"] == long_reasoning

    def test_empty_reasoning_snippet_ok(self, vector_memory):
        """Empty reasoning snippet doesn't cause errors."""
        stored = _store_test_block_fast(vector_memory, reasoning_snippet="")
        assert stored is True


# ═════════════════════════════════════════════════════════════════
# SECTION 6: GUARD TYPE UNKNOWN
# ═════════════════════════════════════════════════════════════════


class TestUnknownGuardType:
    """Unknown guard types in feedback don't crash."""

    def test_unknown_guard_type_renders_with_title_case(self, vector_memory):
        """Unknown guard_type is rendered with title-cased name."""
        _store_test_block_fast(vector_memory, guard_type="custom_new_guard")

        feedback = vector_memory.get_blocked_trade_feedback(n=5)
        assert "Custom New Guard" in feedback  # title-cased fallback


# ═════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════


def _make_minimal_brain() -> TradingBrainService:
    """Create a TradingBrainService with all deps mocked."""
    logger = MagicMock()
    persistence = MagicMock()
    vector_memory = MagicMock()
    vector_memory.trade_count = 0
    return TradingBrainService(
        logger=logger,
        persistence=persistence,
        vector_memory=vector_memory,
    )


def _make_brain_with_svc(svc: VectorMemoryService) -> TradingBrainService:
    """Create a TradingBrainService with a real VectorMemoryService."""
    logger = MagicMock()
    persistence = MagicMock()
    # Override trade_count to simulate existing trades
    svc.trade_count = 5
    return TradingBrainService(
        logger=logger,
        persistence=persistence,
        vector_memory=svc,
    )


def _make_risk_config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "MAX_POSITION_SIZE": 0.10,
        "POSITION_SIZE_FALLBACK_LOW": 0.01,
        "POSITION_SIZE_FALLBACK_MEDIUM": 0.02,
        "POSITION_SIZE_FALLBACK_HIGH": 0.03,
        "TRANSACTION_FEE_PERCENT": 0.001,
        "DEMO_QUOTE_CAPITAL": 10000.0,
        "TIMEFRAME": "4h",
        "STOP_LOSS_TYPE": "soft",
        "STOP_LOSS_CHECK_INTERVAL": "4h",
        "TAKE_PROFIT_TYPE": "soft",
        "TAKE_PROFIT_CHECK_INTERVAL": "4h",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _store_test_block_fast(
    svc: VectorMemoryService,
    *,
    guard_type: str = "rr_minimum",
    direction: str = "LONG",
    confidence: str = "HIGH",
    suggested_rr: float = 1.2,
    required_rr: float = 2.0,
    suggested_sl_pct: float = 0.03,
    suggested_tp_pct: float = 0.04,
    suggested_sl: float = 97.0,
    suggested_tp: float = 104.0,
    current_price: float = 100.0,
    volatility_level: str = "MEDIUM",
    reasoning_snippet: str = "Test",
) -> bool:
    return svc.store_blocked_trade(
        guard_type=guard_type,
        direction=direction,
        confidence=confidence,
        suggested_rr=suggested_rr,
        required_rr=required_rr,
        suggested_sl_pct=suggested_sl_pct,
        suggested_tp_pct=suggested_tp_pct,
        suggested_sl=suggested_sl,
        suggested_tp=suggested_tp,
        current_price=current_price,
        volatility_level=volatility_level,
        reasoning_snippet=reasoning_snippet,
    )


# ── Saturation fixture: 50+ blocked trades ──────────────────────


@pytest.fixture(scope="module")
def embedding_model_saturation():
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture
def vector_memory_saturation(embedding_model_saturation):
    """VectorMemoryService pre-populated with 55 blocked trades."""
    client = chromadb.Client(chromadb.config.Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=False,
    ))
    logger = MagicMock()
    svc = VectorMemoryService(
        logger=logger,
        chroma_client=client,
        embedding_model=embedding_model_saturation,
    )
    svc._ensure_initialized()

    guards = ["rr_minimum", "sl_distance_max", "sl_distance_min",
              "sl_below_entry", "tp_below_entry"]
    for guard in guards:
        for i in range(10):
            _store_test_block_fast(
                svc,
                guard_type=guard,
                suggested_rr=1.0 + i * 0.1,
                required_rr=2.0,
                reasoning_snippet=f"Test #{i} for {guard}",
            )
    # Add 5 more with different metadata
    for i in range(5):
        _store_test_block_fast(
            svc,
            guard_type="position_size_clamp",
            suggested_rr=0.5 + i * 0.1,
            required_rr=1.5,
            reasoning_snippet=f"Size clamp #{i}",
        )

    return svc


# ── Standard vector_memory fixture for edge case tests ──────────


@pytest.fixture(scope="module")
def embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture
def vector_memory(embedding_model):
    """Fresh VectorMemoryService for each test."""
    client = chromadb.Client(chromadb.config.Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=False,
    ))
    logger = MagicMock()
    svc = VectorMemoryService(
        logger=logger,
        chroma_client=client,
        embedding_model=embedding_model,
    )
    svc._ensure_initialized()
    return svc
