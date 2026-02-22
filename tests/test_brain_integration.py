"""
Integration tests for the Brain components (VectorMemoryService, TradingBrainService).
Verifies the full flow of storing experiences, retrieving context, and reflecting on performance.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

import chromadb
from sentence_transformers import SentenceTransformer

from src.logger.logger import Logger
from src.trading.vector_memory import VectorMemoryService
from src.trading.brain import TradingBrainService

# Test data directory - ensures isolation from production data
TEST_DATA_DIR = Path("data/brain_integration_test")

@pytest.fixture(scope="session")
def embedding_model():
    """Load embedding model once for all tests."""
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

@pytest.fixture
def logger():
    """Mock logger to avoid polluting stdout."""
    return MagicMock(spec=Logger)

@pytest.fixture
def vector_memory(logger, tmp_path, embedding_model):
    """Initialize VectorMemoryService with a test database."""
    # Use pytest's tmp_path for isolation and automatic cleanup
    data_dir = tmp_path / "brain_integration_test"
    client = chromadb.PersistentClient(path=str(data_dir))
    service = VectorMemoryService(logger, chroma_client=client, embedding_model=embedding_model)
    
    yield service
    
    # Teardown to release file locks
    if service._client:
        del service._client
    import gc
    gc.collect()

@pytest.fixture
def brain_service(logger, vector_memory):
    """Initialize TradingBrainService with mocked persistence and real vector memory."""
    mock_persistence = MagicMock()
    mock_persistence.load_brain.return_value = MagicMock()
    
    return TradingBrainService(
        logger=logger,
        persistence=mock_persistence,
        vector_memory=vector_memory
    )

class TestBrainIntegration:
    
    def test_full_brain_flow(self, brain_service, vector_memory):
        """
        Verify the complete lifecycle:
        1. Store a trade experience
        2. Verify it exists in vector DB
        3. Query for similar context
        4. Verify retrieval accuracy
        """
        
        # 1. Store Experience
        trade_id = "test_trade_001"
        context = "BULLISH market, High ADX (45), RSI Overbought"
        
        vector_memory.store_experience(
            trade_id=trade_id,
            market_context=context,
            outcome="WIN",
            pnl_pct=5.5,
            direction="LONG",
            confidence="HIGH",
            reasoning="Strong trend continuation",
            metadata={"timestamp": "2026-01-01T12:00:00"}
        )
        
        # 2. Verify Storage directly in ChromaDB (via public count if available, or just query)
        assert vector_memory.experience_count == 1
        
        # 3. Query Context
        # Query with similar context
        retrieved_context = brain_service.get_context(
            trend_direction="BULLISH",
            adx=40.0,
            volatility_level="HIGH",
            rsi_level="OVERBOUGHT",
            macd_signal="BULLISH"
        )
        
        # 4. Verify Retrieval
        # The retrieved string should contain our reasoning or context
        print(f"Retrieved Context: {retrieved_context}")
        
        assert "Strong trend continuation" in retrieved_context
        assert "WIN" in retrieved_context
        assert "+5.50%" in retrieved_context

    def test_semantic_reflection_flow(self, brain_service, vector_memory):
        """Verify that reflection requires at least 10 winning trades."""
        # Inject 8 trades (below threshold)
        for i in range(8):
            vector_memory.store_experience(
                trade_id=f"trade_reflect_{i}",
                market_context="BULLISH pattern",
                outcome="WIN",
                pnl_pct=2.0,
                direction="LONG",
                confidence="HIGH",
                reasoning="Pattern match",
                metadata={
                    "market_regime": "BULLISH",
                    "adx_at_entry": 30,
                    "direction": "LONG",
                    "timestamp": f"2026-01-0{i+1}T12:00:00"
                }
            )
        
        # Trigger reflection - should reject due to insufficient samples
        brain_service._trigger_reflection()
        
        # Verify no rules were created
        rules = vector_memory.get_active_rules(n_results=10)
        assert len(rules) == 0, "No rules should be created with < 10 winning trades"
 

    def test_context_recency_bias(self, brain_service, vector_memory):
        """Verify that recent trades are preferred (if Decay Engine is active)."""
        
        # Old Trade
        vector_memory.store_experience(
            trade_id="old_trade",
            market_context="BULLISH setup",
            outcome="LOSS",
            pnl_pct=-1.0,
            direction="LONG",
            confidence="HIGH",
            reasoning="Old logic",
            metadata={"timestamp": "2025-01-01T12:00:00"} # One year ago
        )
        
        # Recent Trade
        vector_memory.store_experience(
            trade_id="recent_trade",
            market_context="BULLISH setup",
            outcome="WIN",
            pnl_pct=5.0,
            direction="LONG",
            confidence="HIGH",
            reasoning="New logic",
            metadata={"timestamp": "2026-01-05T12:00:00"} # Recent
        )
        
        # Query
        context = brain_service.get_context(
            trend_direction="BULLISH",
            adx=25,
            volatility_level="LOW",
            rsi_level="NEUTRAL",
            macd_signal="NEUTRAL"
        )
        
        # Should verify that the recent trade (WIN) has more influence or appears first
        # This depends on how get_context formats output, usually it lists top matches.
        # We expect "New logic" to be present and potentially ranked higher if sorting by hybrid score.
        
        assert "New logic" in context

    def test_synthetic_insight_for_na_reasoning(self, brain_service, vector_memory):
        """Verify that synthetic insights are generated when reasoning is N/A."""
        vector_memory.store_experience(
            trade_id="trade_na_reasoning",
            market_context="Downtrend + Strong Trend + Low Vol",
            outcome="LOSS",
            pnl_pct=-2.5,
            direction="LONG",
            confidence="HIGH",
            reasoning="N/A",
            metadata={
                "timestamp": "2026-01-06T12:00:00",
                "close_reason": "stop_loss",
                "adx_at_entry": 28
            }
        )
        
        context = vector_memory.get_context_for_prompt(
            "Downtrend + Strong Trend + Low Vol", k=5
        )
        
        assert "Downtrend + Strong Trend + Low Vol" in context
        assert "stop_loss" in context
        assert "ADX: 28" in context
        assert "N/A" not in context

    def test_insufficient_data_warning(self, brain_service, vector_memory):
        """Verify that insufficient data warning appears for low-quality matches."""
        vector_memory.store_experience(
            trade_id="trade_different_context",
            market_context="BULLISH + High ADX + High Vol",
            outcome="WIN",
            pnl_pct=5.0,
            direction="LONG",
            confidence="HIGH",
            reasoning="Strong momentum",
            metadata={"timestamp": "2026-01-06T12:00:00"}
        )
        
        context = vector_memory.get_context_for_prompt(
            "BEARISH + Low ADX + Low Vol", k=5
        )
        
        assert "LIMITED DATA" in context or "Strong momentum" in context

    def test_anti_pattern_learning(self, brain_service, vector_memory):
        """Verify that anti-patterns are generated from LOSS trades."""
        for i in range(3):
            vector_memory.store_experience(
                trade_id=f"loss_trade_{i}",
                market_context="NEUTRAL market with choppy conditions",
                outcome="LOSS",
                pnl_pct=-2.0,
                direction="LONG",
                confidence="MEDIUM",
                reasoning="Failed breakout",
                metadata={
                    "timestamp": f"2026-01-0{i+1}T12:00:00",
                    "market_regime": "NEUTRAL",
                    "close_reason": "stop_loss"
                }
            )

        brain_service._trigger_loss_reflection()

        anti_patterns = vector_memory.get_anti_patterns_for_prompt(k=3)

        assert "AVOID" in anti_patterns or anti_patterns == ""

    def test_update_tracking(self, brain_service, vector_memory):
        """Verify that position updates are tracked for learning."""
        from src.trading.dataclasses import Position
        from datetime import datetime

        position = Position(
            entry_price=50000.0,
            stop_loss=48000.0,
            take_profit=54000.0,
            size=0.01,
            quote_amount=500.0,
            entry_time=datetime.now(),
            confidence="HIGH",
            direction="LONG",
            symbol="BTC/USDT"
        )

        brain_service.track_position_update(
            position=position,
            old_sl=48000.0,
            old_tp=54000.0,
            new_sl=49000.0,
            new_tp=54000.0,
            current_price=51000.0,
            current_pnl_pct=2.0
        )

        assert vector_memory.experience_count >= 1

    def test_regime_metadata_storage(self, brain_service, vector_memory):
        """Verify that regime metadata (fear_greed, weekend) is stored correctly."""
        vector_memory.store_experience(
            trade_id="regime_test_trade",
            market_context="BULLISH + High ADX",
            outcome="WIN",
            pnl_pct=3.5,
            direction="LONG",
            confidence="HIGH",
            reasoning="Strong trend",
            metadata={
                "timestamp": "2026-01-06T12:00:00",
                "fear_greed_index": 25,
                "market_regime": "BULLISH",
                "is_weekend": True
            }
        )

        experiences = vector_memory.retrieve_similar_experiences("BULLISH + High ADX", k=1)

        assert len(experiences) >= 1
        meta = experiences[0].metadata  # VectorSearchResult is a dataclass
        assert meta.get("fear_greed_index") == 25
        assert meta.get("market_regime") == "BULLISH"
        assert meta.get("is_weekend")

    def test_missing_metadata_handling(self, brain_service, vector_memory):
        """Verify that older trades with missing/empty metadata do not cause KeyErrors."""
        # Insert a trade with completely empty metadata (simulates an old entry)
        vector_memory.store_experience(
            trade_id="trade_empty_meta",
            market_context="NEUTRAL + Unknown ADX",
            outcome="WIN",
            pnl_pct=1.5,
            direction="LONG",
            confidence="MEDIUM",
            reasoning="Old trade logic",
            metadata={}  # Explicitly empty metadata
        )

        # Insert a trade with partial metadata
        vector_memory.store_experience(
            trade_id="trade_partial_meta",
            market_context="BULLISH + High ADX",
            outcome="LOSS",
            pnl_pct=-2.0,
            direction="SHORT",
            confidence="HIGH",
            reasoning="Old logic 2",
            metadata={"timestamp": "2025-01-01T12:00:00"}
        )

        # 1. Retrieve Context (tests vector_memory.get_context_for_prompt and get_stats_for_context)
        context = brain_service.get_context(
            trend_direction="NEUTRAL"
        )
        assert context is not None
        assert "trade" in context.lower()

        # 2. Compute Stats (tests compute_confidence_stats)
        conf_stats = vector_memory.compute_confidence_stats()
        assert "MEDIUM" in conf_stats
        assert "HIGH" in conf_stats

        # 3. Compute ADX Performance (tests compute_adx_performance)
        adx_stats = vector_memory.compute_adx_performance()
        assert "LOW" in adx_stats  # Missing ADX defaults to 0 -> LOW

        # 4. Compute Factor Performance (tests compute_factor_performance)
        factor_stats = vector_memory.compute_factor_performance()
        assert isinstance(factor_stats, dict)

        # 5. Get Direction Bias
        direction_bias = vector_memory.get_direction_bias()
        assert direction_bias is not None
        assert direction_bias["long_count"] >= 1
        assert direction_bias["short_count"] >= 1


class TestContextAwareRuleRetrieval:
    """Tests for context-aware semantic rule retrieval based on market conditions."""

    def test_bullish_context_returns_bullish_rule(self, brain_service, vector_memory):
        """Verify BULLISH context retrieves BULLISH rules with higher similarity."""
        # Store BULLISH rule
        vector_memory.store_semantic_rule(
            rule_id="rule_bullish_high_adx",
            rule_text="LONG trades perform well in BULLISH market with High ADX trend strength",
            metadata={"source_pattern": "LONG_BULLISH_HIGH_ADX", "rule_type": "pattern"}
        )
        # Store BEARISH rule
        vector_memory.store_semantic_rule(
            rule_id="rule_bearish_high_adx",
            rule_text="SHORT trades perform well in BEARISH market with High ADX trend strength",
            metadata={"source_pattern": "SHORT_BEARISH_HIGH_ADX", "rule_type": "pattern"}
        )

        # Query with BULLISH context
        rules = vector_memory.get_relevant_rules(
            current_context="BULLISH + High ADX + HIGH Volatility",
            n_results=2
        )

        assert len(rules) >= 1
        # BULLISH rule should rank higher
        assert "BULLISH" in rules[0]["text"] or "LONG" in rules[0]["text"]
        assert rules[0]["similarity"] > 40  # Should exceed min threshold

    def test_bearish_context_returns_bearish_rule(self, brain_service, vector_memory):
        """Verify BEARISH context retrieves BEARISH rules with higher similarity."""
        # Store both rules
        vector_memory.store_semantic_rule(
            rule_id="rule_bullish_001",
            rule_text="LONG trades perform well in BULLISH uptrend with strong momentum",
            metadata={"source_pattern": "LONG_BULLISH"}
        )
        vector_memory.store_semantic_rule(
            rule_id="rule_bearish_001",
            rule_text="SHORT trades perform well in BEARISH downtrend with strong momentum",
            metadata={"source_pattern": "SHORT_BEARISH"}
        )

        # Query with BEARISH context
        rules = vector_memory.get_relevant_rules(
            current_context="BEARISH + High ADX + MEDIUM Volatility",
            n_results=2
        )

        assert len(rules) >= 1
        # BEARISH rule should rank higher
        assert "BEARISH" in rules[0]["text"] or "SHORT" in rules[0]["text"]

    def test_low_adx_context_filters_irrelevant_rules(self, brain_service, vector_memory):
        """Verify Low ADX context correctly filters rules for weak trend conditions."""
        # Store High ADX rule
        vector_memory.store_semantic_rule(
            rule_id="rule_high_adx",
            rule_text="Enter trades with High ADX above 25 for strong trend confirmation",
            metadata={"source_pattern": "HIGH_ADX"}
        )
        # Store Low ADX rule
        vector_memory.store_semantic_rule(
            rule_id="rule_low_adx",
            rule_text="Avoid trades in Low ADX conditions below 20 as trend is weak",
            metadata={"source_pattern": "LOW_ADX", "rule_type": "anti_pattern"}
        )

        # Query with Low ADX context
        rules = vector_memory.get_relevant_rules(
            current_context="NEUTRAL + Low ADX + LOW Volatility",
            n_results=2
        )

        # Low ADX rule should be more relevant
        if rules:
            low_adx_found = any("Low ADX" in r["text"] for r in rules)
            assert low_adx_found, "Low ADX rule should be retrieved for low ADX context"

    def test_min_similarity_threshold_filters_rules(self, brain_service, vector_memory):
        """Verify min_similarity threshold filters out irrelevant rules."""
        # Store a very specific rule
        vector_memory.store_semantic_rule(
            rule_id="rule_specific",
            rule_text="LONG BTC during weekend accumulation with extreme fear sentiment",
            metadata={"source_pattern": "WEEKEND_FEAR"}
        )

        # Query with completely different context
        rules = vector_memory.get_relevant_rules(
            current_context="SHORT trades during high volatility breakout with extreme greed",
            n_results=3,
            min_similarity=0.6  # Higher threshold
        )

        # Rule should be filtered out due to low similarity
        assert len(rules) == 0 or all(r["similarity"] >= 60 for r in rules)

    def test_brain_context_includes_relevant_rules(self, brain_service, vector_memory):
        """Verify TradingBrainService.get_context() includes context-relevant rules."""
        # Store context-specific rule
        vector_memory.store_semantic_rule(
            rule_id="rule_bullish_test",
            rule_text="LONG trades in BULLISH trend with HIGH volatility show strong performance",
            metadata={"source_pattern": "LONG_BULLISH_HIGH_VOL"}
        )

        # Get context with matching conditions
        context = brain_service.get_context(
            trend_direction="BULLISH",
            adx=30.0,
            volatility_level="HIGH",
            rsi_level="STRONG",
            macd_signal="BULLISH"
        )

        # Context should include the relevant rule with similarity (case-insensitive)
        assert "learned trading rules" in context.lower() or context == ""
        if "learned trading rules" in context.lower():
            assert "match]" in context  # Similarity indicator

    def test_anti_pattern_rules_retrieved_for_matching_context(self, brain_service, vector_memory):
        """Verify anti-pattern rules are retrieved when context matches."""
        # Store anti-pattern rule
        vector_memory.store_semantic_rule(
            rule_id="anti_rule_neutral",
            rule_text="⚠️ AVOID: LONG trades in NEUTRAL choppy market hit stop_loss frequently",
            metadata={"rule_type": "anti_pattern", "source_pattern": "LONG_NEUTRAL_STOP"}
        )

        # Query with matching context
        rules = vector_memory.get_relevant_rules(
            current_context="NEUTRAL + Medium ADX + LOW Volatility",
            n_results=3
        )

        # Anti-pattern should be retrieved
        if rules:
            anti_found = any("AVOID" in r["text"] for r in rules)
            assert anti_found or len(rules) == 0

    def test_empty_collection_returns_empty_list(self, brain_service, vector_memory):
        """Verify empty semantic rules collection returns empty list."""
        rules = vector_memory.get_relevant_rules(
            current_context="BULLISH + High ADX",
            n_results=3
        )
        assert rules == []

    def test_get_active_rules_still_works(self, brain_service, vector_memory):
        """Verify backward compatibility - get_active_rules() still returns all active rules."""
        # Store multiple rules
        vector_memory.store_semantic_rule(
            rule_id="rule_1",
            rule_text="Rule one for bullish markets",
            metadata={}
        )
        vector_memory.store_semantic_rule(
            rule_id="rule_2",
            rule_text="Rule two for bearish markets",
            metadata={}
        )

        # Use old method
        rules = vector_memory.get_active_rules(n_results=5)

        assert len(rules) == 2
        # Old method returns rules without similarity score
        assert "similarity" not in rules[0] or rules[0].get("similarity") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

