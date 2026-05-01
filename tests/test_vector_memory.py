"""Tests for vector_memory.py changes: classify_rsi_label integration and _adx_label."""
from unittest.mock import MagicMock, patch
import pytest

from src.trading.vector_memory import VectorMemoryService
from src.trading.data_models import VectorSearchResult
from src.utils.indicator_classifier import classify_rsi_label


def _make_service():
    """Create a VectorMemoryService with mocked dependencies."""
    logger = MagicMock()
    chroma_client = MagicMock()
    embedding_model = MagicMock()
    return VectorMemoryService(logger=logger, chroma_client=chroma_client, embedding_model=embedding_model)


# ── _build_experience_document uses classify_rsi_label ───────────


class TestBuildExperienceDocument:
    """Verify _build_experience_document embeds RSI label via classify_rsi_label."""

    def _build_doc(self, rsi=None, adx=None, **kwargs):
        svc = _make_service()
        defaults = dict(
            direction="LONG",
            symbol="BTC/USDC",
            outcome="WIN",
            pnl_pct=5.0,
            confidence="HIGH",
            reasoning="Test trade",
            close_reason="take_profit",
            market_context="BULLISH + High ADX + MEDIUM Volatility",
            adx=adx,
            rsi=rsi,
            atr_pct=None,
            volatility="MEDIUM",
            macd_signal="BULLISH",
            bb_position="UPPER",
            rr_ratio=2.0,
            sl_pct=1.5,
            tp_pct=3.0,
            market_sentiment="NEUTRAL",
            order_book_bias="BALANCED",
            max_profit_pct=6.0,
            max_drawdown_pct=-1.0,
            factor_scores={},
        )
        defaults.update(kwargs)
        return svc._build_experience_document(**defaults)

    def test_rsi_overbought_label(self):
        doc = self._build_doc(rsi=75)
        assert "RSI=75.0 (OVERBOUGHT)" in doc

    def test_rsi_oversold_label(self):
        doc = self._build_doc(rsi=25)
        assert "RSI=25.0 (OVERSOLD)" in doc

    def test_rsi_neutral_label(self):
        doc = self._build_doc(rsi=50)
        assert "RSI=50.0 (NEUTRAL)" in doc

    def test_rsi_strong_label(self):
        doc = self._build_doc(rsi=65)
        assert "RSI=65.0 (STRONG)" in doc

    def test_rsi_weak_label(self):
        doc = self._build_doc(rsi=35)
        assert "RSI=35.0 (WEAK)" in doc

    def test_rsi_none_omitted(self):
        doc = self._build_doc(rsi=None)
        assert "RSI=" not in doc

    def test_rsi_label_matches_classify_fn(self):
        """The label in the document must match classify_rsi_label."""
        for rsi_val in [10, 30, 40, 50, 60, 70, 90]:
            doc = self._build_doc(rsi=rsi_val)
            expected_label = classify_rsi_label(rsi_val)
            assert f"RSI={rsi_val:.1f} ({expected_label})" in doc

    def test_adx_format(self):
        """ADX should use _adx_label (unified 3-tier matching classify_adx_label)."""
        doc = self._build_doc(adx=45)
        assert "ADX=45.0 (High ADX)" in doc

    def test_adx_medium(self):
        doc = self._build_doc(adx=22)
        assert "ADX=22.0 (Medium ADX)" in doc

    def test_adx_low(self):
        doc = self._build_doc(adx=15)
        assert "ADX=15.0 (Low ADX)" in doc

    def test_adx_none_omitted(self):
        doc = self._build_doc(adx=None)
        assert "ADX=" not in doc

    def test_exit_execution_context_in_structure(self):
        doc = self._build_doc(
            exit_execution_context={
                "stop_loss_type": "hard",
                "stop_loss_check_interval": "15m",
                "take_profit_type": "hard",
                "take_profit_check_interval": "15m",
            }
        )
        assert "Exit Execution: SL hard/15m | TP hard/15m" in doc


# ── _adx_label finer granularity ────────────────────────────────


class TestAdxLabel:
    """_adx_label uses same 3-tier vocabulary as classify_adx_label."""

    def test_very_high_adx(self):
        assert VectorMemoryService._adx_label(45) == "High ADX"

    def test_high_adx(self):
        assert VectorMemoryService._adx_label(30) == "High ADX"

    def test_medium_adx(self):
        assert VectorMemoryService._adx_label(22) == "Medium ADX"

    def test_low_adx(self):
        assert VectorMemoryService._adx_label(15) == "Low ADX"

    def test_boundary_40(self):
        assert VectorMemoryService._adx_label(40) == "High ADX"

    def test_boundary_25(self):
        assert VectorMemoryService._adx_label(25) == "High ADX"

    def test_boundary_20(self):
        assert VectorMemoryService._adx_label(20) == "Medium ADX"


class TestStoreExperience:
    """Verify storage behavior stays stable during refactors."""

    def test_store_experience_does_not_mutate_input_metadata(self):
        svc = _make_service()
        svc._embedding_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

        metadata = {
            "market_regime": "BULLISH",
            "adx_at_entry": 28.0,
            "custom_flag": True,
        }
        original_metadata = dict(metadata)

        stored = svc.store_experience(
            trade_id="trade-1",
            market_context="BULLISH + High ADX",
            outcome="WIN",
            pnl_pct=4.2,
            direction="LONG",
            confidence="HIGH",
            reasoning="Momentum continuation",
            metadata=metadata,
            symbol="BTC/USDC",
            close_reason="take_profit",
        )

        assert stored is True
        assert metadata == original_metadata

    def test_store_experience_persists_exit_execution_metadata_and_document(self):
        svc = _make_service()
        svc._embedding_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

        stored = svc.store_experience(
            trade_id="trade-risk-1",
            market_context="BULLISH + High ADX",
            outcome="WIN",
            pnl_pct=2.5,
            direction="LONG",
            confidence="HIGH",
            reasoning="Momentum continuation",
            metadata={
                "stop_loss_type": "hard",
                "stop_loss_check_interval": "15m",
                "take_profit_type": "hard",
                "take_profit_check_interval": "15m",
            },
            symbol="BTC/USDC",
            close_reason="take_profit",
        )

        assert stored is True
        upsert_kwargs = svc._collection.upsert.call_args.kwargs
        metadata = upsert_kwargs["metadatas"][0]
        assert metadata["stop_loss_type"] == "hard"
        assert metadata["stop_loss_check_interval"] == "15m"
        assert metadata["take_profit_type"] == "hard"
        assert metadata["take_profit_check_interval"] == "15m"
        assert "Exit Execution: SL hard/15m | TP hard/15m" in upsert_kwargs["documents"][0]


class TestComputeFactorPerformance:
    """Verify factor and categorical aggregation remains stable."""

    def test_compute_factor_performance_groups_scores_into_buckets(self):
        svc = _make_service()
        svc._initialized = True
        svc._collection = MagicMock()
        svc._collection.get.return_value = {
            "ids": ["1", "2", "3"],
            "metadatas": [
                {"outcome": "WIN", "pnl_pct": 3.0, "trend_alignment_score": 20},
                {"outcome": "LOSS", "pnl_pct": -1.0, "trend_alignment_score": 55},
                {"outcome": "WIN", "pnl_pct": 5.0, "trend_alignment_score": 82},
            ],
        }

        result = svc.compute_factor_performance()

        assert result["trend_alignment_LOW"]["total_trades"] == 1
        assert result["trend_alignment_LOW"]["win_rate"] == 100.0
        assert result["trend_alignment_MEDIUM"]["total_trades"] == 1
        assert result["trend_alignment_MEDIUM"]["win_rate"] == 0.0
        assert result["trend_alignment_HIGH"]["total_trades"] == 1
        assert result["trend_alignment_HIGH"]["avg_score"] == 82.0

    def test_compute_factor_performance_normalizes_categorical_values(self):
        svc = _make_service()
        svc._initialized = True
        svc._collection = MagicMock()
        svc._collection.get.return_value = {
            "ids": ["1", "2", "3"],
            "metadatas": [
                {"outcome": "WIN", "pnl_pct": 2.0, "market_sentiment": "greed"},
                {"outcome": "LOSS", "pnl_pct": -1.0, "market_sentiment": "EXTREME_GREED"},
                {"outcome": "WIN", "pnl_pct": 1.0, "volatility_level": "high"},
            ],
        }

        result = svc.compute_factor_performance()

        assert result["cat_Sentiment: GREED"]["total_trades"] == 2
        assert result["cat_Sentiment: GREED"]["winning_trades"] == 1
        assert result["cat_Volatility: HIGH VOLATILITY"]["total_trades"] == 1


class TestRetrievalAndPromptContext:
    """Verify retrieval and prompt context behavior after the split."""

    def test_retrieve_similar_experiences_uses_30_day_half_life_by_default(self):
        svc = _make_service()
        svc._initialized = True
        svc._collection = MagicMock()
        svc._collection.count.return_value = 1
        svc._collection.query.return_value = {
            "ids": [["recent-trade"]],
            "documents": [["doc-1"]],
            "metadatas": [[{"timestamp": "2026-03-20T00:00:00+00:00"}]],
            "distances": [[0.10]],
        }
        svc._embedding_model.encode.return_value.tolist.return_value = [0.1, 0.2]

        with patch.object(svc, "_calculate_recency_score", return_value=0.9) as recency_mock:
            svc.retrieve_similar_experiences("BULLISH", k=1)

        assert recency_mock.call_args.args[1] == 30

    def test_retrieve_similar_experiences_reorders_by_hybrid_score(self):
        svc = _make_service()
        svc._initialized = True
        svc._collection = MagicMock()
        svc._collection.count.return_value = 2
        svc._collection.query.return_value = {
            "ids": [["older-better-match", "newer-slightly-weaker"]],
            "documents": [["doc-1", "doc-2"]],
            "metadatas": [[{"timestamp": "2026-01-01T00:00:00+00:00"}, {"timestamp": "2026-03-20T00:00:00+00:00"}]],
            "distances": [[0.10, 0.20]],
        }
        svc._embedding_model.encode.return_value.tolist.return_value = [0.1, 0.2]

        with patch.object(svc, "_calculate_recency_score", side_effect=[0.1, 0.9]):
            results = svc.retrieve_similar_experiences("BULLISH", k=2)

        assert [result.id for result in results] == ["newer-slightly-weaker", "older-better-match"]
        assert results[0].hybrid_score > results[1].hybrid_score

    def test_get_context_for_prompt_includes_limited_data_and_anti_patterns(self):
        svc = _make_service()
        experiences = [
            VectorSearchResult(
                id="exp-1",
                document="doc-1",
                similarity=42.0,
                recency=60.0,
                hybrid_score=47.4,
                metadata={
                    "outcome": "LOSS",
                    "pnl_pct": -1.5,
                    "direction": "SHORT",
                    "market_context": "BEARISH + Low ADX",
                    "reasoning": "Fade failed breakdown",
                },
            ),
            VectorSearchResult(
                id="exp-2",
                document="doc-2",
                similarity=35.0,
                recency=55.0,
                hybrid_score=41.0,
                metadata={
                    "outcome": "WIN",
                    "pnl_pct": 2.3,
                    "direction": "LONG",
                    "market_context": "RANGING + Medium ADX",
                    "reasoning": "Quick mean reversion",
                },
            ),
        ]

        with patch.object(svc, "retrieve_similar_experiences", return_value=experiences), patch.object(
            svc,
            "get_anti_patterns_for_prompt",
            return_value="⚠️ AVOID PATTERNS (learned from losses):\n  - Avoid weak breakouts into resistance",
        ):
            prompt = svc.get_context_for_prompt("BEARISH", k=2, display_context="BEARISH + Low ADX")

        assert "LIMITED DATA" in prompt
        assert "[SIMILARITY 42%] SHORT trade" in prompt
        assert "Avoid weak breakouts into resistance" in prompt


class TestSemanticRules:
    """Verify semantic rule persistence and filtering behavior."""

    def test_store_semantic_rule_persists_active_metadata(self):
        svc = _make_service()
        svc._initialized = True
        svc._semantic_rules_collection = MagicMock()
        svc._embedding_model.encode.return_value.tolist.return_value = [0.4, 0.5]

        stored = svc.store_semantic_rule(
            rule_id="rule-1",
            rule_text="Prefer long setups with aligned momentum",
            metadata={"rule_type": "best_practice", "win_rate": 66.7},
        )

        assert stored is True
        upsert_kwargs = svc._semantic_rules_collection.upsert.call_args.kwargs
        metadata = upsert_kwargs["metadatas"][0]
        assert metadata["active"] is True
        assert metadata["rule_type"] == "best_practice"
        assert metadata["win_rate"] == 66.7
        assert "timestamp" in metadata

    def test_get_relevant_rules_filters_below_similarity_threshold(self):
        svc = _make_service()
        svc._initialized = True
        svc._semantic_rules_collection = MagicMock()
        svc._semantic_rules_collection.count.return_value = 2
        svc._semantic_rules_collection.query.return_value = {
            "ids": [["rule-strong", "rule-weak"]],
            "documents": [["Strong rule", "Weak rule"]],
            "metadatas": [[{"rule_type": "best_practice"}, {"rule_type": "best_practice"}]],
            "distances": [[0.2, 0.75]],
        }
        svc._embedding_model.encode.return_value.tolist.return_value = [0.7, 0.8]

        rules = svc.get_relevant_rules("BULLISH", n_results=3, min_similarity=0.4)

        assert [rule["rule_id"] for rule in rules] == ["rule-strong"]
        assert rules[0]["similarity"] == 80.0

    def test_get_anti_patterns_for_prompt_only_includes_anti_pattern_rules(self):
        svc = _make_service()
        with patch.object(
            svc,
            "get_active_rules",
            return_value=[
                {"text": "Avoid longs into major resistance", "metadata": {"rule_type": "anti_pattern"}},
                {"text": "Favor aligned trends", "metadata": {"rule_type": "best_practice"}},
            ],
        ):
            prompt = svc.get_anti_patterns_for_prompt(k=2)

        assert "Avoid longs into major resistance" in prompt
        assert "Favor aligned trends" not in prompt

    def test_get_anti_patterns_for_prompt_includes_ai_mistake_rules(self):
        svc = _make_service()
        with patch.object(
            svc,
            "get_active_rules",
            return_value=[
                {
                    "text": "AI MISTAKE: HIGH confidence longs failed in sideways markets",
                    "metadata": {
                        "rule_type": "ai_mistake",
                        "failure_reason": "AI expected breakout continuation",
                        "recommended_adjustment": "downgrade confidence until ADX confirms expansion",
                    },
                },
                {"text": "Favor aligned trends", "metadata": {"rule_type": "best_practice"}},
            ],
        ):
            prompt = svc.get_anti_patterns_for_prompt(k=2)

        assert "AI MISTAKE: HIGH confidence longs failed" in prompt
        assert "AI expected breakout continuation" in prompt
        assert "downgrade confidence" in prompt
        assert "Favor aligned trends" not in prompt


class TestAnalyticsAndThresholds:
    """Verify analytics behavior after extracting the mixins."""

    def test_get_direction_bias_excludes_updates(self):
        svc = _make_service()
        svc._initialized = True
        svc._collection = MagicMock()
        svc._collection.get.return_value = {
            "ids": ["1", "2", "3"],
            "metadatas": [
                {"outcome": "WIN", "direction": "LONG"},
                {"outcome": "LOSS", "direction": "SHORT"},
                {"outcome": "UPDATE", "direction": "LONG"},
            ],
        }

        bias = svc.get_direction_bias()

        assert bias == {
            "long_count": 1,
            "short_count": 1,
            "long_pct": 50.0,
            "short_pct": 50.0,
        }

    def test_compute_confidence_stats_maps_unknown_confidence_to_medium(self):
        svc = _make_service()
        svc._initialized = True
        svc._collection = MagicMock()
        svc._collection.get.return_value = {
            "ids": ["1", "2", "3"],
            "metadatas": [
                {"outcome": "WIN", "pnl_pct": 3.0, "confidence": "HIGH"},
                {"outcome": "LOSS", "pnl_pct": -1.0, "confidence": "CUSTOM"},
                {"outcome": "WIN", "pnl_pct": 1.5, "confidence": "MEDIUM"},
            ],
        }

        stats = svc.compute_confidence_stats()

        assert stats["HIGH"]["total_trades"] == 1
        assert stats["MEDIUM"]["total_trades"] == 2
        assert stats["MEDIUM"]["winning_trades"] == 1
        assert stats["MEDIUM"]["avg_pnl_pct"] == 0.25

    def test_compute_optimal_thresholds_learns_expected_values(self):
        svc = _make_service()
        svc._initialized = True
        svc._collection = MagicMock()
        adx_and_confidence_snapshot = {
            "ids": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "metadatas": [
                {"outcome": "WIN", "pnl_pct": 3.0, "confidence": "HIGH", "adx_at_entry": 28},
                {"outcome": "WIN", "pnl_pct": 2.5, "confidence": "HIGH", "adx_at_entry": 27},
                {"outcome": "WIN", "pnl_pct": 1.8, "confidence": "HIGH", "adx_at_entry": 26},
                {"outcome": "LOSS", "pnl_pct": -0.5, "confidence": "HIGH", "adx_at_entry": 29},
                {"outcome": "WIN", "pnl_pct": 1.0, "confidence": "HIGH", "adx_at_entry": 24},
                {"outcome": "LOSS", "pnl_pct": -1.0, "confidence": "HIGH", "adx_at_entry": 24},
                {"outcome": "LOSS", "pnl_pct": -1.2, "confidence": "MEDIUM", "adx_at_entry": 18},
                {"outcome": "LOSS", "pnl_pct": -0.8, "confidence": "MEDIUM", "adx_at_entry": 17},
                {"outcome": "WIN", "pnl_pct": 0.6, "confidence": "MEDIUM", "adx_at_entry": 19},
            ],
        }
        threshold_learning_snapshot = {
            "ids": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
            "metadatas": [
                {"outcome": "WIN", "confidence": "HIGH", "adx_at_entry": 28, "rr_ratio": 2.2, "sl_distance_pct": 0.02},
                {"outcome": "WIN", "confidence": "HIGH", "adx_at_entry": 27, "rr_ratio": 2.0, "sl_distance_pct": 0.025},
                {"outcome": "WIN", "confidence": "HIGH", "adx_at_entry": 26, "rr_ratio": 1.8, "sl_distance_pct": 0.03},
                {"outcome": "LOSS", "confidence": "HIGH", "adx_at_entry": 29, "rr_ratio": 1.2, "sl_distance_pct": 0.015},
                {"outcome": "WIN", "confidence": "HIGH", "adx_at_entry": 30, "rr_ratio": 1.9, "sl_distance_pct": 0.02},
                {"outcome": "LOSS", "confidence": "HIGH", "adx_at_entry": 18, "rr_ratio": 1.1, "sl_distance_pct": 0.02},
                {"outcome": "LOSS", "confidence": "HIGH", "adx_at_entry": 17, "rr_ratio": 1.0, "sl_distance_pct": 0.02},
                {"outcome": "LOSS", "confidence": "HIGH", "adx_at_entry": 16, "rr_ratio": 1.2, "sl_distance_pct": 0.02},
                {"outcome": "WIN", "confidence": "HIGH", "adx_at_entry": 15, "rr_ratio": 1.5, "sl_distance_pct": 0.02},
                {"outcome": "LOSS", "confidence": "HIGH", "adx_at_entry": 19, "rr_ratio": 1.1, "sl_distance_pct": 0.02},
                {"outcome": "LOSS", "confidence": "HIGH", "adx_at_entry": 18, "rr_ratio": 1.3, "sl_distance_pct": 0.02},
            ],
        }
        svc._collection.get.side_effect = [
            adx_and_confidence_snapshot,
            threshold_learning_snapshot,
            adx_and_confidence_snapshot,
        ]

        thresholds = svc.compute_optimal_thresholds(min_sample_size=2)

        assert thresholds["adx_strong_threshold"] == 25
        assert thresholds["adx_weak_threshold"] == 22
        assert thresholds["min_rr_recommended"] == 1.5
        assert thresholds["rr_strong_setup"] == 2.0
        assert thresholds["rr_borderline_min"] == 1.3

    def test_compute_optimal_thresholds_ignores_update_entries_for_rr_boundary(self):
        svc = _make_service()
        svc._initialized = True
        svc._collection = MagicMock()
        svc.compute_adx_performance = MagicMock(return_value={})
        svc.compute_confidence_stats = MagicMock(return_value={})

        # Keep threshold learners focused on RR logic for this test.
        svc._learn_position_size_threshold = MagicMock()
        svc._learn_confluence_thresholds = MagicMock()
        svc._learn_alignment_thresholds = MagicMock()

        svc._collection.get.return_value = {
            "ids": ["1", "2", "3", "4"],
            "metadatas": [
                {"outcome": "WIN", "rr_ratio": 2.0, "sl_distance_pct": 0.02},
                {"outcome": "LOSS", "rr_ratio": 1.0, "sl_distance_pct": 0.02},
                {"outcome": "UPDATE", "rr_ratio": 0.2, "sl_distance_pct": 0.02},
                {"outcome": "UPDATE", "rr_ratio": 0.3, "sl_distance_pct": 0.02},
            ],
        }

        thresholds = svc.compute_optimal_thresholds(min_sample_size=1)

        assert thresholds["min_rr_recommended"] == 1.6
        assert "rr_borderline_min" not in thresholds
