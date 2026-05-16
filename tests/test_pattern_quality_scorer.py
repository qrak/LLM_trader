"""Tests for PatternQualityScorer: deterministic pattern quality scoring.

Covers:
  1. Score components: quantity, confirmation, recency, indicator alignment
  2. Pattern name classification (bullish/bearish/neutral)
  3. Empty/missing patterns → score 0
  4. Pattern extraction from various data shapes
  5. LLM quality comparison and discrepancy detection
  6. overwrite_llm_quality mutates analysis
  7. Component weight verification
  8. Edge cases: no tech data, no recency data
"""

import math
from unittest.mock import MagicMock, patch

import pytest

from src.analyzer.pattern_quality_scorer import (
    QUALITY_DISCREPANCY_THRESHOLD,
    WEIGHT_INDICATOR_ALIGNMENT,
    WEIGHT_PATTERN_CONFIRMATION,
    WEIGHT_PATTERN_QUANTITY,
    WEIGHT_PATTERN_RECENCY,
    PatternQualityScorer,
    QualityScore,
)


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def scorer():
    return PatternQualityScorer()


# ── Helpers ──────────────────────────────────────────────────────


def _make_tech(adx=35.0, rsi=55.0):
    return {"adx": adx, "rsi": rsi}


def _make_patterns(*names_and_indices):
    """Build a patterns dict from (name, bar_index) tuples."""
    patterns: dict[str, list[dict]] = {}
    for name, idx in names_and_indices:
        patterns.setdefault("chart", []).append({"name": name, "bar_index": idx})
    return patterns


# ── Unit: Pattern classification ────────────────────────────────


class TestPatternClassification:
    def test_bullish_engulfing(self, scorer):
        assert scorer._classify_direction("bullish_engulfing") == "bullish"

    def test_bearish_engulfing(self, scorer):
        assert scorer._classify_direction("bearish_engulfing") == "bearish"

    def test_hammer(self, scorer):
        assert scorer._classify_direction("hammer") == "bullish"

    def test_shooting_star(self, scorer):
        assert scorer._classify_direction("shooting_star") == "bearish"

    def test_doji(self, scorer):
        assert scorer._classify_direction("doji") == "neutral"

    def test_golden_cross(self, scorer):
        assert scorer._classify_direction("golden_cross") == "bullish"

    def test_death_cross(self, scorer):
        assert scorer._classify_direction("death_cross") == "bearish"

    def test_double_bottom(self, scorer):
        assert scorer._classify_direction("double_bottom") == "bullish"

    def test_head_and_shoulders(self, scorer):
        assert scorer._classify_direction("head_and_shoulders") == "bearish"

    def test_unknown_pattern(self, scorer):
        assert scorer._classify_direction("unknown_mystery_pattern") == "neutral"

    def test_case_insensitive(self, scorer):
        assert scorer._classify_direction("BULLISH_ENGULFING") == "bullish"

    def test_with_spaces(self, scorer):
        assert scorer._classify_direction("bullish engulfing") == "bullish"


# ── Unit: Pattern name extraction ───────────────────────────────


class TestPatternExtraction:
    def test_dict_patterns(self, scorer):
        patterns = {"chart": [{"name": "hammer"}, {"name": "doji"}]}
        names = scorer._extract_pattern_names(patterns)
        assert len(names) == 2
        assert "hammer" in names

    def test_string_list_patterns(self, scorer):
        patterns = {"indicator": ["rsi_oversold", "macd_bullish_cross"]}
        names = scorer._extract_pattern_names(patterns)
        assert len(names) == 2

    def test_mixed_shapes(self, scorer):
        patterns = {
            "chart": [{"name": "hammer"}, "doji"],
            "indicator": [{"pattern": "golden_cross"}],
        }
        names = scorer._extract_pattern_names(patterns)
        assert len(names) >= 3

    def test_empty_patterns(self, scorer):
        assert scorer._extract_pattern_names({}) == []

    def test_none_patterns(self, scorer):
        assert scorer._extract_pattern_names(None) == []

    def test_pattern_with_type_field(self, scorer):
        patterns = {"chart": [{"type": "bullish_engulfing"}]}
        names = scorer._extract_pattern_names(patterns)
        assert "bullish_engulfing" in names


# ── Quantity scoring ────────────────────────────────────────────


class TestQuantityScoring:
    def test_zero_patterns(self, scorer):
        assert scorer._score_quantity(0) == 0.0

    def test_one_pattern(self, scorer):
        assert scorer._score_quantity(1) == 35.0

    def test_two_patterns(self, scorer):
        assert scorer._score_quantity(2) == 55.0

    def test_three_patterns(self, scorer):
        assert scorer._score_quantity(3) == 70.0

    def test_four_patterns(self, scorer):
        assert scorer._score_quantity(4) == 85.0

    def test_five_patterns(self, scorer):
        assert scorer._score_quantity(5) == 100.0

    def test_many_patterns(self, scorer):
        assert scorer._score_quantity(20) == 100.0


# ── Confirmation scoring ────────────────────────────────────────


class TestConfirmationScoring:
    def test_all_bullish(self, scorer):
        # 5 bullish, 0 bearish → 100% alignment
        assert scorer._score_confirmation(5, 0) == 100.0

    def test_mostly_bullish(self, scorer):
        # 4 bullish, 1 bearish → 80% → ratio 0.8
        assert scorer._score_confirmation(4, 1) == 80.0

    def test_slight_majority(self, scorer):
        # 3 bullish, 2 bearish → 60% → ratio 0.6
        assert scorer._score_confirmation(3, 2) == 50.0

    def test_even_split(self, scorer):
        assert scorer._score_confirmation(2, 2) == 30.0

    def test_no_directional(self, scorer):
        assert scorer._score_confirmation(0, 0) == 0.0

    def test_all_bearish(self, scorer):
        assert scorer._score_confirmation(0, 4) == 100.0


# ── Recency scoring ─────────────────────────────────────────────


class TestRecencyScoring:
    def test_no_recency_data(self, scorer):
        patterns: dict[str, Any] = {"chart": [{"name": "hammer"}]}
        assert scorer._score_recency(patterns) == 50.0  # neutral

    def test_all_recent(self, scorer):
        patterns = _make_patterns(
            ("hammer", 98), ("doji", 99), ("bullish_engulfing", 97),
        )
        score = scorer._score_recency(patterns)
        assert score > 70  # Recent patterns score high

    def test_old_patterns(self, scorer):
        # Patterns at the beginning of data (old relative to dataset of 100 bars)
        patterns = _make_patterns(
            ("hammer", 5), ("doji", 12),
        )
        # Override max_idx: add a "recent" baseline bar at 100
        patterns["reference"] = [{"name": "spinning_top", "bar_index": 100}]
        score = scorer._score_recency(patterns)
        # avg=(5+12+100)/3=39, max=100, ratio=0.39, score=(0.39-0.3)/0.7*100=12.9
        assert score < 30  # Old patterns score low

    def test_mixed_recency(self, scorer):
        patterns = _make_patterns(
            ("hammer", 10), ("bullish_engulfing", 95),
        )
        score = scorer._score_recency(patterns)
        assert 30 < score < 80  # Mixed recency

    def test_zero_max_index(self, scorer):
        patterns = {"chart": [{"name": "doji", "bar_index": 0}]}
        assert scorer._score_recency(patterns) == 50.0  # fallback


# ── Indicator alignment scoring ─────────────────────────────────


class TestIndicatorAlignment:
    def test_bullish_aligned(self, scorer):
        # ADX 40 (trending) + RSI 55 (bullish zone) = max score
        score = scorer._score_indicator_alignment({"adx": 40, "rsi": 55}, "bullish")
        assert score == 100.0

    def test_bullish_oversold_reversal(self, scorer):
        score = scorer._score_indicator_alignment({"adx": 20, "rsi": 25}, "bullish")
        assert score > 30

    def test_bearish_aligned(self, scorer):
        score = scorer._score_indicator_alignment({"adx": 40, "rsi": 45}, "bearish")
        assert score == 100.0

    def test_bearish_overbought_reversal(self, scorer):
        score = scorer._score_indicator_alignment({"adx": 20, "rsi": 75}, "bearish")
        assert score > 30

    def test_neutral_direction(self, scorer):
        score = scorer._score_indicator_alignment({"adx": 40, "rsi": 55}, "neutral")
        assert score == 25.0

    def test_low_adx_no_trend(self, scorer):
        # ADX 15 (<20, no trend) but RSI=50 in bullish zone (40-70)
        score = scorer._score_indicator_alignment({"adx": 15, "rsi": 50}, "bullish")
        # ADX contributes 0 (below 20), RSI contributes 50 (in 40-70)
        assert score == 50

    def test_missing_data(self, scorer):
        # Empty tech_data: ADX defaults to 0, RSI defaults to 50
        # RSI=50 is in 40-70 range for bullish → 50 points, ADX 0 → 0
        score = scorer._score_indicator_alignment({}, "bullish")
        assert score == 50


# ── Main score() method ─────────────────────────────────────────


class TestScoreMethod:
    def test_score_with_strong_bullish_patterns(self, scorer):
        patterns = _make_patterns(
            ("bullish_engulfing", 98),
            ("hammer", 96),
            ("golden_cross", 94),
            ("rsi_bullish_divergence", 95),
        )
        tech = _make_tech(adx=40, rsi=55)

        quality = scorer.score(patterns=patterns, tech_data=tech)
        assert quality.overall >= 60  # Should be strong
        assert quality.label in ("strong", "moderate")
        assert quality.passed

    def test_score_with_mixed_patterns(self, scorer):
        patterns = _make_patterns(
            ("bullish_engulfing", 95),
            ("bearish_engulfing", 94),
            ("doji", 93),
        )
        tech = _make_tech(adx=30, rsi=50)

        quality = scorer.score(patterns=patterns, tech_data=tech)
        assert quality.overall < 60  # Mixed signals
        assert quality.confirmation_score < 60

    def test_score_empty_patterns(self, scorer):
        """No patterns but tech data available: gets baseline from recency (neutral) + indicator."""
        quality = scorer.score(patterns={}, tech_data=_make_tech())
        # quantity=0, confirmation=0, recency=50 (neutral), indicator with neutral dir=25
        # overall = 0*0.3 + 0*0.3 + 50*0.2 + 25*0.2 = 15
        assert 10 <= quality.overall <= 20
        assert quality.label == "negligible"

    def test_score_no_tech_data(self, scorer):
        """Patterns but no tech data: indicator defaults still contribute."""
        patterns = _make_patterns(("hammer", 95))
        quality = scorer.score(patterns=patterns, tech_data={})
        # Single pattern, all same direction, indicator with defaults
        assert quality.quantity_score == 35.0
        assert quality.overall >= 30  # pattern quantity + confirmation contribute

    def test_score_both_none(self, scorer):
        """No patterns and no tech_data: baseline from recency neutral + indicator neutral."""
        quality = scorer.score(patterns=None, tech_data=None)
        # Same as empty: ~15 from recency neutral + indicator neutral
        assert 10 <= quality.overall <= 20
        assert quality.passed

    def test_score_with_llm_match(self, scorer):
        patterns = _make_patterns(("hammer", 95), ("doji", 94))
        tech = _make_tech(adx=35, rsi=55)

        quality = scorer.score(patterns=patterns, tech_data=tech, llm_quality=65)
        # Should be close enough
        assert quality.passed

    def test_score_with_llm_discrepancy(self, scorer):
        patterns = _make_patterns(("hammer", 95), ("doji", 94))
        tech = _make_tech(adx=35, rsi=55)

        # LLM claims 95 but computed is much lower — discrepancy
        quality = scorer.score(patterns=patterns, tech_data=tech, llm_quality=95.0)
        if 95.0 - quality.overall > QUALITY_DISCREPANCY_THRESHOLD:
            assert not quality.passed
            assert len(quality.discrepancies) == 1
        else:
            assert quality.passed  # OK if difference is small

    def test_score_invalid_llm_quality(self, scorer):
        """Non-numeric or out-of-range LLM quality is ignored."""
        quality = scorer.score(
            patterns={}, tech_data={}, llm_quality="not_a_number"
        )
        assert quality.passed  # No comparison made

    def test_score_llm_quality_out_of_range(self, scorer):
        quality = scorer.score(
            patterns={}, tech_data={}, llm_quality=150.0
        )
        assert quality.passed  # Only validated 0-100 range


# ── overwrite_llm_quality ───────────────────────────────────────


class TestOverwriteLlmQuality:
    def test_overwrites_quality(self, scorer):
        analysis = {}
        quality = scorer.score(patterns={}, tech_data={})
        result = scorer.overwrite_llm_quality(analysis, quality)
        # Empty patterns + empty tech → baseline ~15
        assert 10 <= result["pattern_quality"] <= 20

    def test_adds_validation_metadata(self, scorer):
        analysis = {}
        quality = scorer.score(patterns={}, tech_data={})
        result = scorer.overwrite_llm_quality(analysis, quality)
        assert "_pattern_validation" in result


# ── QualityScore dataclass ──────────────────────────────────────


class TestQualityScoreDataclass:
    def test_defaults(self):
        qs = QualityScore()
        assert qs.overall == 0.0
        assert qs.passed
        assert qs.label == "negligible"

    def test_to_dict(self, scorer):
        quality = scorer.score(patterns={}, tech_data={})
        d = quality.to_dict()
        assert "overall" in d
        assert "label" in d
        assert "discrepancies" in d


# ── Weight validation ───────────────────────────────────────────


class TestWeights:
    def test_weights_sum_to_one(self):
        total = (
            WEIGHT_PATTERN_QUANTITY
            + WEIGHT_PATTERN_CONFIRMATION
            + WEIGHT_PATTERN_RECENCY
            + WEIGHT_INDICATOR_ALIGNMENT
        )
        assert abs(total - 1.0) < 0.001
