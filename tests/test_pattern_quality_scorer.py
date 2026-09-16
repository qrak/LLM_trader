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

from typing import Any

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


@pytest.fixture
def scorer():
    return PatternQualityScorer()


def _make_tech(adx=35.0, rsi=55.0):
    return {"adx": adx, "rsi": rsi}


def _make_patterns(*names_and_indices):
    """Build a patterns dict from (name, bar_index) tuples."""
    patterns: dict[str, list[dict]] = {}
    for name, idx in names_and_indices:
        patterns.setdefault("chart", []).append({"name": name, "bar_index": idx})
    return patterns


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


class TestPatternExtraction:
    def test_dict_patterns(self, scorer):
        patterns = {"chart": [{"name": "hammer"}, {"name": "doji"}]}
        names = scorer._extract_pattern_names(patterns)
        assert len(names) == 2
        assert "hammer" in names

    def test_string_list_patterns(self, scorer):
        """All pattern entries are dicts in production — strings never occur."""
        patterns = {"indicator": [{"type": "rsi_oversold"}, {"type": "macd_bullish_cross"}]}
        names = scorer._extract_pattern_names(patterns)
        assert len(names) == 2

    def test_mixed_shapes(self, scorer):
        """Patterns use different key names — 'name', 'pattern', 'type' — but all are dicts."""
        patterns = {
            "chart": [{"name": "hammer"}, {"name": "doji"}],
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


class TestConfirmationScoring:
    def test_all_bullish(self, scorer):
        assert scorer._score_confirmation(5, 0) == 100.0

    def test_mostly_bullish(self, scorer):
        assert scorer._score_confirmation(4, 1) == 80.0

    def test_slight_majority(self, scorer):
        assert scorer._score_confirmation(3, 2) == 50.0

    def test_even_split(self, scorer):
        assert scorer._score_confirmation(2, 2) == 30.0

    def test_no_directional(self, scorer):
        assert scorer._score_confirmation(0, 0) == 0.0

    def test_all_bearish(self, scorer):
        assert scorer._score_confirmation(0, 4) == 100.0


class TestRecencyScoring:
    def test_no_recency_data(self, scorer):
        patterns: dict[str, Any] = {"chart": [{"name": "hammer"}]}
        assert scorer._score_recency(patterns) == 50.0

    def test_all_recent(self, scorer):
        patterns = _make_patterns(
            ("hammer", 98), ("doji", 99), ("bullish_engulfing", 97),
        )
        score = scorer._score_recency(patterns)
        assert score > 70

    def test_old_patterns(self, scorer):
        patterns = _make_patterns(
            ("hammer", 5), ("doji", 12),
        )
        patterns["reference"] = [{"name": "spinning_top", "bar_index": 100}]
        score = scorer._score_recency(patterns)
        assert score < 30

    def test_mixed_recency(self, scorer):
        patterns = _make_patterns(
            ("hammer", 10), ("bullish_engulfing", 95),
        )
        score = scorer._score_recency(patterns)
        assert 30 < score < 80

    def test_zero_max_index(self, scorer):
        patterns = {"chart": [{"name": "doji", "bar_index": 0}]}
        assert scorer._score_recency(patterns) == 50.0


class TestIndicatorAlignment:
    def test_bullish_aligned(self, scorer):
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
        score = scorer._score_indicator_alignment({"adx": 15, "rsi": 50}, "bullish")
        assert score == 50

    def test_missing_data(self, scorer):
        score = scorer._score_indicator_alignment({}, "bullish")
        assert score == 50


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
        assert quality.overall >= 60
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
        assert quality.overall < 60
        assert quality.confirmation_score < 60

    def test_score_empty_patterns(self, scorer):
        """No patterns but tech data available: gets baseline from recency (neutral) + indicator."""
        quality = scorer.score(patterns={}, tech_data=_make_tech())
        assert 10 <= quality.overall <= 20
        assert quality.label == "negligible"

    def test_score_no_tech_data(self, scorer):
        """Patterns but no tech data: indicator defaults still contribute."""
        patterns = _make_patterns(("hammer", 95))
        quality = scorer.score(patterns=patterns, tech_data={})
        assert quality.quantity_score == 35.0
        assert quality.overall >= 30

    def test_score_both_none(self, scorer):
        """No patterns and no tech_data: baseline from recency neutral + indicator neutral."""
        quality = scorer.score(patterns=None, tech_data=None)
        assert 10 <= quality.overall <= 20
        assert quality.passed

    def test_score_with_llm_match(self, scorer):
        patterns = _make_patterns(("hammer", 95), ("doji", 94))
        tech = _make_tech(adx=35, rsi=55)

        quality = scorer.score(patterns=patterns, tech_data=tech, llm_quality=65)
        assert quality.passed

    def test_score_with_llm_discrepancy(self, scorer):
        patterns = _make_patterns(("hammer", 95), ("doji", 94))
        tech = _make_tech(adx=35, rsi=55)

        quality = scorer.score(patterns=patterns, tech_data=tech, llm_quality=95.0)
        if 95.0 - quality.overall > QUALITY_DISCREPANCY_THRESHOLD:
            assert not quality.passed
            assert len(quality.discrepancies) == 1
        else:
            assert quality.passed

    def test_score_invalid_llm_quality(self, scorer):
        """Non-numeric or out-of-range LLM quality is ignored."""
        quality = scorer.score(
            patterns={}, tech_data={}, llm_quality="not_a_number"
        )
        assert quality.passed

    def test_score_llm_quality_out_of_range(self, scorer):
        quality = scorer.score(
            patterns={}, tech_data={}, llm_quality=150.0
        )
        assert quality.passed


class TestOverwriteLlmQuality:
    def test_overwrites_quality(self, scorer):
        analysis = {}
        quality = scorer.score(patterns={}, tech_data={})
        result = scorer.overwrite_llm_quality(analysis, quality)
        assert 10 <= result["pattern_quality"] <= 20

    def test_adds_validation_metadata(self, scorer):
        analysis = {}
        quality = scorer.score(patterns={}, tech_data={})
        result = scorer.overwrite_llm_quality(analysis, quality)
        assert "_pattern_validation" in result


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


class TestWeights:
    def test_weights_sum_to_one(self):
        total = (
            WEIGHT_PATTERN_QUANTITY
            + WEIGHT_PATTERN_CONFIRMATION
            + WEIGHT_PATTERN_RECENCY
            + WEIGHT_INDICATOR_ALIGNMENT
        )
        assert abs(total - 1.0) < 0.001
