"""Pattern Quality Scorer: deterministic quality assessment from pattern detection output.

Computes a 0-100 score based on actual detected patterns and indicator alignment,
independent of LLM-generated scores. This provides a ground-truth baseline that
can be compared against the LLM's self-reported pattern_quality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Weight distribution for quality components (must sum to 1.0)
WEIGHT_PATTERN_QUANTITY = 0.30     # How many patterns were detected
WEIGHT_PATTERN_CONFIRMATION = 0.30  # Multiple patterns agreeing on direction
WEIGHT_PATTERN_RECENCY = 0.20       # How recently patterns formed
WEIGHT_INDICATOR_ALIGNMENT = 0.20   # ADX/RSI alignment with pattern direction

# Maximum allowed delta between LLM-reported and computed quality before flagging
QUALITY_DISCREPANCY_THRESHOLD: float = 25.0


@dataclass(slots=True)
class QualityScore:
    """Breakdown of a deterministic pattern quality score."""

    overall: float = 0.0
    quantity_score: float = 0.0
    confirmation_score: float = 0.0
    recency_score: float = 0.0
    indicator_score: float = 0.0
    discrepancies: list[str] = field(default_factory=list)
    passed: bool = True

    @property
    def label(self) -> str:
        if self.overall >= 75:
            return "strong"
        if self.overall >= 50:
            return "moderate"
        if self.overall >= 25:
            return "weak"
        return "negligible"

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": round(self.overall),
            "quantity_score": round(self.quantity_score, 1),
            "confirmation_score": round(self.confirmation_score, 1),
            "recency_score": round(self.recency_score, 1),
            "indicator_score": round(self.indicator_score, 1),
            "label": self.label,
            "discrepancies": self.discrepancies,
            "passed": self.passed,
        }


class PatternQualityScorer:
    """Computes a deterministic pattern quality score from actual detection output.

    The score reflects how actionable the detected pattern set is, independent
    of any LLM interpretation.

    Usage:
        scorer = PatternQualityScorer()
        quality = scorer.score(
            patterns=context.technical_patterns,
            tech_data=context.technical_data,
        )
        # Compare against LLM's self-reported score
        if llm_quality and abs(llm_quality - quality.overall) > 25:
            logger.warning("Pattern quality discrepancy: LLM=%s, computed=%s",
                           llm_quality, quality.overall)
    """

    # ── Pattern categorization ──────────────────────────────────

    BULLISH_PATTERNS = frozenset({
        "bullish_engulfing", "morning_star", "piercing_line", "hammer",
        "inverted_hammer", "three_white_soldiers", "bullish_harami",
        "bullish_kicker", "dragonfly_doji", "tweezer_bottom",
        "ascending_triangle", "bull_flag", "cup_and_handle",
        "inverse_head_and_shoulders", "double_bottom", "falling_wedge",
        "bullish_divergence", "hidden_bullish_divergence",
        "rsi_bullish_divergence", "macd_bullish_cross",
        "golden_cross", "bullish_macd_divergence",
        "rsi_oversold", "stoch_oversold",
    })

    BEARISH_PATTERNS = frozenset({
        "bearish_engulfing", "evening_star", "dark_cloud_cover",
        "shooting_star", "hanging_man", "three_black_crows",
        "bearish_harami", "bearish_kicker", "gravestone_doji",
        "tweezer_top", "descending_triangle", "bear_flag",
        "head_and_shoulders", "double_top", "rising_wedge",
        "bearish_divergence", "hidden_bearish_divergence",
        "rsi_bearish_divergence", "macd_bearish_cross",
        "death_cross", "bearish_macd_divergence",
        "rsi_overbought", "stoch_overbought",
    })

    # Patterns that are informative but not directional
    NEUTRAL_PATTERNS = frozenset({
        "doji", "spinning_top", "high_wave", "marubozu",
        "symmetrical_triangle", "rectangle", "pennant",
    })

    @classmethod
    def _classify_direction(cls, pattern_name: str) -> str:
        """Classify a pattern name as bullish, bearish, or neutral."""
        name = pattern_name.lower().replace(" ", "_")
        name = name.removesuffix("_pattern")
        if name in cls.BULLISH_PATTERNS:
            return "bullish"
        if name in cls.BEARISH_PATTERNS:
            return "bearish"
        return "neutral"

    @classmethod
    def _extract_pattern_names(cls, patterns: dict[str, Any] | None) -> list[str]:
        """Flatten pattern dict into list of pattern names."""
        if patterns is None:
            return []
        names: list[str] = []
        for category, pattern_list in patterns.items():
            if isinstance(pattern_list, list):
                for p in pattern_list:
                    if isinstance(p, dict):
                        name = p.get("name") or p.get("pattern") or p.get("type", "")
                        if name:
                            names.append(str(name))
                    elif isinstance(p, str):
                        names.append(p)
            elif isinstance(pattern_list, str):
                names.append(pattern_list)
        return names

    # ── Scoring components ──────────────────────────────────────

    @staticmethod
    def _score_quantity(pattern_count: int) -> float:
        """Score based on number of detected patterns.

        0 patterns = 0, 1 = 35, 2 = 55, 3 = 70, 4 = 85, 5+ = 100.
        Capped because too many patterns suggests noise.
        """
        if pattern_count <= 0:
            return 0.0
        if pattern_count == 1:
            return 35.0
        if pattern_count == 2:
            return 55.0
        if pattern_count == 3:
            return 70.0
        if pattern_count == 4:
            return 85.0
        return 100.0  # 5+ patterns

    @staticmethod
    def _score_confirmation(bullish_count: int, bearish_count: int) -> float:
        """Score based on how many same-direction patterns confirm each other.

        High confirmation: majority in one direction. Mixed: penalty.
        No directional patterns: 0.
        """
        total_dir = bullish_count + bearish_count
        if total_dir == 0:
            return 0.0

        majority = max(bullish_count, bearish_count)
        ratio = majority / total_dir

        # Perfect alignment = 100, evenly split = 0
        if ratio >= 0.9:
            return 100.0
        if ratio >= 0.75:
            return 80.0
        if ratio >= 0.6:
            return 50.0
        if ratio >= 0.5:
            return 30.0
        return 0.0  # minority direction (shouldn't happen since majority >= half)

    @staticmethod
    def _score_recency(patterns: dict[str, Any]) -> float:
        """Score based on how recently patterns formed (via bar_index).

        Recent patterns (near the end of the data) score higher.
        """
        bar_indices: list[int] = []
        for _, pattern_list in patterns.items():
            if isinstance(pattern_list, list):
                for p in pattern_list:
                    if isinstance(p, dict):
                        idx = p.get("bar_index") or p.get("index") or p.get("candle")
                        if idx is not None:
                            try:
                                bar_indices.append(int(idx))
                            except (TypeError, ValueError):
                                pass

        if not bar_indices:
            # No recency data available — neutral score
            return 50.0

        max_idx = max(bar_indices)
        if max_idx <= 0:
            return 50.0

        # Average recency: how close to the most recent bar
        # Normalize: recency = avg_index / max_index (closer to 1 = more recent)
        avg_idx = sum(bar_indices) / len(bar_indices)
        recency_ratio = avg_idx / max_idx

        # Scale: 0.3 ratio = 0, 1.0 ratio = 100 (linear)
        if recency_ratio <= 0.3:
            return 0.0
        return min(100.0, (recency_ratio - 0.3) / 0.7 * 100.0)

    @staticmethod
    def _score_indicator_alignment(
        tech_data: dict[str, Any],
        direction: str,
    ) -> float:
        """Score based on ADX and RSI alignment with dominant pattern direction.

        ADX > 25 indicates trending market (patterns more reliable).
        RSI aligned with direction adds confirmation.
        """
        if direction == "neutral":
            return 25.0  # No directional alignment possible

        adx = tech_data.get("adx", 0)
        rsi = tech_data.get("rsi", 50)

        try:
            adx = float(adx)
        except (TypeError, ValueError):
            adx = 0.0
        try:
            rsi = float(rsi)
        except (TypeError, ValueError):
            rsi = 50.0

        score = 0.0

        # ADX component (0-50): higher ADX = trend is real → patterns more reliable
        if math.isfinite(adx):
            if adx >= 40:
                score += 50
            elif adx >= 30:
                score += 35
            elif adx >= 25:
                score += 25
            elif adx >= 20:
                score += 10
            # ADX < 20: no trend, patterns unreliable

        # RSI component (0-50): alignment with pattern direction
        if math.isfinite(rsi):
            if direction == "bullish" and 40 <= rsi <= 70:
                score += 50  # Bullish patterns with healthy RSI
            elif direction == "bullish" and rsi < 30:
                score += 35  # Oversold reversal candidate
            elif direction == "bearish" and 30 <= rsi <= 60:
                score += 50  # Bearish patterns with weakening RSI
            elif direction == "bearish" and rsi > 70:
                score += 35  # Overbought reversal candidate
            else:
                score += 15  # Some RSI data but not strongly aligned

        return min(100.0, score)

    # ── Main scoring method ─────────────────────────────────────

    def score(
        self,
        patterns: dict[str, Any] | None = None,
        tech_data: dict[str, Any] | None = None,
        llm_quality: float | None = None,
    ) -> QualityScore:
        """Compute deterministic pattern quality score.

        Args:
            patterns: Technical patterns dict from PatternAnalyzer output.
            tech_data: Technical indicator dict (must contain 'adx' and 'rsi').
            llm_quality: Optional LLM-reported pattern_quality for comparison.

        Returns:
            QualityScore with component breakdown and any LLM discrepancies.
        """
        result = QualityScore()
        patterns = patterns or {}
        tech_data = tech_data or {}

        # 1. Extract and classify patterns
        pattern_names = self._extract_pattern_names(patterns)
        n_total = len(pattern_names)

        bullish_count = 0
        bearish_count = 0
        for name in pattern_names:
            d = self._classify_direction(name)
            if d == "bullish":
                bullish_count += 1
            elif d == "bearish":
                bearish_count += 1

        # Determine dominant direction
        if bullish_count > bearish_count:
            dominant_dir = "bullish"
        elif bearish_count > bullish_count:
            dominant_dir = "bearish"
        else:
            dominant_dir = "neutral"

        # 2. Compute component scores
        result.quantity_score = self._score_quantity(n_total)
        result.confirmation_score = self._score_confirmation(bullish_count, bearish_count)
        result.recency_score = self._score_recency(patterns)
        result.indicator_score = self._score_indicator_alignment(tech_data, dominant_dir)

        # 3. Weighted overall score
        result.overall = (
            result.quantity_score * WEIGHT_PATTERN_QUANTITY
            + result.confirmation_score * WEIGHT_PATTERN_CONFIRMATION
            + result.recency_score * WEIGHT_PATTERN_RECENCY
            + result.indicator_score * WEIGHT_INDICATOR_ALIGNMENT
        )

        # 4. Compare against LLM if provided
        if llm_quality is not None:
            try:
                llm_q = float(llm_quality)
                if 0 <= llm_q <= 100:
                    delta = abs(llm_q - result.overall)
                    if delta > QUALITY_DISCREPANCY_THRESHOLD:
                        result.passed = False
                        result.discrepancies.append(
                            f"Pattern quality: LLM reported {llm_q:.0f}, "
                            f"computed {result.overall:.0f} ({result.label}). "
                            f"Delta={delta:.0f} > {QUALITY_DISCREPANCY_THRESHOLD:.0f}."
                        )
            except (TypeError, ValueError):
                pass

        return result

    def overwrite_llm_quality(
        self,
        analysis: dict[str, Any],
        quality: QualityScore,
    ) -> dict[str, Any]:
        """Overwrite LLM's pattern_quality with deterministic score.

        Args:
            analysis: The 'analysis' dict from parsed_response.
            quality: Result from score().

        Returns:
            The same analysis dict (mutated in-place).
        """
        analysis["pattern_quality"] = round(quality.overall)

        # Store component breakdown for transparency
        analysis["_pattern_validation"] = quality.to_dict()

        return analysis
