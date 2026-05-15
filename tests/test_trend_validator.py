"""Tests for TrendValidator: system-side ADX validation.

Covers:
  1. Valid LLM values matching computed → passes
  2. LLM values significantly off → flagged
  3. Missing LLM values → computed used
  4. Missing computed values → LLM used
  5. Neither available → fallback defaults
  6. Edge cases: NaN, None, out-of-range values
  7. overwrite_llm_trend() mutates analysis dict correctly
  8. to_dict() / TrendValidation fields
"""

import math
from unittest.mock import MagicMock

import pytest

from src.analyzer.trend_validator import (
    ADX_DISCREPANCY_THRESHOLD,
    TrendValidation,
    TrendValidator,
)


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def validator():
    return TrendValidator()


# ── Unit: _is_valid_adx ─────────────────────────────────────────


class TestIsValidAdx:
    def test_valid_int(self, validator):
        assert validator._is_valid_adx(45)

    def test_valid_float(self, validator):
        assert validator._is_valid_adx(32.7)

    def test_zero(self, validator):
        assert validator._is_valid_adx(0)

    def test_hundred(self, validator):
        assert validator._is_valid_adx(100)

    def test_none(self, validator):
        assert not validator._is_valid_adx(None)

    def test_negative(self, validator):
        assert not validator._is_valid_adx(-5)

    def test_over_100(self, validator):
        assert not validator._is_valid_adx(150)

    def test_nan(self, validator):
        assert not validator._is_valid_adx(math.nan)

    def test_string(self, validator):
        assert not validator._is_valid_adx("forty")


# ── Unit: _adx_label ────────────────────────────────────────────


class TestAdxLabel:
    def test_absent(self, validator):
        assert "absent" in validator._adx_label(10)

    def test_developing(self, validator):
        assert "developing" in validator._adx_label(22)

    def test_strong(self, validator):
        assert "strong" in validator._adx_label(40)

    def test_very_strong(self, validator):
        assert "very strong" in validator._adx_label(60)

    def test_extreme(self, validator):
        assert "extreme" in validator._adx_label(80)


# ── Core validation scenarios ───────────────────────────────────


class TestValidateAllComputed:
    """Both computed values available."""

    def test_perfect_match(self, validator):
        """LLM matches computed exactly — no discrepancies."""
        result = validator.validate(
            strength_4h=35, strength_daily=42,
            computed_adx=35.0, computed_daily_adx=42.0,
        )
        assert result.passed
        assert result.discrepancies == []
        assert result.validated_4h == 35.0
        assert result.validated_daily == 42.0

    def test_small_delta_allowed(self, validator):
        """Delta under threshold is fine."""
        result = validator.validate(
            strength_4h=35 + (ADX_DISCREPANCY_THRESHOLD - 1),
            strength_daily=42,
            computed_adx=35.0,
            computed_daily_adx=42.0,
        )
        assert result.passed

    def test_large_delta_flagged_4h(self, validator):
        """4H ADX discrepancy over threshold is flagged."""
        result = validator.validate(
            strength_4h=80, strength_daily=42,
            computed_adx=35.0, computed_daily_adx=42.0,
        )
        assert not result.passed
        assert len(result.discrepancies) == 1
        assert "4H ADX" in result.discrepancies[0]
        assert result.validated_4h == 35.0  # computed wins

    def test_large_delta_flagged_daily(self, validator):
        """Daily ADX discrepancy over threshold is flagged."""
        result = validator.validate(
            strength_4h=35, strength_daily=90,
            computed_adx=35.0, computed_daily_adx=42.0,
        )
        assert not result.passed
        assert len(result.discrepancies) == 1
        assert "Daily ADX" in result.discrepancies[0]
        assert result.validated_daily == 42.0

    def test_both_discrepant(self, validator):
        """Both 4H and daily are wrong — two discrepancies."""
        result = validator.validate(
            strength_4h=80, strength_daily=90,
            computed_adx=30.0, computed_daily_adx=40.0,
        )
        assert not result.passed
        assert len(result.discrepancies) == 2


class TestValidateMissingLlm:
    """LLM didn't provide values — computed used silently."""

    def test_no_llm_4h(self, validator):
        result = validator.validate(
            strength_4h=None, strength_daily=42,
            computed_adx=35.0, computed_daily_adx=42.0,
        )
        assert result.passed
        assert result.validated_4h == 35.0
        assert result.llm_strength_4h is None

    def test_no_llm_daily(self, validator):
        result = validator.validate(
            strength_4h=35, strength_daily=None,
            computed_adx=35.0, computed_daily_adx=42.0,
        )
        assert result.passed
        assert result.validated_daily == 42.0

    def test_no_llm_any(self, validator):
        result = validator.validate(
            strength_4h=None, strength_daily=None,
            computed_adx=35.0, computed_daily_adx=42.0,
        )
        assert result.passed
        assert result.validated_4h == 35.0
        assert result.validated_daily == 42.0


class TestValidateMissingComputed:
    """No computed data — LLM values trusted."""

    def test_only_llm_available(self, validator):
        result = validator.validate(
            strength_4h=35, strength_daily=42,
            computed_adx=None, computed_daily_adx=None,
        )
        assert result.passed
        assert result.validated_4h == 35.0
        assert result.validated_daily == 42.0

    def test_has_computed_data_false(self, validator):
        result = validator.validate(
            strength_4h=35, strength_daily=42,
            computed_adx=None, computed_daily_adx=None,
        )
        assert not result.has_computed_data


class TestValidateNeither:
    """No data at all — fallback defaults."""

    def test_neither_available(self, validator):
        result = validator.validate(
            strength_4h=None, strength_daily=None,
            computed_adx=None, computed_daily_adx=None,
        )
        assert result.passed
        assert result.validated_4h == 25  # DEFAULT_ADX_FALLBACK
        assert result.validated_daily == 25


# ── Edge Cases ──────────────────────────────────────────────────


class TestValidateEdgeCases:
    def test_invalid_llm_values_ignored(self, validator):
        """NaN/out-of-range LLM values treated as absent."""
        result = validator.validate(
            strength_4h=math.nan, strength_daily=150,
            computed_adx=35.0, computed_daily_adx=42.0,
        )
        assert result.llm_strength_4h is None
        assert result.llm_strength_daily is None
        assert result.validated_4h == 35.0
        assert result.validated_daily == 42.0

    def test_invalid_computed_ignored(self, validator):
        result = validator.validate(
            strength_4h=35, strength_daily=42,
            computed_adx=math.nan, computed_daily_adx=-5,
        )
        assert result.computed_adx is None
        assert result.computed_daily_adx is None
        assert result.validated_4h == 35.0
        assert result.validated_daily == 42.0

    def test_string_llm_values(self, validator):
        """String ADX values that parse as int are accepted."""
        result = validator.validate(
            strength_4h="35", strength_daily="42.0",
            computed_adx=35.0, computed_daily_adx=42.0,
        )
        assert result.llm_strength_4h == 35
        assert result.llm_strength_daily == 42

    def test_zero_adx(self, validator):
        """Zero ADX is valid (means no trend)."""
        result = validator.validate(
            strength_4h=0, computed_adx=0.0,
        )
        assert result.validated_4h == 0.0
        assert result.passed


# ── overwrite_llm_trend ─────────────────────────────────────────


class TestOverwriteLlmTrend:
    def test_overwrites_strengths(self, validator):
        analysis = {"trend": {"direction": "BULLISH", "strength_4h": 99, "strength_daily": 80}}
        validation = validator.validate(
            strength_4h=99, strength_daily=80,
            computed_adx=40.0, computed_daily_adx=45.0,
        )

        result = validator.overwrite_llm_trend(analysis, validation)
        assert result["trend"]["strength_4h"] == 40  # computed
        assert result["trend"]["strength_daily"] == 45  # computed

    def test_adds_validation_metadata(self, validator):
        analysis = {"trend": {"direction": "NEUTRAL"}}
        validation = validator.validate(
            computed_adx=40.0, computed_daily_adx=45.0,
        )

        result = validator.overwrite_llm_trend(analysis, validation)
        assert "_trend_validation" in result
        assert result["_trend_validation"]["passed"] is True

    def test_creates_trend_key_if_missing(self, validator):
        analysis = {}
        validation = validator.validate(
            computed_adx=40.0, computed_daily_adx=45.0,
        )

        result = validator.overwrite_llm_trend(analysis, validation)
        assert "trend" in result
        assert result["trend"]["strength_4h"] == 40


# ── TrendValidation dataclass ───────────────────────────────────


class TestTrendValidationDataclass:
    def test_defaults(self):
        tv = TrendValidation()
        assert tv.validated_4h == 25
        assert tv.validated_daily == 25
        assert tv.passed
        assert tv.discrepancies == []

    def test_to_dict(self, validator):
        result = validator.validate(
            strength_4h=35, strength_daily=42,
            computed_adx=35.0, computed_daily_adx=42.0,
        )
        d = result.to_dict()
        assert d["passed"]
        assert d["validated_4h"] == 35
        assert d["validated_daily"] == 42


# ── Integration: AnalysisResultProcessor validation call ────────


class TestProcessorIntegration:
    """Verify _validate_llm_claims integrates correctly."""

    def test_validate_with_context(self):
        """Full validation runs when context has both tech_data and long_term_data."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        from src.analyzer.analysis_result_processor import AnalysisResultProcessor

        proc = AnalysisResultProcessor(
            model_manager=MagicMock(),
            logger=MagicMock(),
            unified_parser=MagicMock(),
        )

        # Build a minimal context
        proc.context = SimpleNamespace(
            technical_data={"adx": 35.0, "rsi": 55.0},
            long_term_data={"daily_adx": 42.0},
            technical_patterns={
                "bullish_engulfing": [{"name": "bullish_engulfing", "bar_index": 95}],
            },
        )

        parsed = {
            "analysis": {
                "signal": "BUY",
                "confidence": "HIGH",
                "trend": {"direction": "BULLISH", "strength_4h": 35, "strength_daily": 42},
                "pattern_quality": 70,
            }
        }

        proc._validate_llm_claims(parsed)

        # Trend was validated (match → no discrepancy)
        assert parsed["analysis"]["trend"]["strength_4h"] == 35
        assert parsed["analysis"]["trend"]["strength_daily"] == 42
        assert "_trend_validation" in parsed["analysis"]

        # Pattern quality was computed
        assert "pattern_quality" in parsed["analysis"]
        assert "_pattern_validation" in parsed["analysis"]

    def test_validate_no_context_skips(self):
        """No context → validation is skipped."""
        from unittest.mock import MagicMock

        from src.analyzer.analysis_result_processor import AnalysisResultProcessor

        proc = AnalysisResultProcessor(
            model_manager=MagicMock(),
            logger=MagicMock(),
            unified_parser=MagicMock(),
        )
        proc.context = None

        parsed = {"analysis": {"trend": {"strength_4h": 80}}}
        proc._validate_llm_claims(parsed)
        # Should not crash, should not add metadata
        assert "_trend_validation" not in parsed["analysis"]

    def test_validate_no_analysis_skips(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from src.analyzer.analysis_result_processor import AnalysisResultProcessor

        proc = AnalysisResultProcessor(
            model_manager=MagicMock(),
            logger=MagicMock(),
            unified_parser=MagicMock(),
        )
        proc.context = SimpleNamespace(
            technical_data={"adx": 35.0},
            long_term_data={},
            technical_patterns={},
        )

        parsed = {"error": "no analysis"}
        proc._validate_llm_claims(parsed)
        # No crash

    def test_validate_logs_warning_on_discrepancy(self):
        """When ADX is way off, logger.warning is called."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from src.analyzer.analysis_result_processor import AnalysisResultProcessor

        logger = MagicMock()
        proc = AnalysisResultProcessor(
            model_manager=MagicMock(),
            logger=logger,
            unified_parser=MagicMock(),
        )
        proc.context = SimpleNamespace(
            technical_data={"adx": 35.0, "rsi": 55.0},
            long_term_data={"daily_adx": 42.0},
            technical_patterns={},
        )

        parsed = {
            "analysis": {
                "trend": {"strength_4h": 80, "strength_daily": 90},  # way off
            }
        }

        proc._validate_llm_claims(parsed)

        warnings = [c for c in logger.warning.call_args_list if "ADX" in str(c)]
        assert len(warnings) >= 1
