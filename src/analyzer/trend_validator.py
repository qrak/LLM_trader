"""Trend Validator: system-side validation of LLM-reported ADX trend strengths.

Cross-checks the LLM's primary-timeframe and daily ADX claims against
deterministically computed ADX values from the technical indicators.
Flags discrepancies and provides corrected values for downstream use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ADX_DISCREPANCY_THRESHOLD: float = 15.0

DEFAULT_ADX_FALLBACK: int = 25


@dataclass(slots=True)
class TrendValidation:
    """Result of validating LLM trend strength claims against computed ADX.

    Attributes:
        timeframe: Active analysis frame, used only to label the discrepancy.
        llm_strength_primary: The value the LLM reported for the primary frame (or None).
        llm_strength_daily: The value the LLM reported for daily ADX (or None).
        computed_adx: Actual ADX computed from current timeframe OHLCV.
        computed_daily_adx: Actual ADX computed from daily OHLCV data.
        validated_primary: Best available primary-frame ADX (computed wins on discrepancy).
        validated_daily: Best available daily ADX (computed wins on discrepancy).
        discrepancies: List of human-readable discrepancy descriptions.
        passed: True if no significant discrepancies found.
    """

    timeframe: str | None = None
    llm_strength_primary: int | None = None
    llm_strength_daily: int | None = None
    computed_adx: float | None = None
    computed_daily_adx: float | None = None
    validated_primary: float = DEFAULT_ADX_FALLBACK
    validated_daily: float = DEFAULT_ADX_FALLBACK
    discrepancies: list[str] = field(default_factory=list)
    passed: bool = True

    @property
    def has_computed_data(self) -> bool:
        """Whether any computed ADX data was available for validation."""
        return self.computed_adx is not None or self.computed_daily_adx is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "llm_strength_primary": self.llm_strength_primary,
            "llm_strength_daily": self.llm_strength_daily,
            "computed_adx": self.computed_adx,
            "computed_daily_adx": self.computed_daily_adx,
            "validated_primary": self.validated_primary,
            "validated_daily": self.validated_daily,
            "discrepancies": self.discrepancies,
            "passed": self.passed,
        }


class TrendValidator:
    """Validates LLM-reported ADX trend splits against computed indicators.

    Usage:
        validator = TrendValidator()
        result = validator.validate(
            strength_primary=llm_strength_primary,
            strength_daily=llm_strength_daily,
            computed_adx=tech_data.get("adx"),
            computed_daily_adx=long_term_data.get("daily_adx"),
            timeframe=context.timeframe,
        )
        if not result.passed:
            logger.warning("ADX discrepancy: %s", result.discrepancies)
    """

    @staticmethod
    def _adx_label(value: float) -> str:
        """Classify ADX into trend strength label."""
        if value < 20:
            return "absent/weak"
        if value < 25:
            return "developing"
        if value < 50:
            return "strong"
        if value < 75:
            return "very strong"
        return "extreme"

    @staticmethod
    def _is_valid_adx(value: Any) -> bool:
        """Check if a value looks like a plausible ADX reading (0-100)."""
        if value is None:
            return False
        try:
            v = float(value)
            return 0 <= v <= 100
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _primary_label(timeframe: str | None) -> str:
        """Name the primary series in a discrepancy message."""
        return f"{timeframe} ADX" if timeframe else "Primary-timeframe ADX"

    def validate(
        self,
        strength_primary: Any = None,
        strength_daily: Any = None,
        computed_adx: float | None = None,
        computed_daily_adx: float | None = None,
        timeframe: str | None = None,
    ) -> TrendValidation:
        """Validate LLM-reported ADX values against computed indicators.
        Returns:
            TrendValidation with validated values and any discrepancies.
        """
        result = TrendValidation(timeframe=timeframe)

        if self._is_valid_adx(strength_primary):
            result.llm_strength_primary = int(float(strength_primary))
        if self._is_valid_adx(strength_daily):
            result.llm_strength_daily = int(float(strength_daily))
        if computed_adx is not None and self._is_valid_adx(computed_adx):
            result.computed_adx = float(computed_adx)
        if computed_daily_adx is not None and self._is_valid_adx(computed_daily_adx):
            result.computed_daily_adx = float(computed_daily_adx)

        llm_primary = result.llm_strength_primary
        comp_primary = result.computed_adx

        if comp_primary is not None:
            result.validated_primary = comp_primary
            if llm_primary is not None:
                reported_primary = int(llm_primary)
                computed_primary = round(comp_primary)
                delta = abs(reported_primary - computed_primary)
                if delta > ADX_DISCREPANCY_THRESHOLD:
                    result.passed = False
                    result.discrepancies.append(
                        f"{self._primary_label(timeframe)}: LLM reported {reported_primary}, "
                        f"computed {computed_primary} ({self._adx_label(comp_primary)}). "
                        f"Delta={delta} > {ADX_DISCREPANCY_THRESHOLD:.0f}. Using computed value."
                    )
        elif llm_primary is not None:
            result.validated_primary = float(llm_primary)

        llm_daily = result.llm_strength_daily
        comp_daily = result.computed_daily_adx

        if comp_daily is not None:
            result.validated_daily = comp_daily
            if llm_daily is not None:
                reported_daily = int(llm_daily)
                computed_daily = round(comp_daily)
                delta = abs(reported_daily - computed_daily)
                if delta > ADX_DISCREPANCY_THRESHOLD:
                    result.passed = False
                    result.discrepancies.append(
                        f"Daily ADX: LLM reported {reported_daily}, computed {computed_daily} "
                        f"({self._adx_label(comp_daily)}). Delta={delta} > {ADX_DISCREPANCY_THRESHOLD:.0f}. "
                        f"Using computed value."
                    )
        elif llm_daily is not None:
            result.validated_daily = float(llm_daily)

        return result

    def overwrite_llm_trend(
        self,
        analysis: dict[str, Any],
        validation: TrendValidation,
    ) -> dict[str, Any]:
        """Overwrite the LLM's trend strengths with validated values in-place.
        Returns:
            The same analysis dict (mutated in-place for convenience).
        """
        trend = analysis.setdefault("trend", {})

        trend["strength_primary"] = round(validation.validated_primary)
        trend["strength_daily"] = round(validation.validated_daily)
        trend.pop("strength_4h", None)

        analysis["_trend_validation"] = validation.to_dict()

        return analysis
