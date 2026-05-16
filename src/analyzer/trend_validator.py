"""Trend Validator: system-side validation of LLM-reported ADX trend strengths.

Cross-checks the LLM's strength_4h and strength_daily claims against
deterministically computed ADX values from the technical indicators.
Flags discrepancies and provides corrected values for downstream use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Maximum allowed delta between LLM-reported and computed ADX before flagging
ADX_DISCREPANCY_THRESHOLD: float = 15.0

# When LLM provides no ADX value, these defaults are used
DEFAULT_ADX_FALLBACK: int = 25


@dataclass(slots=True)
class TrendValidation:
    """Result of validating LLM trend strength claims against computed ADX.

    Attributes:
        llm_strength_4h: The value the LLM reported for 4H ADX (or None).
        llm_strength_daily: The value the LLM reported for daily ADX (or None).
        computed_adx: Actual ADX computed from current timeframe OHLCV.
        computed_daily_adx: Actual ADX computed from daily OHLCV data.
        validated_4h: Best available 4H ADX (computed overrides LLM on discrepancy).
        validated_daily: Best available daily ADX (computed overrides LLM on discrepancy).
        discrepancies: List of human-readable discrepancy descriptions.
        passed: True if no significant discrepancies found.
    """

    llm_strength_4h: int | None = None
    llm_strength_daily: int | None = None
    computed_adx: float | None = None
    computed_daily_adx: float | None = None
    validated_4h: float = DEFAULT_ADX_FALLBACK
    validated_daily: float = DEFAULT_ADX_FALLBACK
    discrepancies: list[str] = field(default_factory=list)
    passed: bool = True

    @property
    def has_computed_data(self) -> bool:
        """Whether any computed ADX data was available for validation."""
        return self.computed_adx is not None or self.computed_daily_adx is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_strength_4h": self.llm_strength_4h,
            "llm_strength_daily": self.llm_strength_daily,
            "computed_adx": self.computed_adx,
            "computed_daily_adx": self.computed_daily_adx,
            "validated_4h": self.validated_4h,
            "validated_daily": self.validated_daily,
            "discrepancies": self.discrepancies,
            "passed": self.passed,
        }


class TrendValidator:
    """Validates LLM-reported ADX trend splits against computed indicators.

    Usage:
        validator = TrendValidator()
        result = validator.validate(
            strength_4h=llm_strength_4h,
            strength_daily=llm_strength_daily,
            computed_adx=tech_data.get("adx"),
            computed_daily_adx=long_term_data.get("daily_adx"),
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

    def validate(
        self,
        strength_4h: Any = None,
        strength_daily: Any = None,
        computed_adx: float | None = None,
        computed_daily_adx: float | None = None,
    ) -> TrendValidation:
        """Validate LLM-reported ADX values against computed indicators.

        Args:
            strength_4h: LLM-reported 4H trend strength (0-100).
            strength_daily: LLM-reported daily trend strength (0-100).
            computed_adx: Actual ADX from current timeframe technical indicators.
            computed_daily_adx: Actual daily ADX from long-term data.

        Returns:
            TrendValidation with validated values and any discrepancies.
        """
        result = TrendValidation()

        # Store raw inputs
        if self._is_valid_adx(strength_4h):
            result.llm_strength_4h = int(float(strength_4h))
        if self._is_valid_adx(strength_daily):
            result.llm_strength_daily = int(float(strength_daily))
        if self._is_valid_adx(computed_adx):
            result.computed_adx = float(computed_adx)
        if self._is_valid_adx(computed_daily_adx):
            result.computed_daily_adx = float(computed_daily_adx)

        # --- Validate 4H ADX ---
        llm_4h = result.llm_strength_4h
        comp_4h = result.computed_adx

        if comp_4h is not None:
            result.validated_4h = comp_4h
            if llm_4h is not None:
                delta = abs(llm_4h - comp_4h)
                if delta > ADX_DISCREPANCY_THRESHOLD:
                    result.passed = False
                    result.discrepancies.append(
                        f"4H ADX: LLM reported {llm_4h}, computed {comp_4h:.0f} "
                        f"({self._adx_label(comp_4h)}). Delta={delta:.0f} > {ADX_DISCREPANCY_THRESHOLD:.0f}. "
                        f"Using computed value."
                    )
            else:
                # LLM didn't report, but we have computed — no discrepancy, just using computed
                pass
        elif llm_4h is not None:
            # No computed data, trust LLM
            result.validated_4h = float(llm_4h)
        # else: neither — stays at DEFAULT_ADX_FALLBACK

        # --- Validate Daily ADX ---
        llm_daily = result.llm_strength_daily
        comp_daily = result.computed_daily_adx

        if comp_daily is not None:
            result.validated_daily = comp_daily
            if llm_daily is not None:
                delta = abs(llm_daily - comp_daily)
                if delta > ADX_DISCREPANCY_THRESHOLD:
                    result.passed = False
                    result.discrepancies.append(
                        f"Daily ADX: LLM reported {llm_daily}, computed {comp_daily:.0f} "
                        f"({self._adx_label(comp_daily)}). Delta={delta:.0f} > {ADX_DISCREPANCY_THRESHOLD:.0f}. "
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

        Args:
            analysis: The 'analysis' dict from parsed_response.
            validation: Result from validate().

        Returns:
            The same analysis dict (mutated in-place for convenience).
        """
        trend = analysis.setdefault("trend", {})

        # Replace with validated values
        trend["strength_4h"] = int(round(validation.validated_4h))
        trend["strength_daily"] = int(round(validation.validated_daily))

        # Add validation metadata for transparency
        analysis["_trend_validation"] = validation.to_dict()

        return analysis
