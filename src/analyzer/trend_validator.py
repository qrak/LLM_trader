

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TrendValidation:
    timeframe: str | None = None
    computed_adx: float | None = None
    computed_daily_adx: float | None = None
    validated_primary: float | None = None
    validated_daily: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "computed_adx": self.computed_adx,
            "computed_daily_adx": self.computed_daily_adx,
            "validated_primary": self.validated_primary,
            "validated_daily": self.validated_daily,
        }


class TrendValidator:
    @staticmethod
    def _adx_reading(value: Any) -> float | None:

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        reading = float(value)
        if not math.isfinite(reading) or not 0.0 <= reading <= 100.0:
            return None
        return reading

    def validate(
        self,
        computed_adx: Any = None,
        computed_daily_adx: Any = None,
        timeframe: str | None = None,
    ) -> TrendValidation:

        result = TrendValidation(timeframe=timeframe)
        result.computed_adx = self._adx_reading(computed_adx)
        result.computed_daily_adx = self._adx_reading(computed_daily_adx)
        result.validated_primary = result.computed_adx
        result.validated_daily = result.computed_daily_adx
        return result

    def overwrite_llm_trend(
        self,
        analysis: dict[str, Any],
        validation: TrendValidation,
    ) -> dict[str, Any]:

        trend = analysis.setdefault("trend", {})

        trend["strength_primary"] = (
            None if validation.validated_primary is None else round(validation.validated_primary)
        )
        trend["strength_daily"] = (
            None if validation.validated_daily is None else round(validation.validated_daily)
        )
        trend.pop("strength_4h", None)
        trend.pop("strength", None)

        analysis["_trend_validation"] = validation.to_dict()

        return analysis
