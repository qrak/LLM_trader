"""Verify model-quoted risk/reward using the execution layer's arithmetic."""

import math
import re
from typing import Any, ClassVar, cast

from src.indicators.price.risk_reward import risk_reward_ratio_numba


class RiskRewardValidator:
    NUMBER = r"\d+(?:\.\d+)?"
    CHECK = re.compile(
        rf"RR_CHECK side=(LONG|SHORT) entry=({NUMBER}) SL=({NUMBER}) "
        rf"TP=({NUMBER}) reward=({NUMBER}) risk=({NUMBER}) R/R=({NUMBER})",
        re.IGNORECASE,
    )
    SECTION = re.compile(r"(?ms)^11\) RISK/REWARD:.*?(?=^12\)|\Z)")
    RATIO = re.compile(
        r"(?i)(?:\d+(?:[.,]\d+)?\s*R/R|R/R\s*[=~:]\s*\d+(?:[.,]\d+)?|"
        r"(?:gives|offers|ratio)\s*(?:~|≈)?\s*\d+(?:[.,]\d+)?|"
        r"~\s*\d+[.,]\d+\b(?!\s*%))"
    )
    ENTRY_SIDES: ClassVar[dict[str, int]] = {"BUY": 1, "LONG": 1, "SELL": -1, "SHORT": -1}

    def validate(self, response: dict[str, Any]) -> list[str]:
        analysis = response.get("analysis")
        if not isinstance(analysis, dict):
            return []
        issues: list[str] = []
        signal = str(analysis.get("signal", "")).upper()
        entry_invalid = False
        if signal in self.ENTRY_SIDES:
            entry_invalid = self._validate_entry(analysis, signal, issues)

        narrative = response.get("narrative")
        if isinstance(narrative, str):
            response["narrative"] = self._validate_narrative(narrative, issues)
        if entry_invalid:
            analysis.update(
                signal="HOLD", entry_price=None, stop_loss=None, take_profit=None,
                risk_reward_ratio=None, position_size=0.0, quantity=0.0,
                order_type=None, reduce_only=False,
                reasoning="Entry blocked: proposed R/R or price levels failed Python verification.",
            )
            response["narrative"] = re.sub(
                r"(?m)^12\) DECISION:.*$",
                "12) DECISION: HOLD — entry blocked after Python R/R verification.",
                response.get("narrative") or "",
                count=1,
            )
            response["narrative"] = re.sub(
                r"(?m)^13\) EXECUTION NOTE:.*$",
                "13) EXECUTION NOTE: no order — Python R/R verification blocked this entry.",
                response["narrative"],
                count=1,
            )
        if issues:
            response["rr_feedback"] = "Python R/R validation of previous cycle: " + "; ".join(issues[:3])
            response["narrative"] = response["rr_feedback"] + "\n" + (response.get("narrative") or "")
        return issues

    def _validate_entry(self, analysis: dict[str, Any], signal: str, issues: list[str]) -> bool:
        values = [analysis.get(key) for key in ("entry_price", "stop_loss", "take_profit", "risk_reward_ratio")]
        if any(type(value) not in (float, int) for value in values):
            issues.append("entry blocked: entry/SL/TP/R/R must all be numeric")
            return True
        entry, stop, target, quoted = cast(tuple[float, float, float, float], tuple(values))
        actual = risk_reward_ratio_numba(entry, stop, target, self.ENTRY_SIDES[signal])
        if not math.isfinite(actual) or not math.isfinite(quoted) or quoted < 0:
            issues.append("entry blocked: invalid prices or non-finite R/R")
            return True
        if abs(actual - quoted) > 0.055:
            issues.append(f"entry blocked: model R/R {quoted:.2f}, Python R/R {actual:.2f} from entry={entry:g}, SL={stop:g}, TP={target:g}")
            return True
        analysis["risk_reward_ratio"] = actual
        return False

    def _validate_narrative(self, narrative: str, issues: list[str]) -> str:
        section = self.SECTION.search(narrative)
        if section is None:
            return narrative
        text = section.group()
        checks = list(self.CHECK.finditer(text))
        if not checks:
            if self.RATIO.search(text):
                issues.append("unverifiable narrative R/R: no complete RR_CHECK price triple; quote entry, SL and TP")
                return narrative[:section.start()] + "11) RISK/REWARD: N/A — Python could not verify the quoted ratio.\n" + narrative[section.end():]
            return narrative

        corrections: list[str] = []
        for check in checks:
            side, *numbers = check.groups()
            entry, stop, target, quoted_reward, quoted_risk, quoted_ratio = map(float, numbers)
            actual = risk_reward_ratio_numba(entry, stop, target, self.ENTRY_SIDES[side.upper()])
            if not math.isfinite(actual):
                issues.append(f"invalid {side.upper()} R/R price triple in narrative")
                continue
            reward = abs(target - entry)
            risk = abs(entry - stop)
            if (abs(quoted_reward - reward) > 0.011 or abs(quoted_risk - risk) > 0.011
                    or abs(quoted_ratio - actual) > 0.011):
                issues.append(
                    f"{side.upper()} narrative: model reward/risk/R/R "
                    f"{quoted_reward:g}/{quoted_risk:g}/{quoted_ratio:.2f}, "
                    f"Python {reward:.2f}/{risk:.2f}/{actual:.2f} "
                    f"from entry={entry:g}, SL={stop:g}, TP={target:g}"
                )
            corrections.append(
                f"RR_CHECK side={side.upper()} entry={entry:g} SL={stop:g} TP={target:g} "
                f"reward={reward:.2f} risk={risk:.2f} R/R={actual:.2f}"
            )
        if issues or len(self.RATIO.findall(text)) != len(checks):
            if not issues:
                issues.append("narrative contained additional unverified R/R figures")
            corrected = "; ".join(corrections) if corrections else "N/A — invalid price levels"
            return narrative[:section.start()] + f"11) RISK/REWARD: Python verified: {corrected}\n" + narrative[section.end():]
        return narrative
