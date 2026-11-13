

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import httpx

if TYPE_CHECKING:
    from src.analyzer.analysis_context import AnalysisContext
    from src.logger.logger import Logger


@dataclass
class GateVerdict:
    """One Decisions answer reduced to the fields the size policy and the journal need."""

    edge_p: float | None = None
    direction: str | None = None
    direction_confidence: float | None = None
    quality: float | None = None
    quality_confidence: float | None = None
    multiplier: float = 1.0
    latency_ms: int = 0
    input_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


class DecisionsClient:
    """Asks a Decisions endpoint one record of questions about a trading state.

    The alpha endpoint answers `answers` plus `usage.cost`, so the gate never prices a
    token itself. `transport` exists for tests; production passes nothing.
    """

    def __init__(
        self,
        url: str,
        model: str,
        api_key: str,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:

        self.url = url
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    async def ask(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:

        payload = {"model": self.model, "state": state, "questions": questions}
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds, transport=self._transport
        ) as client:
            response = await client.post(
                self.url,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        if response.status_code != httpx.codes.OK:
            raise RuntimeError(
                f"decisions endpoint answered {response.status_code}: {response.text[:200]}"
            )
        body = response.json()
        if not isinstance(body.get("answers"), dict):
            raise TypeError("decisions endpoint returned no answers")
        return body


class DecisionGate:
    """Independent second opinion on one trade decision, spent only on position size.

    The endpoint is an alpha path: the only Decisions shape reachable with the configured
    OpenRouter key. A swap to the GA path touches DecisionsClient alone. A refusal,
    timeout or malformed body leaves the decision and the size untouched.
    """

    SCHEMA = 1
    MODES = ("off", "shadow", "size")
    SCORE_MULTIPLIERS: ClassVar[dict[int, float]] = {0: 0.5, 1: 0.5, 2: 0.75, 3: 1.0}
    DISAGREEMENT_FACTOR = 0.75
    FLOOR = 0.5
    CEILING = 1.0
    CANDLES = 20
    REASONING_CHARS = 600
    ADX_HIGH = 25.0
    ADX_LOW = 20.0
    SIDES: ClassVar[dict[str, str]] = {"BUY": "LONG", "SELL": "SHORT"}
    INDICATOR_KEYS = (
        "adx", "plus_di", "minus_di", "choppiness", "rsi", "macd_hist",
        "atr_percent", "bb_percent_b", "cmf", "mfi", "vwap", "supertrend_direction",
    )

    def __init__(
        self,
        logger: Logger,
        client: DecisionsClient,
        mode: str,
        min_confidence_pct: float,
        journal_path: Path,
    ) -> None:

        if mode not in self.MODES:
            raise ValueError(f"decisions gate mode must be one of {self.MODES}, got {mode!r}")
        if mode != "off" and not client.api_key:
            raise ValueError("decisions gate needs OPENROUTER_API_KEY unless mode = off")
        self.logger = logger
        self.client = client
        self.mode = mode
        self.min_confidence_pct = min_confidence_pct
        self.journal_path = journal_path

    async def review(
        self, response: dict[str, Any], context: AnalysisContext | None
    ) -> GateVerdict | None:

        analysis = response.get("analysis")
        if not isinstance(analysis, dict):
            return None
        started = time.perf_counter()
        state = self._build_state(analysis, context)
        try:
            verdict = self._parse(await self.client.ask(state, self._questions()))
        except Exception as exc:  # noqa: BLE001
            verdict = GateVerdict(error=str(exc)[:200] or exc.__class__.__name__)
            self.logger.warning("Decision gate failed open: %s", verdict.error)
        verdict.latency_ms = int((time.perf_counter() - started) * 1000)
        if verdict.error is None:
            verdict.multiplier = self._multiplier(verdict, analysis)
        applied = self._apply(verdict, response) if self.mode == "size" else False
        response["decision_gate"] = asdict(verdict)
        self._write_journal(verdict, analysis, state["conditions"], applied)
        return verdict

    def _questions(self) -> dict[str, Any]:

        return {
            "edge_now": {
                "type": "noul",
                "instructions": (
                    "Does this trade decision carry positive expectancy over the next "
                    "three candles? Judge the decision as stated, not the market."
                ),
                "criteria": {
                    "true": "positive expectancy after costs",
                    "false": "zero or negative expectancy",
                },
            },
            "direction": {
                "type": "choice",
                "instructions": "Which side do the next three candles favour?",
                "criteria": {
                    "long": "price closes higher than now",
                    "short": "price closes lower than now",
                    "none": "no reliable directional edge",
                },
            },
            "quality": {
                "type": "score",
                "instructions": "Rate the location and structure of this entry.",
                "criteria": ["bad", "weak", "solid", "strong"],
                "legend": "0=bad location 3=strong location",
            },
        }

    def _build_state(
        self, analysis: dict[str, Any], context: AnalysisContext | None
    ) -> dict[str, Any]:

        technical = getattr(context, "technical_data", None) or {}
        indicators = {key: self._number(technical.get(key)) for key in self.INDICATOR_KEYS}
        indicators = {key: value for key, value in indicators.items() if value is not None}
        trend = analysis.get("trend")
        regime = trend.get("direction") if isinstance(trend, dict) else None
        return {
            "symbol": getattr(context, "symbol", None) or analysis.get("symbol"),
            "timeframe": getattr(context, "timeframe", None),
            "price": self._number(getattr(context, "current_price", None)),
            "conditions": {
                "direction": self.SIDES.get(str(analysis.get("signal") or "").upper()),
                "regime": regime,
                "adx_bucket": self._adx_bucket(indicators.get("adx")),
            },
            "indicators": indicators,
            "candles": self._candles(getattr(context, "ohlcv_candles", None)),
            "decision": {
                "signal": analysis.get("signal"),
                "entry_price": self._number(analysis.get("entry_price")),
                "stop_loss": self._number(analysis.get("stop_loss")),
                "take_profit": self._number(analysis.get("take_profit")),
                "risk_reward_ratio": self._number(analysis.get("risk_reward_ratio")),
                "position_size": self._number(analysis.get("position_size")),
                "reasoning": str(analysis.get("reasoning") or "")[: self.REASONING_CHARS],
            },
        }

    def _candles(self, ohlcv: Any) -> list[list[float]]:

        if ohlcv is None or len(ohlcv) == 0:
            return []
        rows: list[list[float]] = []
        for row in ohlcv[-self.CANDLES:]:
            rows.append([self._number(row[index]) or 0.0 for index in range(1, 6)])
        return rows

    def _parse(self, body: dict[str, Any]) -> GateVerdict:

        answers = body["answers"]
        edge = answers.get("edge_now") or {}
        choice = answers.get("direction") or {}
        score = answers.get("quality") or {}
        usage = body.get("usage") or {}
        verdict = GateVerdict(
            edge_p=self._number(edge.get("noul")),
            direction=str(choice.get("choice") or "").lower() or None,
            direction_confidence=self._number(choice.get("confidence")),
            quality=self._number(score.get("score")),
            quality_confidence=self._number(score.get("confidence")),
            input_tokens=int(self._number(usage.get("input_tokens")) or 0),
            cost_usd=self._number(usage.get("cost")) or 0.0,
        )
        if verdict.quality is None:
            raise RuntimeError("decisions response carried no quality score")
        return verdict

    def _multiplier(self, verdict: GateVerdict, analysis: dict[str, Any]) -> float:

        side = self.SIDES.get(str(analysis.get("signal") or "").upper())
        if side is None:
            return self.CEILING
        if (
            verdict.quality_confidence is not None
            and verdict.quality_confidence * 100.0 < self.min_confidence_pct
        ):
            return self.CEILING
        score = max(0, min(3, round(verdict.quality or 0.0)))
        multiplier = self.SCORE_MULTIPLIERS[score]
        expected = "long" if side == "LONG" else "short"
        if verdict.direction is not None and verdict.direction != expected:
            multiplier *= self.DISAGREEMENT_FACTOR
        return round(max(self.FLOOR, min(self.CEILING, multiplier)), 4)

    def _apply(self, verdict: GateVerdict, response: dict[str, Any]) -> bool:

        analysis = response["analysis"]
        if verdict.error is not None or verdict.multiplier >= self.CEILING:
            return False
        size = self._number(analysis.get("position_size"))
        if size is None or size <= 0:
            return False
        analysis["position_size"] = round(min(size * verdict.multiplier, 1.0), 6)
        narrative = response.get("narrative")
        if isinstance(narrative, str):
            response["narrative"] = self._feedback_line(verdict, analysis) + "\n" + narrative
        return True

    def _feedback_line(self, verdict: GateVerdict, analysis: dict[str, Any]) -> str:

        quality = f"{verdict.quality:.2f}/3" if verdict.quality is not None else "n/a"
        edge = f"{verdict.edge_p:.2f}" if verdict.edge_p is not None else "n/a"
        return (
            f"Decision gate (independent model, size only): quality {quality}, "
            f"edge {edge}, direction {verdict.direction or 'n/a'} — position size "
            f"×{verdict.multiplier:g} (now {analysis.get('position_size')})."
        )

    def _write_journal(
        self,
        verdict: GateVerdict,
        analysis: dict[str, Any],
        conditions: dict[str, Any],
        applied: bool,
    ) -> None:

        entry = {
            "schema": self.SCHEMA,
            "ts": datetime.now(timezone.utc).isoformat(),
            "mode": self.mode,
            "model": self.client.model,
            "signal": analysis.get("signal"),
            "confidence": self._number(analysis.get("confidence")),
            "direction": conditions.get("direction"),
            "regime": conditions.get("regime"),
            "adx_bucket": conditions.get("adx_bucket"),
            "applied": applied,
            **asdict(verdict),
        }
        try:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with self.journal_path.open("a", encoding="utf-8") as journal:
                journal.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            self.logger.warning("Decision gate journal write failed: %s", exc)

    def _adx_bucket(self, adx: float | None) -> str | None:

        if adx is None:
            return None
        if adx >= self.ADX_HIGH:
            return "HIGH_ADX"
        if adx < self.ADX_LOW:
            return "LOW_ADX"
        return "MED_ADX"

    @staticmethod
    def _number(value: Any) -> float | None:

        if isinstance(value, bool) or value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None
