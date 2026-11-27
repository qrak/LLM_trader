

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.indicators.price.risk_reward import risk_reward_ratio_numba

SIDES = {"LONG": 1, "SHORT": -1}


@dataclass(frozen=True)
class EVInputs:
    capital: float
    fee_percent: float
    inference_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if not _finite(self.capital) or self.capital < 0:
            raise ValueError("EV capital must be a non-negative finite number")
        if not _finite(self.fee_percent) or self.fee_percent < 0:
            raise ValueError("EV fee_percent must be a non-negative finite number")
        if not _finite(self.inference_cost_usd) or self.inference_cost_usd < 0:
            raise ValueError("EV inference cost must be a non-negative finite number")


@dataclass(frozen=True)
class EVScenario:
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    size_fraction: float
    p_win: float

    def __post_init__(self) -> None:
        side = str(self.side).upper()
        if side not in SIDES:
            raise ValueError(f"EV scenario side {self.side!r} is not LONG or SHORT")
        object.__setattr__(self, "side", side)
        for name in ("entry_price", "stop_loss", "take_profit"):
            value = getattr(self, name)
            if not _finite(value) or value <= 0:
                raise ValueError(f"EV scenario {name} must be a positive finite number")
        if not _finite(self.size_fraction) or not 0 < self.size_fraction <= 1:
            raise ValueError("EV scenario size_fraction must be in (0, 1]")
        if not _finite(self.p_win) or not 0 <= self.p_win <= 1:
            raise ValueError("EV scenario p_win must be in [0, 1]")
        if not math.isfinite(self.risk_reward):
            raise ValueError(
                f"EV scenario {side} prices must satisfy the {side} entry/SL/TP order"
            )

    @property
    def risk_reward(self) -> float:
        return risk_reward_ratio_numba(
            self.entry_price, self.stop_loss, self.take_profit, SIDES[self.side]
        )


@dataclass(frozen=True)
class EVResult:
    scenario: EVScenario
    quantity: float
    gross_win: float
    gross_loss: float
    round_trip_fee: float
    inference_cost: float
    cost: float
    net_ev: float
    threshold: float
    required_p_win: float

    @property
    def clears_threshold(self) -> bool:
        return self.net_ev > self.threshold

    @property
    def reachable(self) -> bool:
        return self.required_p_win < 1.0

    def to_record(self) -> dict[str, Any]:
        return {
            "side": self.scenario.side,
            "entry_price": self.scenario.entry_price,
            "stop_loss": self.scenario.stop_loss,
            "take_profit": self.scenario.take_profit,
            "size_fraction": round(self.scenario.size_fraction, 6),
            "p_win": round(self.scenario.p_win, 4),
            "quantity": round(self.quantity, 8),
            "risk_reward": None if not math.isfinite(self.scenario.risk_reward) else round(self.scenario.risk_reward, 4),
            "gross_win": round(self.gross_win, 4),
            "gross_loss": round(self.gross_loss, 4),
            "round_trip_fee": round(self.round_trip_fee, 4),
            "inference_cost": round(self.inference_cost, 4),
            "cost": round(self.cost, 4),
            "net_ev": round(self.net_ev, 4),
            "threshold": round(self.threshold, 4),
            "required_p_win": round(self.required_p_win, 4),
            "reachable": self.reachable,
            "clears_threshold": self.clears_threshold,
        }


class EVCalculator:
    THRESHOLD_MULTIPLE = 1.5

    def round_trip_fee(self, notional: float, fee_percent: float) -> float:

        return notional * fee_percent * 2

    def cost_base(self, inputs: EVInputs, notional: float) -> tuple[float, float, float, float]:

        fee = self.round_trip_fee(notional, inputs.fee_percent)
        cost = fee + inputs.inference_cost_usd
        return fee, inputs.inference_cost_usd, cost, cost * self.THRESHOLD_MULTIPLE

    def evaluate(self, inputs: EVInputs, scenario: EVScenario) -> EVResult:

        reward = abs(scenario.take_profit - scenario.entry_price)
        risk = abs(scenario.entry_price - scenario.stop_loss)
        notional = inputs.capital * scenario.size_fraction
        quantity = notional / scenario.entry_price
        gross_win = quantity * reward
        gross_loss = quantity * risk
        fee, inference, cost, threshold = self.cost_base(inputs, notional)
        net_ev = scenario.p_win * gross_win - (1.0 - scenario.p_win) * gross_loss - cost
        breadth = gross_win + gross_loss
        required_p_win = (threshold + cost + gross_loss) / breadth if breadth > 0 else math.inf
        return EVResult(
            scenario=scenario,
            quantity=quantity,
            gross_win=gross_win,
            gross_loss=gross_loss,
            round_trip_fee=fee,
            inference_cost=inference,
            cost=cost,
            net_ev=net_ev,
            threshold=threshold,
            required_p_win=required_p_win,
        )


class EVScenarioExtractor:
    def extract(self, payload: dict[str, Any]) -> list[EVScenario]:

        raw = payload.get("ev_scenarios")
        if raw is None:
            return []
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        kept: dict[str, EVScenario] = {}
        for entry in raw:
            scenario = self._parse(entry)
            if scenario is None or scenario.side in kept:
                continue
            kept[scenario.side] = scenario
        return [kept[side] for side in SIDES if side in kept]

    @staticmethod
    def _parse(entry: Any) -> EVScenario | None:

        if not isinstance(entry, dict):
            return None
        try:
            return EVScenario(
                side=str(entry.get("side") or ""),
                entry_price=EVScenarioExtractor._number(entry.get("entry_price")),
                stop_loss=EVScenarioExtractor._number(entry.get("stop_loss")),
                take_profit=EVScenarioExtractor._number(entry.get("take_profit")),
                size_fraction=EVScenarioExtractor._number(entry.get("size_fraction")),
                p_win=EVScenarioExtractor._number(entry.get("p_win")),
            )
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _number(value: Any) -> float:

        if isinstance(value, bool) or value is None:
            return math.nan
        if isinstance(value, str):
            try:
                value = float(value.strip())
            except ValueError:
                return math.nan
        if not isinstance(value, (int, float)):
            return math.nan
        return float(value)


def _finite(value: Any) -> bool:

    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
