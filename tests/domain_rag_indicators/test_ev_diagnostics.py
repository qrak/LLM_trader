

import math

import pytest

from src.analyzer.ev_diagnostics import EVCalculator, EVInputs, EVScenario, EVScenarioExtractor

CAPITAL = 10070.78
ENTRY = 61595.56
STOP = 62750.0
TARGET = 59299.93
SIZE = 0.02
INFERENCE = 0.0116


def inputs(inference_cost_usd: float = INFERENCE) -> EVInputs:
    return EVInputs(capital=CAPITAL, fee_percent=0.00075, inference_cost_usd=inference_cost_usd)


def short_scenario(p_win: float = 0.40) -> EVScenario:
    return EVScenario("SHORT", ENTRY, STOP, TARGET, SIZE, p_win)


def test_a_short_scenario_reproduces_the_1300_cycle_arithmetic():

    result = EVCalculator().evaluate(inputs(), short_scenario())

    assert result.quantity == pytest.approx(0.003269969458837618)
    assert result.gross_win == pytest.approx(7.5066, abs=1e-4)
    assert result.gross_loss == pytest.approx(3.7750, abs=1e-4)
    assert result.round_trip_fee == pytest.approx(0.3021, abs=1e-4)
    assert result.cost == pytest.approx(0.3137, abs=1e-4)
    assert result.threshold == pytest.approx(0.4706, abs=1e-4)
    assert result.net_ev == pytest.approx(0.4239, abs=1e-3)
    assert result.net_ev < result.threshold
    assert result.clears_threshold is False
    assert result.required_p_win == pytest.approx(0.4041, abs=1e-3)


def test_the_threshold_probability_is_reachable_only_below_one():

    result = EVCalculator().evaluate(inputs(), short_scenario(p_win=0.41))
    assert result.clears_threshold is True

    far_target = EVScenario("SHORT", 100.0, 101.0, 99.0, 0.02, 0.99)
    stressed = EVCalculator().evaluate(
        EVInputs(capital=CAPITAL, fee_percent=0.05, inference_cost_usd=1.0), far_target
    )
    assert stressed.required_p_win > 1.0
    assert stressed.reachable is False


def test_a_long_scenario_uses_the_long_price_order():

    long_scenario = EVScenario("LONG", 100.0, 95.0, 110.0, 0.08, 0.5)
    result = EVCalculator().evaluate(inputs(), long_scenario)

    assert result.scenario.risk_reward == pytest.approx(2.0)
    assert result.gross_win == pytest.approx(2 * result.gross_loss)


def test_zero_inference_cost_leaves_only_the_fee_in_the_threshold():

    result = EVCalculator().evaluate(inputs(0.0), short_scenario())

    assert result.inference_cost == 0.0
    assert result.cost == pytest.approx(result.round_trip_fee)
    assert result.threshold == pytest.approx(result.round_trip_fee * 1.5)


@pytest.mark.parametrize(
    ("side", "entry", "stop", "target", "size", "p_win"),
    [
        ("SIDEWAYS", 100.0, 95.0, 110.0, 0.02, 0.5),
        ("LONG", 100.0, 110.0, 95.0, 0.02, 0.5),
        ("SHORT", 100.0, 95.0, 110.0, 0.02, 0.5),
        ("LONG", 100.0, 95.0, 110.0, 0.0, 0.5),
        ("LONG", 100.0, 95.0, 110.0, 1.5, 0.5),
        ("LONG", 100.0, 95.0, 110.0, 0.02, -0.1),
        ("LONG", 100.0, 95.0, 110.0, 0.02, 1.5),
        ("LONG", float("nan"), 95.0, 110.0, 0.02, 0.5),
        ("LONG", 100.0, 0.0, 110.0, 0.02, 0.5),
    ],
    ids=[
        "unknown-side",
        "long-levels-inverted",
        "short-levels-inverted",
        "zero-size",
        "size-above-one",
        "negative-p-win",
        "p-win-above-one",
        "nan-entry",
        "zero-stop",
    ],
)
def test_inconsistent_scenarios_are_rejected(side, entry, stop, target, size, p_win):

    with pytest.raises(ValueError):
        EVScenario(side, entry, stop, target, size, p_win)


@pytest.mark.parametrize(
    "inputs_",
    [
        {"capital": -1.0, "fee_percent": 0.001},
        {"capital": CAPITAL, "fee_percent": -0.001},
        {"capital": CAPITAL, "fee_percent": 0.001, "inference_cost_usd": math.inf},
    ],
    ids=["negative-capital", "negative-fee", "infinite-inference"],
)
def test_invalid_costs_are_rejected(inputs_):

    with pytest.raises(ValueError):
        EVInputs(**inputs_)


def test_the_extractor_keeps_one_valid_scenario_per_side():

    payload = {
        "ev_scenarios": [
            {"side": "SHORT", "entry_price": ENTRY, "stop_loss": STOP, "take_profit": TARGET,
             "size_fraction": SIZE, "p_win": 0.4},
            {"side": "short", "entry_price": ENTRY, "stop_loss": STOP, "take_profit": TARGET,
             "size_fraction": SIZE, "p_win": 0.3},
            {"side": "LONG", "entry_price": 100.0, "stop_loss": 110.0, "take_profit": 90.0,
             "size_fraction": SIZE, "p_win": 0.4},
            {"side": "LONG", "entry_price": 61583.26, "stop_loss": 60458.37,
             "take_profit": 62953.39, "size_fraction": 0.02, "p_win": 0.35},
            "not-a-scenario",
            {"side": "LONG", "entry_price": "n/a", "stop_loss": 1, "take_profit": 2,
             "size_fraction": 0.02, "p_win": 0.5},
        ]
    }

    scenarios = EVScenarioExtractor().extract(payload)

    assert [(scenario.side, scenario.p_win) for scenario in scenarios] == [
        ("LONG", 0.35),
        ("SHORT", 0.4),
    ]


@pytest.mark.parametrize("payload", [{}, {"ev_scenarios": None}, {"ev_scenarios": "x"}])
def test_the_extractor_returns_nothing_without_a_scenario_list(payload):

    assert EVScenarioExtractor().extract(payload) == []
