

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from src.analyzer.decision_gate import DecisionGate, DecisionsClient


ANALYSIS = {
    "signal": "SELL",
    "confidence": 65,
    "entry_price": 73213.07,
    "stop_loss": 75000.0,
    "take_profit": 69450.0,
    "risk_reward_ratio": 2.0,
    "position_size": 0.05,
    "reasoning": "fresh Donchian low on 3.7x volume",
    "trend": {"direction": "BEARISH"},
}

CONTEXT = SimpleNamespace(
    symbol="BTC/USDC",
    timeframe="4h",
    current_price=73213.07,
    technical_data={"adx": 43.18, "rsi": 32.98, "cmf": -0.17},
    ohlcv_candles=[[0, 73010.0, 73480.0, 72800.0, 73152.0, 1200.0]] * 3,
)


def answers(quality=2.0, confidence=0.9, direction="short", edge=0.6):
    return {
        "answers": {
            "edge_now": {"type": "noul", "noul": edge},
            "direction": {
                "type": "choice",
                "choice": direction,
                "probabilities": {"long": 0.1, "short": 0.9, "none": 0.0},
                "confidence": 0.9,
            },
            "quality": {
                "type": "score",
                "score": quality,
                "legend": {"0": "bad", "1": "weak", "2": "solid", "3": "strong"},
                "probabilities": {"0": 0.0, "1": 0.1, "2": 0.9, "3": 0.0},
                "confidence": confidence,
            },
        },
        "usage": {"input_tokens": 700, "output_tokens": 0, "cost": 0.00007},
        "model": "openai/gpt-6-luna-decisions-20261006",
        "provider": "OpenAI",
    }


class StubClient:
    def __init__(self, payload=None, error=None):
        self.api_key = "key"
        self.model = "openai/gpt-6-luna-decisions"
        self.payload = payload or answers()
        self.error = error
        self.calls = []

    async def ask(self, state, questions):
        self.calls.append((state, questions))
        if self.error is not None:
            raise self.error
        return self.payload


def make_gate(tmp_path, mode="shadow", client=None, min_confidence_pct=20):
    logger = MagicMock()
    gate = DecisionGate(
        logger=logger,
        client=client or StubClient(),
        mode=mode,
        min_confidence_pct=min_confidence_pct,
        journal_path=tmp_path / "decisions_gate.jsonl",
    )
    return gate, logger


def journal(tmp_path):
    path = tmp_path / "decisions_gate.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


async def test_shadow_mode_journals_and_leaves_the_decision_untouched(tmp_path):

    gate, _ = make_gate(tmp_path, mode="shadow")
    response = {"analysis": dict(ANALYSIS), "narrative": "narrative"}

    verdict = await gate.review(response, CONTEXT)

    assert response["analysis"] == ANALYSIS
    assert response["narrative"] == "narrative"
    assert verdict.multiplier == 0.75
    assert response["decision_gate"]["quality"] == 2.0
    lines = journal(tmp_path)
    assert len(lines) == 1
    assert lines[0]["mode"] == "shadow"
    assert lines[0]["applied"] is False
    assert lines[0]["adx_bucket"] == "HIGH_ADX"
    assert lines[0]["confidence"] == 65
    assert lines[0]["cost_usd"] == pytest.approx(0.00007)


async def test_size_mode_multiplies_the_size_and_annotates_the_narrative(tmp_path):

    gate, _ = make_gate(tmp_path, mode="size")
    response = {"analysis": dict(ANALYSIS), "narrative": "narrative"}

    await gate.review(response, CONTEXT)

    assert response["analysis"]["position_size"] == pytest.approx(0.0375)
    assert response["narrative"].startswith("Decision gate (independent model, size only):")
    assert response["narrative"].endswith("narrative")
    assert journal(tmp_path)[0]["applied"] is True


@pytest.mark.parametrize(
    "quality,expected",
    [(0.0, 0.5), (0.9, 0.5), (1.4, 0.5), (1.6, 0.75), (1.99, 0.75), (2.6, 1.0), (3.0, 1.0)],
)
async def test_multiplier_follows_the_rounded_score(tmp_path, quality, expected):

    gate, _ = make_gate(tmp_path, client=StubClient(answers(quality=quality)))

    verdict = await gate.review({"analysis": dict(ANALYSIS)}, CONTEXT)

    assert verdict.multiplier == expected


async def test_disagreeing_direction_is_penalised(tmp_path):

    gate, _ = make_gate(tmp_path, client=StubClient(answers(quality=3.0, direction="long")))

    verdict = await gate.review({"analysis": dict(ANALYSIS)}, CONTEXT)

    assert verdict.multiplier == 0.75


async def test_an_unsure_answer_keeps_the_full_size(tmp_path):

    gate, _ = make_gate(tmp_path, client=StubClient(answers(quality=0.0, confidence=0.1)))

    verdict = await gate.review({"analysis": dict(ANALYSIS)}, CONTEXT)

    assert verdict.multiplier == 1.0


async def test_hold_is_journalled_but_never_resized(tmp_path):

    gate, _ = make_gate(tmp_path, mode="size", client=StubClient(answers(quality=0.0)))
    analysis = dict(ANALYSIS, signal="HOLD", position_size=0.0)
    response = {"analysis": analysis, "narrative": "narrative"}

    verdict = await gate.review(response, CONTEXT)

    assert verdict.multiplier == 1.0
    assert response["analysis"]["position_size"] == 0.0
    assert response["narrative"] == "narrative"
    assert journal(tmp_path)[0]["applied"] is False


async def test_a_failure_fails_open_and_logs_once(tmp_path):

    gate, logger = make_gate(tmp_path, mode="size", client=StubClient(error=RuntimeError("boom")))
    response = {"analysis": dict(ANALYSIS), "narrative": "narrative"}

    verdict = await gate.review(response, CONTEXT)

    assert verdict.error == "boom"
    assert verdict.multiplier == 1.0
    assert response["analysis"]["position_size"] == 0.05
    assert response["narrative"] == "narrative"
    logger.warning.assert_called_once()
    assert journal(tmp_path)[0]["error"] == "boom"


async def test_state_hides_the_confidence_and_names_the_rule_conditions(tmp_path):

    client = StubClient()
    gate, _ = make_gate(tmp_path, client=client)

    await gate.review({"analysis": dict(ANALYSIS)}, CONTEXT)

    state = client.calls[0][0]
    assert "confidence" not in json.dumps(state)
    assert state["conditions"] == {
        "direction": "SHORT",
        "regime": "BEARISH",
        "adx_bucket": "HIGH_ADX",
    }
    assert state["indicators"] == {"adx": 43.18, "rsi": 32.98, "cmf": -0.17}
    assert len(state["candles"]) == 3
    assert state["candles"][0] == [73010.0, 73480.0, 72800.0, 73152.0, 1200.0]
    assert state["decision"]["reasoning"] == ANALYSIS["reasoning"]


async def test_client_posts_instructions_and_reads_the_usage():

    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers["authorization"]
        return httpx.Response(200, json=answers())

    client = DecisionsClient(
        url="https://example.test/decisions",
        model="openai/gpt-6-luna-decisions",
        api_key="secret",
        timeout_seconds=5,
        transport=httpx.MockTransport(handler),
    )

    body = await client.ask({"symbol": "BTC/USDC"}, {"quality": {"type": "score", "instructions": "x", "criteria": ["a"]}})

    assert captured["auth"] == "Bearer secret"
    assert captured["body"]["model"] == "openai/gpt-6-luna-decisions"
    assert captured["body"]["state"] == {"symbol": "BTC/USDC"}
    assert body["answers"]["quality"]["score"] == 2.0


async def test_client_raises_on_an_error_body():

    def handler(request):
        return httpx.Response(
            400,
            json={"error": {"message": "OpenAI Decisions needs instructions on every question"}},
        )

    client = DecisionsClient(
        url="https://example.test/decisions",
        model="m",
        api_key="secret",
        timeout_seconds=5,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(RuntimeError, match="400"):
        await client.ask({}, {})


async def test_a_response_without_a_quality_answer_is_an_error(tmp_path):

    payload = answers()
    payload["answers"]["quality"] = {"type": "score"}
    gate, logger = make_gate(tmp_path, client=StubClient(payload))

    verdict = await gate.review({"analysis": dict(ANALYSIS)}, CONTEXT)

    assert verdict.error is not None
    logger.warning.assert_called_once()


def test_the_ini_parser_reads_off_as_a_boolean_and_the_mode_survives_it():

    from src.config.loader import Config

    assert Config._normalize_gate_mode(False) == "off"
    assert Config._normalize_gate_mode("shadow") == "shadow"
    assert Config._normalize_gate_mode(" SIZE ") == "size"
    with pytest.raises(ValueError, match="off, shadow or size"):
        Config._normalize_gate_mode(True)


def test_bad_mode_and_missing_key_are_refused(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        make_gate(tmp_path, mode="loud")

    empty_key = StubClient()
    empty_key.api_key = ""
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        DecisionGate(
            logger=MagicMock(),
            client=empty_key,
            mode="shadow",
            min_confidence_pct=20,
            journal_path=tmp_path / "j.jsonl",
        )
