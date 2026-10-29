

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from src.managers.post_mortem_repository import PostMortemRepository
from src.parsing.unified_parser import UnifiedParser
from src.trading.post_mortem import PostMortemResult, PostMortemService

ENTRY_TIME = datetime(2026, 6, 17, 8, 0, 0, tzinfo=timezone.utc)
EXIT_TIME = ENTRY_TIME + timedelta(hours=16)

LOGGED_RESPONSE = (
    "{\n"
    '  "verdict": "plan_followed",\n'
    '  "llm_analysis": "The trade captured the anticipated impulse continuation out of '
    "consolidation, reaching the 84,500 TP target within 16 hours. Institutional accumulation "
    "and directional strength pushed price smoothly upward without triggering any drawdown "
    'against the position.",\n'
    '  "expected_vs_actual": "Expected a bullish continuation toward an 84,500 measured impulse '
    "target with a 2.0 R/R; actual price action delivered a clean trend run directly to the "
    'limit order without adverse movement.",\n'
    '  "lesson_learned": "When entering a trend continuation backed by expanding ADX and '
    "positive volume accumulation, set predefined limit take-profits at structural extensions "
    'to lock in gains before\n momentum exhausts into overbought territory."\n'
    "}"
)

LOGGED_RESPONSE_WITH_CRLF = LOGGED_RESPONSE.replace("before\n momentum", "before\r\n momentum")

TRUNCATED_RESPONSE = '{"verdict": "plan_followed", "llm_analysis": "cut off in the middle'

SIMPLE_RESPONSE = (
    '{"verdict": "premature_entry", "llm_analysis": "Entry fired before the retest confirmed.", '
    '"expected_vs_actual": "Expected a retest hold, price broke the level instead.", '
    '"lesson_learned": "When a level is untested, wait for the retest before entering."}'
)

MISSING_VERDICT = '{"llm_analysis": "ok", "expected_vs_actual": "ok", "lesson_learned": "ok"}'


def closed_position(**overrides) -> MagicMock:

    position = MagicMock()
    values = {
        "symbol": "BTC/USDC",
        "direction": "LONG",
        "entry_price": 72500.0,
        "stop_loss": 71200.0,
        "take_profit": 84500.0,
        "size_pct": 0.05,
        "confidence": "HIGH",
        "adx_at_entry": 40.9,
        "rsi_at_entry": 61.0,
        "trend_direction_at_entry": "BULLISH",
        "volatility_level": "LOW",
        "rr_ratio_at_entry": 2.0,
        "max_drawdown_pct": -0.2,
        "max_profit_pct": 4.1,
        "entry_time": ENTRY_TIME,
    }
    values.update(overrides)
    for key, value in values.items():
        setattr(position, key, value)
    return position


def decision(reasoning: str, price: float, timestamp: datetime) -> MagicMock:

    double = MagicMock()
    double.reasoning = reasoning
    double.price = price
    double.timestamp = timestamp
    return double


def make_service(response_text, repository=None, real_parser: bool = True) -> PostMortemService:

    manager = MagicMock()
    manager.send_prompt = AsyncMock(return_value=response_text)
    parser = UnifiedParser(logger=MagicMock()) if real_parser else MagicMock()
    return PostMortemService(
        logger=MagicMock(),
        model_manager=manager,
        unified_parser=parser,
        repository=repository or MagicMock(),
    )


def make_repository(tmp_path) -> PostMortemRepository:

    return PostMortemRepository(logger=MagicMock(), db_path=str(tmp_path / "trade_history.db"))


async def analyze(service: PostMortemService, trade_id: int | None = 1):

    return await service.analyze_closed_trade(
        closed_position=closed_position(),
        entry_decision=decision("Bullish continuation out of consolidation.", 72500.0, ENTRY_TIME),
        exit_decision=decision("Take profit filled.", 84500.0, EXIT_TIME),
        pnl=4.07,
        reason="take_profit_filled",
        trade_id=trade_id,
    )


@pytest.mark.parametrize(
    "response_text",
    [LOGGED_RESPONSE, LOGGED_RESPONSE_WITH_CRLF],
    ids=["logged-payload-lf", "logged-payload-crlf"],
)
def test_logged_response_with_raw_newline_in_lesson_is_parsed(response_text):
    service = make_service(response_text)

    result = service._parse_response(response_text)

    assert result is not None
    assert result.verdict == "plan_followed"
    assert result.llm_analysis.startswith("The trade captured the anticipated impulse continuation")
    assert result.expected_vs_actual.startswith("Expected a bullish continuation")
    assert result.lesson_learned.startswith("When entering a trend continuation backed by expanding ADX")
    assert result.lesson_learned.endswith("momentum exhausts into overbought territory.")
    assert "\n" in result.lesson_learned


def test_logged_response_parses_with_the_injected_parser_unavailable():
    parser = MagicMock()
    parser.extract_json_block.return_value = None
    service = make_service(LOGGED_RESPONSE, real_parser=False)
    service.unified_parser = parser

    result = service._parse_response(LOGGED_RESPONSE)

    assert result is not None
    assert result.verdict == "plan_followed"


@pytest.mark.asyncio
async def test_logged_response_is_stored_end_to_end(tmp_path):
    repository = make_repository(tmp_path)
    service = make_service(LOGGED_RESPONSE, repository)

    result = await analyze(service, trade_id=7)

    assert result is not None
    stored = repository.get_recent_post_mortems()
    assert len(stored) == 1
    assert stored[0]["verdict"] == "plan_followed"
    assert "overbought territory" in stored[0]["lesson_learned"]
    found = repository.search_post_mortems("overbought")
    assert len(found) == 1
    assert found[0]["trade_id"] == 7


@pytest.mark.asyncio
async def test_fenced_json_with_raw_newline_is_parsed():
    response = f"Here is the analysis:\n```json\n{LOGGED_RESPONSE}\n```\n"
    service = make_service(response)

    result = await analyze(service)

    assert result is not None
    assert result.verdict == "plan_followed"
    assert "overbought territory" in result.lesson_learned


@pytest.mark.asyncio
async def test_prose_around_json_is_parsed():
    response = f"Sure, here it is.\n{SIMPLE_RESPONSE}\nLet me know if you need more detail."
    service = make_service(response)

    result = await analyze(service)

    assert result is not None
    assert result.verdict == "premature_entry"
    assert result.lesson_learned.endswith("wait for the retest before entering.")


@pytest.mark.asyncio
async def test_prose_wrapped_fenced_json_with_other_fences_is_parsed():
    response = f"```text\nthinking out loud\n```\n```json\n{SIMPLE_RESPONSE}\n```\ntrailing note"
    service = make_service(response)

    result = await analyze(service)

    assert result is not None
    assert result.verdict == "premature_entry"


def test_prose_before_json_with_raw_newline_is_parsed():
    response = f"I analysed the trade.\n{LOGGED_RESPONSE}\nHope that helps."

    result = make_service(response)._parse_response(response)

    assert result is not None
    assert result.verdict == "plan_followed"


def test_raw_tab_inside_string_value_is_parsed():
    response = SIMPLE_RESPONSE.replace("wait for the retest", "wait\tfor the retest")

    result = make_service(response)._parse_response(response)

    assert result is not None
    assert "wait\tfor the retest" in result.lesson_learned


def test_repair_leaves_structural_whitespace_untouched():
    source = '{\n  "verdict": "good_exit",\n  "llm_analysis": "ok"\n}'

    repaired = UnifiedParser.repair_raw_control_characters(source)

    assert repaired == source
    assert UnifiedParser.parse_json_object(source) == {"verdict": "good_exit", "llm_analysis": "ok"}


def test_repair_keeps_escaped_quotes_and_backslashes_intact():
    source = '{"verdict": "good_exit", "llm_analysis": "a \\"quote\\" and a \\\\ slash and \\n newline"}'

    repaired = UnifiedParser.repair_raw_control_characters(source)

    assert repaired == source


@pytest.mark.parametrize(
    "response_text",
    [
        TRUNCATED_RESPONSE,
        '{"verdict": "plan_followed", "llm_analysis": "ok", "expected_vs_actual": "ok",',
        '{"verdict": plan_followed, "llm_analysis": "ok"}',
        "no json here at all",
        "",
        "   \n  ",
        None,
    ],
    ids=["truncated", "truncated-mid-pair", "unquoted-verdict", "prose-only", "empty", "whitespace", "none"],
)
def test_unrecoverable_response_returns_none_without_raising(response_text):
    service = make_service(response_text)

    assert service._parse_response(response_text) is None


@pytest.mark.parametrize(
    "payload",
    [
        MISSING_VERDICT,
        '{"verdict": "plan followed", "llm_analysis": "ok", "expected_vs_actual": "ok", "lesson_learned": "ok"}',
        '{"verdict": "Plan_Followed", "llm_analysis": "ok", "expected_vs_actual": "ok", "lesson_learned": "ok"}',
        '{"verdict": "plan_followed", "llm_analysis": "   ", "expected_vs_actual": "ok", "lesson_learned": "ok"}',
        '{"verdict": "", "llm_analysis": "ok", "expected_vs_actual": "ok", "lesson_learned": "ok"}',
        '{"verdict": "plan_followed", "llm_analysis": null, "expected_vs_actual": "ok", "lesson_learned": "ok"}',
    ],
    ids=[
        "missing-verdict",
        "verdict-with-space",
        "verdict-not-snake-case",
        "blank-analysis",
        "empty-verdict",
        "null-analysis",
    ],
)
def test_hard_validation_rejects_non_contract_payloads(payload):
    service = make_service(payload)

    assert service._parse_response(payload) is None


def test_blank_oversized_and_partial_payloads_are_rejected_not_filled_in():
    valid = {
        "verdict": "plan_followed",
        "llm_analysis": "ok",
        "expected_vs_actual": "ok",
        "lesson_learned": "ok",
    }

    with pytest.raises(ValidationError):
        PostMortemResult(**{**valid, "llm_analysis": "   "})
    with pytest.raises(ValidationError):
        PostMortemResult(**{**valid, "lesson_learned": "x" * 4001})
    with pytest.raises(ValidationError):
        PostMortemResult(**{**valid, "verdict": "plan followed"})


@pytest.mark.asyncio
async def test_failed_parse_stores_nothing_in_the_post_mortem_repository(tmp_path):
    repository = make_repository(tmp_path)
    service = make_service(TRUNCATED_RESPONSE, repository)

    result = await analyze(service)

    assert result is None
    assert repository.get_recent_post_mortems() == []
    assert repository.search_post_mortems("plan_followed") == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [TRUNCATED_RESPONSE, MISSING_VERDICT, "no json here at all"],
    ids=["truncated", "missing-verdict", "prose-only"],
)
async def test_no_repository_write_and_single_warning_on_parse_failure(payload):
    repository = MagicMock()
    service = make_service(payload, repository)

    result = await analyze(service)

    assert result is None
    repository.insert_post_mortem.assert_not_called()
    warnings = service.logger.warning.call_args_list
    assert len(warnings) == 1
    assert warnings[0].args[0] == "Post-mortem: failed to parse LLM response (reason=%s) | preview: %s"
    reason = warnings[0].args[1]
    preview = warnings[0].args[2]
    assert reason
    assert len(preview) <= 203
    assert "\n" not in preview
    assert "Analyze the following closed trade" not in preview


@pytest.mark.asyncio
async def test_parse_failure_touches_no_repository_method():
    repository = MagicMock()
    service = make_service(TRUNCATED_RESPONSE, repository)

    result = await analyze(service)

    assert result is None
    assert repository.method_calls == []


def test_parser_helpers_reject_non_json_input():
    assert UnifiedParser.parse_json_object("") is None
    assert UnifiedParser.parse_json_object("just prose") is None
    assert UnifiedParser.parse_json_object(None) is None


def test_parser_helpers_ignore_earlier_broken_block_and_use_last_parseable():
    response = f"```json\n{{not json}}\n```\n{SIMPLE_RESPONSE}"

    assert UnifiedParser.parse_json_object(response)["verdict"] == "premature_entry"
