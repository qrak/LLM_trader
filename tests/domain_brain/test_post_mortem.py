"""Dense domain tests for the closed-trade post-mortem subsystem.

Covers the PostMortemResult schema contract, PostMortemService.analyze_closed_trade
(success, degradation and prompt contract) and the SQLite + FTS5 repository.
"""

import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from src.managers.post_mortem_repository import PostMortemRepository
from src.trading.post_mortem import (
    POST_MORTEM_SYSTEM_PROMPT,
    PostMortemResult,
    PostMortemService,
)

VALID_RESPONSE = (
    '{"verdict": "good_exit", "llm_analysis": "Trade followed plan.", '
    '"expected_vs_actual": "Expected 75k, hit 71.2k stop.", '
    '"lesson_learned": "When trend reverses, honor the stop."}'
)

ENTRY_TIME = datetime(2026, 6, 17, 8, 0, 0, tzinfo=timezone.utc)
EXIT_TIME = ENTRY_TIME + timedelta(hours=20)

POST_MORTEM_PAYLOAD = {
    "trade_id": 1,
    "symbol": "BTC/USDC",
    "direction": "LONG",
    "verdict": "overestimated_breakout",
    "llm_analysis": "Price rejected resistance twice before dropping. Entry was premature.",
    "expected_vs_actual": "Expected breakout above 72.5k, actual rejection and -3% drop.",
    "lesson_learned": "When price rejects a level twice, wait for confirmation before entering.",
    "pnl_pct": -3.2,
    "close_reason": "stop_loss",
}


def valid_payload(**overrides) -> dict:
    """Minimal valid PostMortemResult payload with targeted overrides."""
    payload = {
        "verdict": "overestimated_breakout",
        "llm_analysis": "Price rejected resistance twice before dropping. Entry was premature.",
        "expected_vs_actual": "Expected breakout above 72.5k, actual rejection and -3% drop.",
        "lesson_learned": "When price rejects a level twice, wait for confirmation before entering.",
    }
    payload.update(overrides)
    return payload


def closed_position(**overrides) -> MagicMock:
    """Closed-position double carrying the fields the post-mortem prompt reads."""
    position = MagicMock()
    values = {
        "symbol": "BTC/USDC",
        "direction": "LONG",
        "entry_price": 72500.0,
        "stop_loss": 71200.0,
        "take_profit": 75000.0,
        "size_pct": 0.05,
        "confidence": "MEDIUM",
        "adx_at_entry": 28.0,
        "rsi_at_entry": 58.0,
        "trend_direction_at_entry": "BULLISH",
        "volatility_level": "MEDIUM",
        "rr_ratio_at_entry": 1.9,
        "max_drawdown_pct": -2.8,
        "max_profit_pct": 1.5,
        "entry_time": ENTRY_TIME,
    }
    values.update(overrides)
    for key, value in values.items():
        setattr(position, key, value)
    return position


def decision(reasoning: str, price: float, timestamp: datetime) -> MagicMock:
    """Entry/exit decision double."""
    double = MagicMock()
    double.reasoning = reasoning
    double.price = price
    double.timestamp = timestamp
    return double


def make_service(manager=None, parser=None, repository=None) -> PostMortemService:
    """PostMortemService wired to doubles (or to a real repository)."""
    service = PostMortemService(
        logger=MagicMock(),
        model_manager=manager or MagicMock(),
        unified_parser=parser or MagicMock(),
        repository=repository or MagicMock(),
    )
    return service


def make_repository(tmp_path, name: str = "trade_history.db") -> PostMortemRepository:
    """Real repository on a temp file."""
    return PostMortemRepository(logger=MagicMock(), db_path=str(tmp_path / name))


def test_post_mortem_result_rejects_empty_and_missing_required_fields():
    for field in ("verdict", "llm_analysis", "expected_vs_actual", "lesson_learned"):
        for bad_value in ("", None):
            with pytest.raises(ValidationError):
                PostMortemResult(**valid_payload(**{field: bad_value}))
        payload = valid_payload()
        del payload[field]
        with pytest.raises(ValidationError):
            PostMortemResult(**payload)


def test_post_mortem_result_accepts_valid_and_minimal_payloads():
    result = PostMortemResult(**valid_payload())
    assert result.verdict == "overestimated_breakout"
    assert result.llm_analysis
    assert result.expected_vs_actual
    assert result.lesson_learned

    minimal = PostMortemResult(verdict="a", llm_analysis="b", expected_vs_actual="c", lesson_learned="d")
    assert (minimal.verdict, minimal.llm_analysis, minimal.expected_vs_actual, minimal.lesson_learned) == (
        "a",
        "b",
        "c",
        "d",
    )

    for verdict in ("overestimated_breakout", "good_exit", "plan_followed", "premature_entry", "held_too_long"):
        assert PostMortemResult(**valid_payload(verdict=verdict)).verdict == verdict


def test_post_mortem_result_drops_unknown_fields():
    result = PostMortemResult(**valid_payload(extra_field="should be ignored"))

    assert "extra_field" not in result.model_dump()
    assert result.verdict == "overestimated_breakout"


@pytest.mark.parametrize(
    ("response", "llm_error", "repo_error", "expected_log"),
    [
        ("", None, False, "Post-mortem: empty LLM response"),
        ("not json at all", None, False, "Post-mortem: failed to parse LLM response"),
        (
            '{"llm_analysis": "ok", "expected_vs_actual": "ok", "lesson_learned": "ok"}',
            None,
            False,
            "Post-mortem: failed to parse LLM response",
        ),
        (None, RuntimeError("API timeout"), False, "Post-mortem analysis failed"),
        (VALID_RESPONSE, None, True, "Post-mortem analysis failed"),
    ],
    ids=["empty-response", "malformed-json", "missing-required-field", "llm-exception", "repository-exception"],
)
async def test_analyze_closed_trade_degrades_to_none(response, llm_error, repo_error, expected_log):
    manager = MagicMock()
    manager.send_prompt = AsyncMock(return_value=response, side_effect=llm_error)
    parser = MagicMock()
    parser.extract_json_block.return_value = None
    repository = MagicMock()
    if repo_error:
        repository.insert_post_mortem.side_effect = RuntimeError("DB locked")
    service = make_service(manager=manager, parser=parser, repository=repository)

    result = await service.analyze_closed_trade(
        closed_position=closed_position(),
        entry_decision=decision("Expected bullish continuation based on ADX strength.", 72500.0, ENTRY_TIME),
        exit_decision=decision("Stop loss hit as price broke below support.", 71200.0, EXIT_TIME),
        pnl=-1.79,
        reason="stop_loss",
    )

    assert result is None
    if not repo_error:
        repository.insert_post_mortem.assert_not_called()
    assert any(call.args[0] == expected_log for call in service.logger.warning.call_args_list)


@pytest.mark.parametrize("trade_id", [42, None], ids=["explicit-trade-id", "legacy-none"])
async def test_analyze_closed_trade_stores_result_and_forwards_trade_id(trade_id):
    manager = MagicMock()
    manager.send_prompt = AsyncMock(return_value=VALID_RESPONSE)
    parser = MagicMock()
    parser.extract_json_block.return_value = None
    repository = MagicMock()
    repository.insert_post_mortem.return_value = 1
    service = make_service(manager=manager, parser=parser, repository=repository)

    result = await service.analyze_closed_trade(
        closed_position=closed_position(),
        entry_decision=decision("Expected bullish continuation based on ADX strength.", 72500.0, ENTRY_TIME),
        exit_decision=decision("Stop loss hit as price broke below support.", 71200.0, EXIT_TIME),
        pnl=-1.79,
        reason="stop_loss",
        trade_id=trade_id,
    )

    assert type(result) is PostMortemResult
    assert result is not None
    assert result.verdict == "good_exit"
    assert result.lesson_learned == "When trend reverses, honor the stop."
    stored = repository.insert_post_mortem.call_args.kwargs
    assert stored["trade_id"] == trade_id
    assert stored["symbol"] == "BTC/USDC"
    assert stored["direction"] == "LONG"
    assert stored["pnl_pct"] == -1.79
    assert stored["close_reason"] == "stop_loss"


async def test_analyze_closed_trade_prompt_carries_full_trade_context():
    manager = MagicMock()
    manager.send_prompt = AsyncMock(return_value=VALID_RESPONSE)
    service = make_service(manager=manager, parser=MagicMock(extract_json_block=MagicMock(return_value=None)))

    await service.analyze_closed_trade(
        closed_position=closed_position(),
        entry_decision=decision("Expected bullish continuation based on ADX strength.", 72500.0, ENTRY_TIME),
        exit_decision=decision("Stop loss hit as price broke below support.", 71200.0, EXIT_TIME),
        pnl=-1.79,
        reason="stop_loss",
    )

    call = manager.send_prompt.call_args.kwargs
    assert call["system_message"] == POST_MORTEM_SYSTEM_PROMPT
    prompt = call["prompt"]
    assert "## Trade: BTC/USDC LONG" in prompt
    assert "## Close Reason: stop_loss" in prompt
    assert "## P&L: -1.79%" in prompt
    assert "- Position Size: 5.0% of capital" in prompt
    assert "- Max Drawdown During Trade: -2.80%" in prompt
    assert "- Max Profit During Trade: 1.50%" in prompt
    assert "## Original Entry Reasoning:" in prompt
    assert "Expected bullish continuation based on ADX strength." in prompt
    assert "## Exit Data:" in prompt
    assert "- Exit Reasoning: Stop loss hit as price broke below support." in prompt
    assert "- Hold Duration: 20:00:00" in prompt
    assert "Produce the JSON post-mortem now." in prompt


async def test_analyze_closed_trade_prompt_tolerates_missing_reasoning_and_timestamps():
    manager = MagicMock()
    manager.send_prompt = AsyncMock(return_value=VALID_RESPONSE)
    service = make_service(manager=manager, parser=MagicMock(extract_json_block=MagicMock(return_value=None)))

    await service.analyze_closed_trade(
        closed_position=closed_position(entry_time=None),
        entry_decision=decision("", 72500.0, ENTRY_TIME),
        exit_decision=decision("Stop hit.", 71200.0, EXIT_TIME),
        pnl=-1.79,
        reason="stop_loss",
    )

    prompt = manager.send_prompt.call_args.kwargs["prompt"]
    assert "## Original Entry Reasoning:\n(no reasoning recorded)" in prompt
    assert "Hold Duration" not in prompt


async def test_analyze_closed_trade_prompt_includes_market_conditions_when_present():
    manager = MagicMock()
    manager.send_prompt = AsyncMock(return_value=VALID_RESPONSE)
    service = make_service(manager=manager, parser=MagicMock(extract_json_block=MagicMock(return_value=None)))
    conditions = MagicMock()
    conditions.__str__ = MagicMock(return_value="MarketConditions(rsi=41.0)")

    await service.analyze_closed_trade(
        closed_position=closed_position(),
        entry_decision=decision("Entry reasoning.", 72500.0, ENTRY_TIME),
        exit_decision=decision("Stop hit.", 71200.0, EXIT_TIME),
        pnl=-1.79,
        reason="stop_loss",
        market_conditions=conditions,
    )

    prompt = manager.send_prompt.call_args.kwargs["prompt"]
    assert "## Market Conditions at Exit:" in prompt
    assert "MarketConditions(rsi=41.0)" in prompt


async def test_analyze_closed_trade_persists_into_real_repository(tmp_path):
    repository = make_repository(tmp_path)
    manager = MagicMock()
    manager.send_prompt = AsyncMock(return_value=VALID_RESPONSE)
    service = make_service(
        manager=manager,
        parser=MagicMock(extract_json_block=MagicMock(return_value=None)),
        repository=repository,
    )

    result = await service.analyze_closed_trade(
        closed_position=closed_position(direction="SHORT"),
        entry_decision=decision("Short into resistance.", 72500.0, ENTRY_TIME),
        exit_decision=decision("Stop hit.", 71200.0, EXIT_TIME),
        pnl=-1.95,
        reason="stop_loss",
        trade_id=7,
    )

    stored = repository.get_recent_post_mortems()
    assert result is not None
    assert len(stored) == 1
    assert stored[0]["direction"] == "SHORT"
    assert stored[0]["verdict"] == "good_exit"
    assert repository.search_post_mortems("honor")[0]["trade_id"] == 7


def test_repository_round_trip_search_and_ordering(tmp_path):
    repository = make_repository(tmp_path)

    assert repository.get_recent_post_mortems() == []
    assert repository.insert_post_mortem(**POST_MORTEM_PAYLOAD) == 1
    repository.insert_post_mortem(**{**POST_MORTEM_PAYLOAD, "trade_id": 2, "verdict": "second", "pnl_pct": -2.0})

    reopened = PostMortemRepository(logger=MagicMock(), db_path=str(tmp_path / "trade_history.db"))
    records = reopened.get_recent_post_mortems(limit=5)
    assert len(records) == 2
    record = next(row for row in records if row["verdict"] == "overestimated_breakout")
    assert record["symbol"] == "BTC/USDC"
    assert record["direction"] == "LONG"
    assert record["pnl_pct"] == -3.2
    assert record["close_reason"] == "stop_loss"
    assert record["lesson_learned"]
    assert record["llm_analysis"]
    assert record["created_at"]
    assert record["id"] == 1
    assert "trade_id" not in record

    found = repository.search_post_mortems("resistance")
    assert len(found) == 2
    assert found[0]["trade_id"] in (1, 2)
    assert "llm_analysis" in found[0]
    assert "expected_vs_actual" in found[0]
    assert "rank" in found[0]
    assert repository.search_post_mortems("nonexistent_term_xyz") == []
    assert repository.search_post_mortems("invalid^^^query!!!") == []
    assert repository.search_post_mortems("confirm*") != []
    assert repository.search_post_mortems("breakout") == []


def test_repository_orders_by_created_at_desc(tmp_path):
    repository = make_repository(tmp_path)
    repository.insert_post_mortem(**{**POST_MORTEM_PAYLOAD, "trade_id": 1, "verdict": "first"})
    repository.insert_post_mortem(**{**POST_MORTEM_PAYLOAD, "trade_id": 2, "verdict": "second"})

    connection = sqlite3.connect(str(tmp_path / "trade_history.db"))
    try:
        connection.execute(
            "UPDATE trade_post_mortem SET created_at = '2026-06-17 12:00:00' WHERE verdict = 'first'"
        )
        connection.execute(
            "UPDATE trade_post_mortem SET created_at = '2026-06-18 12:00:00' WHERE verdict = 'second'"
        )
        connection.commit()
    finally:
        connection.close()

    recent = repository.get_recent_post_mortems(limit=5)
    assert [row["verdict"] for row in recent] == ["second", "first"]
    assert repository.get_recent_post_mortems(limit=1)[0]["verdict"] == "second"
    assert repository.get_recent_post_mortems(limit=0) == []


def test_repository_accepts_nullable_columns(tmp_path):
    repository = make_repository(tmp_path)

    repository.insert_post_mortem(
        **{**POST_MORTEM_PAYLOAD, "trade_id": None, "direction": None, "expected_vs_actual": None, "pnl_pct": None}
    )

    record = repository.get_recent_post_mortems()[0]
    assert record["id"] == 1


def test_repository_write_fails_loudly_on_readonly_database(tmp_path):
    repository = make_repository(tmp_path)
    repository.insert_post_mortem(**POST_MORTEM_PAYLOAD)
    database_file = tmp_path / "trade_history.db"
    database_file.chmod(0o444)

    assert len(repository.get_recent_post_mortems()) == 1
    with pytest.raises(sqlite3.OperationalError):
        repository.insert_post_mortem(**{**POST_MORTEM_PAYLOAD, "trade_id": 99})


def test_repository_construction_fails_loudly_on_unusable_path(tmp_path):
    with pytest.raises(sqlite3.OperationalError):
        PostMortemRepository(logger=MagicMock(), db_path=str(tmp_path / "missing_dir" / "trade_history.db"))
