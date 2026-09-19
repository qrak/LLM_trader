"""Executor decision bridge, handler wire contract, entry rollback and position persistence.

Covers ExecutorHandler payload building / HTTP forward / dead-letter replay, the
TradingStrategy executor handshake (position probe, verdict journal, phantom rollback),
PersistenceManager position and decision round trips, and the blocked-trade store.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import chromadb
import pytest
from sentence_transformers import SentenceTransformer

from src.managers.persistence_manager import PersistenceManager
from src.trading import executor_handler
from src.trading.data_models import MarketConditions, Position, TradeDecision
from src.trading.executor_handler import ACTIONABLE_SIGNALS, ExecutorHandler
from src.trading.trading_strategy import TradingStrategy
from src.trading.vector_memory import VectorMemoryService
from tests.conftest import (
    make_config,
    make_market_conditions,
    make_position,
    mock_brain,
    mock_persistence,
    mock_statistics,
    null_logger,
)

EXECUTOR_URL = "http://127.0.0.1:9199/decision"
POSITION_URL = "http://127.0.0.1:9199/position"
FORWARD_PAYLOAD: dict[str, Any] = {"timestamp": "2026-07-23T12:00:00.000000", "symbol": "BTC/USDC", "signal": "BUY"}


def _handler(**config_overrides: Any) -> tuple[ExecutorHandler, MagicMock]:
    """ExecutorHandler on a config double with the executor enabled, plus its persistence double."""
    config = make_config(**{"EXECUTOR_API_ENABLED": True, "EXECUTOR_API_URL": EXECUTOR_URL, **config_overrides})
    persistence = mock_persistence()
    return ExecutorHandler(persistence=persistence, config=config, logger=null_logger()), persistence


def _decision(**overrides: Any) -> TradeDecision:
    """TradeDecision carrying a full executor payload unless overridden."""
    values: dict[str, Any] = {
        "timestamp": datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc),
        "symbol": "BTC/USDC",
        "action": "BUY",
        "confidence": "HIGH",
        "price": 70000.0,
        "stop_loss": 68000.0,
        "take_profit": 74000.0,
        "quantity": 0.15,
        "reasoning": "Breakout confirmation",
    }
    values.update(overrides)
    return TradeDecision(**values)


def _wire(payload: dict[str, Any]) -> dict[str, Any]:
    """Payload without its wall-clock timestamp, for exact dict comparisons."""
    return {key: value for key, value in payload.items() if key != "timestamp"}


def _response(status_code: int, text: str = "executor down") -> MagicMock:
    """Canned httpx response double."""
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    return response


def _http_client(status_code: int | None = 200, exc: Exception | None = None) -> MagicMock:
    """httpx.AsyncClient double whose POST returns one canned response or raises."""
    client = MagicMock()
    client.is_closed = False
    client.aclose = AsyncMock()
    if exc is not None:
        client.post = AsyncMock(side_effect=exc)
    else:
        client.post = AsyncMock(return_value=_response(status_code))
    return client


def _strategy(
    position: Position | None = None, **config_overrides: Any
) -> tuple[TradingStrategy, MagicMock, MagicMock]:
    """Real TradingStrategy on the shared doubles; returns strategy, logger and persistence."""
    logger = null_logger()
    persistence = mock_persistence()
    statistics = mock_statistics()
    statistics.get_current_capital.return_value = 10000.0
    brain = mock_brain()
    brain.get_dynamic_thresholds.return_value = {"rr_borderline_min": 1.5}
    strategy = TradingStrategy(
        logger=logger,
        persistence=persistence,
        brain_service=brain,
        statistics_service=statistics,
        memory_service=MagicMock(),
        risk_manager=MagicMock(),
        config=make_config(**config_overrides),
        position_extractor=MagicMock(),
    )
    if position is not None:
        strategy.current_position = position
    return strategy, logger, persistence


def _probe_stub(strategy: TradingStrategy, status_code: int = 200, body: Any = None, exc: Exception | None = None) -> MagicMock:
    """Replace the strategy HTTP client factory with one canned position probe."""
    client = MagicMock()
    response = _response(status_code)
    response.json.return_value = {} if body is None else body
    client.get = AsyncMock(side_effect=exc) if exc is not None else AsyncMock(return_value=response)
    strategy._get_http_client = MagicMock(return_value=client)
    return client


def _confirm_window(attempts: int, min_false_reports: int | None = None, delay: float = 0.001) -> ExitStack:
    """Patch the entry-confirmation poll window: attempts, false-report floor, delay."""
    stack = ExitStack()
    stack.enter_context(patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_ATTEMPTS", attempts))
    stack.enter_context(patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_DELAY", delay))
    if min_false_reports is not None:
        stack.enter_context(
            patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_MIN_FALSE_REPORTS", min_false_reports)
        )
    return stack


def _journal(tmp_path: Path, lines: list[str]) -> Path:
    """Executor verdict journal holding the given raw lines."""
    path = tmp_path / "executor_verdicts.jsonl"
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def _verdict_config(path: Path) -> SimpleNamespace:
    """Config double carrying only the executor verdict journal path."""
    return SimpleNamespace(EXECUTOR_VERDICT_PATH=str(path))


def _store_block(service: VectorMemoryService, **overrides: Any) -> bool:
    """Store one blocked-trade rejection event with the full guard payload."""
    values: dict[str, Any] = {
        "guard_type": "rr_minimum",
        "direction": "LONG",
        "confidence": "HIGH",
        "suggested_rr": 1.2,
        "required_rr": 2.0,
        "suggested_sl_pct": 0.03,
        "suggested_tp_pct": 0.04,
        "suggested_sl": 97.0,
        "suggested_tp": 104.0,
        "current_price": 100.0,
        "volatility_level": "MEDIUM",
        "reasoning_snippet": "Test reasoning for LLM feedback",
        "metadata": None,
    }
    values.update(overrides)
    return service.store_blocked_trade(**values)


@pytest.fixture(scope="module")
def embedding_model() -> SentenceTransformer:
    """Shared MiniLM encoder for the blocked-trade store tests."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture
def blocked_memory(embedding_model: SentenceTransformer) -> VectorMemoryService:
    """Empty VectorMemoryService on an ephemeral ChromaDB client."""
    client = chromadb.Client(
        chromadb.config.Settings(anonymized_telemetry=False, allow_reset=True, is_persistent=False)
    )
    client.reset()
    service = VectorMemoryService(
        logger=null_logger(),
        chroma_client=client,
        embedding_model=embedding_model,
        timeframe_minutes=240,
    )
    assert service._ensure_initialized() is True
    return service


class TestExecutorPayload:
    """ExecutorHandler._build: suppression, exact wire schema and LLM type coercion."""

    @pytest.mark.parametrize(
        ("analysis", "decision_overrides", "symbol"),
        [
            pytest.param(None, {}, "BTC/USDC", id="no_analysis"),
            pytest.param({}, {}, "", id="empty_symbol"),
            pytest.param({"signal": "BUY"}, None, "BTC/USDC", id="no_decision"),
            pytest.param({"signal": "HOLD"}, {"action": "HOLD"}, "BTC/USDC", id="hold_signal"),
            pytest.param({"signal": "BUY"}, {"action": "HOLD"}, "BTC/USDC", id="hold_action"),
        ],
    )
    def test_build_suppresses_unusable_inputs(
        self, analysis: dict[str, Any] | None, decision_overrides: dict[str, Any] | None, symbol: str
    ) -> None:
        handler, _ = _handler()
        decision = None if decision_overrides is None else _decision(**decision_overrides)
        assert handler._build(analysis, decision, symbol) is None

    @pytest.mark.parametrize(
        ("config_overrides", "analysis", "decision_overrides", "symbol", "expected", "expected_types"),
        [
            pytest.param(
                {},
                {"signal": "BUY", "reduce_only": False, "leverage": 1},
                {"order_id": "order-20260811160128949603"},
                "BTC/USDC",
                {
                    "symbol": "BTC/USDC",
                    "signal": "BUY",
                    "order_type": "market",
                    "order_id": "order-20260811160128949603",
                    "quantity": 0.15,
                    "entry_price": 70000.0,
                    "stop_loss": 68000.0,
                    "take_profit": 74000.0,
                    "reduce_only": False,
                    "leverage": 1,
                    "confidence": "HIGH",
                    "reasoning": "Breakout confirmation",
                },
                {},
                id="decision_driven",
            ),
            pytest.param(
                {"ENTRY_ORDER_TYPE": "limit"},
                {"signal": "SELL", "reduce_only": "false", "leverage": "3"},
                {
                    "action": "SELL",
                    "quantity": "0.05",
                    "price": "68000.0",
                    "stop_loss": "66000.0",
                    "take_profit": "72000.0",
                    "reasoning": "Strong technical setup",
                },
                "ETH/USDC",
                {
                    "symbol": "ETH/USDC",
                    "signal": "SELL",
                    "order_type": "limit",
                    "order_id": None,
                    "quantity": 0.05,
                    "entry_price": 68000.0,
                    "stop_loss": 66000.0,
                    "take_profit": 72000.0,
                    "reduce_only": False,
                    "leverage": 3,
                    "confidence": "HIGH",
                    "reasoning": "Strong technical setup",
                },
                {
                    "symbol": str,
                    "signal": str,
                    "order_type": str,
                    "quantity": float,
                    "entry_price": float,
                    "stop_loss": float,
                    "take_profit": float,
                    "reduce_only": bool,
                    "leverage": int,
                    "confidence": str,
                    "reasoning": str,
                },
                id="string_payload",
            ),
            pytest.param(
                {},
                {
                    "signal": "BUY",
                    "quantity": 0.5,
                    "entry_price": 50000.0,
                    "stop_loss": 49000.0,
                    "take_profit": 52000.0,
                },
                {
                    "quantity": None,
                    "price": None,
                    "stop_loss": None,
                    "take_profit": None,
                    "reasoning": None,
                    "confidence": None,
                },
                "ETH/USDC",
                {
                    "symbol": "ETH/USDC",
                    "signal": "BUY",
                    "order_type": "market",
                    "order_id": None,
                    "quantity": 0.5,
                    "entry_price": 50000.0,
                    "stop_loss": 49000.0,
                    "take_profit": 52000.0,
                    "reduce_only": False,
                    "leverage": 1,
                    "confidence": "MEDIUM",
                    "reasoning": "",
                },
                {},
                id="analysis_fallback",
            ),
        ],
    )
    def test_build_wire_payload_matrix(
        self,
        config_overrides: dict[str, Any],
        analysis: dict[str, Any],
        decision_overrides: dict[str, Any],
        symbol: str,
        expected: dict[str, Any],
        expected_types: dict[str, type],
    ) -> None:
        """The payload carries exactly the executor schema, no key more, no key less."""
        handler, _ = _handler(**config_overrides)
        payload = handler._build(analysis, _decision(**decision_overrides), symbol)

        assert _wire(payload) == expected
        for key, expected_type in expected_types.items():
            assert type(payload[key]) is expected_type
        stamp = datetime.strptime(payload["timestamp"], "%Y-%m-%dT%H:%M:%S.%f").replace(
            tzinfo=timezone.utc
        )
        elapsed = (datetime.now(timezone.utc) - stamp).total_seconds()
        assert 0 <= elapsed < 60

    @pytest.mark.parametrize(
        ("decision_overrides", "expected"),
        [
            pytest.param(
                {"price": float("inf"), "stop_loss": float("-inf"), "take_profit": None, "quantity": float("nan")},
                {"quantity": 0.0, "entry_price": None, "stop_loss": None, "take_profit": None},
                id="non_finite",
            ),
            pytest.param(
                {"quantity": "abc", "price": "abc", "stop_loss": "abc", "take_profit": "abc"},
                {"quantity": 0.0, "entry_price": None, "stop_loss": None, "take_profit": None},
                id="unparseable",
            ),
            pytest.param(
                {"quantity": -0.5},
                {"quantity": -0.5, "entry_price": 70000.0, "stop_loss": 68000.0, "take_profit": 74000.0},
                id="negative_quantity_unclamped",
            ),
        ],
    )
    def test_build_numeric_guards(
        self, decision_overrides: dict[str, Any], expected: dict[str, Any]
    ) -> None:
        """Non-finite and unparseable numbers collapse to the field defaults; negatives pass through."""
        handler, _ = _handler()
        payload = handler._build({"signal": "BUY"}, _decision(**decision_overrides), "BTC/USDC")
        assert {key: payload[key] for key in expected} == expected

    def test_build_order_id_is_stringified_or_none(self) -> None:
        """A falsy order_id must reach the executor as None, never as the string \"None\"."""
        handler, _ = _handler()
        carried = handler._build({"signal": "SELL"}, _decision(order_id="order-20260811160128949603"), "BTC/USDC")
        assert carried["order_id"] == "order-20260811160128949603"
        assert handler._build({"signal": "BUY"}, _decision(order_id=None), "BTC/USDC")["order_id"] is None
        assert handler._build({"signal": "BUY"}, _decision(order_id=""), "BTC/USDC")["order_id"] is None

    @pytest.mark.parametrize(
        ("reduce_only", "leverage", "expected_reduce_only", "expected_leverage"),
        [
            pytest.param("false", "5x", False, 1, id="string_false_and_unparseable_leverage"),
            pytest.param("true", "0", True, 1, id="string_true_and_below_floor_leverage"),
            pytest.param("on", "3.7", True, 3, id="token_true_and_fractional_leverage"),
            pytest.param("0", float("nan"), False, 1, id="token_false_and_non_finite_leverage"),
            pytest.param(2, 100, True, 100, id="numeric_truthy_and_large_leverage"),
        ],
    )
    def test_build_parses_llm_boolean_and_leverage_tokens(
        self, reduce_only: Any, leverage: Any, expected_reduce_only: bool, expected_leverage: int
    ) -> None:
        """An LLM-supplied \"false\" must never flip a payload into reduce-only."""
        handler, _ = _handler()
        analysis = {"signal": "BUY", "reduce_only": reduce_only, "leverage": leverage}
        payload = handler._build(analysis, _decision(), "BTC/USDC")

        assert payload["reduce_only"] is expected_reduce_only
        assert type(payload["reduce_only"]) is bool
        assert payload["leverage"] == expected_leverage
        assert type(payload["leverage"]) is int

    def test_actionable_signals_contract(self) -> None:
        """The actionable set is frozen and covers the futures signals."""
        assert type(ACTIONABLE_SIGNALS) is frozenset
        assert ACTIONABLE_SIGNALS == frozenset({"BUY", "SELL", "CLOSE", "UPDATE", "LONG", "SHORT"})


class TestDecisionBridge:
    """ExecutorHandler.handle: HTTP fast path, file fallback, dead letter, delivery report."""

    @pytest.mark.parametrize(
        ("forward_outcome", "clear_side_effect", "expected_return", "expected_persist", "expected_clear", "expected_dead_letter"),
        [
            pytest.param(True, None, True, 0, 1, 0, id="http_delivered"),
            pytest.param(False, None, False, 1, 0, 0, id="file_fallback"),
            pytest.param(RuntimeError("network down"), None, False, 1, 0, 1, id="forward_raises"),
            pytest.param(True, RuntimeError("io error"), True, 0, 1, 0, id="stale_file_cleanup_fails"),
        ],
    )
    async def test_handle_outcome_matrix(
        self,
        forward_outcome: Any,
        clear_side_effect: Exception | None,
        expected_return: bool,
        expected_persist: int,
        expected_clear: int,
        expected_dead_letter: int,
    ) -> None:
        """Delivery is reported as delivered; only failures write the fallback and dead-letter."""
        handler, persistence = _handler()
        if isinstance(forward_outcome, Exception):
            handler._forward = AsyncMock(side_effect=forward_outcome)
        else:
            handler._forward = AsyncMock(return_value=forward_outcome)
        handler._write_dead_letter = AsyncMock()
        if clear_side_effect is not None:
            persistence.clear_latest_decision.side_effect = clear_side_effect

        result = await handler.handle({"signal": "SELL"}, _decision(), "ETH/USDC")

        expected_wire = {
            "symbol": "ETH/USDC",
            "signal": "SELL",
            "order_type": "market",
            "order_id": None,
            "quantity": 0.15,
            "entry_price": 70000.0,
            "stop_loss": 68000.0,
            "take_profit": 74000.0,
            "reduce_only": False,
            "leverage": 1,
            "confidence": "HIGH",
            "reasoning": "Breakout confirmation",
        }
        payload = handler._forward.await_args.args[0]
        assert _wire(payload) == expected_wire
        assert result is expected_return
        assert persistence.save_latest_decision.call_count == expected_persist
        assert persistence.clear_latest_decision.call_count == expected_clear
        assert handler._write_dead_letter.await_count == expected_dead_letter
        if expected_persist:
            assert persistence.save_latest_decision.call_args.args[0] == payload
        if expected_dead_letter:
            assert handler._write_dead_letter.await_args.args[0] == payload
            handler.logger.error.assert_called_once_with(
                "Executor forward failed after all retries for %s %s — writing to dead-letter",
                "SELL",
                "ETH/USDC",
            )
        if clear_side_effect is not None:
            handler.logger.warning.assert_called_once_with(
                "Failed to clear stale latest_decision.json after successful forward", exc_info=True
            )

    @pytest.mark.parametrize(
        ("analysis", "decision_overrides", "symbol", "executor_enabled"),
        [
            pytest.param(None, {}, "BTC/USDC", True, id="no_analysis"),
            pytest.param({"signal": "BUY"}, {}, "", True, id="empty_symbol"),
            pytest.param({"signal": "BUY"}, {"action": "HOLD"}, "BTC/USDC", True, id="hold_decision"),
            pytest.param({"signal": "BUY"}, {}, "BTC/USDC", False, id="executor_disabled"),
        ],
    )
    async def test_handle_produces_nothing_for_unbuildable_or_disabled_input(
        self, analysis: dict[str, Any] | None, decision_overrides: dict[str, Any], symbol: str, executor_enabled: bool
    ) -> None:
        """A suppressed decision must not reach the executor through either the API or the file."""
        handler, persistence = _handler(EXECUTOR_API_ENABLED=executor_enabled)
        handler._forward = AsyncMock()

        result = await handler.handle(analysis, _decision(**decision_overrides), symbol)

        assert result is False
        assert handler._forward.await_count == 0
        assert persistence.save_latest_decision.call_count == 0
        assert persistence.clear_latest_decision.call_count == 0


class TestForwarding:
    """ExecutorHandler HTTP forward and the dead-letter journal."""

    @pytest.mark.parametrize(
        ("config_overrides", "status_code", "expected_return", "expected_level", "expected_log_args"),
        [
            pytest.param({}, 200, True, "info", ("Executor queued: %s %s", "BUY", "BTC/USDC"), id="queued"),
            pytest.param(
                {},
                503,
                False,
                "warning",
                ("Executor returned %s for %s %s: %s", 503, "BUY", "BTC/USDC", "executor down"),
                id="executor_down",
            ),
            pytest.param({}, 404, False, "warning", ("Executor returned %s for %s %s: %s", 404, "BUY", "BTC/USDC", "executor down"), id="not_found"),
            pytest.param(
                {"EXECUTOR_API_URL": ""}, None, False, "warning", ("EXECUTOR_API_ENABLED but EXECUTOR_API_URL is empty",), id="url_missing"
            ),
            pytest.param({"EXECUTOR_API_ENABLED": False}, None, False, None, None, id="disabled"),
        ],
    )
    async def test_forward_contract_matrix(
        self,
        config_overrides: dict[str, Any],
        status_code: int | None,
        expected_return: bool,
        expected_level: str | None,
        expected_log_args: tuple[Any, ...] | None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-200 answers are surfaced (never retried) and an unconfigured URL short-circuits."""
        monkeypatch.setattr(executor_handler, "DEAD_LETTER_PATH", tmp_path / "failed_forwards.jsonl")
        handler, _ = _handler(**config_overrides)
        client = _http_client(status_code)
        with patch("src.trading.executor_handler.httpx.AsyncClient", return_value=client):
            result = await handler._forward(dict(FORWARD_PAYLOAD))

        assert result is expected_return
        if status_code is None:
            assert client.post.await_count == 0
        else:
            assert client.post.await_args.args == (EXECUTOR_URL,)
            assert client.post.await_args.kwargs == {"json": FORWARD_PAYLOAD}
        if expected_level is not None:
            logger_double = {"info": handler.logger.info, "warning": handler.logger.warning}[expected_level]
            assert logger_double.call_args.args == expected_log_args

    async def test_forward_reuses_one_client_until_close(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two forwards share one TCP pool; close() releases it exactly once."""
        monkeypatch.setattr(executor_handler, "DEAD_LETTER_PATH", tmp_path / "failed_forwards.jsonl")
        handler, _ = _handler()
        client = _http_client(200)

        with patch("src.trading.executor_handler.httpx.AsyncClient", return_value=client) as client_class:
            assert await handler._forward(dict(FORWARD_PAYLOAD)) is True
            assert await handler._forward(dict(FORWARD_PAYLOAD)) is True
            assert client_class.call_count == 1
            assert client.post.await_count == 2
            assert handler._http_client is client
            await handler.close()

        assert client.aclose.await_count == 1
        assert handler._http_client is None

    async def test_dead_letter_appends_and_logs_write_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every unreachable entry lands in the journal; an unwritable path is logged, not raised."""
        journal = tmp_path / "failed_forwards.jsonl"
        monkeypatch.setattr(executor_handler, "DEAD_LETTER_PATH", journal)
        handler, _ = _handler()
        first = {"timestamp": "2026-07-23T12:00:00.000000", "symbol": "BTC/USDC", "signal": "BUY"}
        second = {"timestamp": "2026-07-23T12:05:00.000000", "symbol": "ETH/USDC", "signal": "SELL"}

        await handler._write_dead_letter(first)
        await handler._write_dead_letter(second)
        assert [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()] == [first, second]

        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        monkeypatch.setattr(executor_handler, "DEAD_LETTER_PATH", blocker / "sub" / "failed_forwards.jsonl")
        await handler._write_dead_letter(first)

        handler.logger.error.assert_called_once_with("Failed to write dead-letter entry", exc_info=True)
        assert not (blocker / "sub").exists()

    async def test_dead_letter_replay_keeps_unacknowledged_entries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Replay drops the journal only once every entry was acknowledged; a corrupt one aborts it."""
        journal = tmp_path / "failed_forwards.jsonl"
        monkeypatch.setattr(executor_handler, "DEAD_LETTER_PATH", journal)
        first = {"timestamp": "2026-07-23T12:00:00.000000", "symbol": "BTC/USDC", "signal": "BUY"}
        second = {"timestamp": "2026-07-23T12:05:00.000000", "symbol": "ETH/USDC", "signal": "SELL"}
        journal.write_text("".join(f"{json.dumps(entry)}\n" for entry in (first, second)), encoding="utf-8")
        handler, _ = _handler()
        client = _http_client(200)
        client.post = AsyncMock(side_effect=[_response(200), _response(500), _response(200)])
        with patch("src.trading.executor_handler.httpx.AsyncClient", return_value=client):
            assert await handler._forward(dict(FORWARD_PAYLOAD)) is True
        assert [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()] == [first]

        handler, _ = _handler()
        client = _http_client(200)
        with patch("src.trading.executor_handler.httpx.AsyncClient", return_value=client):
            assert await handler._forward(dict(FORWARD_PAYLOAD)) is True
        assert journal.exists() is False
        assert client.post.await_count == 2

        journal.write_text("not json\n", encoding="utf-8")
        handler, _ = _handler()
        client = _http_client(200)
        with patch("src.trading.executor_handler.httpx.AsyncClient", return_value=client):
            assert await handler._forward(dict(FORWARD_PAYLOAD)) is True
        assert journal.exists() is True
        handler.logger.error.assert_called_once_with("Dead-letter replay failed", exc_info=True)


class TestPositionPersistence:
    """PersistenceManager: position snapshot round trip, refusal paths and decision writes."""

    def test_position_round_trip_preserves_the_entry_snapshot(self, tmp_path: Path) -> None:
        """Exit-execution, ATR and condition snapshots must all survive save/load verbatim."""
        snapshot = make_market_conditions(
            trend_direction="BULLISH",
            adx=24.78,
            rsi=40.78,
            volatility="LOW",
            atr=451.47,
            atr_percentage=0.71,
            choppiness=62.7,
            mfi=38.0,
            cmf=-0.0559,
            vwap=63120.5,
            fear_greed_index=23,
            is_weekend=True,
            social_sentiment_reddit="BEARISH",
            portfolio_pnl_pct=-1.3,
        )
        position = make_position(
            entry_price=63408.99,
            stop_loss=62720.0,
            take_profit=64000.0,
            size=1.0,
            entry_time=datetime(2026, 8, 13, 12, 6, tzinfo=timezone.utc),
            atr_at_entry=451.47,
            atr_percentage_at_entry=0.71,
            stop_loss_type_at_entry="hard",
            stop_loss_check_interval_at_entry="15m",
            take_profit_type_at_entry="soft",
            take_profit_check_interval_at_entry="4h",
            order_book_bias_at_entry="BUY_PRESSURE",
            conditions_at_entry=snapshot,
        )
        manager = PersistenceManager(null_logger(), data_dir=str(tmp_path))
        manager.save_position(position)
        loaded = PersistenceManager(null_logger(), data_dir=str(tmp_path)).load_position()

        assert loaded.entry_price == 63408.99
        assert loaded.entry_time == position.entry_time
        assert loaded.size == 1.0
        assert loaded.atr_at_entry == 451.47
        assert loaded.atr_percentage_at_entry == 0.71
        assert loaded.stop_loss_type_at_entry == "hard"
        assert loaded.stop_loss_check_interval_at_entry == "15m"
        assert loaded.take_profit_type_at_entry == "soft"
        assert loaded.take_profit_check_interval_at_entry == "4h"
        assert loaded.order_book_bias_at_entry == "BUY_PRESSURE"
        assert loaded.conditions_at_entry.trend_direction == "BULLISH"
        assert loaded.conditions_at_entry.adx == 24.78
        assert loaded.conditions_at_entry.choppiness == 62.7
        assert loaded.conditions_at_entry.mfi == 38.0
        assert loaded.conditions_at_entry.cmf == -0.0559
        assert loaded.conditions_at_entry.vwap == 63120.5
        assert loaded.conditions_at_entry.fear_greed_index == 23
        assert loaded.conditions_at_entry.is_weekend is True
        assert loaded.conditions_at_entry.social_sentiment_reddit == "BEARISH"
        assert loaded.conditions_at_entry.portfolio_pnl_pct == -1.3
        assert manager.validate_loaded_position() == []

    def test_position_file_without_conditions_snapshot_is_refused(self, tmp_path: Path) -> None:
        """A file lacking the entry snapshot is rejected instead of rebuilt from defaults."""
        manager = PersistenceManager(null_logger(), data_dir=str(tmp_path))
        manager.positions_file.write_text(
            json.dumps(
                {
                    "entry_price": 100.0,
                    "stop_loss": 95.0,
                    "take_profit": 110.0,
                    "size": 1.0,
                    "entry_time": "2026-04-30T00:00:00+00:00",
                    "confidence": "HIGH",
                    "direction": "LONG",
                    "symbol": "BTC/USDC",
                }
            ),
            encoding="utf-8",
        )
        assert manager.load_position() is None

    def test_position_file_without_exit_snapshot_defaults_to_unknown(self, tmp_path: Path) -> None:
        """Old files without the typed-exit fields load as unknown, not as a fabricated type."""
        manager = PersistenceManager(null_logger(), data_dir=str(tmp_path))
        manager.positions_file.write_text(
            json.dumps(
                {
                    "entry_price": 100.0,
                    "stop_loss": 95.0,
                    "take_profit": 110.0,
                    "size": 1.0,
                    "entry_time": "2026-04-30T00:00:00+00:00",
                    "confidence": "HIGH",
                    "direction": "LONG",
                    "symbol": "BTC/USDC",
                    "conditions_at_entry": make_market_conditions().to_dict(),
                }
            ),
            encoding="utf-8",
        )
        loaded = manager.load_position()

        assert loaded.stop_loss_type_at_entry == "unknown"
        assert loaded.stop_loss_check_interval_at_entry == "unknown"
        assert loaded.take_profit_type_at_entry == "unknown"
        assert loaded.take_profit_check_interval_at_entry == "unknown"
        assert loaded.order_book_bias_at_entry == "BALANCED"
        assert loaded.conditions_at_entry.trend_direction == "NEUTRAL"

    def test_position_file_corrupt_json_loads_as_none(self, tmp_path: Path) -> None:
        """A truncated write must surface as no position, never as a half-built one."""
        manager = PersistenceManager(null_logger(), data_dir=str(tmp_path))
        manager.positions_file.write_text('{"entry_price": 100.0, "stop_loss":', encoding="utf-8")
        assert manager.load_position() is None
        manager.logger.error.assert_called_once()

    def test_failed_position_write_keeps_the_cache_invalid_then_recovers(self, tmp_path: Path) -> None:
        """A failed atomic replace must not publish a valid cache entry."""
        manager = PersistenceManager(null_logger(), data_dir=str(tmp_path))
        position = make_position()

        with patch("src.managers.persistence_manager.os.replace", side_effect=OSError("disk full")):
            manager.save_position(position)
        assert manager._position_cache_valid is False
        assert manager.positions_file.exists() is False
        assert manager.load_position() is None

        manager.save_position(position)
        assert manager._position_cache_valid is True
        assert manager.load_position() is position
        reloaded = PersistenceManager(null_logger(), data_dir=str(tmp_path)).load_position()
        assert reloaded.entry_price == 50000.0
        assert reloaded.stop_loss == 49000.0

        manager.save_position(None)
        assert manager._position_cache_valid is True
        assert manager.positions_file.exists() is False
        assert manager.load_position() is None

    def test_save_trade_decision_returns_monotonic_row_ids(self, tmp_path: Path) -> None:
        """The returned rowid is what links the post-mortem journal back to trade_history."""
        manager = PersistenceManager(null_logger(), data_dir=str(tmp_path))
        closed = _decision(
            action="CLOSE_SHORT",
            price=63671.19,
            stop_loss=None,
            take_profit=None,
            quantity=0.0,
            fee=0.3702,
            reasoning="Position closed: stop_loss",
        )
        row_id = manager.save_trade_decision(closed)
        second_id = manager.save_trade_decision(_decision(action="BUY", price=63826.23, quantity=0.0))

        assert type(row_id) is int
        assert row_id > 0
        assert second_id > row_id
        rows = {row["id"]: row for row in manager.sqlite_history.query()}
        assert rows[row_id]["action"] == "CLOSE_SHORT"
        assert rows[row_id]["fee"] == 0.3702
        assert rows[second_id]["action"] == "BUY"
        assert manager.get_last_execution_timestamp() == closed.timestamp

    def test_last_execution_timestamp_propagates_the_sqlite_failure(self, tmp_path: Path) -> None:
        """A dead history database must surface as an error, not as 'no trades yet'."""
        logger = null_logger()
        manager = PersistenceManager(logger, data_dir=str(tmp_path))
        failure = RuntimeError("db down")
        manager.sqlite_history.get_last_execution_timestamp = MagicMock(side_effect=failure)

        with pytest.raises(RuntimeError, match="db down"):
            manager.get_last_execution_timestamp()
        logger.error.assert_called_once_with(
            "Failed to read last execution timestamp from SQLite: %s", failure
        )

    @pytest.mark.parametrize(
        ("rows", "expected_reasoning"),
        [
            pytest.param(
                [
                    {"timestamp": "2026-04-30T12:00:00.010000+00:00", "symbol": "ETH/USDC", "action": "BUY", "confidence": "HIGH", "price": 2000.0, "reasoning": "Wrong symbol"},
                    {"timestamp": "2026-04-30T12:00:00.020000+00:00", "symbol": "BTC/USDC", "action": "BUY", "confidence": "MEDIUM", "price": 100.0, "reasoning": "Expected entry"},
                    {"timestamp": "2026-04-30T12:00:00.400000+00:00", "symbol": "BTC/USDC", "action": "BUY", "confidence": "LOW", "price": 101.0, "reasoning": "Later entry"},
                ],
                "Expected entry",
                id="nearest_symbol_match",
            ),
            pytest.param(
                [{"timestamp": "2026-04-30T12:00:00.600000+00:00", "symbol": "BTC/USDC", "action": "BUY", "confidence": "HIGH", "price": 100.0, "reasoning": "Too late"}],
                None,
                id="outside_match_tolerance",
            ),
            pytest.param(
                [{"timestamp": "2026-04-30T12:00:00.000000+00:00", "symbol": "BTC/USDC", "action": "CLOSE", "confidence": "HIGH", "price": 100.0, "reasoning": "Exit row"}],
                None,
                id="non_entry_action_ignored",
            ),
        ],
    )
    def test_entry_decision_lookup_matrix(
        self, tmp_path: Path, rows: list[dict[str, Any]], expected_reasoning: str | None
    ) -> None:
        """Only a BUY/SELL row within 0.5s of the entry stamp can describe that entry."""
        manager = PersistenceManager(null_logger(), data_dir=str(tmp_path))
        for record in rows:
            manager.sqlite_history.insert(record)

        decision = manager.get_entry_decision_for_position(
            datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc), symbol="BTC/USDC"
        )

        if expected_reasoning is None:
            assert decision is None
        else:
            assert decision.reasoning == expected_reasoning
            assert decision.symbol == "BTC/USDC"
            assert decision.price == 100.0

    def test_latest_decision_fallback_file_lifecycle(self, tmp_path: Path) -> None:
        """The fallback file is written atomically, leaves no temp files and clears idempotently."""
        manager = PersistenceManager(null_logger(), data_dir=str(tmp_path))
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": "BTC/USDC",
            "signal": "BUY",
            "order_type": "limit",
            "quantity": 0.01,
            "entry_price": 10.0,
            "stop_loss": 9.0,
            "take_profit": 12.0,
            "reduce_only": False,
            "leverage": 1,
            "confidence": "HIGH",
            "reasoning": "unit",
        }
        manager.save_latest_decision(payload)
        path = tmp_path / "latest_decision.json"

        assert json.loads(path.read_text(encoding="utf-8")) == payload
        assert list(tmp_path.glob("*.json")) == [path]

        manager.clear_latest_decision()
        assert path.exists() is False
        manager.clear_latest_decision()
        assert path.exists() is False


class TestExecutorHandshake:
    """TradingStrategy executor position probe and the CLOSE/UPDATE gate."""

    @pytest.mark.parametrize(
        ("config_overrides", "status_code", "body", "probe_exc", "expected", "expected_query_url"),
        [
            pytest.param({"EXECUTOR_API_ENABLED": False}, None, None, None, True, None, id="disabled_assumes_open"),
            pytest.param({"EXECUTOR_API_URL": ""}, None, None, None, True, None, id="unconfigured_url_assumes_open"),
            pytest.param({}, 200, {"open": False}, None, False, POSITION_URL, id="decision_suffix_stripped"),
            pytest.param({"EXECUTOR_API_URL": "http://127.0.0.1:9199/decision/"}, 200, {"open": False}, None, False, POSITION_URL, id="trailing_slash_stripped"),
            pytest.param({"EXECUTOR_API_URL": "http://executor:8000"}, 200, {"open": True}, None, True, "http://executor:8000/position", id="base_url_without_suffix"),
            pytest.param({}, 200, {}, None, False, POSITION_URL, id="unknown_body_reads_closed"),
            pytest.param({}, 200, {"open": "false"}, None, True, POSITION_URL, id="stringified_false_is_truthy"),
            pytest.param({}, 404, None, None, None, POSITION_URL, id="http_error_unverifiable"),
            pytest.param({}, None, None, RuntimeError("connection refused"), None, POSITION_URL, id="network_error_unverifiable"),
        ],
    )
    async def test_position_probe_matrix(
        self,
        config_overrides: dict[str, Any],
        status_code: int | None,
        body: Any,
        probe_exc: Exception | None,
        expected: bool | None,
        expected_query_url: str | None,
    ) -> None:
        """Only an explicit open:false counts as flat; every other failure is unverifiable."""
        config = {"EXECUTOR_API_ENABLED": True, "EXECUTOR_API_URL": EXECUTOR_URL, **config_overrides}
        strategy, _, _ = _strategy(make_position(), **config)
        client = _probe_stub(strategy, status_code if status_code is not None else 200, body, probe_exc)

        assert await strategy._executor_has_position("BTC/USDC") is expected
        if expected_query_url is None:
            assert client.get.await_count == 0
        else:
            assert client.get.await_args.args == (expected_query_url,)
            assert client.get.await_args.kwargs == {"params": {"symbol": "BTC/USDC"}}

    @pytest.mark.parametrize(
        ("signal", "position_overrides", "stop_loss", "take_profit", "probe", "update_age_hours", "expected_action", "expected_sl", "expected_tp", "expected_save"),
        [
            pytest.param("CLOSE", {}, None, None, "real", None, "CLOSE", None, None, "none", id="close_signal_exits"),
            pytest.param("CLOSE_LONG", {}, None, None, "real", None, "CLOSE", None, None, "none", id="close_long_signal_exits"),
            pytest.param("UPDATE", {"stop_loss": 40000.0, "take_profit": 60000.0}, 39000.0, 60000.0, "real", 0.0, None, 40000.0, 60000.0, "absent", id="update_rejected_too_soon"),
            pytest.param("UPDATE", {"stop_loss": 40000.0, "take_profit": 60000.0}, 39000.0, 60000.0, "real", 100000.0, "UPDATE", 39000.0, 60000.0, "position", id="update_sl_applied"),
            pytest.param("UPDATE", {"stop_loss": 40000.0, "take_profit": 60000.0}, 40000.0, 61000.0, "real", 100000.0, "UPDATE", 40000.0, 61000.0, "position", id="update_tp_applied"),
            pytest.param("CLOSE", {"stop_loss": 40000.0, "take_profit": 60000.0}, None, None, False, None, None, None, None, "none", id="close_skipped_when_executor_flat"),
            pytest.param("UPDATE", {"stop_loss": 40000.0, "take_profit": 60000.0}, 39000.0, 60000.0, False, 100000.0, None, None, None, "none", id="update_skipped_when_executor_flat"),
            pytest.param("CLOSE", {"stop_loss": 40000.0, "take_profit": 60000.0}, None, None, None, None, None, 40000.0, 60000.0, "absent", id="close_skipped_when_unverifiable"),
            pytest.param("UPDATE", {"stop_loss": 40000.0, "take_profit": 60000.0}, 39000.0, 60000.0, None, 100000.0, None, 40000.0, 60000.0, "absent", id="update_skipped_when_unverifiable"),
        ],
    )
    async def test_existing_position_signal_matrix(
        self,
        signal: str,
        position_overrides: dict[str, Any],
        stop_loss: float | None,
        take_profit: float | None,
        probe: Any,
        update_age_hours: float | None,
        expected_action: str | None,
        expected_sl: float | None,
        expected_tp: float | None,
        expected_save: str,
    ) -> None:
        """CLOSE exits, UPDATE needs a verified executor position and a matured interval."""
        position = make_position(**position_overrides)
        strategy, _, persistence = _strategy(position)
        if probe != "real":
            strategy._executor_has_position = AsyncMock(return_value=probe)
        if update_age_hours is not None:
            strategy._last_position_update_time = datetime.now(timezone.utc) - timedelta(hours=update_age_hours)

        result = await strategy._handle_existing_position(
            signal=signal,
            confidence="MEDIUM",
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=50500.0,
            symbol="BTC/USDC",
            reasoning="Signal reasoning",
            market_conditions=MarketConditions(),
        )

        if expected_action is None:
            assert result is None
        else:
            assert result.action == expected_action
            assert result.symbol == "BTC/USDC"
            assert result.confidence == "MEDIUM"
            assert result.price == 50500.0
        if expected_sl is None:
            assert strategy.current_position is None
        else:
            assert strategy.current_position.stop_loss == expected_sl
            assert strategy.current_position.take_profit == expected_tp
        if expected_save == "none":
            assert persistence.async_save_position.await_args_list[-1].args == (None,)
        elif expected_save == "position":
            assert persistence.async_save_position.await_count == 1
            assert persistence.async_save_position.await_args.args[0].stop_loss == stop_loss
        else:
            assert persistence.async_save_position.await_count == 0

    def test_position_context_reports_progress_and_empty_state(self) -> None:
        """A SHORT position reports real progress; without a position the status line is None."""
        position = make_position(entry_price=50000.0, stop_loss=49000.0, take_profit=52000.0, direction="SHORT")
        strategy, _, _ = _strategy(position)
        strategy.brain_service.get_dynamic_thresholds = MagicMock(return_value={})

        context = strategy.get_position_context(current_price=49500.0)
        progress_lines = [line for line in context.split("\n") if "progress" in line.lower()]

        assert "Current price progress: 25.0%" in context
        assert progress_lines[0] == "- Effective minimum progress: 20% of entry-to-TP (source: config)"
        assert "0.0%" not in context.split("## SL Tightening Policy")[1]
        assert "Tightening eligible: YES" in context

        flat_strategy, _, _ = _strategy()
        assert "Status: None" in flat_strategy.get_position_context()

    @pytest.mark.parametrize(
        ("direction", "stop_loss", "take_profit", "new_stop_loss", "current_price", "expected_updated"),
        [
            pytest.param("LONG", 40000.0, 60000.0, 39000.0, 50500.0, True, id="long_widening_within_cap"),
            pytest.param("SHORT", 60000.0, 40000.0, 61000.0, 49500.0, True, id="short_widening_within_cap"),
            pytest.param("LONG", 40000.0, 60000.0, 1000.0, 50500.0, False, id="long_widening_beyond_cap"),
            pytest.param("SHORT", 60000.0, 40000.0, 99000.0, 49500.0, False, id="short_widening_beyond_cap"),
        ],
    )
    async def test_sl_widening_is_directional_and_capped(
        self,
        direction: str,
        stop_loss: float,
        take_profit: float,
        new_stop_loss: float,
        current_price: float,
        expected_updated: bool,
    ) -> None:
        """Widening the SL past 150% of its distance is refused; within it the move is logged."""
        position = make_position(direction=direction, stop_loss=stop_loss, take_profit=take_profit)
        strategy, logger, _ = _strategy(position)

        updated = await strategy._update_position_parameters(
            stop_loss=new_stop_loss, take_profit=None, current_price=current_price
        )

        assert updated is expected_updated
        widening_logs = [call.args[0] for call in logger.info.call_args_list if "Widening" in str(call.args[0])]
        if expected_updated:
            assert strategy.current_position.stop_loss == new_stop_loss
            assert widening_logs == [f"AI Widening Stop Loss for {direction}: $%.2f -> $%.2f (Risk Increased)"]
        else:
            assert strategy.current_position.stop_loss == stop_loss
            assert widening_logs == []


class TestEntryConfirmation:
    """TradingStrategy confirm_entry_with_executor: /position polling and the verdict journal."""

    @pytest.mark.parametrize(
        ("probe_values", "attempts", "min_false_reports", "expected", "expected_polls", "expected_warning"),
        [
            pytest.param([True], 10, 6, True, 1, None, id="executor_confirms"),
            pytest.param([False, False, False, False, True], 10, 6, True, 5, None, id="slow_executor_keeps_polling"),
            pytest.param([False], 2, 2, False, 2, "entry was likely blocked", id="stable_false_rolls_back"),
            pytest.param([False], 10, 6, False, 6, "entry was likely blocked", id="true_block_after_false_streak"),
            pytest.param([False], 3, 6, True, 3, "keeping local position (fail-open)", id="window_shorter_than_false_floor_fails_open"),
            pytest.param([None], 2, None, True, 2, "keeping local position (fail-open)", id="unverifiable_fails_open"),
            pytest.param([None, None, True], 5, None, True, 3, None, id="transient_errors_then_confirmed"),
        ],
    )
    async def test_confirm_entry_poll_matrix(
        self,
        probe_values: list[Any],
        attempts: int,
        min_false_reports: int | None,
        expected: bool,
        expected_polls: int,
        expected_warning: str | None,
    ) -> None:
        """Rolling back needs a full false streak; a short window or a query error fails open."""
        strategy, logger, _ = _strategy(make_position())
        strategy._executor_has_position = AsyncMock(side_effect=probe_values * attempts)

        with _confirm_window(attempts, min_false_reports):
            result = await strategy.confirm_entry_with_executor("BTC/USDC")

        assert result is expected
        assert strategy._executor_has_position.await_count == expected_polls
        if expected_warning is None:
            assert logger.warning.call_args_list == []
        else:
            assert expected_warning in logger.warning.call_args.args[0]

    @pytest.mark.parametrize(
        ("verdict", "expected"),
        [
            pytest.param("executed", True, id="executed"),
            pytest.param("blocked", False, id="blocked"),
            pytest.param("error", False, id="error"),
        ],
    )
    async def test_confirm_entry_reads_the_verdict_journal(
        self, tmp_path: Path, verdict: str, expected: bool
    ) -> None:
        """A per-order verdict decides the entry state without racing the executor queue tick."""
        journal = _journal(tmp_path, [json.dumps({"order_id": "order-abc", "verdict": verdict, "reason": "reported"})])
        strategy, logger, _ = _strategy(make_position())
        strategy.config = _verdict_config(journal)

        with _confirm_window(2):
            result = await strategy.confirm_entry_with_executor("BTC/USDC", order_id="order-abc")

        assert result is expected
        if verdict == "executed":
            assert logger.warning.call_args_list == []
        else:
            assert logger.warning.call_args.args[0] == "Executor verdict for %s: %s — entry was %s"

    async def test_confirm_entry_fails_open_without_a_verdict(self, tmp_path: Path) -> None:
        """An unreadable or absent journal must never order a rollback."""
        missing = tmp_path / "does_not_exist.jsonl"
        strategy, logger, _ = _strategy(make_position())
        strategy.config = _verdict_config(missing)

        with _confirm_window(2):
            result = await strategy.confirm_entry_with_executor("BTC/USDC", order_id="order-abc")

        assert result is True
        assert logger.warning.call_args.args == (
            "No executor verdict for %s after %d polls — keeping local position (fail-open)",
            "order-abc",
            2,
        )

    async def test_confirm_entry_fails_open_on_a_corrupt_journal(self, tmp_path: Path) -> None:
        """A truncated journal line hides the verdict behind it — fail open, never roll back."""
        journal = _journal(
            tmp_path,
            [json.dumps({"order_id": "order-abc", "verdict": "blocked", "reason": "Notional exceeds max"}), "truncated {"],
        )
        strategy, logger, _ = _strategy(make_position())
        strategy.config = _verdict_config(journal)

        with _confirm_window(2):
            result = await strategy.confirm_entry_with_executor("BTC/USDC", order_id="order-abc")

        assert result is True
        logger.warning.assert_any_call("Failed to read executor verdict journal at %s", journal)

    async def test_confirm_entry_polls_through_a_late_verdict(self, tmp_path: Path) -> None:
        """A verdict written after several polls must be picked up instead of read as absent."""
        journal = tmp_path / "executor_verdicts.jsonl"
        strategy, _, _ = _strategy(make_position())
        strategy.config = _verdict_config(journal)

        async def _write_late_verdict() -> None:
            await asyncio.sleep(0.005)
            journal.write_text(
                f"{json.dumps({'order_id': 'order-abc', 'verdict': 'executed', 'reason': ''})}\n",
                encoding="utf-8",
            )

        written = asyncio.create_task(_write_late_verdict())
        try:
            with _confirm_window(10, delay=0.002):
                assert await strategy.confirm_entry_with_executor("BTC/USDC", order_id="order-abc") is True
        finally:
            await written


class TestEntryRollback:
    """TradingStrategy rollback_blocked_entry: only a delivered, unconfirmed entry is rolled back."""

    @pytest.mark.parametrize(
        ("has_position", "delivered", "probe", "verdict", "expected_kept", "expected_close_row"),
        [
            pytest.param(False, True, False, None, False, False, id="without_position"),
            pytest.param(True, False, False, None, True, False, id="forward_undelivered"),
            pytest.param(True, True, True, None, True, False, id="probe_confirms_position"),
            pytest.param(True, True, False, None, False, True, id="probe_reports_flat"),
            pytest.param(True, True, None, "blocked", False, True, id="verdict_blocked"),
            pytest.param(True, True, None, "executed", True, False, id="verdict_executed"),
        ],
    )
    async def test_rollback_matrix(
        self,
        tmp_path: Path,
        has_position: bool,
        delivered: bool,
        probe: bool | None,
        verdict: str | None,
        expected_kept: bool,
        expected_close_row: bool,
    ) -> None:
        """A phantom is cleared only after an HTTP-delivered entry the executor denies."""
        position = make_position(size_pct=0.05, quote_amount=500.0)
        strategy, logger, persistence = _strategy(position if has_position else None)
        order_id = None
        if verdict is not None:
            order_id = "order-abc"
            strategy.config = _verdict_config(
                _journal(tmp_path, [json.dumps({"order_id": order_id, "verdict": verdict, "reason": "reported"})])
            )
        if probe is not None:
            strategy._executor_has_position = AsyncMock(return_value=probe)

        with _confirm_window(2, 2):
            await strategy.rollback_blocked_entry("BTC/USDC", forward_delivered=delivered, order_id=order_id)

        if expected_kept:
            assert strategy.current_position is position
            assert persistence.async_save_position.await_count == 0
            assert persistence.async_save_trade_decision.await_count == 0
        elif has_position:
            assert strategy.current_position is None
            assert persistence.async_save_position.await_args.args == (None,)
        else:
            assert strategy.current_position is None
            assert persistence.async_save_position.await_count == 0
            assert persistence.async_save_trade_decision.await_count == 0
        if probe is not None and not delivered:
            assert strategy._executor_has_position.await_count == 0
        if expected_close_row:
            decision = persistence.async_save_trade_decision.await_args.args[0]
            assert decision.action == "CLOSE"
            assert decision.symbol == position.symbol
            assert decision.confidence == position.confidence
            assert decision.quantity == position.size
            assert decision.price == position.entry_price
            assert decision.fee == 0.0
            assert decision.position_size == position.size_pct
            assert decision.quote_amount == position.quote_amount
            assert decision.stop_loss == position.stop_loss
            assert decision.take_profit == position.take_profit
            assert "blocked" in decision.reasoning.lower()
            assert position.entry_time.isoformat() in decision.reasoning
            assert any("rolled back local phantom" in call.args[0] for call in logger.warning.call_args_list)


class TestBlockedTradeStore:
    """VectorMemoryService rejections that feed the brain's feedback loop."""

    def test_blocked_trade_round_trip(self, blocked_memory: VectorMemoryService) -> None:
        """Every rejection keeps its guard payload, id and delta through ChromaDB."""
        stored = [
            _store_block(blocked_memory, guard_type=f"guard_{index}", suggested_rr=1.0 + index * 0.1)
            for index in range(3)
        ]
        assert stored == [True, True, True]
        assert blocked_memory.BLOCKED_TRADES_COLLECTION == "system_constraints_rejections"
        assert blocked_memory.get_blocked_trade_count() == 3
        assert blocked_memory._blocked_collection.count() == 3

        results = blocked_memory.get_recent_blocked_trades(n=5)
        assert len(results) == 3
        ids = [result["id"] for result in results]
        assert len(set(ids)) == 3
        assert all(identifier.startswith("blocked_") for identifier in ids)

        result = blocked_memory.get_recent_blocked_trades(n=1)[0]
        assert set(result) >= {
            "id",
            "document",
            "guard_type",
            "direction",
            "confidence",
            "suggested_rr",
            "required_rr",
            "rr_delta",
            "suggested_sl_pct",
            "suggested_tp_pct",
            "suggested_sl",
            "suggested_tp",
            "current_price",
            "volatility_level",
            "reasoning_snippet",
            "timestamp",
            "event_type",
        }
        assert result["guard_type"] == "guard_2"
        assert result["suggested_rr"] == pytest.approx(1.2)
        assert result["required_rr"] == pytest.approx(2.0)
        assert result["rr_delta"] == pytest.approx(1.2 - 2.0)
        assert result["event_type"] == "system_rejection"
        assert result["reasoning_snippet"] == "Test reasoning for LLM feedback"
        assert result["direction"] == "LONG"
        assert result["volatility_level"] == "MEDIUM"

    def test_blocked_trade_metadata_drops_none_values(self, blocked_memory: VectorMemoryService) -> None:
        """ChromaDB rejects None metadata, so those keys must be sanitised away, not persisted."""
        assert _store_block(blocked_memory, metadata={"extra_field": None}) is True
        assert "extra_field" not in blocked_memory.get_recent_blocked_trades(n=1)[0]

        assert (
            _store_block(
                blocked_memory,
                metadata={"custom_tag": "important", "nullable_field": None, "score": 42},
            )
            is True
        )
        result = blocked_memory.get_recent_blocked_trades(n=1)[0]
        assert result["custom_tag"] == "important"
        assert result["score"] == 42
        assert "nullable_field" not in result

    def test_blocked_trade_retrieval_order_limits_and_filters(self, blocked_memory: VectorMemoryService) -> None:
        """Retrieval is newest-first, capped by n, and filterable by guard and age."""
        for index in range(10):
            guard = "rr_minimum" if index % 2 == 0 else "sl_distance_max"
            assert _store_block(blocked_memory, guard_type=guard) is True
        assert blocked_memory.get_blocked_trade_count() == 10

        newest = blocked_memory.get_recent_blocked_trades(n=3)
        timestamps = [result["timestamp"] for result in newest]
        assert len(newest) == 3
        assert timestamps == sorted(timestamps, reverse=True)

        rejected = blocked_memory.get_recent_blocked_trades(n=10, guard_type="rr_minimum")
        assert len(rejected) == 5
        assert all(result["guard_type"] == "rr_minimum" for result in rejected)
        assert blocked_memory.get_recent_blocked_trades(n=10, guard_type="nonexistent") == []
        assert blocked_memory.get_recent_blocked_trades(n=10, max_age_hours=0) == []
        assert len(blocked_memory.get_recent_blocked_trades(n=10, max_age_hours=9999)) == 10

    def test_blocked_trade_feedback_rendering(self, blocked_memory: VectorMemoryService) -> None:
        """The feedback prompt groups rejections per guard and states the R/R gap."""
        assert blocked_memory.get_blocked_trade_feedback(n=5) == ""
        assert (
            _store_block(
                blocked_memory,
                suggested_rr=1.2,
                required_rr=2.5,
                reasoning_snippet="Expecting breakout above resistance",
            )
            is True
        )

        feedback = blocked_memory.get_blocked_trade_feedback(n=5)
        assert "CRITICAL FEEDBACK" in feedback
        assert "R/R Minimum Guard" in feedback
        assert "PRE-FLIGHT CHECKLIST" in feedback
        assert "R/R >= required minimum" in feedback
        assert "gap:" in feedback
        assert "1.20" in feedback
        assert "2.50" in feedback
        assert "Expecting breakout" in feedback

        assert _store_block(blocked_memory, guard_type="sl_distance_max") is True
        grouped = blocked_memory.get_blocked_trade_feedback(n=10)
        assert "R/R Minimum Guard" in grouped
        assert "SL Too Far (max 10%)" in grouped

    def test_blocked_trade_empty_state(self, blocked_memory: VectorMemoryService) -> None:
        """An untouched store reports zero, no rows and no feedback."""
        assert blocked_memory.get_blocked_trade_count() == 0
        assert blocked_memory.get_recent_blocked_trades(n=5) == []
        assert blocked_memory.get_blocked_trade_feedback() == ""
