"""Wave-5: a lesson needs PROOF — no evidence, no post-mortem, no brain update.

The 2026-09-21 incident wrote a lesson into the brain from a closure the bot could not
prove (an ``unknown`` booking with a corrupted local size). This suite pins the guard:

* a close whose exit fact is not proven (no evidence, unknown price/amount, a
  non-confirming source, no stable event id) must NOT ask the post-mortem and must NOT
  update the brain — the skip is logged explicitly, and the headline names the finding:
  "Lesson skipped: no exit evidence" when nothing was measured, "Lesson skipped: the
  venue fill price is not a price the real market made" when the fill was measured and
  found to be an off-market print;
* a post-mortem that fails (returns None or raises) must NOT produce a brain entry
  either — nothing is written that could pass for an analysis;
* a proven exit + a validated post-mortem teaches the brain exactly ONCE per exit
  event (idempotency), and the CLOSE row carries the FILL amount, not the local size.

Every test uses doubles and temporary paths: no live repo, no exchange, no credentials.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.dashboard.routers.performance import PerformanceRouter
from src.managers.persistence_manager import PersistenceManager
from src.trading.data_models import (
    FILL_DIVERGENCE_MARKER,
    MAX_MARKET_REFERENCE_WINDOW_SECONDS,
    market_verification_is_verified,
)
from src.trading.executor_reconciliation import (
    ExitEvidence,
    LESSON_SKIPPED_FILL_DIVERGENCE,
    LESSON_SKIPPED_NO_EVIDENCE,
    exit_evidence_is_learnable,
    fill_divergence_reason,
    market_verification_gap_reason,
)
from src.trading.memory import TradingMemoryService
from src.trading.statistics_calculator import StatisticsCalculator
from src.trading.trading_strategy import TradingStrategy
from tests.conftest import (
    make_config,
    make_market_conditions,
    make_position,
    mock_brain,
    mock_persistence,
    mock_statistics,
    null_logger,
    verified_market_evidence,
)
from tests.domain_trading.test_bookkeeping_authority import (
    TestExecutedEntrySettlement as _EntrySettlement,
)

ENTRY_TIME = datetime(2026, 9, 21, 8, 12, 0, tzinfo=timezone.utc)


def proven_exit(**overrides: Any) -> ExitEvidence:
    """Evidence of a PROVEN exit (real fill price + amount + stable event id)."""
    values: dict[str, Any] = {
        "source": "executor_exit_journal",
        "price": 84500.0,
        "quantity": 0.00554,
        "event_id": "executor_exit_journal#event-1",
        "fill_verification": "verified",
        "market_verification": verified_market_evidence(),
        "fee": 0.0,
        "fee_currency": "USDC",
    }
    values.update(overrides)
    return ExitEvidence(**values)


def _position(**overrides: Any):
    values: dict[str, Any] = {
        "symbol": "BTC/USDC",
        "direction": "LONG",
        "entry_price": 81200.45,
        "stop_loss": 78800.0,
        "take_profit": 84500.0,
        "size": 0.00554,
        "entry_time": ENTRY_TIME,
        "conditions_at_entry": make_market_conditions(),
    }
    values.update(overrides)
    return make_position(**values)


def _strategy(
    tmp_path: Path,
    position: Any | None = None,
    *,
    post_mortem_result: Any = "valid",
    entry_decision: bool = True,
    post_mortem_raises: bool = False,
) -> tuple[TradingStrategy, MagicMock, MagicMock, MagicMock]:
    """Real strategy with an isolated intent journal and scripted post-mortem.

    Returns ``(strategy, logger, persistence, post_mortem)``.
    """
    logger = null_logger()
    persistence = mock_persistence()
    persistence.async_save_trade_decision = AsyncMock(return_value=42)
    if entry_decision:
        entry = MagicMock()
        entry.reasoning = "Breakout continuation."
        persistence.get_entry_decision_for_position = MagicMock(return_value=entry)
    else:
        persistence.get_entry_decision_for_position = MagicMock(return_value=None)
    statistics = mock_statistics()
    statistics.get_current_capital.return_value = 10000.0
    brain = mock_brain()
    config = make_config(BOT_INTENT_JOURNAL_PATH=str(tmp_path / "bot_position_intents.jsonl"))
    strategy = TradingStrategy(
        logger=logger,
        persistence=persistence,
        brain_service=brain,
        statistics_service=statistics,
        memory_service=MagicMock(),
        risk_manager=MagicMock(),
        config=config,
        position_extractor=MagicMock(),
    )
    post_mortem = MagicMock()
    if post_mortem_raises:
        post_mortem.analyze_closed_trade = AsyncMock(side_effect=RuntimeError("llm down"))
    elif post_mortem_result == "valid":
        post_mortem.analyze_closed_trade = AsyncMock(return_value=MagicMock())
    else:
        post_mortem.analyze_closed_trade = AsyncMock(return_value=None)
    strategy.post_mortem_service = post_mortem
    if position is not None:
        strategy.current_position = position
    return strategy, logger, persistence, post_mortem


def _logged(logger: MagicMock, phrase: str) -> bool:
    """True when any log call (any level) carried ``phrase``."""
    for level in ("warning", "info", "critical", "error"):
        for call in getattr(logger, level).call_args_list:
            if phrase in str(call.args):
                return True
    return False


class TestNoEvidenceMeansNoLesson:
    """A close whose exit fact is unproven never reaches the post-mortem or the brain."""

    @pytest.mark.parametrize(
        "evidence",
        [
            pytest.param(None, id="no-evidence-at-all"),
            pytest.param(proven_exit(price=None), id="unknown-price"),
            pytest.param(proven_exit(quantity=None), id="unknown-quantity"),
            pytest.param(proven_exit(source="local_monitor"), id="local-monitor-source"),
            pytest.param(proven_exit(source="unknown"), id="unknown-source"),
            pytest.param(proven_exit(event_id=None), id="no-stable-event-id"),
        ],
    )
    async def test_unproven_close_skips_post_mortem_and_brain(self, tmp_path: Path, evidence: Any) -> None:
        position = _position()
        strategy, logger, persistence, post_mortem = _strategy(tmp_path, position)

        await strategy.close_position("stop_loss", 78800.0, make_market_conditions(), evidence=evidence)

        persistence.async_save_trade_decision.assert_awaited_once()
        assert strategy.current_position is None
        post_mortem.analyze_closed_trade.assert_not_awaited()
        strategy.brain_service.update_from_closed_trade.assert_not_called()
        assert _logged(logger, "Lesson skipped: no exit evidence")


class TestFailedPostMortemMeansNoLesson:
    """A post-mortem that did not produce a validated analysis produces no brain entry."""

    @pytest.mark.parametrize(
        ("post_mortem_result", "post_mortem_raises"),
        [
            pytest.param(None, False, id="post-mortem-returned-none"),
            pytest.param("valid", True, id="post-mortem-raised"),
        ],
    )
    async def test_failed_post_mortem_writes_no_lesson(
        self, tmp_path: Path, post_mortem_result: Any, post_mortem_raises: bool
    ) -> None:
        position = _position()
        strategy, logger, _, post_mortem = _strategy(
            tmp_path, position,
            post_mortem_result=post_mortem_result, post_mortem_raises=post_mortem_raises,
        )

        await strategy.close_position(
            "take_profit", 84500.0, make_market_conditions(),
            filled_quantity=0.00554, evidence=proven_exit(),
        )

        post_mortem.analyze_closed_trade.assert_awaited_once()
        strategy.brain_service.update_from_closed_trade.assert_not_called()
        assert _logged(logger, "no confirmed post-mortem analysis")

    async def test_missing_entry_decision_blocks_the_lesson(self, tmp_path: Path) -> None:
        """Without the entry decision there is nothing to analyse — and nothing to teach."""
        position = _position()
        strategy, logger, _, post_mortem = _strategy(tmp_path, position, entry_decision=False)

        await strategy.close_position(
            "take_profit", 84500.0, make_market_conditions(),
            filled_quantity=0.00554, evidence=proven_exit(),
        )

        post_mortem.analyze_closed_trade.assert_not_awaited()
        strategy.brain_service.update_from_closed_trade.assert_not_called()
        assert _logged(logger, "no confirmed post-mortem analysis")

    async def test_missing_post_mortem_service_blocks_the_lesson(self, tmp_path: Path) -> None:
        position = _position()
        strategy, logger, _, _ = _strategy(tmp_path, position)
        strategy.post_mortem_service = None

        await strategy.close_position(
            "take_profit", 84500.0, make_market_conditions(),
            filled_quantity=0.00554, evidence=proven_exit(),
        )

        strategy.brain_service.update_from_closed_trade.assert_not_called()
        assert _logged(logger, "no confirmed post-mortem analysis")


class TestDivergentFillMeansNoLesson:
    """A fill the real market never made is BOOKED for history, but never TAUGHT.

    2026-10-17: a sandbox take-profit filled at 77506 while the real market traded at
    81325. The bot booked +4.68% (the best trade of the whole run), the post-mortem
    called it "plan_followed" with "no meaningful adverse excursion", and the brain was
    fed a WIN experience. The executor's divergence marking now books the close (history
    stays paired, the position is released) and refuses the lesson.
    """

    DIVERGENT_ARTIFACT: ClassVar[dict[str, Any]] = {
        "fill_verification": "divergent",
        "reference_price": 81325.09,
        "deviation_pct": -4.6963,
    }

    async def test_divergent_close_is_booked_but_never_teaches(self, tmp_path: Path) -> None:
        position = _position()
        strategy, logger, persistence, post_mortem = _strategy(tmp_path, position)

        await strategy.close_position(
            "take_profit", 77506.0, make_market_conditions(),
            filled_quantity=0.00554, evidence=proven_exit(**self.DIVERGENT_ARTIFACT),
        )

        persistence.async_save_trade_decision.assert_awaited_once()
        assert strategy.current_position is None
        post_mortem.analyze_closed_trade.assert_not_awaited()
        strategy.brain_service.update_from_closed_trade.assert_not_called()
        assert _logged(logger, LESSON_SKIPPED_FILL_DIVERGENCE)
        assert not _logged(logger, LESSON_SKIPPED_NO_EVIDENCE)
        assert _logged(logger, "the venue's fill price is not a price the real market made")

    async def test_marker_lands_on_the_close_row_so_the_model_sees_it(self, tmp_path: Path) -> None:
        """The CLOSE row's reasoning is what the prompt renders — the artifact must show."""
        position = _position()
        strategy, _, persistence, _ = _strategy(tmp_path, position)

        await strategy.close_position(
            f"{FILL_DIVERGENCE_MARKER}: take_profit_filled @ 77506.0", 77506.0,
            make_market_conditions(),
            filled_quantity=0.00554, evidence=proven_exit(**self.DIVERGENT_ARTIFACT),
        )

        decision = persistence.async_save_trade_decision.await_args.args[0]
        assert FILL_DIVERGENCE_MARKER in decision.reasoning

    async def test_verified_fill_still_teaches(self, tmp_path: Path) -> None:
        """No false positives: an ordinary close keeps teaching exactly as before."""
        position = _position()
        strategy, _, _, post_mortem = _strategy(tmp_path, position)

        await strategy.close_position(
            "take_profit", 84500.0, make_market_conditions(),
            filled_quantity=0.00554,
            evidence=proven_exit(
                fill_verification="verified", reference_price=84510.0, deviation_pct=-0.0118
            ),
        )

        post_mortem.analyze_closed_trade.assert_awaited_once()
        strategy.brain_service.update_from_closed_trade.assert_called_once()


FILL_AT = datetime(2026, 10, 22, 0, 0, 33, 281000, tzinfo=timezone.utc)


def _market_evidence(age_seconds: float, window_seconds: float | None) -> dict[str, Any]:
    """A ``verified`` payload whose reference print sits ``age_seconds`` before the fill."""
    evidence: dict[str, Any] = {
        "fill_verification": "verified",
        "reference_price": 84037.19,
        "reference_source": "binance_spot_agg_trades",
        "fill_timestamp": FILL_AT.isoformat(),
        "reference_timestamp": (FILL_AT - timedelta(seconds=age_seconds)).isoformat(),
    }
    if window_seconds is not None:
        evidence["reference_window_seconds"] = window_seconds
        evidence["reference_window_start"] = (FILL_AT - timedelta(seconds=window_seconds)).isoformat()
        evidence["reference_window_end"] = FILL_AT.isoformat()
    return evidence


class TestReferenceWindowIsBounded:
    """A widened window may never pass a print from minutes ago as "the price at the fill"."""

    def test_print_inside_the_window_verifies(self) -> None:
        assert market_verification_is_verified(_market_evidence(9.0, 10.0))

    def test_print_outside_the_window_is_not_verification(self) -> None:
        assert not market_verification_is_verified(_market_evidence(11.0, 10.0))

    def test_unbounded_window_is_refused(self) -> None:
        assert not market_verification_is_verified(
            _market_evidence(60.0, MAX_MARKET_REFERENCE_WINDOW_SECONDS + 1)
        )

    def test_window_bounds_alone_are_enough(self) -> None:
        payload = _market_evidence(9.0, None)
        payload["reference_window_start"] = (FILL_AT - timedelta(seconds=10)).isoformat()
        payload["reference_window_end"] = FILL_AT.isoformat()
        assert market_verification_is_verified(payload)

    def test_legacy_payload_without_a_window_keeps_the_one_second_rule(self) -> None:
        assert market_verification_is_verified(_market_evidence(0.5, None))
        assert not market_verification_is_verified(_market_evidence(2.0, None))

    def test_future_reference_print_is_never_verification(self) -> None:
        assert not market_verification_is_verified(_market_evidence(-1.0, 10.0))


class TestMissingSampleIsNotADivergence:
    """A divergence says the venue's price was wrong; a gap says nothing was measured.

    2026-10-22: the executor had no public print to compare against (a one-second
    window on a thin pair), and the operator-facing line read "Lesson skipped: no exit
    evidence" — the same headline as a missing fill. The two findings are now named.
    """

    UNAVAILABLE: ClassVar[dict[str, Any]] = {
        "fill_verification": "unavailable",
        "reference_window_seconds": 10.0,
        "verification_reason": "no_reference_print",
    }

    def test_gap_reason_never_claims_the_venue_printed_a_wrong_price(self) -> None:
        reason = market_verification_gap_reason(self.UNAVAILABLE)

        assert "no public print" in reason
        assert "10s" in reason
        assert "not a price the real market made" not in reason

    def test_gap_reason_names_a_switched_off_check(self) -> None:
        assert "switched off" in market_verification_gap_reason({"fill_verification": "disabled"})

    def test_missing_sample_is_refused_as_unverified_not_as_divergent(self) -> None:
        learnable, reason = exit_evidence_is_learnable(
            proven_exit(fill_verification="unavailable", market_verification=self.UNAVAILABLE)
        )

        assert learnable is False
        assert "no public print" in reason
        assert "not a price the real market made" not in reason

    async def test_unavailable_close_keeps_the_no_evidence_headline(self, tmp_path: Path) -> None:
        position = _position()
        strategy, logger, persistence, post_mortem = _strategy(tmp_path, position)

        await strategy.close_position(
            "take_profit", 84500.0, make_market_conditions(),
            filled_quantity=0.00554,
            evidence=proven_exit(fill_verification="unavailable", market_verification=self.UNAVAILABLE),
        )

        decision = persistence.async_save_trade_decision.await_args.args[0]
        assert FILL_DIVERGENCE_MARKER not in decision.reasoning
        assert "EXECUTION UNVERIFIED" in decision.reasoning
        assert decision.execution_quality["eligible"] is False
        post_mortem.analyze_closed_trade.assert_not_awaited()
        strategy.brain_service.update_from_closed_trade.assert_not_called()
        assert _logged(logger, LESSON_SKIPPED_NO_EVIDENCE)
        assert not _logged(logger, LESSON_SKIPPED_FILL_DIVERGENCE)
        assert _logged(logger, "no public print")


class TestDivergenceVocabulary:
    """The refusal must name the numbers: an operator (and the model) gets the evidence."""

    def test_reason_carries_both_prices(self) -> None:
        evidence = proven_exit(
            price=77506.0, fill_verification="divergent",
            reference_price=81325.09, deviation_pct=-4.6963,
        )

        reason = fill_divergence_reason(evidence)

        assert "77,506.00" in reason
        assert "81,325.09" in reason
        assert "-4.70%" in reason

    def test_guard_refuses_a_divergent_evidence_directly(self) -> None:
        learnable, reason = exit_evidence_is_learnable(
            proven_exit(
                price=77506.0, fill_verification="divergent",
                reference_price=81325.09, deviation_pct=-4.6963,
            )
        )

        assert learnable is False
        assert "not a price the real market made" in reason

    @pytest.mark.parametrize("value", ["verified", "unavailable", "disabled", None, ""])
    def test_only_verified_market_evidence_can_teach(self, value: Any) -> None:
        """Unknown, disabled and legacy prices never authorize learning."""
        learnable, reason = exit_evidence_is_learnable(
            proven_exit(price=84500.0, fill_verification=value)
        )

        assert learnable is (value == "verified"), reason


class TestDivergentFillIsNotCounted:
    """A marked CLOSE never moves a statistic — the pair is dropped, not counted."""

    @staticmethod
    def _entry(price: float, quantity: float = 0.00554) -> dict[str, Any]:
        return {"action": "SELL", "price": price, "quantity": quantity,
                "quote_amount": price * quantity, "reasoning": "entry"}

    @staticmethod
    def _close(price: float, reasoning: str = "Position closed") -> dict[str, Any]:
        return {"action": "CLOSE_SHORT", "price": price, "reasoning": reasoning}

    def test_marked_close_pair_is_excluded_from_statistics(self) -> None:
        history = [
            self._entry(81308.48),
            self._close(77506.0, f"{FILL_DIVERGENCE_MARKER}: take_profit_filled @ 77506.0"),
            self._entry(81308.48),
            self._close(82000.0),  # an ordinary, honest loss
        ]

        stats = StatisticsCalculator.calculate_from_history(history, 10000.0)

        assert stats.total_trades == 1
        assert stats.losing_trades == 1
        assert stats.winning_trades == 0
        assert stats.best_trade_pct < 0  # the +4.68% artifact is gone

    def test_without_the_marker_the_same_pair_counts(self) -> None:
        """The guard keys on the marker only — an unmarked close still counts."""
        history = [self._entry(81308.48), self._close(77506.0, "Position closed")]

        stats = StatisticsCalculator.calculate_from_history(history, 10000.0)

        assert stats.total_trades == 1
        assert stats.best_trade_pct == pytest.approx(4.6766, abs=1e-3)


class TestMarketEvidenceQuarantine:
    async def test_verified_label_without_historical_provenance_cannot_teach(self, tmp_path):
        strategy, _, persistence, post_mortem = _strategy(tmp_path, _position())
        await strategy.close_position("take_profit", 84500.0, make_market_conditions(),
                                     evidence=proven_exit(market_verification=None))
        assert persistence.async_save_trade_decision.await_args.args[0].execution_quality["eligible"] is False
        post_mortem.analyze_closed_trade.assert_not_awaited()
        strategy.brain_service.update_from_closed_trade.assert_not_called()

    def test_equity_curve_keeps_excluded_exit_flat_without_misassigning_next_result(self):
        history = [
            {"action": "SELL", "price": 100.0, "quantity": 1.0, "timestamp": "entry1"},
            {"action": "CLOSE_SHORT", "price": 50.0, "timestamp": "exit1", "execution_quality": {"eligible": False}},
            {"action": "SELL", "price": 100.0, "quantity": 1.0, "timestamp": "entry2"},
            {"action": "CLOSE_SHORT", "price": 110.0, "timestamp": "exit2"},
        ]
        router = PerformanceRouter(make_config(), null_logger(), MagicMock())
        curve = router._process_trade_history(history, {"initial_capital": 1000.0})
        assert [p["value"] for p in curve] == [1000.0, 1000.0, 1000.0, 990.0]

    @pytest.mark.parametrize("status", ["unavailable", "disabled", None, "", "divergent"])
    async def test_missing_or_divergent_market_reference_books_without_learning_or_statistics(self, tmp_path, status):
        strategy, _, persistence, post_mortem = _strategy(tmp_path, _position())
        await strategy.close_position("take_profit", 77506.0, make_market_conditions(),
                                     evidence=proven_exit(price=77506, fill_verification=status))
        decision = persistence.async_save_trade_decision.await_args.args[0]
        assert decision.execution_quality["eligible"] is False
        history = [{"action": "BUY", "price": 81200.45, "quantity": 0.00554}, decision.to_dict()]
        assert StatisticsCalculator.calculate_from_history(history).total_trades == 0
        assert strategy.current_position is None
        post_mortem.analyze_closed_trade.assert_not_awaited()
        strategy.brain_service.update_from_closed_trade.assert_not_called()

    async def test_divergent_entry_survives_sqlite_and_position_restart_and_excludes_normal_exit(self, tmp_path):
        strategy, persistence, intent, receipt = _EntrySettlement.setup_entry(tmp_path)
        receipt.update(average_price=90000, fill_verification="divergent", reference_price=81325.09)
        Path(strategy.config.EXECUTOR_VERDICT_PATH).write_text(json.dumps(receipt) + "\n", encoding="utf-8")
        await strategy.resolve_position_intents_after_forward(intent.order_id, True)
        restarted = PersistenceManager(null_logger(), str(persistence.data_dir))
        position = restarted.load_position()
        assert position.entry_execution_quality["fill_verification"] == "divergent"
        assert restarted.load_trade_history()[0]["execution_quality"]["eligible"] is False
        strategy.current_position = position
        strategy.persistence = restarted
        strategy.memory_service = TradingMemoryService(null_logger(), restarted)
        post_mortem = MagicMock()
        post_mortem.analyze_closed_trade = AsyncMock(return_value=MagicMock())
        strategy.post_mortem_service = post_mortem
        await strategy.close_position("normal_exit", 81000, make_market_conditions(),
                                     evidence=proven_exit(price=81000, quantity=position.size))
        post_mortem.analyze_closed_trade.assert_not_awaited()
        strategy.brain_service.update_from_closed_trade.assert_not_called()
        history = restarted.load_trade_history()
        assert len(history) == 2
        assert StatisticsCalculator.calculate_from_history(history).total_trades == 0
        memory = TradingMemoryService(null_logger(), restarted)
        assert "Overall Performance" not in memory.get_context_summary()


class TestProvenExitTeachesOnce:
    """Evidence + validated post-mortem = exactly one brain entry per exit event."""

    async def test_the_brain_is_updated_with_the_exit_evidence(self, tmp_path: Path) -> None:
        position = _position()
        strategy, _, persistence, post_mortem = _strategy(tmp_path, position)

        await strategy.close_position(
            "take_profit", 84500.0, make_market_conditions(),
            filled_quantity=0.00554, evidence=proven_exit(),
        )

        post_mortem.analyze_closed_trade.assert_awaited_once()
        brain = strategy.brain_service
        brain.update_from_closed_trade.assert_called_once()
        evidence = brain.update_from_closed_trade.call_args.kwargs["evidence"]
        assert (evidence.price, evidence.quantity) == (84500.0, 0.00554)
        assert evidence.event_id == "executor_exit_journal#event-1"
        assert evidence.source == "executor_exit_journal"
        decision = persistence.async_save_trade_decision.await_args.args[0]
        assert decision.quantity == pytest.approx(0.00554)
        assert decision.fee is None

    async def test_the_same_exit_event_is_taught_only_once(self, tmp_path: Path) -> None:
        """A re-entry of the close path for one exit event must not double-teach."""
        position = _position()
        strategy, logger, _, post_mortem = _strategy(tmp_path, position)

        await strategy.close_position(
            "take_profit", 84500.0, make_market_conditions(),
            filled_quantity=0.00554, evidence=proven_exit(),
        )
        strategy.current_position = position
        await strategy.close_position(
            "take_profit", 84500.0, make_market_conditions(),
            filled_quantity=0.00554, evidence=proven_exit(),
        )

        assert post_mortem.analyze_closed_trade.await_count == 2
        assert strategy.brain_service.update_from_closed_trade.call_count == 1
        assert _logged(logger, "already processed")

    async def test_a_failed_brain_write_may_be_retried(self, tmp_path: Path) -> None:
        """The dedup key is not kept when the lesson did not land (retry still teaches)."""
        position = _position()
        strategy, _, _, _ = _strategy(tmp_path, position)
        brain = strategy.brain_service
        brain.update_from_closed_trade.side_effect = [RuntimeError("brain down"), None]

        await strategy.close_position(
            "take_profit", 84500.0, make_market_conditions(),
            filled_quantity=0.00554, evidence=proven_exit(),
        )
        strategy.current_position = position
        await strategy.close_position(
            "take_profit", 84500.0, make_market_conditions(),
            filled_quantity=0.00554, evidence=proven_exit(),
        )

        assert brain.update_from_closed_trade.call_count == 2


class TestBookedAmountComesFromTheFill:
    """The CLOSE row carries the actual filled amount, never the local size alone."""

    async def test_the_evidence_amount_is_booked_without_an_explicit_quantity(self, tmp_path: Path) -> None:
        position = _position(size=0.045)
        strategy, logger, persistence, _ = _strategy(tmp_path, position)

        await strategy.close_position(
            "take_profit", 84500.0, make_market_conditions(),
            evidence=proven_exit(quantity=0.00554),
        )

        decision = persistence.async_save_trade_decision.await_args.args[0]
        assert decision.quantity == pytest.approx(0.00554)
        assert decision.quantity != pytest.approx(0.045)
        assert "0.00554000" in decision.reasoning and "0.04500000" in decision.reasoning
        assert _logged(logger, "QUANTITY DIVERGENCE")

    async def test_without_any_quantity_the_legacy_local_size_is_kept(self, tmp_path: Path) -> None:
        """Backward compatibility: no evidence fields at all behaves exactly as before."""
        position = _position()
        strategy, _, persistence, post_mortem = _strategy(tmp_path, position)

        await strategy.close_position("take_profit", 84500.0, make_market_conditions())

        decision = persistence.async_save_trade_decision.await_args.args[0]
        assert decision.quantity == pytest.approx(position.size)
        assert "WARNING: the executor fill quantity" not in decision.reasoning
        post_mortem.analyze_closed_trade.assert_not_awaited()
        strategy.brain_service.update_from_closed_trade.assert_not_called()

    async def test_a_matching_fill_amount_adds_no_divergence_note(self, tmp_path: Path) -> None:
        position = _position()
        strategy, _, persistence, _ = _strategy(tmp_path, position)

        await strategy.close_position(
            "take_profit", 84500.0, make_market_conditions(),
            filled_quantity=0.00554, evidence=proven_exit(),
        )

        decision = persistence.async_save_trade_decision.await_args.args[0]
        assert decision.quantity == pytest.approx(0.00554)
        assert "WARNING: the executor fill quantity" not in decision.reasoning
