"""Tests for PostMortemService.analyze_closed_trade with mocked LLM."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.trading.post_mortem import PostMortemService, PostMortemResult


class TestAnalyzeClosedTrade:
    """Tests for PostMortemService.analyze_closed_trade."""

    @staticmethod
    def _make_service(model_manager=None, unified_parser=None, repository=None):
        """Factory: create PostMortemService with mocked deps."""
        return PostMortemService(
            logger=MagicMock(),
            model_manager=model_manager or MagicMock(),
            unified_parser=unified_parser or MagicMock(),
            repository=repository or MagicMock(),
        )

    @staticmethod
    def _make_position(**overrides):
        """Factory: create a mock closed position."""
        pos = MagicMock()
        pos.symbol = "BTC/USDC"
        pos.direction = "LONG"
        pos.entry_price = 72500.0
        pos.stop_loss = 71200.0
        pos.take_profit = 75000.0
        pos.size_pct = 0.05
        pos.confidence = "MEDIUM"
        pos.adx_at_entry = 28.0
        pos.rsi_at_entry = 58.0
        pos.trend_direction_at_entry = "BULLISH"
        pos.volatility_level = "MEDIUM"
        pos.rr_ratio_at_entry = 1.9
        pos.max_drawdown_pct = -2.8
        pos.max_profit_pct = 1.5
        pos.entry_time = datetime(2026, 6, 17, 8, 0, 0, tzinfo=timezone.utc)
        for k, v in overrides.items():
            setattr(pos, k, v)
        return pos

    @staticmethod
    def _make_entry_decision(**overrides):
        """Factory: create a mock entry decision."""
        d = MagicMock()
        d.reasoning = "Expected bullish continuation based on ADX strength and volume."
        d.price = 72500.0
        d.timestamp = datetime(2026, 6, 17, 8, 0, 0, tzinfo=timezone.utc)
        for k, v in overrides.items():
            setattr(d, k, v)
        return d

    @staticmethod
    def _make_exit_decision(**overrides):
        """Factory: create a mock exit decision."""
        d = MagicMock()
        d.reasoning = "Stop loss hit as price broke below support."
        d.price = 71200.0
        d.timestamp = datetime(2026, 6, 18, 4, 0, 0, tzinfo=timezone.utc)
        for k, v in overrides.items():
            setattr(d, k, v)
        return d

    @pytest.mark.asyncio
    async def test_successful_analysis_stores_result(self):
        """Valid LLM response should be parsed and stored in repository."""
        parser = MagicMock()
        parser.extract_json_block.return_value = None  # force raw JSON path
        manager = MagicMock()
        manager.send_prompt = AsyncMock(return_value=(
            '{"verdict": "good_exit", "llm_analysis": "Trade followed plan.", '
            '"expected_vs_actual": "Expected 75k, hit 71.2k stop.", '
            '"lesson_learned": "When trend reverses, honor the stop."}'
        ))
        repo = MagicMock()
        repo.insert_post_mortem.return_value = 1

        service = self._make_service(model_manager=manager, unified_parser=parser, repository=repo)
        result = await service.analyze_closed_trade(
            closed_position=self._make_position(),
            entry_decision=self._make_entry_decision(),
            exit_decision=self._make_exit_decision(),
            pnl=-1.79,
            reason="stop_loss",
        )

        assert result is not None
        assert isinstance(result, PostMortemResult)
        assert result.verdict == "good_exit"
        assert result.lesson_learned
        repo.insert_post_mortem.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_llm_response_returns_none(self):
        """Empty LLM response should return None and not store."""
        manager = MagicMock()
        manager.send_prompt = AsyncMock(return_value="")
        repo = MagicMock()

        service = self._make_service(model_manager=manager, repository=repo)
        result = await service.analyze_closed_trade(
            closed_position=self._make_position(),
            entry_decision=self._make_entry_decision(),
            exit_decision=self._make_exit_decision(),
            pnl=-1.79,
            reason="stop_loss",
        )

        assert result is None
        repo.insert_post_mortem.assert_not_called()

    @pytest.mark.asyncio
    async def test_malformed_json_returns_none(self):
        """Unparseable LLM response should return None gracefully."""
        parser = MagicMock()
        parser.extract_json_block.return_value = None
        manager = MagicMock()
        manager.send_prompt = AsyncMock(return_value="not json at all")
        repo = MagicMock()

        service = self._make_service(model_manager=manager, unified_parser=parser, repository=repo)
        result = await service.analyze_closed_trade(
            closed_position=self._make_position(),
            entry_decision=self._make_entry_decision(),
            exit_decision=self._make_exit_decision(),
            pnl=-1.79,
            reason="stop_loss",
        )

        assert result is None
        repo.insert_post_mortem.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_required_field_returns_none(self):
        """JSON missing 'verdict' should fail Pydantic validation, return None."""
        parser = MagicMock()
        parser.extract_json_block.return_value = None
        manager = MagicMock()
        manager.send_prompt = AsyncMock(return_value=(
            '{"llm_analysis": "ok", "expected_vs_actual": "ok", "lesson_learned": "ok"}'
        ))
        repo = MagicMock()

        service = self._make_service(model_manager=manager, unified_parser=parser, repository=repo)
        result = await service.analyze_closed_trade(
            closed_position=self._make_position(),
            entry_decision=self._make_entry_decision(),
            exit_decision=self._make_exit_decision(),
            pnl=-1.79,
            reason="stop_loss",
        )

        assert result is None
        repo.insert_post_mortem.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_exception_returns_none(self):
        """LLM call raising exception should return None, not propagate."""
        manager = MagicMock()
        manager.send_prompt = AsyncMock(side_effect=RuntimeError("API timeout"))
        repo = MagicMock()

        service = self._make_service(model_manager=manager, repository=repo)
        result = await service.analyze_closed_trade(
            closed_position=self._make_position(),
            entry_decision=self._make_entry_decision(),
            exit_decision=self._make_exit_decision(),
            pnl=-1.79,
            reason="stop_loss",
        )

        assert result is None
        repo.insert_post_mortem.assert_not_called()

    @pytest.mark.asyncio
    async def test_repository_exception_returns_none(self):
        """Repository insert failure should return None, not propagate."""
        parser = MagicMock()
        parser.extract_json_block.return_value = None
        manager = MagicMock()
        manager.send_prompt = AsyncMock(return_value=(
            '{"verdict": "good_exit", "llm_analysis": "ok", '
            '"expected_vs_actual": "ok", "lesson_learned": "ok"}'
        ))
        repo = MagicMock()
        repo.insert_post_mortem.side_effect = RuntimeError("DB locked")

        service = self._make_service(model_manager=manager, unified_parser=parser, repository=repo)
        result = await service.analyze_closed_trade(
            closed_position=self._make_position(),
            entry_decision=self._make_entry_decision(),
            exit_decision=self._make_exit_decision(),
            pnl=-1.79,
            reason="stop_loss",
        )

        assert result is None
