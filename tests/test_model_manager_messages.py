"""Unit tests for ModelManager._prepare_messages() role separation."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.managers.model_manager import ModelManager
from src.managers.provider_types import InvocationResult
from src.platforms.ai_providers.response_models import ChatResponseModel


def _make_manager() -> ModelManager:
    mgr = ModelManager.__new__(ModelManager)
    mgr.logger = MagicMock()
    counter = MagicMock()
    counter.count_tokens.return_value = 10
    mgr.token_counter = counter
    return mgr


class TestPrepareMessagesRoles:
    """Tests that _prepare_messages returns properly role-tagged message lists."""

    def setup_method(self) -> None:
        self.mgr = _make_manager()

    def test_with_system_returns_two_messages(self) -> None:
        messages = self.mgr._prepare_messages("user query", system_message="system instructions")
        assert len(messages) == 2

    def test_with_system_first_message_is_system_role(self) -> None:
        messages = self.mgr._prepare_messages("user query", system_message="system instructions")
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "system instructions"

    def test_with_system_second_message_is_user_role(self) -> None:
        messages = self.mgr._prepare_messages("user query", system_message="system instructions")
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "user query"

    def test_without_system_returns_single_user_message(self) -> None:
        messages = self.mgr._prepare_messages("user query only")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "user query only"

    def test_reset_session_stats_called_once(self) -> None:
        self.mgr._prepare_messages("prompt", system_message="sys")
        self.mgr.token_counter.reset_session_stats.assert_called_once()

    def test_count_tokens_called_for_system_and_prompt(self) -> None:
        self.mgr._prepare_messages("my prompt", system_message="my system")
        calls = [call.args[0] for call in self.mgr.token_counter.count_tokens.call_args_list]
        assert "my system" in calls
        assert "my prompt" in calls

    def test_count_tokens_called_once_without_system(self) -> None:
        self.mgr._prepare_messages("only prompt")
        assert self.mgr.token_counter.count_tokens.call_count == 1


class TestContractRepairMessages:
    """Tests for send_contract_repair() message construction."""

    def _make_repair_manager(self, response_text: str, provider: str = "deepseekai"):
        mgr = _make_manager()
        mgr.provider = "deepseek"
        mgr.cost_storage = MagicMock()
        mgr.model_pricing = MagicMock()
        orchestrator = MagicMock()
        orchestrator.get_text_response = AsyncMock(return_value=InvocationResult(
            success=True,
            response=ChatResponseModel.from_content(response_text),
            provider=provider,
            model="deepseek-flash",
        ))
        mgr._orchestrator = orchestrator
        return mgr, orchestrator

    @pytest.mark.asyncio
    async def test_contract_repair_replays_turn_and_requests_json_block(self) -> None:
        json_block = '```json\n{"analysis": {"signal": "HOLD", "confidence": 60}}\n```'
        mgr, orchestrator = self._make_repair_manager(json_block)

        text = await mgr.send_contract_repair(
            system_message="system instructions",
            prompt="original user prompt",
            previous_response="narrative without the block",
            provider=None,
            model=None,
        )

        assert text == json_block
        provider_arg, messages, model_arg = orchestrator.get_text_response.await_args.args
        assert provider_arg == "deepseek"
        assert model_arg is None
        assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
        assert messages[0]["content"] == "system instructions"
        assert messages[1]["content"] == "original user prompt"
        assert messages[2]["content"] == "narrative without the block"
        assert "json block" in messages[3]["content"]

    @pytest.mark.asyncio
    async def test_contract_repair_honors_provider_and_model_override(self) -> None:
        mgr, orchestrator = self._make_repair_manager("```json\n{}\n```", provider="googleai")

        await mgr.send_contract_repair(
            system_message="s",
            prompt="p",
            previous_response="r",
            provider="googleai",
            model="gemini-3.8-flash",
        )

        provider_arg, _, model_arg = orchestrator.get_text_response.await_args.args
        assert provider_arg == "googleai"
        assert model_arg == "gemini-3.8-flash"
