"""Unit tests for ModelManager._prepare_messages() role separation."""
from unittest.mock import MagicMock, patch

import pytest

from src.managers.model_manager import ModelManager


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
