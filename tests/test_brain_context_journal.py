"""Tests for post-mortem journal injection into brain context."""

from unittest.mock import MagicMock

from src.trading.brain_context import BrainContextProvider


class TestTradeJournalContext:
    """Tests for post-mortem injection into brain context."""

    @staticmethod
    def _make_repo(post_mortems=None):
        """Factory: create a mock PostMortemRepository."""
        repo = MagicMock()
        repo.get_recent_post_mortems.return_value = post_mortems or []
        return repo

    @staticmethod
    def _make_provider(post_mortem_repo=None):
        """Factory: create BrainContextProvider with mocked deps."""
        vm = MagicMock()
        vm.trade_count = 0
        vm.experience_count = 0
        vm.get_context_for_prompt.return_value = ""
        vm.get_blocked_trade_feedback.return_value = None
        vm.get_relevant_rules.return_value = []
        vm.compute_confidence_stats.return_value = {}
        vm.get_direction_bias.return_value = None
        vm.get_confidence_recommendation.return_value = None
        exit_profiles = MagicMock()
        exit_profiles.replace_unknown_exit_profile_text.return_value = ""
        exit_profiles.render_rule_text.return_value = ""
        return BrainContextProvider(
            vector_memory=vm,
            exit_profiles=exit_profiles,
            post_mortem_repo=post_mortem_repo,
        )

    def test_journal_section_appears_when_post_mortems_exist(self):
        """get_context() should include '### Trade Journal' when post-mortems exist."""
        repo = self._make_repo(post_mortems=[
            {
                "verdict": "overestimated_breakout",
                "created_at": "2026-06-18 12:00:00",
                "symbol": "BTC/USDC",
                "lesson_learned": "Wait for confirmation before entering.",
                "pnl_pct": -3.2,
            },
        ])
        provider = self._make_provider(post_mortem_repo=repo)
        context = provider.get_context()
        assert "### Trade Journal (Recent Post-Mortem Lessons):" in context
        assert "overestimated_breakout" in context
        assert "BTC/USDC" in context
        assert "Wait for confirmation" in context

    def test_journal_section_absent_when_no_post_mortems(self):
        """get_context() should NOT include journal section when empty."""
        repo = self._make_repo(post_mortems=[])
        provider = self._make_provider(post_mortem_repo=repo)
        context = provider.get_context()
        assert "### Trade Journal (Recent Post-Mortem Lessons):" not in context

    def test_journal_section_absent_when_repo_is_none(self):
        """get_context() should NOT include journal section when repo not configured."""
        provider = self._make_provider(post_mortem_repo=None)
        context = provider.get_context()
        assert "### Trade Journal (Recent Post-Mortem Lessons):" not in context

    def test_journal_section_absent_on_repo_exception(self):
        """get_context() should gracefully skip journal on repo error."""
        repo = self._make_repo()
        repo.get_recent_post_mortems.side_effect = RuntimeError("DB error")
        provider = self._make_provider(post_mortem_repo=repo)
        context = provider.get_context()
        assert "### Trade Journal (Recent Post-Mortem Lessons):" not in context

    def test_multiple_post_mortems_rendered(self):
        """Multiple post-mortems should all appear in the journal section."""
        repo = self._make_repo(post_mortems=[
            {
                "verdict": "overestimated_breakout",
                "created_at": "2026-06-18 12:00:00",
                "symbol": "BTC/USDC",
                "lesson_learned": "Lesson A",
                "pnl_pct": -3.2,
            },
            {
                "verdict": "good_exit",
                "created_at": "2026-06-17 12:00:00",
                "symbol": "ETH/USDC",
                "lesson_learned": "Lesson B",
                "pnl_pct": 2.1,
            },
        ])
        provider = self._make_provider(post_mortem_repo=repo)
        context = provider.get_context()
        assert "Lesson A" in context
        assert "Lesson B" in context
        assert "good_exit" in context
        assert "overestimated_breakout" in context

    def test_pnl_formatted_in_journal(self):
        """P&L should be formatted with sign and percent."""
        repo = self._make_repo(post_mortems=[
            {
                "verdict": "test",
                "created_at": "2026-06-18 12:00:00",
                "symbol": "BTC/USDC",
                "lesson_learned": "Lesson",
                "pnl_pct": 5.5,
            },
        ])
        provider = self._make_provider(post_mortem_repo=repo)
        context = provider.get_context()
        assert "P&L: +5.5%" in context

    def test_pnl_omitted_when_none(self):
        """P&L should be omitted from journal entry when None."""
        repo = self._make_repo(post_mortems=[
            {
                "verdict": "test",
                "created_at": "2026-06-18 12:00:00",
                "symbol": "BTC/USDC",
                "lesson_learned": "Lesson",
                "pnl_pct": None,
            },
        ])
        provider = self._make_provider(post_mortem_repo=repo)
        context = provider.get_context()
        assert "P&L:" not in context
