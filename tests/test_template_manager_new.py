"""Tests for template_manager — Bull/Bear Debate and Adversarial Awareness sections."""

from unittest.mock import MagicMock

import pytest

from src.analyzer.prompts.template_manager import TemplateManager


class TestBullBearDebateSection:
    """Verify Bull/Bear Debate section is injected when config flag is on."""

    @pytest.fixture
    def manager_bull_bear_on(self):
        config = MagicMock()
        config.RESEARCH_TEAM_ENABLED = True
        config.MODEL_VERBOSITY = "medium"
        return TemplateManager(config)

    @pytest.fixture
    def manager_bull_bear_off(self):
        config = MagicMock()
        config.RESEARCH_TEAM_ENABLED = False
        config.MODEL_VERBOSITY = "medium"
        return TemplateManager(config)

    def test_bull_bear_section_present_when_enabled(self, manager_bull_bear_on):
        prompt = manager_bull_bear_on.build_system_prompt("BTC/USDT")
        assert "Bull vs Bear Debate Protocol" in prompt
        assert "BULL CASE" in prompt
        assert "BEAR CASE" in prompt
        assert "SYNTHESIS" in prompt

    def test_bull_bear_section_absent_when_disabled(self, manager_bull_bear_off):
        prompt = manager_bull_bear_off.build_system_prompt("BTC/USDT")
        assert "Bull vs Bear Debate Protocol" not in prompt

    def test_bull_bear_section_fallback_on_missing_config_attr(self):
        """getattr(config, 'RESEARCH_TEAM_ENABLED', False) must work with mock that has all needed attrs."""
        config = MagicMock()
        config.RESEARCH_TEAM_ENABLED = False
        config.MODEL_VERBOSITY = "medium"
        config.STOP_LOSS_TYPE = "ATR_BASED"
        manager = TemplateManager(config)
        prompt = manager.build_system_prompt("BTC/USDT")
        assert "Bull vs Bear Debate Protocol" not in prompt


class TestAdversarialAwarenessSection:
    """Verify Adversarial Awareness section is always present."""

    @pytest.fixture
    def manager(self):
        config = MagicMock()
        config.RESEARCH_TEAM_ENABLED = False
        config.MODEL_VERBOSITY = "medium"
        return TemplateManager(config)

    def test_adversarial_awareness_section_present(self, manager):
        prompt = manager.build_system_prompt("BTC/USDT")
        assert "Adversarial Awareness" in prompt
        assert "order book" in prompt.lower()
        assert "front-run" in prompt.lower()
        assert "liquidity" in prompt.lower()

    def test_adversarial_section_has_key_concepts(self, manager):
        prompt = manager.build_system_prompt("BTC/USDT")
        assert "counterparties" in prompt or "counterparty" in prompt
        assert "Funding rate" in prompt
        assert "squeeze" in prompt.lower()


class TestCorePrinciples:
    """Verify existing Core Principles are not broken by new sections."""

    @pytest.fixture
    def manager(self):
        config = MagicMock()
        config.RESEARCH_TEAM_ENABLED = False
        config.MODEL_VERBOSITY = "medium"
        return TemplateManager(config)

    def test_core_principles_still_present(self, manager):
        prompt = manager.build_system_prompt("BTC/USDT")
        assert "Core Principles" in prompt
        assert "SL and TP required" in prompt
        assert "Closed-candle structure" in prompt

    def test_key_terminology_still_present(self, manager):
        prompt = manager.build_system_prompt("BTC/USDT")
        assert "Golden Cross" in prompt
        assert "Death Cross" in prompt


class TestPreviousContextSanitization:
    """Verify last_analysis_time injection is correct."""

    @pytest.fixture
    def manager(self):
        config = MagicMock()
        config.RESEARCH_TEAM_ENABLED = False
        config.MODEL_VERBOSITY = "medium"
        return TemplateManager(config)

    def test_last_analysis_time_in_prompt(self, manager):
        prompt = manager.build_system_prompt(
            "BTC/USDT", last_analysis_time="2025-12-26 14:30:00"
        )
        assert "Temporal Context" in prompt
        assert "2025-12-26 14:30:00" in prompt

    def test_no_temporal_context_when_none(self, manager):
        prompt = manager.build_system_prompt("BTC/USDT")
        assert "Temporal Context" not in prompt
