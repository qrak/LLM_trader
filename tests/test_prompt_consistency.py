"""Prompt consistency tests for continuity sanitization and contract wording."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.analyzer.prompts.template_manager import TemplateManager
from src.utils.timeframe_validator import TimeframeValidator


def _make_manager() -> TemplateManager:
    """Create a TemplateManager with minimal production-like config."""
    config = SimpleNamespace(
        STOP_LOSS_TYPE="soft",
        STOP_LOSS_CHECK_INTERVAL="1h",
        TAKE_PROFIT_TYPE="soft",
        TAKE_PROFIT_CHECK_INTERVAL="1h",
        MAX_POSITION_SIZE=0.10,
        AI_CHART_CANDLE_LIMIT=120,
        MODEL_VERBOSITY="high",
    )
    return TemplateManager(
        config=config,
        logger=MagicMock(),
        timeframe_validator=TimeframeValidator,
    )


def _previous_context(system_prompt: str) -> str:
    """Extract only the previous-analysis context from a system prompt."""
    section = system_prompt.split("## PREVIOUS ANALYSIS CONTEXT", 1)[1]
    return section.split("### DETERMINISTIC TIME CHECK", 1)[0]


class TestPreviousContextSanitization:
    """Regression tests for stale prompt instruction leakage."""

    def setup_method(self) -> None:
        self.manager = _make_manager()

    def test_extract_previous_analysis_uses_last_valid_analysis_block(self) -> None:
        schema_example = {"analysis": {"signal": "HOLD", "confidence": 10}}
        actual_analysis = {"analysis": {"signal": "SELL", "confidence": 82, "reasoning": "Breakdown confirmed."}}
        previous_response = (
            "```json\n"
            + json.dumps(schema_example)
            + "\n```\n"
            "1) MARKET STRUCTURE: Bearish continuation.\n"
            "```json\n"
            + json.dumps(actual_analysis)
            + "\n```"
        )

        system_prompt = self.manager.build_system_prompt("BTC/USDT", previous_response=previous_response)

        previous_section = _previous_context(system_prompt)
        assert "Signal: SELL" in previous_section
        assert "confidence: 82" in previous_section
        assert "Signal: HOLD" not in previous_section

    def test_previous_reasoning_removes_echoed_prompt_contract_sections(self) -> None:
        previous_response = """
1) MARKET STRUCTURE: Bearish but range-bound.
## Response Format
Allowed signals: BUY, SELL, HOLD, CLOSE, UPDATE.
CONFLUENCE SCORING:
- trend_alignment: score this factor.
POSITION SIZING FORMULA (calculate before finalizing):
- Suggested minimum for normal valid entries: 0.080.
2) DECISION: HOLD because invalidation is unclear.
```json
{"analysis": {"signal": "HOLD", "confidence": 73, "reasoning": "Invalidation is unclear."}}
```
"""

        system_prompt = self.manager.build_system_prompt("BTC/USDT", previous_response=previous_response)

        previous_section = _previous_context(system_prompt)
        assert "1) MARKET STRUCTURE: Bearish but range-bound." in previous_section
        assert "2) DECISION: HOLD because invalidation is unclear." in previous_section
        assert "Allowed signals" not in previous_section
        assert "CONFLUENCE SCORING" not in previous_section
        assert "POSITION SIZING FORMULA" not in previous_section
        assert "0.080" not in previous_section

    def test_previous_reasoning_strict_mode_keeps_only_compact_output_lines(self) -> None:
        previous_response = """
You are an Institutional-Grade Crypto Trading Analyst managing BTC/USDT.
Analyze technical indicators, price action, volume, patterns, and news.
## Response Format
JSON rules: valid JSON only.
1) MARKET STRUCTURE: Prior answer line survives.
5) EXECUTION NOTE: Wait for closed-candle confirmation.
```json
{"analysis": {"signal": "HOLD", "confidence": 70}}
```
"""

        system_prompt = self.manager.build_system_prompt("BTC/USDT", previous_response=previous_response)

        previous_section = _previous_context(system_prompt)
        assert "Prior answer line survives" in previous_section
        assert "Wait for closed-candle confirmation" in previous_section
        assert "Institutional-Grade" not in previous_section
        assert "Analyze technical indicators" not in previous_section
        assert "JSON rules" not in previous_section


class TestPromptContractWording:
    """Regression tests for prompt contract consistency."""

    def setup_method(self) -> None:
        self.manager = _make_manager()

    def test_markdown_heading_rule_is_output_only(self) -> None:
        system_prompt = self.manager.build_system_prompt("BTC/USDT")

        assert "Output rule:" in system_prompt
        assert "prompt headings are organizational only" in system_prompt

    def test_update_progress_rule_has_no_competing_percentage_thresholds(self) -> None:
        system_prompt = self.manager.build_system_prompt(
            "BTC/USDT",
            performance_context="Recent trade performance available.",
        )
        response_template = self.manager.build_response_template()
        combined = f"{system_prompt}\n{response_template}"

        assert ">40%" not in combined
        assert "50%+ of the entry-to-TP distance" not in combined
        assert "hybrid tightening policy" in combined
        assert "material structure change" in combined

    def test_hold_open_position_contract_is_explicit(self) -> None:
        response_template = self.manager.build_response_template()

        assert "HOLD(open position) means no execution change" in response_template
        assert "must not repeat stale SL/TP values" in response_template
        assert "UPDATE is for an open position only" in response_template


class TestVerbosityParserContract:
    """Verify all verbosity levels preserve required parser contract sections."""

    def _make_mgr(self, level: str) -> TemplateManager:
        from types import SimpleNamespace
        config = SimpleNamespace(
            STOP_LOSS_TYPE="soft",
            STOP_LOSS_CHECK_INTERVAL="1h",
            TAKE_PROFIT_TYPE="soft",
            TAKE_PROFIT_CHECK_INTERVAL="1h",
            MAX_POSITION_SIZE=0.10,
            AI_CHART_CANDLE_LIMIT=120,
            MODEL_VERBOSITY=level,
        )
        return TemplateManager(config=config, logger=MagicMock(), timeframe_validator=TimeframeValidator)

    import pytest

    @pytest.mark.parametrize("level", ["low", "medium", "high"])
    def test_response_format_header_present(self, level: str) -> None:
        tmpl = self._make_mgr(level).build_response_template()
        assert "## Response Format" in tmpl

    @pytest.mark.parametrize("level", ["low", "medium", "high"])
    def test_fenced_json_block_present(self, level: str) -> None:
        tmpl = self._make_mgr(level).build_response_template()
        assert "```json" in tmpl

    @pytest.mark.parametrize("level", ["low", "medium", "high"])
    def test_analysis_wrapper_key_present(self, level: str) -> None:
        tmpl = self._make_mgr(level).build_response_template()
        assert '"analysis"' in tmpl

    @pytest.mark.parametrize("level", ["low", "medium", "high"])
    def test_allowed_signals_present(self, level: str) -> None:
        tmpl = self._make_mgr(level).build_response_template()
        assert "Allowed signals:" in tmpl
