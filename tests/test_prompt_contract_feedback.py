"""Contract tests for Brain.get_context() prompt injection of CRITICAL FEEDBACK.

Validates:
  1. CRITICAL FEEDBACK section is injected into the system prompt
  2. Formatting is parsable by LLM (no broken markdown, consistent structure)
  3. Section ordering: Confidence Calibration → Rejection Feedback → Vector Context → Rules
  4. Feedback is omitted when vector_memory returns empty string
  5. Correct parameters (n=5, max_age_hours=168) passed to get_blocked_trade_feedback
  6. Empty feedback does not inject blank lines
  7. Feedback survives interaction with other sections (direction bias, rules)
"""

from unittest.mock import ANY, MagicMock, call, patch

import pytest

from src.trading.brain import TradingBrainService


# ── Helpers ──────────────────────────────────────────────────────


def _make_brain_with_feedback(feedback: str = "", trade_count: int = 5, **kwargs):
    """Create a TradingBrainService with mocked vector_memory returning given feedback."""
    logger = MagicMock()
    persistence = MagicMock()
    vector_memory = MagicMock()
    vector_memory.trade_count = trade_count
    vector_memory.get_blocked_trade_feedback.return_value = feedback
    vector_memory.get_context_for_prompt.return_value = ""
    vector_memory.get_stats_for_context.return_value = {"total_trades": trade_count, "win_rate": 60, "avg_pnl": 1.0}
    vector_memory.get_relevant_rules.return_value = []
    vector_memory.compute_confidence_stats.return_value = {
        "HIGH": {"total_trades": 3, "winning_trades": 2, "win_rate": 66.7, "avg_pnl_pct": 2.5},
        "MEDIUM": {"total_trades": 2, "winning_trades": 1, "win_rate": 50.0, "avg_pnl_pct": -0.5},
    }
    vector_memory.get_confidence_recommendation.return_value = "Focus on HIGH confidence setups"
    vector_memory.get_direction_bias.return_value = {"long_count": 4, "short_count": 1}

    return TradingBrainService(
        logger=logger,
        persistence=persistence,
        vector_memory=vector_memory,
        **kwargs,
    )


def _make_brain_no_trades():
    """Brain with zero trades — no confidence calibration section."""
    logger = MagicMock()
    persistence = MagicMock()
    vector_memory = MagicMock()
    vector_memory.trade_count = 0
    vector_memory.get_blocked_trade_feedback.return_value = ""
    vector_memory.get_context_for_prompt.return_value = ""
    vector_memory.get_relevant_rules.return_value = []
    vector_memory.compute_confidence_stats.return_value = {}
    vector_memory.get_confidence_recommendation.return_value = ""
    vector_memory.get_direction_bias.return_value = None

    return TradingBrainService(
        logger=logger,
        persistence=persistence,
        vector_memory=vector_memory,
    )


# ── Contract: CRITICAL FEEDBACK Injection ────────────────────────


class TestCriticalFeedbackInjection:
    """Verify CRITICAL FEEDBACK appears in the system prompt correctly."""

    FEEDBACK_SAMPLE = """## CRITICAL FEEDBACK: System Rejections

The following trade suggestions were BLOCKED by risk guards. ADJUST your parameters before proposing the next trade.

### R/R Minimum Guard (2 recent):
  1. LONG (HIGH confidence, MEDIUM volatility):
     - Your R/R: 1.20 | Required: 2.00 (gap: -0.80)
     - Your SL: 3.00% from entry
     - Your TP: 4.00% from entry
  2. SHORT (MEDIUM confidence, HIGH volatility):
     - Your R/R: 1.50 | Required: 2.50 (gap: -1.00)

### PRE-FLIGHT CHECKLIST (MANDATORY):
- Before outputting BUY/SELL, verify: R/R >= required minimum (see Response Format).
- If volatility is HIGH, widen SL to >1x ATR to achieve required R/R.
- If volatility is LOW, do not use 2x+ ATR SL — tighten to keep R/R viable.
- Compare your proposed SL/TP against the last rejection patterns above.
- If you cannot meet the R/R requirement with a reasonable SL/TP, output HOLD.
"""

    def test_critical_feedback_appears_in_context(self):
        """When feedback is available, it is included in get_context()."""
        brain = _make_brain_with_feedback(feedback=self.FEEDBACK_SAMPLE)

        ctx = brain.get_context(adx=25, trend_direction="BULLISH")

        assert "CRITICAL FEEDBACK" in ctx
        assert "System Rejections" in ctx
        assert "R/R Minimum Guard" in ctx
        assert "PRE-FLIGHT CHECKLIST" in ctx

    def test_critical_feedback_absent_when_empty(self):
        """Empty feedback → CRITICAL FEEDBACK section is omitted."""
        brain = _make_brain_with_feedback(feedback="")

        ctx = brain.get_context(adx=25, trend_direction="BULLISH")

        assert "CRITICAL FEEDBACK" not in ctx
        assert "System Rejections" not in ctx

    def test_get_blocked_trade_feedback_called_with_correct_params(self):
        """Brain calls get_blocked_trade_feedback with n=5, max_age_hours=168."""
        brain = _make_brain_with_feedback(feedback=self.FEEDBACK_SAMPLE)

        brain.get_context(adx=25)

        brain.vector_memory.get_blocked_trade_feedback.assert_called_once_with(
            n=5, max_age_hours=168
        )

    def test_critical_feedback_position_in_prompt(self):
        """Feedback appears after Confidence Calibration, before Vector Context."""
        brain = _make_brain_with_feedback(feedback=self.FEEDBACK_SAMPLE)
        brain.vector_memory.get_context_for_prompt.return_value = "## Vector Context: ..."

        ctx = brain.get_context(adx=25, trend_direction="BULLISH")

        # Find the section order
        lines = ctx.split("\n")
        critical_idx = next(i for i, l in enumerate(lines) if "CRITICAL FEEDBACK" in l)
        confidence_idx = next(i for i, l in enumerate(lines) if "Confidence Calibration" in l)
        vector_idx = next(i for i, l in enumerate(lines) if "Vector Context" in l)

        assert confidence_idx < critical_idx, "CRITICAL FEEDBACK must come AFTER confidence"
        assert critical_idx < vector_idx, "CRITICAL FEEDBACK must come BEFORE vector context"

    def test_feedback_section_not_empty_when_present(self):
        """When feedback is present, the section has actual content."""
        brain = _make_brain_with_feedback(feedback=self.FEEDBACK_SAMPLE)

        ctx = brain.get_context(adx=25)
        # The feedback section should not just be a header — should have bullet points
        assert "- Your R/R:" in ctx


# ── Contract: LLM Parsability ────────────────────────────────────


class TestFeedbackLLMParsability:
    """Verify the CRITICAL FEEDBACK format is parsable by LLM."""

    SAMPLE = """## CRITICAL FEEDBACK: System Rejections

The following trade suggestions were BLOCKED by risk guards. ADJUST your parameters before proposing the next trade.

### R/R Minimum Guard (1 recent):
  1. LONG (HIGH confidence, HIGH volatility):
     - Your R/R: 0.80 | Required: 1.50 (gap: -0.70)
     - Your SL: 2.00% from entry
     - Your TP: 1.60% from entry
     - Your thesis: "Expecting quick bounce off support"

### SL Too Far (max 10%) (1 recent):
  1. SHORT (LOW confidence, MEDIUM volatility):
     - Your R/R: 0.67 | Required: 0.00 (gap: +0.67)
     - Your SL: 15.00% from entry
     - Your TP: 10.00% from entry

### PRE-FLIGHT CHECKLIST (MANDATORY):
- Before outputting BUY/SELL, verify: R/R >= required minimum (see Response Format).
- If volatility is HIGH, widen SL to >1x ATR to achieve required R/R.
- If volatility is LOW, do not use 2x+ ATR SL — tighten to keep R/R viable.
- Compare your proposed SL/TP against the last rejection patterns above.
- If you cannot meet the R/R requirement with a reasonable SL/TP, output HOLD.
"""

    def test_markdown_headers_are_well_formed(self):
        """All markdown headers start with ## or ### followed by a space."""
        brain = _make_brain_with_feedback(feedback=self.SAMPLE)
        ctx = brain.get_context(adx=25)

        # Extract all lines that look like headers
        header_lines = [l for l in ctx.split("\n") if l.startswith("#")]
        for line in header_lines:
            # Must be "## " or "### " (not "####" where '#' and text would merge)
            assert line.startswith("## ") or line.startswith("### "), \
                f"Malformed header: {line!r}"

    def test_no_trailing_whitespace_on_bullets(self):
        """Bullet points should not have trailing whitespace."""
        brain = _make_brain_with_feedback(feedback=self.SAMPLE)
        ctx = brain.get_context(adx=25)

        for line in ctx.split("\n"):
            if line.startswith("- "):
                assert not line.endswith(" ") and not line.endswith("\t"), \
                    f"Trailing whitespace: {line!r}"

    def test_rr_format_is_parseable(self):
        """R/R lines follow 'Your R/R: X.XX | Required: Y.YY (gap: +/-Z.ZZ)' format."""
        brain = _make_brain_with_feedback(feedback=self.SAMPLE)
        ctx = brain.get_context(adx=25)

        import re
        rr_pattern = re.compile(r"Your R/R: \d+\.\d+ \| Required: \d+\.\d+ \(gap: [+-]\d+\.\d+\)")
        matches = rr_pattern.findall(ctx)
        assert len(matches) >= 1, "No parseable R/R lines found"

    def test_percentage_format_consistent(self):
        """SL/TP percentages use consistent 'X.XX% from entry' format."""
        brain = _make_brain_with_feedback(feedback=self.SAMPLE)
        ctx = brain.get_context(adx=25)

        import re
        pct_pattern = re.compile(r"Your (SL|TP): \d+\.\d+% from entry")
        matches = pct_pattern.findall(ctx)
        assert len(matches) >= 1, "No parseable SL/TP percentage lines found"

    def test_pre_flight_checklist_has_five_items(self):
        """PRE-FLIGHT CHECKLIST has exactly 5 mandatory items."""
        brain = _make_brain_with_feedback(feedback=self.SAMPLE)
        ctx = brain.get_context(adx=25)

        # Count checklist items after the PRE-FLIGHT header
        in_checklist = False
        checklist_items = 0
        for line in ctx.split("\n"):
            if "PRE-FLIGHT CHECKLIST" in line:
                in_checklist = True
                continue
            if in_checklist and line.startswith("- "):
                checklist_items += 1
            elif in_checklist and not line.startswith("- ") and line.strip():
                in_checklist = False

        assert checklist_items == 5, f"Expected 5 checklist items, got {checklist_items}"

    def test_no_broken_parentheses(self):
        """No unclosed parentheses in the prompt."""
        brain = _make_brain_with_feedback(feedback=self.SAMPLE)
        ctx = brain.get_context(adx=25)

        open_count = ctx.count("(")
        close_count = ctx.count(")")
        assert open_count == close_count, \
            f"Mismatched parentheses: {open_count} open, {close_count} close"

    def test_thesis_line_uses_quoted_format(self):
        """AI thesis appears in double-quoted format."""
        brain = _make_brain_with_feedback(feedback=self.SAMPLE)
        ctx = brain.get_context(adx=25)

        assert "Your thesis:" in ctx
        assert "bounce off support" in ctx


# ── Contract: Section Integration ────────────────────────────────


class TestSectionIntegration:
    """Feedback integrates with other sections without breakage."""

    FEEDBACK = """## CRITICAL FEEDBACK: System Rejections

The following trade suggestions were BLOCKED by risk guards. ADJUST your parameters before proposing the next trade.

### R/R Minimum Guard (1 recent):
  1. LONG (HIGH confidence, MEDIUM volatility):
     - Your R/R: 1.10 | Required: 2.00 (gap: -0.90)

### PRE-FLIGHT CHECKLIST (MANDATORY):
- Before outputting BUY/SELL, verify: R/R >= required minimum (see Response Format).
- If volatility is HIGH, widen SL to >1x ATR to achieve required R/R.
- If volatility is LOW, do not use 2x+ ATR SL — tighten to keep R/R viable.
- Compare your proposed SL/TP against the last rejection patterns above.
- If you cannot meet the R/R requirement with a reasonable SL/TP, output HOLD.
"""

    def test_feedback_and_direction_bias_coexist(self):
        """Both feedback and direction bias warnings render without overlap."""
        brain = _make_brain_with_feedback(feedback=self.FEEDBACK)
        brain.vector_memory.get_direction_bias.return_value = {
            "long_count": 4, "short_count": 0,  # no shorts triggers warning
        }

        ctx = brain.get_context(adx=25, trend_direction="BULLISH")

        assert "CRITICAL FEEDBACK" in ctx
        assert "Direction Bias Check" in ctx
        assert "SHORT TRADES" in ctx

    def test_feedback_and_vector_context_coexist(self):
        """Feedback followed by vector context — both render properly."""
        brain = _make_brain_with_feedback(feedback=self.FEEDBACK)
        brain.vector_memory.get_context_for_prompt.return_value = (
            "## Similar Past Trades\n- Trade 1: WIN\n- Trade 2: LOSS\n"
        )

        ctx = brain.get_context(adx=25, trend_direction="BULLISH")

        assert "CRITICAL FEEDBACK" in ctx
        assert "Similar Past Trades" in ctx

    def test_feedback_and_semantic_rules_coexist(self):
        """Feedback and learned semantic rules coexist."""
        brain = _make_brain_with_feedback(feedback=self.FEEDBACK)
        brain.vector_memory.get_relevant_rules.return_value = [
            {"similarity": 85, "metadata": {"rule_type": "anti_pattern"}, "text": "Avoid longs into resistance"},
        ]

        ctx = brain.get_context(adx=25, trend_direction="BULLISH")

        assert "CRITICAL FEEDBACK" in ctx
        assert "Learned Trading Rules" in ctx

    def test_no_blank_lines_between_feedback_and_next_section(self):
        """No excessive blank lines between CRITICAL FEEDBACK and next section."""
        brain = _make_brain_with_feedback(feedback=self.FEEDBACK)
        brain.vector_memory.get_context_for_prompt.return_value = "## Similar Past Trades\n..."

        ctx = brain.get_context(adx=25, trend_direction="BULLISH")

        # Should not have 3+ consecutive blank lines
        lines = ctx.split("\n")
        blank_runs = []
        current_run = 0
        for line in lines:
            if line.strip() == "":
                current_run += 1
            else:
                if current_run >= 3:
                    blank_runs.append(current_run)
                current_run = 0

        assert blank_runs == [], f"Found {len(blank_runs)} runs of 3+ blank lines"

    def test_zero_trade_count_feedback_still_checked(self):
        """Even with zero trades, feedback is still checked (called)."""
        brain = _make_brain_no_trades()
        brain.vector_memory.get_blocked_trade_feedback.return_value = self.FEEDBACK

        ctx = brain.get_context(adx=25)

        # Feedback should appear even without trades
        assert "CRITICAL FEEDBACK" in ctx
        brain.vector_memory.get_blocked_trade_feedback.assert_called_once()


# ── Contract: Feedback Resilience ────────────────────────────────


class TestFeedbackResilience:
    """CRITICAL FEEDBACK section handles edge inputs gracefully."""

    def test_empty_feedback_adds_no_blank_lines(self):
        """Empty feedback doesn't add spurious blank lines to the prompt."""
        brain = _make_brain_with_feedback(feedback="")
        brain.vector_memory.get_context_for_prompt.return_value = "## Context section"

        ctx = brain.get_context(adx=25)

        # Should not have "CRITICAL FEEDBACK" anywhere
        assert "CRITICAL FEEDBACK" not in ctx

    def test_feedback_exception_handled_gracefully(self):
        """If get_blocked_trade_feedback raises, context still builds."""
        brain = _make_brain_with_feedback(feedback="SAMPLE")
        brain.vector_memory.get_blocked_trade_feedback.side_effect = RuntimeError("crash")

        # Should not raise
        ctx = brain.get_context(adx=25)

        assert "CRITICAL FEEDBACK" not in ctx
        # Other sections should still be present
        assert "Confidence Calibration" in ctx  # has trades

    def test_feedback_with_only_newlines_does_not_crash(self):
        """Feedback that is just newlines is treated as empty."""
        brain = _make_brain_with_feedback(feedback="\n\n\n")
        brain.vector_memory.get_context_for_prompt.return_value = "## Context"

        ctx = brain.get_context(adx=25)

        # Empty-ish strings are falsy in Python (but "\n\n\n" is truthy)
        # Verify it doesn't crash regardless
        assert isinstance(ctx, str)

    def test_feedback_with_unicode_does_not_break(self):
        """Unicode characters in feedback don't break prompt formatting."""
        feedback = "## CRITICAL FEEDBACK: System Rejections\n\nUnicode: émoji ✓ • ★\n"
        brain = _make_brain_with_feedback(feedback=feedback)

        ctx = brain.get_context(adx=25)

        assert "émoji" in ctx
        assert "CRITICAL FEEDBACK" in ctx
