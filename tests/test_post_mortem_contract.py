"""Contract tests for PostMortemResult Pydantic model."""

import pytest
from pydantic import ValidationError

from src.trading.post_mortem import PostMortemResult


class TestPostMortemResultContract:
    """Contract tests for PostMortemResult Pydantic model."""

    @staticmethod
    def _make_valid_data(**overrides):
        data = {
            "verdict": "overestimated_breakout",
            "llm_analysis": "Price rejected resistance twice before dropping. Entry was premature.",
            "expected_vs_actual": "Expected breakout above 72.5k, actual rejection and -3% drop.",
            "lesson_learned": "When price rejects a level twice, wait for confirmation before entering.",
        }
        data.update(overrides)
        return data

    def test_valid_fields_accepted(self):
        """All required fields with valid values should pass."""
        result = PostMortemResult(**self._make_valid_data())
        assert result.verdict == "overestimated_breakout"
        assert result.lesson_learned
        assert result.llm_analysis
        assert result.expected_vs_actual

    def test_empty_verdict_rejected(self):
        """Empty verdict string should raise ValidationError."""
        with pytest.raises(ValidationError):
            PostMortemResult(**self._make_valid_data(verdict=""))

    def test_empty_lesson_learned_rejected(self):
        """Empty lesson_learned string should raise ValidationError."""
        with pytest.raises(ValidationError):
            PostMortemResult(**self._make_valid_data(lesson_learned=""))

    def test_empty_llm_analysis_rejected(self):
        """Empty llm_analysis string should raise ValidationError."""
        with pytest.raises(ValidationError):
            PostMortemResult(**self._make_valid_data(llm_analysis=""))

    def test_empty_expected_vs_actual_rejected(self):
        """Empty expected_vs_actual string should raise ValidationError."""
        with pytest.raises(ValidationError):
            PostMortemResult(**self._make_valid_data(expected_vs_actual=""))

    def test_missing_verdict_rejected(self):
        """Missing 'verdict' should raise ValidationError."""
        data = self._make_valid_data()
        del data["verdict"]
        with pytest.raises(ValidationError):
            PostMortemResult(**data)

    def test_missing_lesson_learned_rejected(self):
        """Missing 'lesson_learned' should raise ValidationError."""
        data = self._make_valid_data()
        del data["lesson_learned"]
        with pytest.raises(ValidationError):
            PostMortemResult(**data)

    def test_extra_fields_ignored(self):
        """Extra fields beyond the schema should be ignored."""
        data = self._make_valid_data(extra_field="should be ignored")
        result = PostMortemResult(**data)
        assert not hasattr(result, "extra_field")
        assert result.verdict == "overestimated_breakout"

    def test_verdict_snake_case_accepted(self):
        """Snake_case verdict strings are valid."""
        for verdict in ["overestimated_breakout", "good_exit", "plan_followed", "premature_entry", "held_too_long"]:
            result = PostMortemResult(**self._make_valid_data(verdict=verdict))
            assert result.verdict == verdict

    def test_minimal_valid_fields(self):
        """Minimum valid fields should work (single char strings)."""
        result = PostMortemResult(
            verdict="a",
            llm_analysis="b",
            expected_vs_actual="c",
            lesson_learned="d",
        )
        assert result.verdict == "a"
        assert result.llm_analysis == "b"
