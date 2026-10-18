"""Response-model tolerance for provider quirks that must not discard usable data."""

import pytest
from pydantic import ValidationError

from src.platforms.ai_providers.response_models import RedditAssessmentModel


def _payload(evidence):
    return {"outlook": "BULLISH", "evidence": evidence, "reason": "titles lean bullish"}


def test_reddit_evidence_accepts_decimal_strings():
    """Providers quote title IDs as strings ("12"); that must not discard the assessment."""
    model = RedditAssessmentModel.model_validate(_payload(["1", "2", "12"]))
    assert model.evidence == [1, 2, 12]


def test_reddit_evidence_still_rejects_non_numeric_strings():
    with pytest.raises(ValidationError):
        RedditAssessmentModel.model_validate(_payload(["1", "two"]))


def test_reddit_evidence_still_rejects_floats_and_bools():
    with pytest.raises(ValidationError):
        RedditAssessmentModel.model_validate(_payload([1.5]))
    with pytest.raises(ValidationError):
        RedditAssessmentModel.model_validate(_payload([True]))


def test_reddit_evidence_uniqueness_survives_coercion():
    with pytest.raises(ValidationError):
        RedditAssessmentModel.model_validate(_payload(["3", 3]))


def test_reddit_insufficient_evidence_needs_no_ids():
    model = RedditAssessmentModel.model_validate(
        {"outlook": "INSUFFICIENT_EVIDENCE", "evidence": [], "reason": "titles are off-topic"}
    )
    assert model.evidence == []
