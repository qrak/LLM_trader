

import pytest
from pydantic import ValidationError

from src.platforms.ai_providers.response_models import SocialAssessmentModel


def _payload(evidence):
    return {"outlook": "BULLISH", "evidence": evidence, "reason": "titles lean bullish"}


def test_social_evidence_accepts_decimal_strings():

    model = SocialAssessmentModel.model_validate(_payload(["1", "2", "12"]))
    assert model.evidence == [1, 2, 12]


def test_social_evidence_still_rejects_non_numeric_strings():
    with pytest.raises(ValidationError):
        SocialAssessmentModel.model_validate(_payload(["1", "two"]))


def test_social_evidence_still_rejects_floats_and_bools():
    with pytest.raises(ValidationError):
        SocialAssessmentModel.model_validate(_payload([1.5]))
    with pytest.raises(ValidationError):
        SocialAssessmentModel.model_validate(_payload([True]))


def test_social_evidence_uniqueness_survives_coercion():
    with pytest.raises(ValidationError):
        SocialAssessmentModel.model_validate(_payload(["3", 3]))


def test_social_insufficient_evidence_needs_no_ids():
    model = SocialAssessmentModel.model_validate(
        {"outlook": "INSUFFICIENT_EVIDENCE", "evidence": [], "reason": "titles are off-topic"}
    )
    assert model.evidence == []
