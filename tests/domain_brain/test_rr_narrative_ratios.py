"""Narrative R/R ratios: a verified ratio quoted in prose must not look unverified.

Regression for the every-cycle warning
``R/R validation: narrative contained additional unverified R/R figures``:
``risk=819.51 R/R=1.953`` inside ``RR_CHECK`` made the ``<number> R/R`` pattern
count the RISK in USD (819.51) as a second, unverifiable ratio.
"""

from src.analyzer.risk_reward_validator import RiskRewardValidator

CHECK = (
    "RR_CHECK side=SHORT entry=81300.49 SL=82120 TP=79700 "
    "reward=1600.49 risk=819.51 R/R=1.953"
)


def _response(narrative: str) -> dict:
    return {"narrative": narrative, "analysis": {"signal": "HOLD"}}


def test_verified_ratio_quoted_in_prose_is_not_flagged():
    narrative = (
        "10) POSITION & RISK: flat\n"
        f"11) RISK/REWARD: {CHECK}, but because entry sits near the low the minimum-1 pct "
        "stop yields just 1.95 R/R with P(win) near 0.44, so the hypothetical short fails "
        "the EV test at this location.\n"
        "12) DECISION: HOLD\n"
        "13) EXECUTION NOTE: no order"
    )
    response = _response(narrative)

    issues = RiskRewardValidator().validate(response)

    assert issues == []
    assert response["narrative"] == narrative


def test_repeated_verified_ratio_in_prose_is_not_flagged():
    narrative = (
        "11) RISK/REWARD: " + CHECK + ", i.e. 1.95 R/R, and 1.95 R/R again at the same "
        "levels; no second scenario is claimed.\n"
        "12) DECISION: HOLD"
    )
    response = _response(narrative)

    assert RiskRewardValidator().validate(response) == []
    assert response["narrative"] == narrative


def test_invented_ratio_is_flagged_with_its_value_and_section_rewritten():
    narrative = (
        "11) RISK/REWARD: " + CHECK + ", an alternative target would give 3.4 R/R.\n"
        "12) DECISION: HOLD"
    )
    response = _response(narrative)

    issues = RiskRewardValidator().validate(response)

    assert any("3.4" in issue for issue in issues)
    assert "Python verified" in response["narrative"]
    assert "3.4 R/R" not in response["narrative"]
    assert "12) DECISION: HOLD" in response["narrative"]


def test_ratio_without_price_triple_is_still_refused():
    response = _response("11) RISK/REWARD: 0.62 R/R\n12) DECISION: HOLD")

    issues = RiskRewardValidator().validate(response)

    assert issues
    assert "N/A" in response["narrative"]


def test_near_miss_ratio_beyond_rounding_tolerance_is_still_unverified():
    """Rounding tolerance is 0.055, not 0.5: 2.35 against a verified 1.953 is a lie."""
    narrative = (
        "11) RISK/REWARD: " + CHECK + ", a wider stop would allow a 2.35 R/R read.\n"
        "12) DECISION: HOLD"
    )
    response = _response(narrative)

    issues = RiskRewardValidator().validate(response)

    assert any("2.35" in issue for issue in issues), issues
    assert "Python verified" in response["narrative"]
    assert "2.35 R/R" not in response["narrative"]

