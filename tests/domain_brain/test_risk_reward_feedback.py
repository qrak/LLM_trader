"""Python R/R validation and one-cycle correction feedback."""

import json
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.analyzer.analysis_result_processor import AnalysisResultProcessor
from src.analyzer.pattern_quality_scorer import PatternQualityScorer
from src.analyzer.prompts.template_manager import TemplateManager
from src.analyzer.risk_reward_validator import RiskRewardValidator
from src.analyzer.trend_validator import TrendValidator
from src.managers.persistence_manager import PersistenceManager
from src.managers.risk_manager import RiskManager
from src.parsing.unified_parser import UnifiedParser
from src.utils.format_utils import FormatUtils
from tests.conftest import make_config


@pytest.mark.asyncio
async def test_bad_hold_ratio_is_corrected_and_reaches_only_next_prompt(tmp_path):
    reply = json.dumps({
        "narrative": (
            "10) POSITION & RISK: flat\n"
            "11) RISK/REWARD: RR_CHECK side=SHORT entry=84380 SL=84750 TP=83726 "
            "reward=654 risk=370 R/R=0.62\n"
            "12) DECISION: HOLD\n13) EXECUTION NOTE: no order"
        ),
        "analysis": {"signal": "HOLD", "confidence": 72,
                     "confluence_factors": {"trend_alignment": 63, "momentum_strength": 71,
                                            "volume_support": 55, "pattern_quality": 67,
                                            "support_resistance_strength": 78},
                     "entry_price": None, "stop_loss": None, "take_profit": None,
                     "risk_reward_ratio": None, "position_size": 0.0,
                     "quantity": 0.0, "order_type": None, "reasoning": "No trade.",
                     "key_levels": {"support": [83726.0], "resistance": [84750.0]},
                     "trend": {"direction": "NEUTRAL", "strength_4h": 20,
                               "strength_daily": 15, "timeframe_alignment": "DIVERGENT"},
                     "symbol": "BTC/USDC", "reduce_only": False, "leverage": 1},
    })
    manager = MagicMock()
    manager.supports_image_analysis.return_value = False
    manager.send_prompt_streaming = AsyncMock(return_value=reply)
    logger = MagicMock()
    parser = UnifiedParser(logger=logger, format_utils=FormatUtils())
    processor = AnalysisResultProcessor(
        manager, logger, parser, TrendValidator(), PatternQualityScorer(), RiskRewardValidator()
    )
    result = await processor.process_analysis("system", "user")
    assert result["analysis"]["signal"] == "HOLD"
    assert "model reward/risk/R/R 654/370/0.62, Python 654.00/370.00/1.77" in result["rr_feedback"]
    assert "R/R=1.77" in result["raw_response"]
    assert "R/R=0.62" not in result["raw_response"]
    assert parser.validate_trading_response(result)["status"] == "valid"

    persistence = PersistenceManager(logger, data_dir=str(tmp_path / "trading"))
    persistence.save_previous_response(result["raw_response"], prompt="old user prompt")
    loaded = persistence.load_previous_response()
    assert loaded is not None
    template = TemplateManager(cast(Any, make_config()), logger)
    next_prompt = template.build_system_prompt("BTC/USDC", previous_response=loaded["response"])
    assert next_prompt.count("### PYTHON R/R FEEDBACK (previous cycle)") == 1
    assert next_prompt.count("model reward/risk/R/R 654/370/0.62, Python 654.00/370.00/1.77") == 1
    assert "R/R=0.62" not in next_prompt
    persistence.save_previous_response("11) RISK/REWARD: N/A\n```json\n{\"analysis\":{\"signal\":\"HOLD\"}}\n```")
    following_data = persistence.load_previous_response()
    assert following_data is not None
    following = template.build_system_prompt(
        "BTC/USDC", previous_response=following_data["response"]
    )
    assert "PYTHON R/R FEEDBACK" not in following


@pytest.mark.parametrize("quoted", [0.62, float("nan")])
def test_wrong_trade_ratio_is_blocked_before_strategy_receives_signal(quoted):
    response = {"analysis": {"signal": "SELL", "entry_price": 84380.0,
                             "stop_loss": 84750.0, "take_profit": 83726.0,
                             "risk_reward_ratio": quoted, "position_size": 0.08,
                             "quantity": 0.01, "order_type": "market"},
                "narrative": "11) RISK/REWARD: 0.62 R/R\n12) DECISION: SELL\n13) EXECUTION NOTE: place order"}
    issues = RiskRewardValidator().validate(response)
    assert issues
    assert response["analysis"]["signal"] == "HOLD"
    assert response["analysis"]["quantity"] == 0.0
    assert response["analysis"]["stop_loss"] is None
    assert "12) DECISION: HOLD" in response["narrative"]
    assert "13) EXECUTION NOTE: no order" in response["narrative"]
    assert "0.62 R/R" not in response["narrative"]
    rendered = AnalysisResultProcessor._render_response_text(response, json.dumps(response))
    parser = UnifiedParser(logger=MagicMock(), format_utils=FormatUtils())
    assert parser.parse_ai_response(rendered)["analysis"]["signal"] == "HOLD"


@pytest.mark.parametrize("text", [
    "short to 83726 with SL 84750 gives ~0.62 R/R",
    "hypothetical long to 85257 gives ~0.58 and short ~1.70",
])
def test_unverifiable_ratio_is_removed_and_flagged(text):
    response = {"analysis": {"signal": "HOLD"},
                "narrative": f"11) RISK/REWARD: {text}\n12) DECISION: HOLD"}
    RiskRewardValidator().validate(response)
    assert "unverifiable narrative R/R" in response["rr_feedback"]
    assert text not in response["narrative"]
    assert response["analysis"]["signal"] == "HOLD"


def test_correct_trade_ratio_and_risk_manager_post_clamp_are_distinct():
    response = {"analysis": {"signal": "BUY", "entry_price": 100.0,
                             "stop_loss": 99.5, "take_profit": 102.0, "risk_reward_ratio": 4.0}}
    assert RiskRewardValidator().validate(response) == []
    assert response["analysis"]["risk_reward_ratio"] == 4.0
    risk = RiskManager(MagicMock(), cast(Any, make_config())).calculate_entry_parameters(
        "BUY", current_price=100.0, capital=10000.0, confidence="HIGH",
        stop_loss=99.5, take_profit=102.0, position_size=0.05
    )
    assert risk.stop_loss == pytest.approx(99.0)
    assert risk.rr_ratio == pytest.approx(2.0)


def test_valid_hypothetical_ratio_does_not_emit_feedback():
    response = {"analysis": {"signal": "HOLD"},
                "narrative": "11) RISK/REWARD: RR_CHECK side=SHORT entry=84380 SL=84750 TP=83726 "
                             "reward=654 risk=370 R/R=1.77\n12) DECISION: HOLD"}
    assert RiskRewardValidator().validate(response) == []
    assert "rr_feedback" not in response


@pytest.mark.parametrize(("reward", "risk", "ratio"), [
    (654, 300, 1.77),
    (654, 370, 0.62),
    (600, 370, 1.77),
])
def test_incorrect_hypothetical_components_are_replaced(reward, risk, ratio):
    response = {"analysis": {"signal": "HOLD"},
                "narrative": (f"11) RISK/REWARD: RR_CHECK side=SHORT entry=84380 SL=84750 TP=83726 "
                              f"reward={reward} risk={risk} R/R={ratio}\n12) DECISION: HOLD")}
    assert RiskRewardValidator().validate(response)
    assert "reward=654.00 risk=370.00 R/R=1.77" in response["narrative"]


def test_bad_level_order_rejects_actionable_trade():
    response = {"analysis": {"signal": "LONG", "entry_price": 100,
                             "stop_loss": 105, "take_profit": 110, "risk_reward_ratio": 2.0}}
    RiskRewardValidator().validate(response)
    assert response["analysis"]["signal"] == "HOLD"
    assert "invalid prices" in response["rr_feedback"]
