

import asyncio
import io
import json
import math
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.analyzer.analysis_context import AnalysisContext
from src.analyzer.analysis_engine import AnalysisEngine
from src.analyzer.analysis_result_processor import AnalysisResultProcessor
from src.analyzer.decision_gate import DecisionGate
from src.analyzer.pattern_quality_scorer import PatternQualityScorer
from src.analyzer.risk_reward_validator import RiskRewardValidator
from src.analyzer.trend_validator import TrendValidator
from src.dashboard.dashboard_state import DashboardState
from src.dashboard.decision_presenter import build_decision_synopsis_data
from src.dashboard.routers.brain import BrainRouter
from src.managers.provider_types import InvocationResult
from src.parsing.unified_parser import UnifiedParser
from src.trading.data_models import MarketConditions, Position
from src.trading.market_conditions_extractor import MarketConditionsExtractor
from src.utils.format_utils import FormatUtils
from src.utils.indicator_classifier import (
    format_adx_readings,
    resolve_primary_trend_strength,
)
from tests.conftest import make_config
from tests.domain_brain.test_prompt_contracts import make_builder, manager_for

VALID_ANALYSIS_RESPONSE = '{"analysis": {"signal": "HOLD"}}'
NARRATIVE_ONLY = "1) MARKET STRUCTURE: price below both SMAs; 2) DECISION: HOLD - no edge."


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "error", "cancelled"])
async def test_repeated_analysis_cycles_do_not_accumulate_news_after_failures(monkeypatch, outcome):
    config = make_config()
    builder = make_builder(config)
    rag = MagicMock()
    rag.retrieve_context = AsyncMock()
    rag.get_latest_article_urls_snapshot.return_value = {}
    manager = MagicMock()
    engine = AnalysisEngine(MagicMock(), rag, manager, MagicMock(), config, prompt_builder=builder)
    engine.symbol = "BTC/USDC"
    engine.context = AnalysisContext(symbol=engine.symbol)
    captured = []

    async def generate(*args, **kwargs):
        captured.append(tuple(builder.custom_instructions))
        if outcome == "error":
            raise RuntimeError("Synthetic analysis failure")
        if outcome == "cancelled":
            raise asyncio.CancelledError()
        return {"analysis": {"signal": "HOLD"}}

    monkeypatch.setattr(engine, "_collect_market_data", AsyncMock(return_value=True))
    monkeypatch.setattr(engine, "_enrich_market_context", AsyncMock())
    monkeypatch.setattr(engine, "_perform_technical_analysis", AsyncMock())
    monkeypatch.setattr(engine, "_generate_ai_analysis", generate)
    expected = []
    for index in range(30):
        news = f"News for cycle {index:03d}: " + "x" * 1000
        expected.append((news,))
        rag.retrieve_context.return_value = news
        if outcome == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await engine.analyze_market()
        else:
            result = await engine.analyze_market()
            if outcome == "error":
                assert result == {"error": "Synthetic analysis failure", "recommendation": "HOLD"}
            else:
                assert result == {"analysis": {"signal": "HOLD"}}
    assert captured == expected
    assert builder.custom_instructions == []



def make_processor() -> AnalysisResultProcessor:

    model_manager = MagicMock()
    model_manager.describe_provider_and_model.return_value = ("google", "gemini-3.5-flash")
    model_manager.send_prompt = AsyncMock(return_value=VALID_ANALYSIS_RESPONSE)

    unified_parser = MagicMock()
    unified_parser.parse_ai_response.return_value = {"analysis": {"signal": "HOLD", "trend": {}}}
    unified_parser.validate_ai_response.return_value = True

    return AnalysisResultProcessor(
        model_manager=model_manager,
        logger=MagicMock(),
        unified_parser=unified_parser,
        trend_validator=TrendValidator(),
        quality_scorer=PatternQualityScorer(),
        risk_reward_validator=RiskRewardValidator(),
    )


def test_sl_atr_distance_is_computed_by_python():
    processor = make_processor()

    analysis = {"signal": "BUY", "entry_price": 100.0, "stop_loss": 95.0}
    processor._record_sl_atr_distance(analysis, {"atr": 4.0})
    assert analysis["sl_atr_distance"] == 1.25
    processor.logger.warning.assert_not_called()

    tight = {"signal": "SELL", "entry_price": 100.0, "stop_loss": 103.0}
    processor._record_sl_atr_distance(tight, {"atr": 4.0})
    assert tight["sl_atr_distance"] == 0.75
    processor.logger.warning.assert_called_once()

    hold = {"signal": "HOLD", "entry_price": None, "stop_loss": None}
    processor._record_sl_atr_distance(hold, {"atr": 4.0})
    assert "sl_atr_distance" not in hold

    no_atr = {"signal": "BUY", "entry_price": 100.0, "stop_loss": 95.0}
    processor._record_sl_atr_distance(no_atr, {})
    assert "sl_atr_distance" not in no_atr


def make_json_parser_processor(first_response: str, decision_gate=None):
    model_manager = MagicMock()
    model_manager.send_prompt = AsyncMock(return_value=first_response)
    processor = AnalysisResultProcessor(
        model_manager=model_manager,
        logger=MagicMock(),
        unified_parser=UnifiedParser(logger=MagicMock(), format_utils=FormatUtils()),
        trend_validator=TrendValidator(),
        quality_scorer=PatternQualityScorer(),
        risk_reward_validator=RiskRewardValidator(),
        decision_gate=decision_gate,
    )
    return processor, model_manager


@pytest.fixture
def brain_config(tmp_path):
    trading = tmp_path / "trading"
    trading.mkdir()
    (trading / "previous_response.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-07-09T12:00:00+00:00",
                "technical_data": {"adx": 28.0, "rsi": 55.0, "atr_percent": 2.0, "plus_di": 30.0, "minus_di": 12.0},
                "response": {
                    "text_analysis": (
                        "SIGNAL: HOLD\nConfidence: 75%\n"
                        "Thesis: Range-bound under resistance; wait for break.\n"
                        "Invalidation: daily close above 71000."
                    ),
                    "current_price": 70000.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        DATA_DIR=str(tmp_path),
        TIMEFRAME="4h",
        STOP_LOSS_TYPE="hard",
        STOP_LOSS_CHECK_INTERVAL="15m",
        TAKE_PROFIT_TYPE="hard",
        TAKE_PROFIT_CHECK_INTERVAL="15m",
        CRYPTO_PAIR="BTC/USDC",
    )


def default_vector_memory() -> MagicMock:

    memory = MagicMock()
    memory.trade_count = 2
    memory.semantic_rule_count = 1
    memory.experience_count = 2
    memory.compute_confidence_stats.return_value = {"HIGH": 1, "LOW": 1}
    memory.compute_adx_performance.return_value = {}
    memory.compute_factor_performance.return_value = {}
    memory.get_active_rules.return_value = [
        {
            "text": "Avoid longs into low ADX bear flags",
            "metadata": {
                "rule_type": "anti_pattern",
                "win_rate": 0.2,
                "final_score": 2.0,
                "recommended_adjustment": "Wait for ADX>25",
            },
            "final_score": 2.0,
        }
    ]
    memory.retrieve_similar_experiences.return_value = []
    memory.get_all_experiences.return_value = []
    memory.get_recent_blocked_trades.return_value = []
    memory.get_blocked_trade_count.return_value = 0
    return memory


def default_post_mortem_repo() -> MagicMock:
    repo = MagicMock()
    repo.get_recent_post_mortems.return_value = [
        {
            "verdict": "premature_entry",
            "symbol": "BTCUSDC",
            "direction": "LONG",
            "pnl_pct": -1.2,
            "close_reason": "stop_loss",
            "lesson_learned": "Do not long into falling volume.",
            "created_at": "2026-07-01T00:00:00+00:00",
        }
    ]
    repo.search_post_mortems.return_value = []
    return repo


UNSET = object()


def make_brain_router(brain_config, *, dashboard_state: DashboardState | None = None, vector_memory: Any = UNSET,
                      post_mortem_repo: Any = UNSET, persistence: Any = None) -> BrainRouter:

    if persistence is None:
        persistence = MagicMock()
        persistence.load_position.return_value = None
        persistence.load_trade_history.return_value = []
    return BrainRouter(
        config=brain_config,
        logger=MagicMock(),
        dashboard_state=dashboard_state or DashboardState(),
        vector_memory=default_vector_memory() if vector_memory is UNSET else vector_memory,
        unified_parser=None,
        persistence=persistence,
        exchange_manager=None,
        post_mortem_repo=default_post_mortem_repo() if post_mortem_repo is UNSET else post_mortem_repo,
    )


def test_adx_reading_accepts_only_finite_values_inside_the_scale():
    validator = TrendValidator()

    for valid in (45, 32.7, 0, 100, 0.0, 100.0):
        assert validator._adx_reading(valid) == float(valid), valid
    for invalid in (None, -5, 150, math.nan, math.inf, True, "forty", "35"):
        assert validator._adx_reading(invalid) is None, invalid


def test_validate_uses_only_the_computed_readings():
    validator = TrendValidator()

    both = validator.validate(computed_adx=35.0, computed_daily_adx=42.0)
    assert (both.validated_primary, both.validated_daily) == (35.0, 42.0)
    assert both.timeframe is None

    missing_daily = validator.validate(computed_adx=30.83)
    assert (missing_daily.validated_primary, missing_daily.validated_daily) == (30.83, None)

    missing_primary = validator.validate(computed_daily_adx=48.65)
    assert (missing_primary.validated_primary, missing_primary.validated_daily) == (None, 48.65)

    nothing = validator.validate()
    assert (nothing.validated_primary, nothing.validated_daily) == (None, None)

    zero = validator.validate(computed_adx=0.0, computed_daily_adx=0)
    assert (zero.validated_primary, zero.validated_daily) == (0.0, 0.0)

    invalid = validator.validate(computed_adx=math.nan, computed_daily_adx=150)
    assert (invalid.computed_adx, invalid.computed_daily_adx) == (None, None)
    assert (invalid.validated_primary, invalid.validated_daily) == (None, None)


def test_overwrite_sets_both_fields_and_nulls_a_missing_reading():
    validator = TrendValidator()
    analysis = {"trend": {"direction": "BULLISH", "strength_primary": 2, "strength_daily": 90, "strength_4h": 99}}
    validation = validator.validate(computed_adx=34.19, computed_daily_adx=48.56, timeframe="4h")

    overwritten = validator.overwrite_llm_trend(analysis, validation)
    assert overwritten["trend"]["strength_primary"] == 34
    assert overwritten["trend"]["strength_daily"] == 49
    assert overwritten["trend"]["direction"] == "BULLISH"
    assert "strength_4h" not in overwritten["trend"]
    assert overwritten["_trend_validation"] == {
        "timeframe": "4h",
        "computed_adx": 34.19,
        "computed_daily_adx": 48.56,
        "validated_primary": 34.19,
        "validated_daily": 48.56,
    }

    missing_daily = validator.overwrite_llm_trend(
        {"trend": {"strength_primary": 2, "strength_daily": 73}},
        validator.validate(computed_adx=30.83),
    )
    assert missing_daily["trend"]["strength_primary"] == 31
    assert missing_daily["trend"]["strength_daily"] is None

    created = validator.overwrite_llm_trend({}, validator.validate())
    assert created["trend"]["strength_primary"] is None
    assert created["trend"]["strength_daily"] is None


async def test_model_supplied_strength_never_surfaces_without_python_adx():
    reply = json.dumps({
        "narrative": NARRATIVE_ONLY,
        "analysis": {"signal": "HOLD", "confidence": 62, "trend": {"direction": "NEUTRAL", "strength": 73}},
    })
    processor, _ = make_json_parser_processor(reply)
    processor.context = SimpleNamespace(
        timeframe="4h",
        technical_data={"rsi": 55.0},
        long_term_data={},
        technical_patterns={},
        current_price=100.0,
    )

    result = await processor.process_analysis(system_prompt="system", prompt="prompt")

    trend = result["analysis"]["trend"]
    assert "strength" not in trend
    assert trend["strength_primary"] is None
    assert resolve_primary_trend_strength(trend) is None
    assert format_adx_readings(trend) == "ADX n/a"
    assert MarketConditionsExtractor(MagicMock()).extract_market_conditions(result).trend_strength == 0.0
    assert '"strength": 73' not in result["raw_response"]


async def test_a_real_zero_adx_survives_processing():
    reply = json.dumps({
        "narrative": NARRATIVE_ONLY,
        "analysis": {"signal": "HOLD", "confidence": 62, "trend": {"direction": "NEUTRAL", "strength": 73}},
    })
    processor, _ = make_json_parser_processor(reply)
    processor.context = SimpleNamespace(
        timeframe="4h",
        technical_data={"adx": 0.0, "rsi": 55.0},
        long_term_data={},
        technical_patterns={},
        current_price=100.0,
    )

    result = await processor.process_analysis(system_prompt="system", prompt="prompt")

    trend = result["analysis"]["trend"]
    assert trend["strength_primary"] == 0
    assert resolve_primary_trend_strength(trend) == 0
    assert format_adx_readings(trend) == "ADX 0"


async def test_process_analysis_returns_error_payload_when_validation_fails():
    processor = make_processor()
    processor.unified_parser.validate_ai_response.return_value = False

    result = await processor.process_analysis(system_prompt="system", prompt="prompt")

    assert result["error"] == "Invalid response format"
    assert "raw_response" in result


def json_object_reply(signal: str = "HOLD") -> str:

    return json.dumps({
        "narrative": NARRATIVE_ONLY,
        "analysis": {
            "signal": signal,
            "confidence": 82,
            "entry_price": 77880,
            "stop_loss": 76500,
            "take_profit": 80640,
            "position_size": 0.08,
            "risk_reward_ratio": 2.0,
            "reasoning": "Valid setup.",
        },
    })


async def test_json_object_reply_keeps_the_narrative_and_a_trailing_block():

    processor, model_manager = make_json_parser_processor(json_object_reply("BUY"))

    result = await processor.process_analysis(system_prompt="system", prompt="prompt")

    assert result["analysis"]["signal"] == "BUY"
    assert result["response_validation"]["status"] == "valid"
    assert result["raw_response"].startswith(NARRATIVE_ONLY)
    assert "```json" in result["raw_response"]
    block = result["raw_response"].split("```json", 1)[1].rsplit("```", 1)[0]
    assert json.loads(block)["analysis"]["signal"] == "BUY"
    model_manager.send_prompt.assert_awaited_once()


class GateClient:
    api_key = "key"
    model = "openai/gpt-6-luna-decisions"

    async def ask(self, state, questions):
        return {
            "answers": {
                "edge_now": {"noul": 0.35},
                "direction": {"choice": "short", "confidence": 0.8},
                "quality": {"score": 0.2, "confidence": 0.8},
            },
            "usage": {"input_tokens": 640, "output_tokens": 0, "cost": 0.00006},
        }


async def test_the_gate_line_lands_above_the_narrative_and_the_block_still_parses(tmp_path):

    gate = DecisionGate(
        logger=MagicMock(),
        client=GateClient(),
        mode="size",
        min_confidence_pct=20,
        journal_path=tmp_path / "gate.jsonl",
    )
    processor, _ = make_json_parser_processor(json_object_reply("BUY"), decision_gate=gate)

    result = await processor.process_analysis(system_prompt="system", prompt="prompt")

    assert result["raw_response"].startswith("Decision gate (independent model, size only):")
    assert NARRATIVE_ONLY in result["raw_response"]
    assert result["analysis"]["position_size"] == 0.04
    block = result["raw_response"].split("```json", 1)[1].rsplit("```", 1)[0]
    assert json.loads(block)["analysis"]["position_size"] == 0.04
    assert result["decision_gate"]["multiplier"] == 0.5


async def test_reply_without_a_json_block_falls_back_without_a_second_request():
    processor, model_manager = make_json_parser_processor(NARRATIVE_ONLY)

    result = await processor.process_analysis(system_prompt="system", prompt="prompt")

    assert result["parse_error"] == "Failed to parse response"
    assert result["response_validation"]["status"] == "invalid"
    assert result["raw_response"] == NARRATIVE_ONLY
    model_manager.send_prompt.assert_awaited_once()


def test_invocation_result_error_surface():
    result = InvocationResult(
        success=False,
        response=None,
        provider="google",
        model="gemini-3.5-flash",
        error_message="Empty or invalid response content from Google AI",
    )

    assert result.error == "Empty or invalid response content from Google AI"


async def test_analysis_validation_flags_discrepant_llm_claims():
    logger = MagicMock()
    processor = AnalysisResultProcessor(
        model_manager=MagicMock(),
        logger=logger,
        unified_parser=MagicMock(),
        trend_validator=TrendValidator(),
        quality_scorer=PatternQualityScorer(),
        risk_reward_validator=RiskRewardValidator(),
    )
    processor.context = SimpleNamespace(
        timeframe="4h",
        technical_data={"adx": 35.0, "rsi": 55.0},
        long_term_data={"daily_adx": 42.0},
        technical_patterns={"rsi": [{"type": "rsi_oversold", "bar_index": 95}]},
    )
    parsed = {
        "analysis": {
            "signal": "BUY",
            "confidence": "HIGH",
            "trend": {"direction": "BULLISH", "strength_4h": 35, "strength_daily": 42},
            "confluence_factors": {"pattern_quality": 51},
        }
    }

    processor._validate_llm_claims(parsed)

    assert parsed["analysis"]["signal"] == "BUY"
    assert parsed["analysis"]["trend"]["strength_primary"] == 35
    assert "strength_4h" not in parsed["analysis"]["trend"]
    assert parsed["analysis"]["trend"]["strength_daily"] == 42
    assert "_trend_validation" in parsed["analysis"]
    assert parsed["analysis"]["confluence_factors"]["pattern_quality"] == 78
    assert parsed["analysis"]["_pattern_validation"]["passed"] is False
    logger.warning.assert_called_once_with(
        "Pattern quality validation: %s",
        "Pattern quality: LLM reported 51, computed 78 (strong). Delta=27 > 25.",
    )


def test_timeframe_alignment_is_computed_from_di_dominance():
    logger = MagicMock()
    processor = AnalysisResultProcessor(
        model_manager=MagicMock(),
        logger=logger,
        unified_parser=MagicMock(),
        trend_validator=TrendValidator(),
        quality_scorer=PatternQualityScorer(),
        risk_reward_validator=RiskRewardValidator(),
    )
    processor.context = SimpleNamespace(
        timeframe="4h",
        technical_data={"adx": 35.0, "plus_di": 10.0, "minus_di": 30.0},
        long_term_data={"daily_adx": 42.0, "daily_plus_di": 10.0, "daily_minus_di": 30.0},
        technical_patterns={},
    )
    parsed = {"analysis": {"trend": {"direction": "BEARISH", "timeframe_alignment": "MIXED"}}}

    processor._validate_llm_claims(parsed)

    assert parsed["analysis"]["trend"]["timeframe_alignment"] == "ALIGNED"
    logger.info.assert_called_once_with(
        "Timeframe alignment: model %s, computed %s (DI dominance); using computed",
        "MIXED", "ALIGNED",
    )

    processor.context = SimpleNamespace(
        timeframe="1d",
        technical_data={"adx": 35.0, "plus_di": 10.0, "minus_di": 30.0},
        long_term_data={"daily_adx": 42.0, "daily_plus_di": 10.0, "daily_minus_di": 30.0},
        technical_patterns={},
    )
    daily = {"analysis": {"trend": {"direction": "BEARISH", "timeframe_alignment": "MIXED"}}}

    processor._validate_llm_claims(daily)

    assert daily["analysis"]["trend"]["timeframe_alignment"] == "MIXED"

    processor.context = SimpleNamespace(
        timeframe="4h",
        technical_data={"adx": 35.0, "plus_di": 10.0},
        long_term_data={"daily_adx": 42.0, "daily_plus_di": 10.0, "daily_minus_di": 30.0},
        technical_patterns={},
    )
    incomplete = {"analysis": {"trend": {"direction": "BEARISH", "timeframe_alignment": "DIVERGENT"}}}

    processor._validate_llm_claims(incomplete)

    assert incomplete["analysis"]["trend"]["timeframe_alignment"] is None


def test_analysis_validation_is_skipped_without_context_or_analysis():
    processor = AnalysisResultProcessor(
        model_manager=MagicMock(),
        logger=MagicMock(),
        unified_parser=MagicMock(),
        trend_validator=MagicMock(),
        quality_scorer=MagicMock(),
        risk_reward_validator=RiskRewardValidator(),
    )

    processor.context = None
    without_context = {"analysis": {"trend": {"strength_4h": 80}}}
    processor._validate_llm_claims(without_context)
    assert "_trend_validation" not in without_context["analysis"]

    processor.context = SimpleNamespace(timeframe="4h", technical_data={"adx": 35.0}, long_term_data={}, technical_patterns={})
    without_analysis = {"error": "no analysis"}
    processor._validate_llm_claims(without_analysis)
    assert "_trend_validation" not in without_analysis


async def test_decision_summary_sections_and_flat_synopsis(brain_config):
    router = make_brain_router(brain_config)

    result = await router.get_decision_summary()

    for key in ("generated_at", "synopsis_data", "now", "last_decision", "position", "memory", "journal", "counts"):
        assert key in result
    assert result["position"]["has_position"] is False
    assert result["journal"]["count"] >= 1
    assert result["synopsis_data"]["has_position"] is False


async def test_decision_summary_tolerates_missing_dependencies(brain_config):
    router = make_brain_router(brain_config, vector_memory=None, post_mortem_repo=None)

    result = await router.get_decision_summary()

    assert result["memory"]["top_experiences"] == []
    assert result["journal"]["items"] == []
    assert "synopsis_data" in result
    assert result["counts"]["experiences"] == 0
    assert result["counts"]["journal"] == 0


async def test_decision_summary_counts_match_the_sections(brain_config):
    router = make_brain_router(brain_config)

    result = await router.get_decision_summary()
    counts = result["counts"]

    assert counts["experiences"] == (result["memory"].get("experience_count") or 0)
    assert counts["rules"] == (result["memory"].get("rule_count") or 0)
    assert counts["journal"] == (result["journal"].get("count") or 0)
    assert counts["blocked"] == ((result["memory"].get("blocked") or {}).get("blocked_count") or 0)


async def test_decision_summary_reports_open_position(brain_config):
    position = Position(
        entry_price=68000.0,
        stop_loss=67000.0,
        take_profit=71000.0,
        size=0.01,
        entry_time=datetime.now(timezone.utc),
        confidence="HIGH",
        direction="LONG",
        symbol="BTC/USDC",
        sl_distance_pct=0.0147,
        tp_distance_pct=0.0441,
        rr_ratio_at_entry=2.0,
        stop_loss_type_at_entry="hard",
        stop_loss_check_interval_at_entry="15m",
        take_profit_type_at_entry="hard",
        take_profit_check_interval_at_entry="15m",
        conditions_at_entry=MarketConditions(),
    )
    persistence = MagicMock()
    persistence.load_position.return_value = position
    persistence.load_trade_history.return_value = []
    router = make_brain_router(brain_config, persistence=persistence)

    result = await router.get_decision_summary()

    assert result["position"]["has_position"] is True
    assert result["synopsis_data"]["direction"] == "LONG"


async def test_decision_summary_cache_invalidation(brain_config):
    state = DashboardState()
    router = make_brain_router(brain_config, dashboard_state=state)

    first = await router.get_decision_summary()
    state.invalidate_brain_caches()
    second = await router.get_decision_summary()

    assert first["generated_at"] and second["generated_at"]
    assert "decision_summary" not in state._cache or state.get_cached("decision_summary", ttl_seconds=15.0)


def test_decision_synopsis_is_a_pure_function():
    data = build_decision_synopsis_data(
        now={"action": "HOLD", "confidence": 75, "trend": "BEARISH"},
        position={"has_position": False},
        memory={"current_context": "BEARISH + Low ADX", "top_rules": [], "blocked": {}},
        journal={"items": []},
        last_decision={"signal": "HOLD", "confidence": 75},
    )
    assert data["has_position"] is False
    assert data["action"] == "HOLD"
    assert data["confidence"] == 75
    assert data["trend"] == "BEARISH"
    assert data["top_rule"] is None
    assert data["latest_journal"] is None

@pytest.mark.parametrize("json_mode", [True, False])
@pytest.mark.parametrize("valid_entry", [True, False])
async def test_full_markdown_report_survives_processing_and_python_rr_checks(json_mode, valid_entry):
    payload = json.loads(json_object_reply("BUY"))
    narrative = (
        "## 1) Technical Analysis Overview\n\n"
        + "**Metrics:** closed-candle evidence; interpretation remains separate.\n\n" * 200
        + "## 10) Risk Considerations & Final Synthesis\n\nThe current setup has explicit invalidation.\n\n"
        + "11) RISK/REWARD: RR_CHECK side=LONG entry=100 SL=90 TP=120 reward=20 risk=10 R/R=2\n"
        + "12) DECISION: BUY\n13) EXECUTION NOTE: current cycle only."
    )
    payload["narrative"] = narrative
    if not valid_entry:
        payload["analysis"]["risk_reward_ratio"] = 9.0
    raw = json.dumps(payload) if json_mode else narrative + "\n```json\n" + json.dumps({"analysis": payload["analysis"]}) + "\n```"
    processor, model_manager = make_json_parser_processor(raw)
    result = await processor.process_analysis(system_prompt="system", prompt="prompt")
    assert result["response_validation"]["status"] == "valid"
    assert result["analysis"]["signal"] == ("BUY" if valid_entry else "HOLD")
    assert "## 1) Technical Analysis Overview" in result["raw_response"]
    assert result["raw_response"].count("**Metrics:**") == 200
    assert "## 10) Risk Considerations & Final Synthesis" in result["raw_response"]
    if not valid_entry:
        assert "12) DECISION: HOLD" in result["raw_response"]
        assert "13) EXECUTION NOTE: no order" in result["raw_response"]
        assert "12) DECISION: BUY" not in result["raw_response"]
    block = result["raw_response"].split("```json", 1)[1].rsplit("```", 1)[0]
    assert json.loads(block)["analysis"]["signal"] == result["analysis"]["signal"]
    model_manager.send_prompt.assert_awaited_once()


class TestCommaDecimalPrices:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param("63500,50", None, id="comma_decimal"),
            pytest.param("$63,500.50", None, id="thousands_separator"),
            pytest.param("63500.50", 63500.5, id="plain_decimal_string"),
            pytest.param(63500.5, 63500.5, id="float"),
        ],
    )
    def test_a_price_with_a_comma_is_refused(self, value: Any, expected: float | None) -> None:

        parser = UnifiedParser(logger=MagicMock(), format_utils=FormatUtils())

        assert parser._parse_finite_number(value, None) == expected


class TestNonObjectTrendField:
    @pytest.mark.parametrize("trend", [None, "BULLISH", ["BULLISH"], {"strength_4h": 35}])
    def test_claim_validation_survives_a_non_object_trend(self, trend: Any) -> None:

        processor = AnalysisResultProcessor(
            model_manager=MagicMock(),
            logger=MagicMock(),
            unified_parser=MagicMock(),
            trend_validator=TrendValidator(),
            quality_scorer=PatternQualityScorer(),
            risk_reward_validator=RiskRewardValidator(),
        )
        processor.context = SimpleNamespace(
            timeframe="4h",
            technical_data={"adx": 35.0},
            long_term_data={},
            technical_patterns={},
        )
        parsed: dict[str, Any] = {"analysis": {"signal": "HOLD", "trend": trend}}

        processor._validate_llm_claims(parsed)

        assert isinstance(parsed["analysis"]["trend"], dict)
        assert "_trend_validation" in parsed["analysis"]


@pytest.mark.asyncio
async def test_the_engine_renders_the_chart_for_humans_without_sending_it(monkeypatch):
    config = make_config()
    builder = make_builder(config)
    rag = MagicMock()
    rag.retrieve_context = AsyncMock(return_value=None)
    rag.get_latest_article_urls_snapshot.return_value = {}
    manager = MagicMock()
    engine = AnalysisEngine(MagicMock(), rag, manager, MagicMock(), config, prompt_builder=builder)
    engine.symbol = "BTC/USDC"
    engine.context = AnalysisContext(symbol=engine.symbol)
    captured: dict[str, Any] = {}

    async def generate(*args, **kwargs):
        captured.update(kwargs)
        return {"analysis": {"signal": "HOLD"}}

    monkeypatch.setattr(engine, "_collect_market_data", AsyncMock(return_value=True))
    monkeypatch.setattr(engine, "_enrich_market_context", AsyncMock())
    monkeypatch.setattr(engine, "_perform_technical_analysis", AsyncMock())
    monkeypatch.setattr(engine, "_generate_chart_image", AsyncMock(return_value=io.BytesIO(b"png")))
    monkeypatch.setattr(engine, "_generate_ai_analysis", generate)

    await engine.analyze_market()

    assert "precomputed_chart" not in captured
    assert engine.last_chart_buffer is not None
    assert engine.last_chart_buffer.getvalue() == b"png"


@pytest.mark.asyncio
async def test_a_failed_chart_render_keeps_the_previous_dashboard_image(monkeypatch):
    config = make_config()
    builder = make_builder(config)
    rag = MagicMock()
    rag.retrieve_context = AsyncMock(return_value=None)
    rag.get_latest_article_urls_snapshot.return_value = {}
    engine = AnalysisEngine(MagicMock(), rag, MagicMock(), MagicMock(), config, prompt_builder=builder)
    engine.symbol = "BTC/USDC"
    engine.context = AnalysisContext(symbol=engine.symbol)
    engine.last_chart_buffer = io.BytesIO(b"previous")

    monkeypatch.setattr(engine, "_collect_market_data", AsyncMock(return_value=True))
    monkeypatch.setattr(engine, "_enrich_market_context", AsyncMock())
    monkeypatch.setattr(engine, "_perform_technical_analysis", AsyncMock())
    monkeypatch.setattr(engine, "_generate_chart_image", AsyncMock(return_value=None))
    monkeypatch.setattr(engine, "_generate_ai_analysis", AsyncMock(return_value={"analysis": {}}))

    await engine.analyze_market()

    assert engine.last_chart_buffer.getvalue() == b"previous"


@pytest.mark.asyncio
async def test_the_engine_reports_the_transport_that_produced_the_answer():
    config = make_config()
    builder = make_builder(config)
    builder.template_manager = manager_for(config)
    processor = make_processor()
    manager = processor.model_manager
    manager.token_counter.count_tokens.return_value = 100
    engine = AnalysisEngine(
        MagicMock(), MagicMock(), manager, MagicMock(), config,
        prompt_builder=builder, result_processor=processor,
    )
    engine.symbol = "BTC/USDC"
    engine.context = AnalysisContext(
        symbol=engine.symbol,
        technical_data={"adx": 45.0, "rsi": 55.0},
        technical_patterns={"trend": [{"name": "death_cross", "bar_index": 5}]},
    )

    result = await engine._generate_ai_analysis(None, None)

    assert "chart_analysis" not in result
    assert "chart" not in manager.describe_provider_and_model.call_args.kwargs
    assert result["analysis"]["pattern_quality"] == 80
    system_message = manager.send_prompt.call_args.kwargs["system_message"]
    assert '"pattern_quality": 80,' in system_message
    assert "Copy the system-computed pattern_quality" in system_message
    assert "supporting/total" not in system_message
    assert "CHART" not in system_message
    assert "P1-price" not in system_message


@pytest.mark.asyncio
async def test_the_engine_hands_the_computed_alignment_to_the_prompt():
    config = make_config()
    builder = make_builder(config)
    builder.template_manager = manager_for(config)
    processor = make_processor()
    manager = processor.model_manager
    manager.token_counter.count_tokens.return_value = 100
    engine = AnalysisEngine(
        MagicMock(), MagicMock(), manager, MagicMock(), config,
        prompt_builder=builder, result_processor=processor,
    )
    engine.symbol = "BTC/USDC"
    engine.context = AnalysisContext(
        symbol=engine.symbol,
        timeframe="4h",
        technical_data={"adx": 45.0, "plus_di": 10.0, "minus_di": 30.0},
        technical_patterns={},
    )
    processor.compute_timeframe_alignment = MagicMock(return_value="ALIGNED")

    await engine._generate_ai_analysis(None, None)

    system_message = manager.send_prompt.call_args.kwargs["system_message"]
    assert '"timeframe_alignment": "ALIGNED"' in system_message
    assert "copy the system-computed value shown in the JSON template" in system_message


@pytest.mark.asyncio
async def test_a_contract_retry_reuses_the_same_system_prompt():
    processor, model_manager = make_json_parser_processor('{"analysis": {"signal": "BUY"}}')
    model_manager.send_prompt = AsyncMock(
        side_effect=['{"analysis": {"signal": "BUY"}}', '{"analysis": {"signal": "HOLD"}}']
    )

    await processor.process_analysis(system_prompt="system prompt", prompt="prompt")

    sent = model_manager.send_prompt.call_args.kwargs
    assert sent["system_message"] == "system prompt"
