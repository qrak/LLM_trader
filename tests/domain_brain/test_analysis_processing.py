"""Dense domain tests for the analysis pipeline: LLM claim validation, the
result processor execution paths and the decision-summary aggregation.
"""

import json
import math
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.analyzer.analysis_result_processor import AnalysisResultProcessor
from src.analyzer.pattern_quality_scorer import PatternQualityScorer
from src.analyzer.trend_validator import (
    ADX_DISCREPANCY_THRESHOLD,
    TrendValidation,
    TrendValidator,
)
from src.dashboard.dashboard_state import DashboardState
from src.dashboard.decision_presenter import (
    build_decision_graph,
    build_decision_synopsis,
)
from src.dashboard.routers.brain import BrainRouter
from src.managers.provider_types import InvocationResult
from src.parsing.unified_parser import UnifiedParser
from src.trading.data_models import MarketConditions, Position
from src.utils.format_utils import FormatUtils

VALID_ANALYSIS_RESPONSE = '{"analysis": {"signal": "HOLD"}}'
NARRATIVE_ONLY = "1) MARKET STRUCTURE: price below both SMAs; 2) DECISION: HOLD - no edge."


def json_block(signal: str = "HOLD") -> str:
    """LLM reply carrying the required JSON block."""
    return (
        "```json\n"
        f'{{"analysis": {{"signal": "{signal}", "confidence": 82, "entry_price": 77880, '
        '"stop_loss": 76500, "take_profit": 80640, "position_size": 0.08, '
        '"risk_reward_ratio": 2.0, "reasoning": "Valid setup."}}\n'
        "```"
    )


def make_processor(*, supports_image: bool = True, chart_error: Exception | None = None) -> AnalysisResultProcessor:
    """AnalysisResultProcessor wired to a configurable model manager."""
    model_manager = MagicMock()
    model_manager.supports_image_analysis.return_value = supports_image
    model_manager.describe_provider_and_model.return_value = ("google", "gemini-3.5-flash")
    model_manager.send_prompt_with_chart_analysis = (
        AsyncMock(return_value=VALID_ANALYSIS_RESPONSE)
        if chart_error is None
        else AsyncMock(side_effect=chart_error)
    )
    model_manager.send_prompt_streaming = AsyncMock(return_value=VALID_ANALYSIS_RESPONSE)

    unified_parser = MagicMock()
    unified_parser.parse_ai_response.return_value = {"analysis": {"signal": "HOLD", "trend": {}}}
    unified_parser.validate_ai_response.return_value = True

    return AnalysisResultProcessor(
        model_manager=model_manager,
        logger=MagicMock(),
        unified_parser=unified_parser,
        trend_validator=TrendValidator(),
        quality_scorer=PatternQualityScorer(),
    )


def make_repair_processor(first_response: str, repair_response: str):
    """Processor wired to the real parser so the JSON-repair contract is exercised."""
    model_manager = MagicMock()
    model_manager.supports_image_analysis.return_value = False
    model_manager.send_prompt_streaming = AsyncMock(return_value=first_response)
    model_manager.send_contract_repair = AsyncMock(return_value=repair_response)
    processor = AnalysisResultProcessor(
        model_manager=model_manager,
        logger=MagicMock(),
        unified_parser=UnifiedParser(logger=MagicMock(), format_utils=FormatUtils()),
        trend_validator=TrendValidator(),
        quality_scorer=PatternQualityScorer(),
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
    """Vector-memory double with one experience, one rule and one journal entry."""
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
    """BrainRouter with flattened position and default memory/journal doubles.

    Passing vector_memory=None / post_mortem_repo=None means "dependency absent",
    which is a different case from leaving the default double in place.
    """
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


def test_adx_validity_and_label_bands():
    validator = TrendValidator()

    for valid in (45, 32.7, 0, 100):
        assert validator._is_valid_adx(valid) is True, valid
    for invalid in (None, -5, 150, math.nan, "forty"):
        assert validator._is_valid_adx(invalid) is False, invalid

    for adx, expected in [(10, "absent"), (22, "developing"), (40, "strong"), (60, "very strong"), (80, "extreme")]:
        assert expected in validator._adx_label(adx), adx


def test_validate_with_both_computed_values():
    validator = TrendValidator()

    matched = validator.validate(strength_4h=35, strength_daily=42, computed_adx=35.0, computed_daily_adx=42.0)
    assert matched.passed is True
    assert matched.discrepancies == []
    assert (matched.validated_4h, matched.validated_daily) == (35.0, 42.0)
    assert matched.has_computed_data is True

    near = validator.validate(
        strength_4h=35 + (ADX_DISCREPANCY_THRESHOLD - 1), strength_daily=42, computed_adx=35.0, computed_daily_adx=42.0
    )
    assert near.passed is True

    off_4h = validator.validate(strength_4h=80, strength_daily=42, computed_adx=35.0, computed_daily_adx=42.0)
    assert off_4h.passed is False
    assert len(off_4h.discrepancies) == 1
    assert "4H ADX" in off_4h.discrepancies[0]
    assert off_4h.validated_4h == 35.0

    off_daily = validator.validate(strength_4h=35, strength_daily=90, computed_adx=35.0, computed_daily_adx=42.0)
    assert off_daily.passed is False
    assert len(off_daily.discrepancies) == 1
    assert "Daily ADX" in off_daily.discrepancies[0]
    assert off_daily.validated_daily == 42.0

    both_off = validator.validate(strength_4h=80, strength_daily=90, computed_adx=30.0, computed_daily_adx=40.0)
    assert both_off.passed is False
    assert len(both_off.discrepancies) == 2


def test_validate_missing_input_matrix():
    validator = TrendValidator()

    no_llm = validator.validate(strength_4h=None, strength_daily=42, computed_adx=35.0, computed_daily_adx=42.0)
    assert no_llm.passed is True
    assert (no_llm.validated_4h, no_llm.validated_daily) == (35.0, 42.0)
    assert no_llm.llm_strength_4h is None

    no_llm_daily = validator.validate(strength_4h=35, strength_daily=None, computed_adx=35.0, computed_daily_adx=42.0)
    assert (no_llm_daily.validated_4h, no_llm_daily.validated_daily) == (35.0, 42.0)

    no_llm_at_all = validator.validate(strength_4h=None, strength_daily=None, computed_adx=35.0, computed_daily_adx=42.0)
    assert (no_llm_at_all.validated_4h, no_llm_at_all.validated_daily) == (35.0, 42.0)

    llm_only = validator.validate(strength_4h=35, strength_daily=42, computed_adx=None, computed_daily_adx=None)
    assert llm_only.passed is True
    assert (llm_only.validated_4h, llm_only.validated_daily) == (35.0, 42.0)
    assert llm_only.has_computed_data is False

    nothing = validator.validate(strength_4h=None, strength_daily=None, computed_adx=None, computed_daily_adx=None)
    assert nothing.passed is True
    assert (nothing.validated_4h, nothing.validated_daily) == (25, 25)


def test_validate_ignores_invalid_values_and_parses_numeric_strings():
    validator = TrendValidator()

    invalid_llm = validator.validate(strength_4h=math.nan, strength_daily=150, computed_adx=35.0, computed_daily_adx=42.0)
    assert invalid_llm.llm_strength_4h is None
    assert invalid_llm.llm_strength_daily is None
    assert (invalid_llm.validated_4h, invalid_llm.validated_daily) == (35.0, 42.0)

    invalid_computed = validator.validate(strength_4h=35, strength_daily=42, computed_adx=math.nan, computed_daily_adx=-5)
    assert invalid_computed.computed_adx is None
    assert invalid_computed.computed_daily_adx is None
    assert (invalid_computed.validated_4h, invalid_computed.validated_daily) == (35.0, 42.0)

    numeric_strings = validator.validate(strength_4h="35", strength_daily="42.0", computed_adx=35.0, computed_daily_adx=42.0)
    assert (numeric_strings.llm_strength_4h, numeric_strings.llm_strength_daily) == (35, 42)

    zero = validator.validate(strength_4h=0, computed_adx=0.0)
    assert zero.validated_4h == 0.0
    assert zero.passed is True


def test_overwrite_llm_trend_and_dataclass_contract():
    validator = TrendValidator()
    analysis = {"trend": {"direction": "BULLISH", "strength_4h": 99, "strength_daily": 80}}
    validation = validator.validate(strength_4h=99, strength_daily=80, computed_adx=40.0, computed_daily_adx=45.0)

    overwritten = validator.overwrite_llm_trend(analysis, validation)
    assert overwritten["trend"]["strength_4h"] == 40
    assert overwritten["trend"]["strength_daily"] == 45
    assert overwritten["trend"]["direction"] == "BULLISH"
    assert overwritten["_trend_validation"]["passed"] is False
    assert overwritten["_trend_validation"]["discrepancies"]

    created = validator.overwrite_llm_trend({}, validator.validate(computed_adx=40.0, computed_daily_adx=45.0))
    assert created["trend"]["strength_4h"] == 40
    assert created["_trend_validation"]["passed"] is True

    defaults = TrendValidation()
    assert (defaults.validated_4h, defaults.validated_daily) == (25, 25)
    assert defaults.passed is True
    assert defaults.discrepancies == []

    serialized = validation.to_dict()
    assert serialized["passed"] is False
    assert serialized["validated_4h"] == 40
    assert serialized["validated_daily"] == 45


async def test_process_analysis_routes_between_chart_and_text_paths():
    chart = make_processor(supports_image=True)
    await chart.process_analysis(system_prompt="system", prompt="prompt", chart_image=b"img")
    chart.model_manager.send_prompt_with_chart_analysis.assert_awaited_once()
    chart.model_manager.send_prompt_streaming.assert_not_awaited()

    text_only = make_processor(supports_image=False)
    await text_only.process_analysis(system_prompt="system", prompt="prompt", chart_image=b"img")
    text_only.model_manager.send_prompt_with_chart_analysis.assert_not_awaited()
    text_only.model_manager.send_prompt_streaming.assert_awaited_once()

    falling_back = make_processor(supports_image=True, chart_error=ValueError("Empty response content from Google AI"))
    await falling_back.process_analysis(system_prompt="system", prompt="prompt", chart_image=b"img")
    falling_back.model_manager.send_prompt_with_chart_analysis.assert_awaited_once()
    falling_back.model_manager.send_prompt_streaming.assert_awaited_once()

    falling_back.logger.warning.assert_called_once()
    warning_args = falling_back.logger.warning.call_args[0]
    assert warning_args[0] == "Chart analysis failed: %s. Falling back to text-only analysis."
    assert "Empty response content from Google AI" in str(warning_args[1])
    assert not str(warning_args[1]).startswith("Chart analysis failed:")


async def test_process_analysis_returns_error_payload_when_validation_fails():
    processor = make_processor(supports_image=False)
    processor.unified_parser.validate_ai_response.return_value = False

    result = await processor.process_analysis(system_prompt="system", prompt="prompt")

    assert result["error"] == "Invalid response format"
    assert "raw_response" in result


async def test_contract_repair_recovers_missing_json_block():
    processor, model_manager = make_repair_processor(NARRATIVE_ONLY, json_block("HOLD"))

    result = await processor.process_analysis(system_prompt="system", prompt="prompt")

    model_manager.send_contract_repair.assert_awaited_once()
    repair_kwargs = model_manager.send_contract_repair.await_args.kwargs
    assert repair_kwargs["system_message"] == "system"
    assert repair_kwargs["prompt"] == "prompt"
    assert repair_kwargs["previous_response"] == NARRATIVE_ONLY
    assert result["analysis"]["signal"] == "HOLD"
    assert result["response_validation"]["status"] == "valid"
    assert "parse_error" not in result
    assert result["raw_response"].startswith(NARRATIVE_ONLY)
    assert "```json" in result["raw_response"]


async def test_contract_repair_is_skipped_when_block_present_and_kept_when_repair_fails():
    compliant, compliant_manager = make_repair_processor(json_block("BUY"), "unused")
    compliant_result = await compliant.process_analysis(system_prompt="system", prompt="prompt")

    compliant_manager.send_contract_repair.assert_not_awaited()
    assert compliant_result["analysis"]["signal"] == "BUY"
    assert compliant_result["response_validation"]["status"] == "valid"

    unrepairable, unrepairable_manager = make_repair_processor("narrative only", "still no json here")
    fallback_result = await unrepairable.process_analysis(system_prompt="system", prompt="prompt")

    unrepairable_manager.send_contract_repair.assert_awaited_once()
    assert fallback_result["parse_error"] == "Failed to parse response"
    assert fallback_result["response_validation"]["status"] == "invalid"
    assert fallback_result["raw_response"] == "narrative only"


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
    )
    processor.context = SimpleNamespace(
        technical_data={"adx": 35.0, "rsi": 55.0},
        long_term_data={"daily_adx": 42.0},
        technical_patterns={"bullish_engulfing": [{"name": "bullish_engulfing", "bar_index": 95}]},
    )
    parsed = {
        "analysis": {
            "signal": "BUY",
            "confidence": "HIGH",
            "trend": {"direction": "BULLISH", "strength_4h": 35, "strength_daily": 42},
            "pattern_quality": 70,
        }
    }

    processor._validate_llm_claims(parsed)

    assert parsed["analysis"]["trend"]["strength_4h"] == 35
    assert parsed["analysis"]["trend"]["strength_daily"] == 42
    assert "_trend_validation" in parsed["analysis"]
    assert "_pattern_validation" in parsed["analysis"]

    discrepant = {"analysis": {"trend": {"strength_4h": 80, "strength_daily": 90}}}
    processor._validate_llm_claims(discrepant)
    assert any("ADX" in str(call) for call in logger.warning.call_args_list)


def test_analysis_validation_is_skipped_without_context_or_analysis():
    processor = AnalysisResultProcessor(
        model_manager=MagicMock(),
        logger=MagicMock(),
        unified_parser=MagicMock(),
        trend_validator=MagicMock(),
        quality_scorer=MagicMock(),
    )

    processor.context = None
    without_context = {"analysis": {"trend": {"strength_4h": 80}}}
    processor._validate_llm_claims(without_context)
    assert "_trend_validation" not in without_context["analysis"]

    processor.context = SimpleNamespace(technical_data={"adx": 35.0}, long_term_data={}, technical_patterns={})
    without_analysis = {"error": "no analysis"}
    processor._validate_llm_claims(without_analysis)
    assert "_trend_validation" not in without_analysis


async def test_decision_summary_sections_and_flat_synopsis(brain_config):
    router = make_brain_router(brain_config)

    result = await router.get_decision_summary()

    for key in ("generated_at", "synopsis", "now", "last_decision", "position", "memory", "journal", "counts", "graph"):
        assert key in result
    assert result["position"]["has_position"] is False
    assert result["journal"]["count"] >= 1
    assert type(result["synopsis"]) is str and len(result["synopsis"]) > 20
    assert "nodes" in result["graph"] and "edges" in result["graph"]

    lowered = result["synopsis"].lower()
    assert "no open position" in lowered or "flat" in lowered or "no active position" in lowered


async def test_decision_summary_tolerates_missing_dependencies(brain_config):
    router = make_brain_router(brain_config, vector_memory=None, post_mortem_repo=None)

    result = await router.get_decision_summary()

    assert result["memory"]["top_experiences"] == []
    assert result["journal"]["items"] == []
    assert "synopsis" in result
    node_ids = {node["id"] for node in result["graph"]["nodes"]}
    assert "hub_now" in node_ids
    assert "hub_position" in node_ids


async def test_decision_summary_graph_topology_and_labels(brain_config):
    router = make_brain_router(brain_config)

    result = await router.get_decision_summary()
    nodes = result["graph"]["nodes"]
    edges = result["graph"]["edges"]
    by_id = {node["id"]: node for node in nodes}

    for required in ("hub_now", "hub_position", "hub_memory", "hub_rules", "hub_journal", "hub_context"):
        assert required in by_id, required
        assert by_id[required]["label"]
        assert by_id[required]["type"]

    assert by_id["hub_position"]["label"] == "FLAT"
    assert any(edge["from"] == "hub_now" and edge["to"] == "hub_journal" for edge in edges)
    assert any(node["id"].startswith("pm_") for node in nodes)
    assert any(node["id"].startswith("rule_") for node in nodes)
    for node in nodes:
        assert node["label"], node
        assert node["type"], node


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
    assert "LONG" in result["synopsis"] or "long" in result["synopsis"].lower()
    position_node = next(node for node in result["graph"]["nodes"] if node["id"] == "hub_position")
    assert "LONG" in position_node["label"]


async def test_decision_summary_cache_invalidation(brain_config):
    state = DashboardState()
    router = make_brain_router(brain_config, dashboard_state=state)

    first = await router.get_decision_summary()
    state.invalidate_brain_caches()
    second = await router.get_decision_summary()

    assert first["generated_at"] and second["generated_at"]
    assert "decision_summary" not in state._cache or state.get_cached("decision_summary", ttl_seconds=15.0)


def test_decision_graph_and_synopsis_are_pure_functions():
    graph = build_decision_graph(
        now={"action": "HOLD", "confidence": 80, "trend": "BEARISH", "adx": 20, "rsi": 40},
        last_decision={"signal": "HOLD", "confidence": 80, "reasoning_excerpt": "Wait"},
        position={"has_position": False},
        memory={
            "current_context": "BEARISH + Low ADX",
            "experience_count": 2,
            "rule_count": 1,
            "top_experiences": [
                {"outcome": "WIN", "pnl_pct": 2.1, "direction": "LONG", "similarity": 0.9, "document_excerpt": "good long"}
            ],
            "top_rules": [{"rule_type": "anti_pattern", "rule_text": "Avoid chop longs", "final_score": 1}],
            "blocked": {"blocked_count": 0, "items": []},
        },
        journal={"count": 1, "items": [{"verdict": "premature_entry", "symbol": "BTC", "lesson_learned": "wait"}]},
    )
    ids = {node["id"] for node in graph["nodes"]}
    assert {"hub_now", "exp_0", "rule_0", "pm_0"} <= ids
    assert all(node["label"] for node in graph["nodes"])

    text = build_decision_synopsis(
        now={"action": "HOLD", "confidence": 75, "trend": "BEARISH"},
        position={"has_position": False},
        memory={"current_context": "BEARISH + Low ADX", "top_rules": [], "blocked": {}},
        journal={"items": []},
        last_decision={"signal": "HOLD", "confidence": 75},
    )
    assert "flat" in text.lower() or "no open position" in text.lower()
    assert "HOLD" in text
