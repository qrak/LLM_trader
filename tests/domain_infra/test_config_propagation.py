"""A configured timeframe reaches every consumer, and no consumer substitutes its own.

The regression this pins: a change that reads ``Config.TIMEFRAME`` in one place and leaves a
plausible literal (``4h``/``1h``/``240``) in another, so the bot runs two frames at once and only
the log disagrees. Every assertion here is derived from the ini, never from a constant, which is
why a literal can no longer pass by matching the production value.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.analyzer.analysis_engine import AnalysisEngine
from src.analyzer.market_data_collector import MarketDataCollector
from src.analyzer.market_metrics_calculator import MarketMetricsCalculator
from src.analyzer.prompts.prompt_builder import PromptBuilder
from src.analyzer.prompts.template_manager import TemplateManager
from src.config import loader
from src.config.loader import Config
from src.trading.brain_context import BrainContextProvider
from src.trading.regime_risk_profile import RegimeRiskProfile, RegimeRiskProfileSelector
from src.utils.timeframe_validator import TimeframeValidator
from tests.conftest import null_logger

CONFIGURED_FRAMES = [
    ("6h", 360, {"12h": 2, "24h": 4, "3d": 12, "7d": 28}),
    ("12h", 720, {"24h": 2, "3d": 6, "7d": 14}),
    ("1d", 1440, {"3d": 3, "7d": 7}),
]


def write_config(
    tmp_path: Path,
    timeframe: str,
    interval: str = "15m",
    max_position_size: str | None = None,
) -> Path:
    ini = tmp_path / "config.ini"
    position_cap = (
        f"max_position_size = {max_position_size}\n" if max_position_size is not None else ""
    )
    ini.write_text(
        "[general]\n"
        "crypto_pair = BTC/USDC\n"
        f"timeframe = {timeframe}\n"
        "candle_limit = 999\n"
        "ai_chart_candle_limit = 125\n"
        "market_type = spot\n"
        "timezone = UTC\n"
        "[ai_providers]\n"
        "provider = local\n"
        "[model_config]\n"
        "max_tokens = 1024\n"
        "[risk_management]\n"
        "stop_loss_type = hard\n"
        f"stop_loss_check_interval = {interval}\n"
        "take_profit_type = hard\n"
        f"take_profit_check_interval = {interval}\n"
        f"{position_cap}",
        encoding="utf-8",
    )
    return ini


@pytest.fixture
def build_config(tmp_path, monkeypatch):
    def build(
        timeframe: str = "6h",
        interval: str = "15m",
        max_position_size: str | None = None,
    ) -> Config:
        ini = write_config(tmp_path, timeframe, interval, max_position_size)
        monkeypatch.setattr(loader, "CONFIG_INI_PATH", ini)
        monkeypatch.setattr(loader, "KEYS_ENV_PATH", tmp_path / "missing.env")
        with patch.dict(os.environ, {}, clear=True):
            return Config()

    return build


@pytest.mark.parametrize(("timeframe", "minutes", "periods"), CONFIGURED_FRAMES)
def test_the_configured_frame_and_its_minutes_come_from_the_ini(build_config, timeframe, minutes, periods):
    config = build_config(timeframe)

    assert config.TIMEFRAME == timeframe
    assert config.TIMEFRAME_MINUTES == minutes


def frame_tokens(text: str) -> set[str]:
    """Every supported frame token that appears as a standalone word in the text."""
    return {
        frame
        for frame in TimeframeValidator.SUPPORTED_TIMEFRAMES
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(frame)}(?![A-Za-z0-9_])", text)
    }


def test_the_complete_system_prompt_carries_the_configured_frame_and_no_other(build_config):
    """The whole prompt, not just its header: the response contract is a consumer too.

    This is the regression that the first timeframe pass missed. Its system-prompt test rendered
    only ``build_system_prompt()``, so the hardcoded ``4h`` in ``build_response_template()`` kept
    passing 2 619 green tests while the model was told to analyse a frame the bot was not running.
    """
    config = build_config("6h")
    manager = TemplateManager(config, null_logger(), TimeframeValidator)

    prompt = "\n\n".join([
        manager.build_system_prompt("BTC/USDC"),
        manager.build_decision_rules(),
        manager.build_analysis_steps("BTC/USDC"),
        manager.build_response_template(),
    ])

    assert re.search(r"\b6h\b", prompt)
    assert not re.search(r"\b4h\b", prompt)
    assert not re.search(r"\b1h\b", prompt)


@pytest.mark.parametrize("timeframe", ["15m", "1h", "6h", "12h", "1d"])
def test_the_response_contract_names_only_the_configured_frame(build_config, timeframe):
    """The response template must not carry a second frame or the legacy 4h field name."""
    config = build_config(timeframe)
    manager = TemplateManager(config, null_logger(), TimeframeValidator)

    template = manager.build_response_template()

    assert frame_tokens(template) == {timeframe}
    assert "strength_4h" not in template


def test_previous_snapshot_labels_the_configured_frame_for_a_legacy_trend_key(build_config):
    """A prior response written before the rename still gets today's frame label."""
    config = build_config("6h")
    manager = TemplateManager(config, null_logger(), TimeframeValidator)
    prior = json.dumps({
        "analysis": {
            "trend": {
                "direction": "BULLISH",
                "strength_4h": 70,
                "strength_daily": 55,
                "timeframe_alignment": "ALIGNED",
            },
        },
    })

    prompt = manager.build_system_prompt(
        "BTC/USDC", previous_response=f"1) MARKET STRUCTURE: prior\n```json\n{prior}\n```"
    )

    assert "- Trend: BULLISH | 6h: 70 | daily: 55 | alignment: ALIGNED" in prompt
    assert "4h: 70" not in prompt


def prompt_builder_arguments(config: Config) -> dict:
    return {
        "logger": null_logger(),
        "config": config,
        "format_utils": MagicMock(),
        "overview_formatter": MagicMock(),
        "long_term_formatter": MagicMock(),
        "technical_formatter": MagicMock(),
        "market_formatter": MagicMock(),
        "timeframe_validator": TimeframeValidator,
        "template_manager": TemplateManager(config, null_logger(), TimeframeValidator),
    }


def test_the_prompt_builder_resolves_the_frame_from_config(build_config):
    config = build_config("6h")
    builder = PromptBuilder(**prompt_builder_arguments(config))

    assert builder.timeframe == "6h"
    assert builder._calculate_period_candles() == {"12h": 2, "24h": 4, "3d": 12, "7d": 28}


def test_the_prompt_builder_refuses_to_invent_a_frame():
    with pytest.raises(ValueError, match="timeframe is required"):
        PromptBuilder(
            logger=null_logger(),
            format_utils=MagicMock(),
            overview_formatter=MagicMock(),
            long_term_formatter=MagicMock(),
            technical_formatter=MagicMock(),
            market_formatter=MagicMock(),
        )


def test_the_prompt_builder_refuses_to_invent_minutes_per_candle(build_config):
    arguments = prompt_builder_arguments(build_config("6h"))
    arguments.pop("timeframe_validator")

    with pytest.raises(ValueError, match="timeframe_validator is required"):
        PromptBuilder(**arguments)


def test_the_data_collector_refuses_to_invent_a_frame():
    collector = MarketDataCollector(null_logger(), MagicMock())

    with pytest.raises(TypeError):
        collector.initialize(MagicMock(), "BTC/USDC", MagicMock())  # type: ignore[call-arg]


def test_the_data_collector_keeps_the_frame_it_was_given():
    collector = MarketDataCollector(null_logger(), MagicMock())

    collector.initialize(MagicMock(), "BTC/USDC", MagicMock(), timeframe="6h")

    assert collector.timeframe == "6h"


def test_period_metrics_refuse_a_context_without_a_frame():
    calculator = MarketMetricsCalculator(null_logger())
    context = SimpleNamespace(ohlcv_candles=np.zeros((100, 6)), timeframe=None)

    with pytest.raises(ValueError, match="context.timeframe is required"):
        calculator.update_period_metrics(context)


def test_an_exit_interval_longer_than_the_frame_is_rejected_at_load(tmp_path, monkeypatch):
    ini = write_config(tmp_path, "1h", interval="4h")
    monkeypatch.setattr(loader, "CONFIG_INI_PATH", ini)
    monkeypatch.setattr(loader, "KEYS_ENV_PATH", tmp_path / "missing.env")

    with patch.dict(os.environ, {}, clear=True), pytest.raises(ValueError, match="must not be greater than timeframe"):
        Config()


def make_engine(config: Config, logger) -> AnalysisEngine:
    return AnalysisEngine(
        logger,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        config,
        technical_calculator=MagicMock(),
        pattern_analyzer=MagicMock(),
        prompt_builder=MagicMock(),
        data_collector=MagicMock(),
        metrics_calculator=MagicMock(),
        result_processor=MagicMock(),
        chart_generator=MagicMock(),
    )


def test_the_engine_uses_the_configured_frame_when_none_is_passed(build_config):
    engine = make_engine(build_config("6h"), MagicMock())

    engine.initialize_for_symbol("BTC/USDC", SimpleNamespace(name="binance", timeframes={}))

    assert engine.context is not None
    assert engine.context.timeframe == "6h"


def test_the_engine_logs_an_override_instead_of_swallowing_it(build_config):
    logger = MagicMock()
    engine = make_engine(build_config("6h"), logger)

    engine.initialize_for_symbol("BTC/USDC", SimpleNamespace(name="binance", timeframes={}), timeframe="4h")

    assert engine.context is not None
    assert engine.context.timeframe == "4h"
    assert any("Timeframe override" in str(call) for call in logger.warning.call_args_list)


def test_the_engine_refuses_an_unsupported_override(build_config):
    engine = make_engine(build_config("6h"), MagicMock())

    with pytest.raises(ValueError, match="Unsupported timeframe"):
        engine.initialize_for_symbol("BTC/USDC", SimpleNamespace(name="binance", timeframes={}), timeframe="13h")


def test_the_prompt_sizing_cap_is_the_configured_one(build_config):
    """A 5% hard cap must not be described as a 10% profile cap in the same prompt.

    The prompt is the model's only view of the cap. When it quoted the profile table
    instead of the effective cap, the model was told it could size 10% while RiskManager
    clamped to 5% — the same failure shape as the hardcoded ``4h``: two sources of truth,
    a plausible one winning over the configured one.
    """
    rules = TemplateManager(
        build_config("6h", max_position_size="0.05"), null_logger(), TimeframeValidator
    ).build_decision_rules()

    assert "AGGRESSIVE 5% / NEUTRAL 5% / CONSERVATIVE 5%" in rules
    assert "fall back to 0.05 (5%)" in rules
    assert "capped by max_position_size" in rules
    assert "AGGRESSIVE 10%" not in rules


def test_the_prompt_sizing_cap_is_unchanged_when_the_config_does_not_bind(build_config):
    """The default 10% cap keeps the historic profile sentence byte for byte."""
    rules = TemplateManager(
        build_config("6h"), null_logger(), TimeframeValidator
    ).build_decision_rules()

    assert "AGGRESSIVE 10% / NEUTRAL 8% / CONSERVATIVE 5%" in rules
    assert "capped by max_position_size" not in rules


def test_an_unusable_position_cap_is_not_replaced_by_a_plausible_one(build_config):
    manager = TemplateManager(
        build_config("6h", max_position_size="0"), null_logger(), TimeframeValidator
    )

    with pytest.raises(ValueError, match="MAX_POSITION_SIZE"):
        manager.build_decision_rules()


@pytest.mark.parametrize("profile", list(RegimeRiskProfile), ids=lambda profile: profile.value)
def test_the_risk_profile_context_names_the_effective_cap(profile):
    text = RegimeRiskProfileSelector.build_profile_context(
        profile, "test reason", max_position_size=0.05
    )

    assert "max position 5%" in text


def test_the_brain_context_provider_forwards_the_configured_cap():
    memory = MagicMock()
    memory.trade_count = 0
    provider = BrainContextProvider(
        vector_memory=memory,
        exit_profiles=MagicMock(),
        max_position_size=0.05,
    )

    text = provider._build_regime_risk_profile_context(choppiness=None, atr_percentage=2.0)

    assert "max position 5%" in text


def test_period_metrics_refuse_an_unknown_frame():
    """The old fallback mapped 1D to 24 candles, i.e. it silently assumed an hourly frame."""
    calculator = MarketMetricsCalculator(null_logger())
    context = SimpleNamespace(ohlcv_candles=np.zeros((100, 6)), timeframe="13h")

    with pytest.raises(ValueError, match="13h"):
        calculator.update_period_metrics(context)
