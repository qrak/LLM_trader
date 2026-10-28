"""Configuration contract gates: the loader, the template, config.ini, the docs and the code agree.

These tests pin the two scripts that keep a second source of truth from appearing, plus the
invariants the shipped configuration must satisfy. A failure here means an agent (or a person)
changed one of the four artefacts without the others, which is exactly the omission this
repository cannot afford.
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.check_config_contract import ConfigContractChecker
from scripts.check_hardcoded_timeframes import TIMEFRAMES, HardcodedTimeframeChecker
from src.config import loader
from src.config.loader import Config
from src.utils.timeframe_validator import TimeframeValidator
from tests.conftest import make_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ACTIVE_CONFIG = PROJECT_ROOT / "config" / "config.ini"

FAKE_LOADER = '''
class Config:
    REQUIRED_CONFIG_KEYS = {"general": ("timeframe",)}

    def TIMEFRAME(self):
        return self.get_config("general", "timeframe", "1h")
'''

CLEAN_EXAMPLE = "[general]\ntimeframe = 4h\n"


def write_contract_tree(root: Path, example: str, loader_source: str = FAKE_LOADER) -> None:
    (root / "src" / "config").mkdir(parents=True)
    (root / "src" / "config" / "loader.py").write_text(loader_source, encoding="utf-8")
    (root / "config").mkdir(parents=True)
    (root / "config" / "config.ini.example").write_text(example, encoding="utf-8")


def test_the_repository_config_contract_is_clean():
    assert ConfigContractChecker().check() == []


def test_the_repository_has_no_hardcoded_timeframes():
    assert HardcodedTimeframeChecker().check() == []


def test_the_literal_gate_knows_every_supported_frame():
    """A frame added to the validator must not silently escape the AST gate."""
    assert TIMEFRAMES == tuple(TimeframeValidator.SUPPORTED_TIMEFRAMES)


def test_a_loader_key_missing_from_the_template_is_reported(tmp_path):
    loader_source = FAKE_LOADER.replace(
        'return self.get_config("general", "timeframe", "1h")',
        'return self.get_config("general", "timeframe", "1h") + str(self.get_config("general", "future_key"))',
    )
    write_contract_tree(tmp_path, CLEAN_EXAMPLE, loader_source)

    violations = ConfigContractChecker(tmp_path).check()

    assert any("future_key" in line and "config.ini.example" in line for line in violations)


def test_a_typo_in_the_active_config_is_reported(tmp_path):
    write_contract_tree(tmp_path, CLEAN_EXAMPLE)
    (tmp_path / "config" / "config.ini").write_text(
        "[general]\ntimefrmae = 6h\n", encoding="utf-8"
    )

    violations = ConfigContractChecker(tmp_path).check()

    assert any("timefrmae" in line and "silently ignored" in line for line in violations)


def test_a_required_key_missing_from_the_template_is_reported(tmp_path):
    write_contract_tree(tmp_path, "[general]\ncrypto_pair = BTC/USDC\n")

    violations = ConfigContractChecker(tmp_path).check()

    assert any(line.startswith("REQUIRED_CONFIG_KEYS") and "timeframe" in line for line in violations)


def test_a_documented_key_absent_from_the_template_is_reported(tmp_path):
    write_contract_tree(tmp_path, CLEAN_EXAMPLE)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "CONFIGURATION.md").write_text(
        "## Core settings\n\n| Setting | Default | Description |\n|---|---|---|\n"
        "| `ghost_key` | 1 | documented but not shipped |\n",
        encoding="utf-8",
    )

    violations = ConfigContractChecker(tmp_path).check()

    assert any("ghost_key" in line for line in violations)


def test_a_literal_timeframe_in_src_is_reported(tmp_path):
    module = tmp_path / "src" / "pkg" / "analysis.py"
    module.parent.mkdir(parents=True)
    module.write_text('FALLBACK_TIMEFRAME = "4h"\n', encoding="utf-8")

    violations = HardcodedTimeframeChecker(tmp_path).check()

    assert violations
    assert "analysis.py:1" in violations[0]
    assert "'4h'" in violations[0]


def test_the_allowlisted_definition_module_is_exempt(tmp_path):
    module = tmp_path / "src" / "utils" / "timeframe_validator.py"
    module.parent.mkdir(parents=True)
    module.write_text('SUPPORTED_TIMEFRAMES = ["5m", "4h", "1d"]\n', encoding="utf-8")

    assert HardcodedTimeframeChecker(tmp_path).check() == []


def test_a_timeframe_inside_a_docstring_is_not_reported(tmp_path):
    module = tmp_path / "src" / "pkg" / "notes.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        '"""The bot was designed for 4h candles, never for 1h ones."""\n\nVALUE = 1\n',
        encoding="utf-8",
    )

    assert HardcodedTimeframeChecker(tmp_path).check() == []


def test_missing_timeframe_key_fails_startup_by_name(tmp_path, monkeypatch):
    ini = tmp_path / "config.ini"
    ini.write_text(
        "[general]\ncrypto_pair = BTC/USDC\ncandle_limit = 999\n"
        "ai_chart_candle_limit = 125\nmarket_type = spot\n"
        "[ai_providers]\nprovider = local\n[model_config]\nmax_tokens = 1024\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loader, "CONFIG_INI_PATH", ini)
    monkeypatch.setattr(loader, "KEYS_ENV_PATH", tmp_path / "missing.env")
    with patch.dict(os.environ, {}, clear=True), pytest.raises(RuntimeError, match="timeframe"):
        Config()


def test_a_missing_general_section_is_named_key_by_key(tmp_path, monkeypatch):
    ini = tmp_path / "config.ini"
    ini.write_text(
        "[ai_providers]\nprovider = local\n[model_config]\nmax_tokens = 1024\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loader, "CONFIG_INI_PATH", ini)
    monkeypatch.setattr(loader, "KEYS_ENV_PATH", tmp_path / "missing.env")
    with patch.dict(os.environ, {}, clear=True), pytest.raises(RuntimeError, match=r"\[general\] crypto_pair"):
        Config()


@pytest.mark.skipif(not ACTIVE_CONFIG.exists(), reason="config.ini is local and gitignored")
def test_the_shipped_configuration_is_internally_consistent():
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))
    parser.optionxform = lambda option: option
    parser.read(ACTIVE_CONFIG, encoding="utf-8")
    timeframe = TimeframeValidator.validate_and_normalize(parser.get("general", "timeframe"))
    timeframe_minutes = TimeframeValidator.to_minutes(timeframe)

    assert int(parser.get("general", "ai_chart_candle_limit")) <= int(parser.get("general", "candle_limit"))
    for key in ("stop_loss_check_interval", "take_profit_check_interval"):
        if parser.has_option("risk_management", key):
            assert TimeframeValidator.parse_period_to_minutes(parser.get("risk_management", key)) <= timeframe_minutes
    provider = parser.get("ai_providers", "provider")
    model_key = {
        "deepseek": "deepseek_model",
        "googleai": "google_studio_model",
        "openrouter": "openrouter_base_model",
        "local": "lm_studio_model",
    }.get(provider)
    if model_key and parser.has_option("ai_providers", model_key) and parser.get("ai_providers", model_key):
        target = f"{provider}/{parser.get('ai_providers', model_key)}"
        assert target in parser["model_context_windows"], f"{target} has no context window"


def test_the_test_config_double_keeps_frame_and_minutes_in_step():
    config = make_config()

    assert TimeframeValidator.validate(config.TIMEFRAME)
    assert config.TIMEFRAME_MINUTES == TimeframeValidator.to_minutes(config.TIMEFRAME)
    assert config.AI_CHART_CANDLE_LIMIT <= config.CANDLE_LIMIT
