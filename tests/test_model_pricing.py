"""Tests for model pricing metadata."""
from pathlib import Path
from types import ModuleType
from typing import Any

from src.utils.token_counter import ModelPricing


_CONFIG_CLASS: type | None = None


def test_gemini_3_5_flash_pricing_uses_configured_rates() -> None:
    pricing = ModelPricing()

    cost = pricing.get_cost("google", "gemini-3.5-flash", input_tokens=1_000_000, output_tokens=1_000_000)

    assert cost == 10.50  # $1.50 input + $9.00 output per million tokens (ai.google.dev 2026-05-19)


def test_config_model_mapping_prefers_canonical_penalty_names_over_legacy_aliases() -> None:
    config = _make_config({
        "ai_providers": {"provider": "openrouter", "openrouter_fallback_model": "fallback/model"},
        "model_config": {
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "frequency_penalty": 0.3,
            "freq_penalty": 0.1,
            "presence_penalty": 0.4,
            "pres_penalty": 0.2,
        },
    })

    model_config = config.get_model_config("openrouter/model")

    assert model_config["frequency_penalty"] == 0.3
    assert model_config["presence_penalty"] == 0.4
    assert "freq_penalty" not in model_config
    assert "pres_penalty" not in model_config
    assert config.OPENROUTER_FALLBACK_MODEL == "fallback/model"


def test_config_model_mapping_falls_back_to_legacy_penalty_aliases() -> None:
    config = _make_config({
        "ai_providers": {"provider": "openrouter"},
        "model_config": {
            "max_tokens": 256,
            "freq_penalty": 0.1,
            "pres_penalty": 0.2,
        },
    })

    model_config = config.get_model_config("openrouter/model")

    assert model_config["frequency_penalty"] == 0.1
    assert model_config["presence_penalty"] == 0.2
    assert config.OPENROUTER_FALLBACK_MODEL == "deepseek/deepseek-r1:free"


def test_config_model_mapping_uses_google_runtime_keys() -> None:
    config = _make_config({
        "ai_providers": {"provider": "googleai", "google_studio_model": "gemini-3.5-flash"},
        "model_config": {
            "max_tokens": 256,
            "google_max_tokens": 128,
            "google_thinking_level": "medium",
            "google_code_execution": True,
        },
    })

    model_config = config.get_model_config("gemini-3.5-flash")

    assert model_config["max_tokens"] == 128
    assert model_config["thinking_level"] == "medium"
    assert model_config["google_code_execution"] is True
    assert "temperature" not in model_config
    assert "top_p" not in model_config
    assert "top_k" not in model_config


def _make_config(config_data: dict[str, dict[str, Any]]) -> Any:
    config_class = _load_config_class_without_global_instance()
    config = object.__new__(config_class)
    config._env_vars = {}
    config._config_data = config_data
    config._build_model_configs()
    return config


def _load_config_class_without_global_instance() -> type:
    global _CONFIG_CLASS
    if _CONFIG_CLASS is not None:
        return _CONFIG_CLASS
    source_path = Path(__file__).resolve().parents[1] / "src" / "config" / "loader.py"
    source = source_path.read_text(encoding="utf-8")
    source = source.replace("\n# Create global config instance\nconfig = Config()\n", "\n")
    module = ModuleType("config_loader_under_test")
    module.__file__ = str(source_path)
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    _CONFIG_CLASS = module.Config
    return _CONFIG_CLASS