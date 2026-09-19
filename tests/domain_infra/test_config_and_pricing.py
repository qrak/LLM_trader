"""Dense domain tests for configuration loading, model pricing metadata and
JSON serialization of the data models.
"""

import io
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from src.config import loader
from src.config.loader import Config
from src.trading.data_models import MarketConditions, Position, TradeDecision
from src.trading.statistics_calculator import StatisticsCalculator, TradingStatistics
from src.utils.data_utils import (
    SerializableMixin,
    get_last_n_valid,
    get_last_valid_value,
    serialize_for_json,
)
from src.utils.token_counter import ModelPricing


@dataclass
class TupleModel(SerializableMixin):
    """Dataclass with a nested tuple field, used to exercise tuple restoration."""

    pairs: tuple[tuple[str, float], ...]


@dataclass
class DatetimeModel(SerializableMixin):
    """Dataclass with a datetime field, used to exercise strict parsing.

    The annotation must stay bare `datetime`: the mixin converts a field whose declared
    type IS datetime, and a `datetime | None` annotation skips conversion entirely.
    """

    ts: datetime = None  # type: ignore[assignment]
    name: str = ""


def config_with(config_data: dict[str, dict[str, Any]]) -> Config:
    """Config instance backed by an in-memory ini mapping."""
    config = object.__new__(Config)
    config._config_data = config_data
    config._env_vars = {}
    if "model_config" in config_data:
        config._build_model_configs()
    return config


def test_convert_value_coercion_matrix():
    convert = Config._convert_value

    assert convert("") == ""
    assert convert("42") == 42 and type(convert("42")) is int
    assert convert("075") == 75 and type(convert("075")) is int
    assert convert("-1") == -1.0 and type(convert("-1")) is float
    assert convert("3.14") == 3.14
    assert convert("1e5") == 100000.0
    assert convert("2.5e-3") == 0.0025
    for raw in ("true", "True", "TRUE", "yes", "on", "1"):
        assert convert(raw) is True, raw
    for raw in ("false", "False", "FALSE", "no", "off", "0"):
        assert convert(raw) is False, raw
    assert convert("a, b, c") == ["a", "b", "c"]
    assert convert("1.0,2.0") == ["1.0", "2.0"]
    assert convert("some-value") == "some-value"
    assert convert("4h") == "4h"
    assert math.isnan(convert("NaN"))
    assert math.isinf(convert("inf")) and convert("inf") > 0
    assert math.isinf(convert("-inf")) and convert("-inf") < 0


def test_get_config_falls_back_on_missing_section_or_key():
    config = Config.__new__(Config)
    config._config_data = {}

    assert config.get_config("nonexistent", "key", "fallback") == "fallback"

    config._config_data["existing"] = {}
    assert config.get_config("existing", "missing_key", 42) == 42

    config._config_data["test"] = {"key": "real_value"}
    assert config.get_config("test", "key") == "real_value"


def test_ini_loading_strips_inline_comments_and_keeps_percent_values(tmp_path, monkeypatch):
    config_path = tmp_path / "config.ini"
    config_path.write_text(
        """
[rag]
update_interval_hours = 4               # hours between RAG rebuilds
news_sources = coindesk, cointelegraph  # enabled feeds

[demo_trading]
transaction_fee_percent = 0.00075       # 0.075% maker fee

[risk_management]
sl_tightening_scalping = 0.25           # < 1h timeframe

[debug]
note = keep 20% buffer                  # literal percent in value is valid
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(loader, "CONFIG_INI_PATH", config_path)
    config = Config.__new__(Config)
    config._config_data = {}

    config._load_ini_config()

    assert config._config_data["rag"]["update_interval_hours"] == 4
    assert config._config_data["rag"]["news_sources"] == ["coindesk", "cointelegraph"]
    assert config._config_data["demo_trading"]["transaction_fee_percent"] == 0.00075
    assert config._config_data["risk_management"]["sl_tightening_scalping"] == 0.25
    assert config._config_data["debug"]["note"] == "keep 20% buffer"


def test_shipped_example_config_loads_without_comment_text(monkeypatch):
    monkeypatch.setattr(loader, "CONFIG_INI_PATH", loader.CONFIG_DIR / "config.ini.example")
    config = Config.__new__(Config)
    config._config_data = {}

    config._load_ini_config()

    assert config._config_data["rag"]["update_interval_hours"] == 4
    assert config._config_data["rag"]["news_max_concurrency"] == 6
    assert config._config_data["risk_management"]["sl_tightening_scalping"] == 0.25
    assert config._config_data["cooldowns"]["file_message_expiry"] == 168
    assert config._config_data["exchanges"]["supported"] == [
        "binance",
        "kucoin",
        "gateio",
        "mexc",
        "hyperliquid",
    ]


def test_executor_max_position_usdc_coercion():
    def build(max_value):
        return config_with({"executor_api": {"max_position_usdc": max_value}})

    parsed = build("100.0").EXECUTOR_MAX_POSITION_USDC
    assert parsed == 100.0 and type(parsed) is float
    assert build("not-a-number").EXECUTOR_MAX_POSITION_USDC == 0.0
    assert config_with({}).EXECUTOR_MAX_POSITION_USDC == 0.0


def test_model_pricing_uses_configured_rates():
    pricing = ModelPricing()

    assert pricing.get_cost("google", "gemini-3.5-flash", input_tokens=1_000_000, output_tokens=1_000_000) == 10.50


def test_model_pricing_degrades_to_empty_tables_on_missing_or_corrupt_file():
    with patch("builtins.open", side_effect=FileNotFoundError):
        missing = ModelPricing()

    assert missing._pricing == {"google": {}, "openrouter": {}}
    assert missing.get_cost("google", "gemini-3.5-flash", input_tokens=1000, output_tokens=1000) is None

    with patch("builtins.open", return_value=io.StringIO("{not-json")):
        corrupt = ModelPricing()

    assert corrupt._pricing == {"google": {}, "openrouter": {}}
    assert corrupt.get_cost("openrouter", "any-model", input_tokens=1000, output_tokens=1000) is None


def test_model_config_mapping_penalty_aliases_and_google_runtime_keys():
    canonical = config_with(
        {
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
        }
    ).get_model_config("openrouter/model")
    assert canonical["frequency_penalty"] == 0.3
    assert canonical["presence_penalty"] == 0.4
    assert "freq_penalty" not in canonical
    assert "pres_penalty" not in canonical
    assert canonical["max_tokens"] == 256

    legacy = config_with(
        {"ai_providers": {"provider": "openrouter"}, "model_config": {"max_tokens": 256, "freq_penalty": 0.1, "pres_penalty": 0.2}}
    )
    legacy_config = legacy.get_model_config("openrouter/model")
    assert legacy_config["frequency_penalty"] == 0.1
    assert legacy_config["presence_penalty"] == 0.2
    assert legacy.OPENROUTER_FALLBACK_MODEL == "deepseek/deepseek-v4.1-flash"

    google = config_with(
        {
            "ai_providers": {"provider": "googleai", "google_studio_model": "gemini-3.5-flash"},
            "model_config": {
                "max_tokens": 256,
                "google_max_tokens": 128,
                "google_thinking_level": "medium",
                "google_code_execution": True,
            },
        }
    ).get_model_config("gemini-3.5-flash")
    assert google["max_tokens"] == 128
    assert google["thinking_level"] == "medium"
    assert google["google_code_execution"] is True
    assert "temperature" not in google
    assert "top_p" not in google
    assert "top_k" not in google


def test_serialize_for_json_handles_primitives_numpy_and_non_finite_values():
    data = {
        "str": "hello",
        "int": 10,
        "bool": True,
        "none": None,
        "float": 3.14,
        "nan": float("nan"),
        "inf": float("inf"),
        "np_int": np.int64(42),
        "np_float": np.float64(3.14),
        "np_nan": np.float64(np.nan),
        "array": np.array([1.0, 2.0]),
    }

    serialized = serialize_for_json(data)

    assert serialized["str"] == "hello"
    assert serialized["int"] == 10
    assert serialized["bool"] is True
    assert serialized["none"] is None
    assert serialized["float"] == 3.14
    assert serialized["nan"] is None
    assert serialized["inf"] is None
    assert serialized["np_int"] == 42
    assert serialized["np_float"] == 3.14
    assert serialized["np_nan"] is None
    assert serialized["array"] == [1.0, 2.0]


def test_last_valid_value_helpers_skip_nan_and_empty_arrays():
    assert get_last_valid_value(np.array([1.0, np.nan, 3.5, np.nan])) == 3.5
    assert get_last_valid_value(np.array([np.nan, np.nan]), default=0.0) == 0.0
    assert get_last_valid_value(np.array([]), default=-1.0) == -1.0
    assert get_last_valid_value(np.array([np.inf, 2.0])) == 2.0

    np.testing.assert_array_equal(get_last_n_valid(np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan]), 3), [2.0, 4.0, 5.0])
    assert len(get_last_n_valid(np.array([]), 5)) == 0
    assert get_last_n_valid(np.array([np.nan, 7.0]), 1).tolist() == [7.0]


def test_serializable_mixin_restores_tuples_and_rejects_corrupt_datetime():
    model = TupleModel.from_dict({"pairs": [["trend", 80.0], ["volume", 55.5]]})

    assert model.pairs == (("trend", 80.0), ("volume", 55.5))
    assert type(model.pairs) is tuple
    assert type(model.pairs[0]) is tuple

    with pytest.raises(ValueError, match="Cannot convert"):
        DatetimeModel.from_dict({"ts": "not-a-date", "name": "test"})


def test_statistics_calculator_trade_pairing_and_pnl():
    calculator = StatisticsCalculator()

    def history(*rows):
        return [{"action": action, "price": price, "quantity": quantity} for action, price, quantity in rows]

    assert calculator.calculate_from_history(history(("BUY", 100.0, 1.0), ("CLOSE", 110.0, 1.0))).total_pnl_quote == 10.0

    short = calculator.calculate_from_history(history(("SELL", 110.0, 1.0), ("CLOSE_SHORT", 100.0, 1.0)))
    assert (short.total_trades, short.winning_trades, short.total_pnl_quote) == (1, 1, 10.0)

    assert calculator.calculate_from_history(history(("CLOSE", 110.0, 1.0), ("BUY", 100.0, 1.0), ("CLOSE", 105.0, 1.0))).total_trades == 1
    assert calculator.calculate_from_history(history(("BUY", 100.0, 1.0), ("CLOSE", 110.0, 1.0), ("CLOSE", 115.0, 1.0))).total_trades == 1
    assert calculator.calculate_from_history(history(("BUY", 100.0, 1.0), ("BUY", 95.0, 1.0))).total_trades == 0
    assert calculator.calculate_from_history(history(("BUY", 0.0, 1.0), ("CLOSE", 110.0, 1.0))).total_trades == 0


def test_statistics_calculator_degenerate_metrics():
    calculator = StatisticsCalculator()

    def history(*rows):
        return [{"action": action, "price": price, "quantity": quantity} for action, price, quantity in rows]

    empty = calculator.calculate_from_history([])
    assert (empty.total_trades, empty.win_rate) == (0, 0.0)

    single_win = calculator.calculate_from_history(history(("BUY", 100.0, 1.0), ("CLOSE", 110.0, 1.0)))
    assert math.isinf(single_win.profit_factor)
    assert single_win.sharpe_ratio == 0.0
    assert single_win.sortino_ratio == 0.0

    all_wins = calculator.calculate_from_history(
        history(("BUY", 100.0, 1.0), ("CLOSE", 110.0, 1.0), ("BUY", 100.0, 1.0), ("CLOSE", 105.0, 1.0))
    )
    assert math.isinf(all_wins.sortino_ratio) and all_wins.sortino_ratio > 0

    single_loss = calculator.calculate_from_history(history(("BUY", 100.0, 1.0), ("CLOSE", 90.0, 1.0)))
    assert single_loss.max_drawdown_pct < 0
    assert single_loss.worst_trade_pct < 0

    zero_capital = calculator.calculate_from_history(history(("BUY", 100.0, 1.0), ("CLOSE", 110.0, 1.0)), initial_capital=0.0)
    assert zero_capital.total_pnl_pct == 0.0


def test_statistics_and_models_survive_json_round_trip_with_non_finite_values():
    stats = TradingStatistics(total_trades=5, winning_trades=5, sortino_ratio=float("inf"), profit_factor=float("inf"))
    loaded_raw = json.loads(json.dumps(serialize_for_json(stats.to_dict())))
    assert loaded_raw["sortino_ratio"] is None
    assert loaded_raw["profit_factor"] is None
    restored = TradingStatistics.from_dict(loaded_raw)
    assert (restored.sortino_ratio, restored.profit_factor) == (0.0, 0.0)

    nan_stats = TradingStatistics(win_rate=float("nan"), sharpe_ratio=float("nan"))
    restored_nan = TradingStatistics.from_dict(json.loads(json.dumps(serialize_for_json(nan_stats.to_dict()))))
    assert (restored_nan.win_rate, restored_nan.sharpe_ratio) == (0.0, 0.0)

    corrupted = TradingStatistics.from_dict(
        {field: None for field in ("total_trades", "winning_trades", "losing_trades", "win_rate", "total_pnl_pct",
                                   "total_pnl_quote", "initial_capital", "current_capital", "avg_trade_pct",
                                   "best_trade_pct", "worst_trade_pct", "max_drawdown_pct", "avg_drawdown_pct",
                                   "sharpe_ratio", "sortino_ratio", "profit_factor")}
    )
    assert (corrupted.total_trades, corrupted.win_rate, corrupted.initial_capital) == (0, 0.0, 0.0)


def test_position_and_trade_decision_round_trip_through_json():
    position = Position(
        symbol="BTC/USDC",
        direction="LONG",
        confidence="HIGH",
        entry_price=50000.0,
        stop_loss=49000.0,
        take_profit=52000.0,
        size=0.001,
        entry_time=datetime(2026, 5, 25, 12, 0, 0, tzinfo=timezone.utc),
        entry_fee=3.75,
        size_pct=0.05,
        tp_distance_pct=4.0,
        sl_distance_pct=2.0,
        confluence_factors=(("trend", 0.8), ("momentum", 0.6)),
        conditions_at_entry=MarketConditions(),
    )

    restored = Position.from_dict(json.loads(json.dumps(serialize_for_json(position.to_dict()))))

    assert restored.symbol == "BTC/USDC"
    assert restored.direction == "LONG"
    assert restored.entry_price == 50000.0
    assert restored.stop_loss == 49000.0
    assert restored.size_pct == 0.05
    assert type(restored.entry_time) is datetime
    assert restored.confluence_factors == (("trend", 0.8), ("momentum", 0.6))
    assert all(type(factor) is tuple for factor in restored.confluence_factors)

    decision = TradeDecision(
        timestamp=datetime(2026, 5, 25, 12, 0, 0, tzinfo=timezone.utc),
        symbol="BTC/USDC",
        action="HOLD",
        confidence="MEDIUM",
        price=50000.0,
        stop_loss=None,
        take_profit=None,
        position_size=0.0,
        reasoning="Test reasoning",
    )
    restored_decision = TradeDecision.from_dict(json.loads(json.dumps(serialize_for_json(decision.to_dict()))))

    assert restored_decision.stop_loss is None
    assert restored_decision.take_profit is None
    assert restored_decision.action == "HOLD"


def test_position_from_dict_restores_confluence_tuples_from_plain_lists():
    position = Position.from_dict(
        {
            "symbol": "BTC/USDC",
            "direction": "LONG",
            "confidence": "HIGH",
            "entry_price": 50000.0,
            "stop_loss": 49000.0,
            "take_profit": 52000.0,
            "size": 0.001,
            "entry_time": "2026-05-25T12:00:00+00:00",
            "confluence_factors": [["trend", 0.8], ["momentum", 0.6]],
            "conditions_at_entry": MarketConditions().to_dict(),
        }
    )

    assert type(position.confluence_factors) is tuple
    assert position.confluence_factors == (("trend", 0.8), ("momentum", 0.6))
    assert all(type(factor) is tuple for factor in position.confluence_factors)
