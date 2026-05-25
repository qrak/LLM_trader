"""Silent-failure tests for Config loader — type coercion, missing keys, edge values.

These tests bypass the conftest.py mock by importing the source module directly.
"""

import sys
import pytest


@pytest.fixture(autouse=True)
def _unmock_config():
    """Remove conftest's mock for this module so we can test the real Config."""
    mock = sys.modules.pop('src.config.loader', None)
    yield
    if mock is not None:
        sys.modules['src.config.loader'] = mock


# ═══════════════════════════════════════════════════════════════════════
# _convert_value unit tests
# ═══════════════════════════════════════════════════════════════════════

class TestConvertValue:
    """Config._convert_value — the central type-coercion function."""

    @pytest.fixture
    def convert_value(self):
        from src.config.loader import Config
        return Config._convert_value

    def test_empty_string_returns_empty_string(self, convert_value):
        assert convert_value("") == ""

    def test_positive_integer(self, convert_value):
        result = convert_value("42")
        assert result == 42
        assert isinstance(result, int)

    def test_leading_zero_integer(self, convert_value):
        """'075' is isdigit() True → int(75)."""
        result = convert_value("075")
        assert result == 75
        assert isinstance(result, int)

    def test_negative_number_returns_float(self, convert_value):
        """'-1' is NOT isdigit() → falls through to float()."""
        result = convert_value("-1")
        assert result == -1.0
        assert isinstance(result, float)

    def test_float_with_decimal(self, convert_value):
        result = convert_value("3.14")
        assert result == 3.14

    def test_scientific_notation(self, convert_value):
        assert convert_value("1e5") == 100000.0
        assert convert_value("2.5e-3") == 0.0025

    def test_boolean_true_variants(self, convert_value):
        for v in ("true", "True", "TRUE", "yes", "on", "1"):
            assert convert_value(v) is True, f"'{v}' should be True"

    def test_boolean_false_variants(self, convert_value):
        for v in ("false", "False", "FALSE", "no", "off", "0"):
            assert convert_value(v) is False, f"'{v}' should be False"

    def test_comma_list_becomes_string_list(self, convert_value):
        result = convert_value("a, b, c")
        assert result == ["a", "b", "c"]

    def test_numeric_comma_list_is_string_list(self, convert_value):
        """'1.0,2.0' is not isdigit(), not valid float → split into strings."""
        result = convert_value("1.0,2.0")
        assert result == ["1.0", "2.0"]
        assert all(isinstance(x, str) for x in result)

    def test_plain_string_passthrough(self, convert_value):
        assert convert_value("some-value") == "some-value"
        assert convert_value("4h") == "4h"

    def test_nan_string_falls_through(self, convert_value):
        """'NaN' — float('NaN') succeeds in Python! Returns float('nan')."""
        import math
        result = convert_value("NaN")
        assert math.isnan(result), f"Expected NaN, got {result!r}"

    def test_inf_string_becomes_float_inf(self, convert_value):
        """'inf' → float('inf'). DANGER: this breaks JSON serialization downstream."""
        import math
        result = convert_value("inf")
        assert math.isinf(result) and result > 0

    def test_negative_inf(self, convert_value):
        import math
        result = convert_value("-inf")
        assert math.isinf(result) and result < 0


# ═══════════════════════════════════════════════════════════════════════
# get_config fallback tests
# ═══════════════════════════════════════════════════════════════════════

class TestGetConfig:
    """get_config fallback behavior with missing sections/keys."""

    @pytest.fixture
    def cfg(self):
        from src.config.loader import Config
        c = Config.__new__(Config)
        # _config_data is a dict[section, dict[key, value]] populated from ini
        c._config_data = {}
        return c

    def test_missing_section_returns_default(self, cfg):
        assert cfg.get_config("nonexistent", "key", "fallback") == "fallback"

    def test_missing_key_returns_default(self, cfg):
        cfg._config_data["existing"] = {}
        assert cfg.get_config("existing", "missing_key", 42) == 42

    def test_present_key_returns_value(self, cfg):
        cfg._config_data["test"] = {"key": "real_value"}
        assert cfg.get_config("test", "key") == "real_value"


class TestIniLoading:
    """Real INI parsing behavior around inline comments and interpolation."""

    def test_load_ini_config_strips_inline_comments_and_allows_percent_signs(self, tmp_path, monkeypatch):
        import src.config.loader as loader

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
        cfg = loader.Config.__new__(loader.Config)
        cfg._config_data = {}

        cfg._load_ini_config()

        assert cfg._config_data["rag"]["update_interval_hours"] == 4
        assert cfg._config_data["rag"]["news_sources"] == ["coindesk", "cointelegraph"]
        assert cfg._config_data["demo_trading"]["transaction_fee_percent"] == 0.00075
        assert cfg._config_data["risk_management"]["sl_tightening_scalping"] == 0.25
        assert cfg._config_data["debug"]["note"] == "keep 20% buffer"

    def test_example_config_loads_values_without_inline_comment_text(self, monkeypatch):
        import src.config.loader as loader

        monkeypatch.setattr(loader, "CONFIG_INI_PATH", loader.CONFIG_DIR / "config.ini.example")
        cfg = loader.Config.__new__(loader.Config)
        cfg._config_data = {}

        cfg._load_ini_config()

        assert cfg._config_data["rag"]["update_interval_hours"] == 4
        assert cfg._config_data["rag"]["news_max_concurrency"] == 6
        assert cfg._config_data["exchanges"]["supported"] == [
            "binance", "kucoin", "gateio", "mexc", "hyperliquid",
        ]
        assert cfg._config_data["risk_management"]["sl_tightening_scalping"] == 0.25
        assert cfg._config_data["cooldowns"]["file_message_expiry"] == 168
