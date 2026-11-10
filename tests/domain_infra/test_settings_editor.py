import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.config.settings_editor import SettingsEditor, SettingsError

CONFIG_TEXT = (
    "[general]\r\n"
    "# Pair comment\r\n"
    "crypto_pair = BTC/USDC\r\n"
    "# Frame comment\r\n"
    "timeframe = 4h  # candle frame\r\n"
    "candle_limit = 999\r\n"
    "ai_chart_candle_limit = 125\r\n"
    "market_type = spot\r\n"
    "\r\n"
    "[ai_providers]\r\n"
    "# Provider comment\r\n"
    "provider = local\r\n"
    "\r\n"
    "[model_config]\r\n"
    "# Token comment\r\n"
    "max_tokens = 1024\r\n"
)

KEYS_EXAMPLE = (
    "# Provider keys\n"
    "DEEPSEEK_API_KEY=\n"
    "OPENCODE_API_KEY=\n"
    "# Discord\n"
    "BOT_TOKEN_DISCORD=\n"
)

KEYS_TEXT = (
    "# Provider keys\n"
    "DEEPSEEK_API_KEY=file-secret-value\n"
    "OPENCODE_API_KEY=other-secret-value\n"
    "\n"
    "# Discord\n"
    "BOT_TOKEN_DISCORD=\n"
)


def write(path, text):
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


@pytest.fixture
def desk(tmp_path, monkeypatch):
    for name in (
        "DEEPSEEK_API_KEY",
        "OPENCODE_API_KEY",
        "OPENROUTER_API_KEY",
        "BOT_TOKEN_DISCORD",
    ):
        monkeypatch.delenv(name, raising=False)
    config = tmp_path / "config.ini"
    keys = tmp_path / "keys.env"
    example = tmp_path / "keys.env.example"
    write(config, CONFIG_TEXT)
    write(keys, KEYS_TEXT)
    write(example, KEYS_EXAMPLE)
    logger = MagicMock()
    editor = SettingsEditor(
        config_path=config,
        keys_path=keys,
        keys_example_path=example,
        logger=logger,
    )
    return SimpleNamespace(
        editor=editor,
        config=config,
        keys=keys,
        example=example,
        logger=logger,
    )


def test_read_config_exposes_comments_types_and_required(desk):
    data = desk.editor.read_config()

    assert [section["name"] for section in data["sections"]] == [
        "general",
        "ai_providers",
        "model_config",
    ]
    general = data["sections"][0]
    pair = general["keys"][0]
    frame = general["keys"][1]
    provider = data["sections"][1]["keys"][0]

    assert pair["help"] == "Pair comment"
    assert pair["required"] is True
    assert frame["type"] == "enum"
    assert "4h" in frame["options"]
    assert frame["inline_help"] == "# candle frame"
    assert provider["type"] == "enum"
    assert "local" in provider["options"]


def test_config_update_preserves_comments_inline_comments_and_crlf(desk):
    snapshot = desk.editor.read_config()
    result = desk.editor.update_config(
        snapshot["etag"],
        [{"section": "general", "key": "crypto_pair", "value": "ETH/USDC"}],
    )

    text = desk.config.read_bytes().decode("utf-8")
    assert result["restart_required"] is True
    assert result["etag"] != snapshot["etag"]
    assert "\r\n" in text
    assert "# Pair comment\r\n" in text
    assert "# Frame comment\r\n" in text
    assert "crypto_pair = ETH/USDC\r\n" in text
    assert "timeframe = 4h  # candle frame\r\n" in text
    assert "max_tokens = 1024\r\n" in text


def test_config_update_rejects_unknown_key(desk):
    snapshot = desk.editor.read_config()

    with pytest.raises(SettingsError) as info:
        desk.editor.update_config(
            snapshot["etag"],
            [{"section": "general", "key": "no_such_key", "value": "x"}],
        )

    assert info.value.status == 400
    assert info.value.key == "no_such_key"


def test_config_update_rejects_bad_provider_without_touching_the_file(desk):
    before = desk.config.read_bytes()
    snapshot = desk.editor.read_config()

    with pytest.raises(SettingsError) as info:
        desk.editor.update_config(
            snapshot["etag"],
            [{"section": "ai_providers", "key": "provider", "value": "nope"}],
        )

    assert info.value.status == 400
    assert desk.config.read_bytes() == before


def test_config_update_rejects_a_stale_etag(desk):
    snapshot = desk.editor.read_config()
    with desk.config.open("a", encoding="utf-8", newline="") as handle:
        handle.write("# manual edit\r\n")

    with pytest.raises(SettingsError) as info:
        desk.editor.update_config(
            snapshot["etag"],
            [{"section": "general", "key": "crypto_pair", "value": "ETH/USDC"}],
        )

    assert info.value.status == 409
    assert "# manual edit" in desk.config.read_text(encoding="utf-8")
    assert "crypto_pair = BTC/USDC" in desk.config.read_text(encoding="utf-8")


def test_config_update_rejects_a_value_that_configparser_would_truncate(desk):
    before = desk.config.read_bytes()
    snapshot = desk.editor.read_config()

    with pytest.raises(SettingsError) as info:
        desk.editor.update_config(
            snapshot["etag"],
            [{"section": "general", "key": "crypto_pair", "value": "BTC # USDC"}],
        )

    assert info.value.status == 400
    assert desk.config.read_bytes() == before


def test_read_config_rejects_duplicate_options(desk):
    write(
        desk.config,
        CONFIG_TEXT.replace(
            "crypto_pair = BTC/USDC\r\n",
            "crypto_pair = BTC/USDC\r\ncrypto_pair = ETH/USDC\r\n",
        ),
    )

    with pytest.raises(SettingsError) as info:
        desk.editor.read_config()

    assert "Duplicate option" in str(info.value)


def test_read_keys_never_returns_values(desk):
    payload = desk.editor.read_keys()
    serialized = json.dumps(payload)

    assert "file-secret-value" not in serialized
    assert "other-secret-value" not in serialized
    row = next(item for item in payload["keys"] if item["key"] == "DEEPSEEK_API_KEY")
    assert row["set"] is True
    assert row["source"] == "file"


def test_update_key_writes_only_the_value_and_keeps_comments(desk):
    snapshot = desk.editor.read_keys()
    result = desk.editor.update_key(
        snapshot["etag"],
        "DEEPSEEK_API_KEY",
        "brand-new-secret",
    )

    text = desk.keys.read_text(encoding="utf-8")
    assert "DEEPSEEK_API_KEY=brand-new-secret" in text
    assert "OPENCODE_API_KEY=other-secret-value" in text
    assert "# Provider keys" in text
    assert "file-secret-value" not in text
    assert result["etag"] != snapshot["etag"]
    assert result["set"] is True
    for call in desk.logger.method_calls:
        assert "brand-new-secret" not in str(call)


def test_keys_environment_wins_and_stays_out_of_the_response(desk, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-secret")
    monkeypatch.setenv("BOT_TOKEN_DISCORD", "environment-only-secret")

    payload = desk.editor.read_keys()
    serialized = json.dumps(payload)
    deepseek = next(item for item in payload["keys"] if item["key"] == "DEEPSEEK_API_KEY")
    discord = next(item for item in payload["keys"] if item["key"] == "BOT_TOKEN_DISCORD")

    assert "environment-secret" not in serialized
    assert "environment-only-secret" not in serialized
    assert deepseek["source"] == "both"
    assert discord["source"] == "environment"
    result = desk.editor.update_key(
        payload["etag"],
        "DEEPSEEK_API_KEY",
        "file-now",
    )
    assert result["source"] == "both"
    assert "file-now" not in json.dumps(result)


def test_create_keys_file_is_exclusive_and_copies_the_template(desk):
    desk.keys.unlink()

    first = desk.editor.create_keys_file()
    second = desk.editor.create_keys_file()

    assert first == {"created": True}
    assert second == {"created": False}
    assert desk.keys.read_bytes() == desk.example.read_bytes()


def test_update_key_rejects_unknown_and_duplicate_keys(desk):
    snapshot = desk.editor.read_keys()
    with pytest.raises(SettingsError) as unknown:
        desk.editor.update_key(snapshot["etag"], "NOT_IN_TEMPLATE", "x")
    assert unknown.value.key == "NOT_IN_TEMPLATE"

    write(desk.keys, "DEEPSEEK_API_KEY=one\nDEEPSEEK_API_KEY=two\n")
    duplicate = desk.editor.read_keys()
    with pytest.raises(SettingsError) as duplicated:
        desk.editor.update_key(duplicate["etag"], "DEEPSEEK_API_KEY", "three")
    assert "more than once" in str(duplicated.value)
