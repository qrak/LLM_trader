from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from src.config import loader
from src.config.settings_editor import SettingsEditor
from src.dashboard.docs import DocsLibrary
from src.dashboard.routers.settings import SettingsRouter

CONFIG_TEXT = (
    "[general]\n"
    "# Pair comment\n"
    "crypto_pair = BTC/USDC\n"
    "# Frame comment\n"
    "timeframe = 4h\n"
    "candle_limit = 999\n"
    "ai_chart_candle_limit = 125\n"
    "market_type = spot\n"
    "\n"
    "[ai_providers]\n"
    "provider = local\n"
    "\n"
    "[model_config]\n"
    "max_tokens = 1024\n"
)

KEYS_EXAMPLE = "DEEPSEEK_API_KEY=\nOPENCODE_API_KEY=\nBOT_TOKEN_DISCORD=\n"

KEYS_TEXT = "DEEPSEEK_API_KEY=file-secret-value\nOPENCODE_API_KEY=other-secret-value\n"

DOC_README = (
    "# Documentation\n"
    "\n"
    "| a | b |\n"
    "|---|---|\n"
    "| 1 | 2 |\n"
    "\n"
    "[Setup](SETUP.md) [Bad](javascript:alert(1))\n"
    "\n"
    "<script>alert(1)</script>\n"
    "\n"
    "![Shot](assets/x.png)\n"
)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


@pytest.fixture
def desk(tmp_path, monkeypatch):
    config = tmp_path / "config.ini"
    keys = tmp_path / "keys.env"
    example = tmp_path / "keys.env.example"
    write(config, CONFIG_TEXT)
    write(keys, KEYS_TEXT)
    write(example, KEYS_EXAMPLE)
    write(tmp_path / "docs" / "README.md", DOC_README)
    write(tmp_path / "docs" / "SETUP.md", "# Setup\n")
    write(tmp_path / "docs" / "assets" / "x.png", "png")
    write(tmp_path / "README.md", "# Overview\n")
    write(tmp_path / "CONTRIBUTING.md", "# Contributing\n")
    monkeypatch.setattr(loader, "KEYS_ENV_PATH", tmp_path / "missing.env")
    editor = SettingsEditor(
        config_path=config,
        keys_path=keys,
        keys_example_path=example,
        logger=MagicMock(),
    )
    docs = DocsLibrary(root=tmp_path)
    router = SettingsRouter(logger=MagicMock(), settings_editor=editor, docs_library=docs)
    app = FastAPI()
    app.include_router(router.router)
    return SimpleNamespace(
        app=app,
        config=config,
        keys=keys,
        editor=editor,
        docs=docs,
    )


@pytest.fixture
def local_client(desk):
    with TestClient(
        desk.app,
        client=("127.0.0.1", 50000),
        headers={"host": "localhost:8000", "origin": "http://localhost:8000"},
    ) as client:
        yield client


@pytest.fixture
def remote_client(desk):
    with TestClient(desk.app) as client:
        yield client


def test_status_reports_local_for_loopback(local_client):
    response = local_client.get("/api/settings/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["local"] is True
    assert payload["docs"][0]["slug"] == "docs"


def test_status_reports_remote_for_non_loopback(remote_client):
    response = remote_client.get("/api/settings/status")

    assert response.status_code == 200
    assert response.json()["local"] is False


def test_remote_settings_are_forbidden(remote_client):
    etag = "0" * 64
    update = {
        "etag": etag,
        "updates": [
            {"section": "general", "key": "crypto_pair", "value": "ETH/USDC"}
        ],
    }
    key = {"etag": etag, "key": "DEEPSEEK_API_KEY", "value": "new-secret"}

    assert remote_client.get("/api/settings/config").status_code == 403
    assert remote_client.get("/api/settings/keys").status_code == 403
    assert remote_client.post("/api/settings/config", json=update).status_code == 403
    assert remote_client.post("/api/settings/keys", json=key).status_code == 403
    assert remote_client.post("/api/settings/keys/init").status_code == 403


def test_tunnel_header_is_forbidden(local_client):
    response = local_client.get(
        "/api/settings/config",
        headers={"cf-connecting-ip": "203.0.113.7"},
    )

    assert response.status_code == 403


def test_public_forwarded_for_is_forbidden(local_client):
    response = local_client.get(
        "/api/settings/config",
        headers={"x-forwarded-for": "203.0.113.7"},
    )

    assert response.status_code == 403


def test_foreign_host_is_forbidden(local_client):
    for host in ("evil.example", "[::1"):
        response = local_client.get(
            "/api/settings/config",
            headers={"host": host},
        )
        assert response.status_code == 403


def test_foreign_origin_is_forbidden(local_client):
    response = local_client.get(
        "/api/settings/config",
        headers={"origin": "https://evil.example"},
    )

    assert response.status_code == 403


def test_ipv6_loopback_is_allowed(desk):
    with TestClient(
        desk.app,
        client=("::1", 50000),
        headers={"host": "[::1]:8000"},
    ) as client:
        response = client.get("/api/settings/config")

    assert response.status_code == 200


def test_local_can_read_config_and_keys_without_values(local_client):
    config = local_client.get("/api/settings/config")
    keys = local_client.get("/api/settings/keys")

    assert config.status_code == 200
    assert config.json()["sections"][0]["keys"][0]["key"] == "crypto_pair"
    assert keys.status_code == 200
    assert "file-secret-value" not in keys.text
    assert "other-secret-value" not in keys.text


def test_config_save_persists_the_value_and_keeps_comments(local_client, desk):
    etag = local_client.get("/api/settings/config").json()["etag"]
    response = local_client.post(
        "/api/settings/config",
        json={
            "etag": etag,
            "updates": [
                {"section": "general", "key": "crypto_pair", "value": "ETH/USDC"}
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["updated"][0]["value"] == "ETH/USDC"
    text = desk.config.read_text(encoding="utf-8")
    assert "crypto_pair = ETH/USDC" in text
    assert "# Pair comment" in text


def test_config_save_returns_the_key_on_validation_error(local_client, desk):
    before = desk.config.read_bytes()
    etag = local_client.get("/api/settings/config").json()["etag"]
    response = local_client.post(
        "/api/settings/config",
        json={
            "etag": etag,
            "updates": [
                {"section": "ai_providers", "key": "provider", "value": "nope"}
            ],
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["section"] == "ai_providers"
    assert payload["key"] == "provider"
    assert desk.config.read_bytes() == before


def test_config_save_rejects_a_stale_etag(local_client, desk):
    etag = local_client.get("/api/settings/config").json()["etag"]
    with desk.config.open("a", encoding="utf-8", newline="") as handle:
        handle.write("# manual edit\n")
    response = local_client.post(
        "/api/settings/config",
        json={
            "etag": etag,
            "updates": [
                {"section": "general", "key": "crypto_pair", "value": "ETH/USDC"}
            ],
        },
    )

    assert response.status_code == 409
    assert "# manual edit" in desk.config.read_text(encoding="utf-8")


def test_keys_save_never_returns_the_value(local_client, desk):
    etag = local_client.get("/api/settings/keys").json()["etag"]
    response = local_client.post(
        "/api/settings/keys",
        json={"etag": etag, "key": "DEEPSEEK_API_KEY", "value": "brand-new-secret"},
    )

    assert response.status_code == 200
    assert "brand-new-secret" not in response.text
    assert "brand-new-secret" in desk.keys.read_text(encoding="utf-8")


def test_docs_are_public_and_rewrite_links(remote_client):
    response = remote_client.get("/api/settings/docs/docs")

    assert response.status_code == 200
    payload = response.json()
    html = payload["html"]
    assert payload["title"] == "Documentation home"
    assert "<table>" in html
    assert 'href="#docs/setup"' in html
    assert 'src="/docs-assets/x.png"' in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert 'href="javascript:' not in html


def test_unknown_doc_is_404(remote_client):
    response = remote_client.get("/api/settings/docs/nope")

    assert response.status_code == 404
    assert response.json()["error"] == "Document not found."
