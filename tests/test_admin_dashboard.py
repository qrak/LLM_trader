"""Tests for the admin dashboard — auth, config, system control, log streaming.

Covers:
- Authentication: login, logout, session verification, 401 on missing/invalid auth
- Config CRUD: read schema, update values, batch update, validation errors
- System control: force analysis, toggle feed, human input
- Log streaming: recent logs REST endpoint
- WritableConfig: atomic writes, validation, schema metadata
"""

import asyncio
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from src.dashboard.auth import (
    check_credentials,
    create_session,
    hash_password,
    init_auth,
    verify_admin_session,
    _sign_token,
    _verify_token,
    _is_lan_ip,
    _get_real_client_ip,
    AdminAuthMiddleware,
)
from src.dashboard.log_stream import LogStreamHandler, LogStreamManager
from src.config.writable_config import WritableConfig, SettingMeta, _validate_and_coerce


def _has_numpy() -> bool:
    """Check if numpy is available (needed for full DashboardServer import)."""
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


# ─── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def tmp_config(tmp_path):
    """Create a temporary config.ini for testing."""
    config_content = """[general]
crypto_pair = BTC/USDC
timeframe = 4h
candle_limit = 999
discord_bot = true

[debug]
save_chart_images = false
logger_debug = true

[ai_providers]
provider = googleai

[model_config]
temperature = 1.0
top_p = 0.9

[dashboard]
enabled = true
host = 0.0.0.0
port = 8000
"""
    config_path = tmp_path / "config.ini"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def writable_config(tmp_config):
    return WritableConfig(tmp_config)


@pytest.fixture
def auth_env():
    """Initialize auth with test credentials."""
    password_hash = hash_password("testpass123")
    init_auth(
        signing_key="test-signing-key-secret",
        admin_username="admin",
        admin_password_hash=password_hash,
    )
    return {"username": "admin", "password": "testpass123"}


@pytest.fixture
def log_manager():
    return LogStreamManager(max_queue_size=100)


@pytest.fixture
def mock_config():
    """Mock bot Config object."""
    config = MagicMock()
    config.ADMIN_USERNAME = "admin"
    config.ADMIN_PASSWORD_HASH = hash_password("testpass123")
    config.ADMIN_SIGNING_KEY = "test-signing-key"
    config.DASHBOARD_ENABLE_CORS = False
    config.DASHBOARD_CORS_ORIGINS = []
    return config


@pytest.fixture
def admin_app(writable_config, log_manager, mock_config, auth_env):
    """Create a FastAPI test app with admin router mounted."""
    from src.dashboard.routers.admin import AdminRouter

    force_event = asyncio.Event()

    app = FastAPI()

    admin_router = AdminRouter(
        writable_config=writable_config,
        log_stream_manager=log_manager,
        config=mock_config,
        logger=MagicMock(),
        force_analysis_event=force_event,
    )
    app.include_router(admin_router.router)
    app.add_middleware(AdminAuthMiddleware)

    return app


@pytest.fixture
def client(admin_app):
    """Create a test HTTP client that simulates a LAN connection."""
    return TestClient(admin_app, raise_server_exceptions=False, client=("192.168.1.100", 54321))


# ─── Auth Tests ──────────────────────────────────────────────────────

class TestPasswordHashing:
    def test_hash_and_verify(self):
        h = hash_password("mypassword")
        assert ":" in h
        salt_hex, hash_hex = h.split(":")
        assert len(salt_hex) == 32  # 16 bytes = 32 hex chars
        assert len(hash_hex) == 64  # 32 bytes = 64 hex chars

    def test_different_salts(self):
        h1 = hash_password("same_password")
        h2 = hash_password("same_password")
        assert h1 != h2  # Different salts

    def test_verify_correct_password(self, auth_env):
        assert check_credentials("admin", "testpass123")

    def test_verify_wrong_password(self, auth_env):
        assert not check_credentials("admin", "wrongpassword")

    def test_verify_wrong_username(self, auth_env):
        assert not check_credentials("wronguser", "testpass123")


class TestSessionTokens:
    def test_sign_and_verify(self):
        token = _sign_token("admin", time.time())
        assert _verify_token(token) == "admin"

    def test_expired_token(self):
        token = _sign_token("admin", time.time() - 100000)
        assert _verify_token(token) is None

    def test_tampered_token(self):
        token = _sign_token("admin", time.time())
        parts = token.split(":")
        parts[2] = "tampered"
        assert _verify_token(":".join(parts)) is None

    def test_invalid_format(self):
        assert _verify_token("invalid") is None
        assert _verify_token("") is None
        assert _verify_token("a:b") is None


class TestLoginEndpoint:
    def test_login_success(self, client, auth_env):
        resp = client.post("/api/admin/login", json={"username": "admin", "password": "testpass123"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "admin_session" in resp.cookies

    def test_login_invalid_credentials(self, client, auth_env):
        resp = client.post("/api/admin/login", json={"username": "admin", "password": "wrong"})
        assert resp.status_code == 401

    def test_login_missing_fields(self, client):
        resp = client.post("/api/admin/login", json={"username": "admin"})
        assert resp.status_code == 422  # Pydantic validation


class TestAuthMiddleware:
    def test_unauthenticated_api_returns_401(self, client, auth_env):
        resp = client.get("/api/admin/config")
        assert resp.status_code == 401

    def test_health_endpoint_no_auth(self, client, auth_env):
        resp = client.get("/api/admin/health")
        assert resp.status_code == 200

    def test_authenticated_request_succeeds(self, client, auth_env):
        # Login first
        client.post("/api/admin/login", json={"username": "admin", "password": "testpass123"})
        resp = client.get("/api/admin/config")
        assert resp.status_code == 200


class TestWsToken:
    def test_ws_token_requires_auth(self, client, auth_env):
        resp = client.get("/api/admin/ws-token")
        assert resp.status_code == 401

    def test_ws_token_returns_token_when_authenticated(self, client, auth_env):
        client.post("/api/admin/login", json={"username": "admin", "password": "testpass123"})
        resp = client.get("/api/admin/ws-token")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["username"] == "admin"
        assert _verify_token(data["token"]) == "admin"


# ─── Config Tests ────────────────────────────────────────────────────

class TestWritableConfig:
    def test_get_full_schema(self, writable_config):
        schema = writable_config.get_full_schema()
        assert "general" in schema
        assert "ai_providers" in schema
        assert "debug" in schema
        assert schema["general"]["title"] == "General"

    def test_get_section_schema(self, writable_config):
        section = writable_config.get_section_schema("general")
        assert section is not None
        assert "crypto_pair" in section["keys"]

    def test_get_unknown_section(self, writable_config):
        assert writable_config.get_section_schema("nonexistent") is None

    @pytest.mark.asyncio
    async def test_set_value_hot(self, writable_config):
        category = await writable_config.set_value("debug", "logger_debug", "false")
        assert category == "hot"
        assert writable_config.get_value("debug", "logger_debug") == "false"

    @pytest.mark.asyncio
    async def test_set_value_cycle(self, writable_config):
        category = await writable_config.set_value("general", "timeframe", "1h")
        assert category == "cycle"
        assert writable_config.get_value("general", "timeframe") == "1h"

    @pytest.mark.asyncio
    async def test_set_value_invalid_enum(self, writable_config):
        with pytest.raises(ValueError, match="Invalid option"):
            await writable_config.set_value("general", "timeframe", "invalid")

    @pytest.mark.asyncio
    async def test_set_value_unknown_key(self, writable_config):
        with pytest.raises(ValueError, match="Unknown config key"):
            await writable_config.set_value("general", "nonexistent", "value")

    @pytest.mark.asyncio
    async def test_set_value_persists_to_disk(self, writable_config, tmp_config):
        await writable_config.set_value("general", "crypto_pair", "ETH/USDC")
        # Re-read from disk
        wc2 = WritableConfig(tmp_config)
        assert wc2.get_value("general", "crypto_pair") == "ETH/USDC"

    @pytest.mark.asyncio
    async def test_batch_update(self, writable_config):
        results = await writable_config.set_values([
            ("general", "timeframe", "1d"),
            ("debug", "logger_debug", "false"),
        ])
        assert len(results) == 2
        assert results[0]["category"] == "cycle"
        assert results[1]["category"] == "hot"

    @pytest.mark.asyncio
    async def test_reload_from_disk(self, writable_config, tmp_config):
        # Modify file externally
        tmp_config.write_text("[general]\ncrypto_pair = SOL/USDC\n")
        await writable_config.reload_from_disk()
        assert writable_config.get_value("general", "crypto_pair") == "SOL/USDC"

    def test_reload_event(self, writable_config):
        assert not writable_config.read_reload_event()


class TestValidation:
    def test_validate_bool(self):
        meta = SettingMeta(key="test", type="bool", category="hot", description="test")
        assert _validate_and_coerce("true", meta) == "true"
        assert _validate_and_coerce("false", meta) == "false"
        assert _validate_and_coerce("1", meta) == "true"
        assert _validate_and_coerce("0", meta) == "false"
        with pytest.raises(ValueError):
            _validate_and_coerce("maybe", meta)

    def test_validate_int(self):
        meta = SettingMeta(key="test", type="int", category="hot", description="test", min_val=1, max_val=100)
        assert _validate_and_coerce(50, meta) == "50"
        assert _validate_and_coerce("50", meta) == "50"
        with pytest.raises(ValueError, match="below minimum"):
            _validate_and_coerce(0, meta)
        with pytest.raises(ValueError, match="above maximum"):
            _validate_and_coerce(101, meta)

    def test_validate_float(self):
        meta = SettingMeta(key="test", type="float", category="hot", description="test", min_val=0.0, max_val=1.0, step=0.1)
        assert _validate_and_coerce(0.5, meta) == "0.5"
        with pytest.raises(ValueError):
            _validate_and_coerce(1.5, meta)

    def test_validate_enum(self):
        meta = SettingMeta(key="test", type="enum", category="hot", description="test", options=("a", "b", "c"))
        assert _validate_and_coerce("a", meta) == "a"
        with pytest.raises(ValueError, match="Invalid option"):
            _validate_and_coerce("d", meta)

    def test_validate_string(self):
        meta = SettingMeta(key="test", type="string", category="hot", description="test")
        assert _validate_and_coerce("hello", meta) == "hello"
        with pytest.raises(ValueError, match="too long"):
            _validate_and_coerce("x" * 256, meta)


# ─── API Endpoint Tests ─────────────────────────────────────────────

class TestConfigAPI:
    def test_get_config(self, client, auth_env):
        client.post("/api/admin/login", json={"username": "admin", "password": "testpass123"})
        resp = client.get("/api/admin/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "general" in data
        assert "ai_providers" in data

    def test_get_section(self, client, auth_env):
        client.post("/api/admin/login", json={"username": "admin", "password": "testpass123"})
        resp = client.get("/api/admin/config/general")
        assert resp.status_code == 200
        data = resp.json()
        assert "crypto_pair" in data["keys"]

    def test_get_unknown_section(self, client, auth_env):
        client.post("/api/admin/login", json={"username": "admin", "password": "testpass123"})
        resp = client.get("/api/admin/config/nonexistent")
        assert resp.status_code == 404

    def test_update_config(self, client, auth_env, writable_config):
        client.post("/api/admin/login", json={"username": "admin", "password": "testpass123"})
        resp = client.patch(
            "/api/admin/config/general/timeframe",
            json={"value": "1d"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["category"] == "cycle"
        assert writable_config.get_value("general", "timeframe") == "1d"

    def test_update_invalid_value(self, client, auth_env):
        client.post("/api/admin/login", json={"username": "admin", "password": "testpass123"})
        resp = client.patch(
            "/api/admin/config/general/timeframe",
            json={"value": "invalid_timeframe"},
        )
        assert resp.status_code == 400

    def test_batch_update(self, client, auth_env, writable_config):
        client.post("/api/admin/login", json={"username": "admin", "password": "testpass123"})
        resp = client.post("/api/admin/config/batch", json={
            "updates": [
                {"section": "general", "key": "timeframe", "value": "1h"},
                {"section": "debug", "key": "logger_debug", "value": "false"},
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2


class TestSystemControlAPI:
    def test_trigger_analysis(self, client, auth_env):
        client.post("/api/admin/login", json={"username": "admin", "password": "testpass123"})
        resp = client.post("/api/admin/system/trigger-analysis")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_toggle_feed(self, client, auth_env):
        client.post("/api/admin/login", json={"username": "admin", "password": "testpass123"})
        resp = client.post("/api/admin/system/toggle-feed")
        assert resp.status_code == 200
        data = resp.json()
        assert "feed_enabled" in data

    def test_system_status(self, client, auth_env):
        client.post("/api/admin/login", json={"username": "admin", "password": "testpass123"})
        resp = client.get("/api/admin/system/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "uptime_seconds" in data
        assert "feed_enabled" in data

    def test_human_input_crud(self, client, auth_env):
        client.post("/api/admin/login", json={"username": "admin", "password": "testpass123"})

        # Set input
        resp = client.post("/api/admin/system/human-input", json={"text": "Focus on RSI"})
        assert resp.status_code == 200
        assert resp.json()["text"] == "Focus on RSI"

        # Get input
        resp = client.get("/api/admin/system/human-input")
        assert resp.status_code == 200
        assert resp.json()["text"] == "Focus on RSI"

        # Clear input
        resp = client.delete("/api/admin/system/human-input")
        assert resp.status_code == 200

        # Verify cleared
        resp = client.get("/api/admin/system/human-input")
        assert resp.json()["text"] == ""


class TestLogAPI:
    def test_recent_logs(self, client, auth_env, log_manager):
        client.post("/api/admin/login", json={"username": "admin", "password": "testpass123"})
        resp = client.get("/api/admin/logs/recent")
        assert resp.status_code == 200
        data = resp.json()
        assert "lines" in data


# ─── LogStreamManager Tests ─────────────────────────────────────────

class TestLogStreamManager:
    def test_handler_subscribe(self, log_manager):
        sid, queue = log_manager.handler.subscribe()
        assert sid is not None
        assert queue is not None
        log_manager.handler.unsubscribe(sid)

    def test_handler_emit(self, log_manager):
        import logging
        sid, queue = log_manager.handler.subscribe()

        record = logging.LogRecord("test", logging.INFO, "", 0, "Hello world", (), None)
        log_manager.handler.emit(record)

        # Should have the line in the queue
        assert not queue.empty()
        line = queue.get_nowait()
        assert "Hello world" in line

        log_manager.handler.unsubscribe(sid)

    def test_recent_logs(self, log_manager):
        import logging
        for i in range(5):
            record = logging.LogRecord("test", logging.INFO, "", 0, f"Line {i}", (), None)
            log_manager.handler.emit(record)

        recent = log_manager.get_recent_logs(count=3)
        assert len(recent) == 3
        assert "Line 4" in recent[-1]

    def test_subscriber_count(self, log_manager):
        assert log_manager.subscriber_count == 0
        sid, _ = log_manager.handler.subscribe()
        assert log_manager.subscriber_count == 1
        log_manager.handler.unsubscribe(sid)
        assert log_manager.subscriber_count == 0


# ─── Integration: Full Dashboard Server ──────────────────────────────

class TestDashboardServerIntegration:
    """Test that DashboardServer correctly initializes admin components."""

    @pytest.mark.skipif(
        not _has_numpy(),
        reason="Full bot deps (numpy) not available in test venv",
    )
    def test_server_has_admin_router(self):
        """Verify server.py creates admin router and middleware."""
        from src.dashboard.server import DashboardServer

        mock_config = MagicMock()
        mock_config.DASHBOARD_ENABLE_CORS = False
        mock_config.DASHBOARD_CORS_ORIGINS = []
        mock_config.ADMIN_USERNAME = "admin"
        mock_config.ADMIN_PASSWORD_HASH = hash_password("test")
        mock_config.ADMIN_SIGNING_KEY = "key"

        server = DashboardServer(
            brain_service=None,
            vector_memory=None,
            analysis_engine=None,
            config=mock_config,
            logger=MagicMock(),
            admin_credentials={
                "username": "admin",
                "password_hash": hash_password("test"),
                "signing_key": "key",
            },
        )

        # Verify admin components exist
        assert hasattr(server, 'writable_config')
        assert hasattr(server, 'log_stream_manager')
        assert server.app.state.admin_router is not None
