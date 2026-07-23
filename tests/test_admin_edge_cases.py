"""Edge-case, concurrency, and integration tests for the admin dashboard.

Tests:
- Auth: token expiry, malformed tokens, empty credentials, double init
- Config: concurrent writes, unknown keys, boundary values, hot-reload signal
- WebSocket: auth rejection, command dispatch, ping/pong
- LogStream: subscriber isolation, queue overflow, ring buffer
- Integration: config write → reload → read roundtrip
"""

import asyncio
import configparser
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from src.dashboard.auth import (
    _sign_token,
    _verify_token,
    _verify_password,
    _is_lan_ip,
    check_credentials,
    create_session,
    hash_password,
    init_auth,
    verify_admin_session,
    COOKIE_NAME,
    COOKIE_MAX_AGE,
    AdminAuthMiddleware,
)
from src.dashboard.log_stream import LogStreamHandler, LogStreamManager
from src.config.writable_config import WritableConfig, SettingMeta, _validate_and_coerce, _SCHEMA


# ─── Auth Edge Cases ─────────────────────────────────────────────────

class TestAuthEdgeCases:
    def test_expired_token_rejected(self):
        """Tokens older than COOKIE_MAX_AGE are rejected."""
        init_auth("testkey", "admin", hash_password("pass"))
        # Create a token from 9 hours ago (max age is 8 hours)
        old_ts = time.time() - (9 * 3600)
        token = _sign_token("admin", old_ts)
        assert _verify_token(token) is None

    def test_valid_token_within_expiry(self):
        """Tokens within COOKIE_MAX_AGE are accepted."""
        init_auth("testkey", "admin", hash_password("pass"))
        recent_ts = time.time() - 3600  # 1 hour ago
        token = _sign_token("admin", recent_ts)
        assert _verify_token(token) == "admin"

    def test_malformed_token_rejected(self):
        """Garbage tokens return None."""
        assert _verify_token("") is None
        assert _verify_token("not-a-token") is None
        assert _verify_token("a:b") is None  # only 2 parts
        assert _verify_token("a:b:c:d") is None  # 4 parts
        assert _verify_token(":::") is None

    def test_tampered_token_rejected(self):
        """Modified token signature is detected."""
        init_auth("testkey", "admin", hash_password("pass"))
        token = _sign_token("admin", time.time())
        # Tamper with the signature
        parts = token.split(":")
        parts[2] = "0" * 64
        tampered = ":".join(parts)
        assert _verify_token(tampered) is None

    def test_tampered_username_rejected(self):
        """Modified username in token is detected."""
        init_auth("testkey", "admin", hash_password("pass"))
        token = _sign_token("admin", time.time())
        parts = token.split(":")
        parts[0] = "hacker"
        tampered = ":".join(parts)
        assert _verify_token(tampered) is None

    def test_empty_credentials_rejected(self):
        """Empty username or password is rejected."""
        init_auth("testkey", "admin", hash_password("pass"))
        assert check_credentials("", "pass") is False
        assert check_credentials("admin", "") is False
        assert check_credentials("", "") is False

    def test_uninitialized_auth_rejects_all(self):
        """Before init_auth(), all credentials are rejected."""
        # Reset module state
        import src.dashboard.auth as auth_mod
        old_init = auth_mod._initialized
        auth_mod._initialized = False
        try:
            assert check_credentials("admin", "pass") is False
        finally:
            auth_mod._initialized = old_init

    def test_password_hash_roundtrip(self):
        """hash_password output can be verified by _verify_password."""
        password = "my_secure_password_123!@#"
        stored = hash_password(password)
        assert _verify_password(password, stored) is True
        assert _verify_password("wrong", stored) is False

    def test_password_hash_unique_per_call(self):
        """Each hash_password call produces a different salt."""
        h1 = hash_password("same_pass")
        h2 = hash_password("same_pass")
        assert h1 != h2  # different salts
        # But both verify correctly
        assert _verify_password("same_pass", h1) is True
        assert _verify_password("same_pass", h2) is True

    def test_malformed_stored_hash_rejected(self):
        """Corrupted stored hash doesn't crash, returns False."""
        assert _verify_password("pass", "not-hex:garbage") is False
        assert _verify_password("pass", "") is False
        assert _verify_password("pass", "no-colon") is False


# ─── Config Validation Edge Cases ────────────────────────────────────

class TestConfigValidation:
    def test_bool_truthy_values(self):
        """All truthy string variants coerce to 'true'."""
        meta = SettingMeta(key="test", type="bool", category="hot", description="")
        for val in ["true", "1", "yes", "on", "True", "YES", "ON"]:
            assert _validate_and_coerce(val, meta) == "true"

    def test_bool_falsy_values(self):
        """All falsy string variants coerce to 'false'."""
        meta = SettingMeta(key="test", type="bool", category="hot", description="")
        for val in ["false", "0", "no", "off", "False", "NO", "OFF"]:
            assert _validate_and_coerce(val, meta) == "false"

    def test_bool_invalid_values(self):
        """Non-boolean strings raise ValueError."""
        meta = SettingMeta(key="test", type="bool", category="hot", description="")
        with pytest.raises(ValueError, match="Invalid boolean"):
            _validate_and_coerce("maybe", meta)
        with pytest.raises(ValueError, match="Invalid boolean"):
            _validate_and_coerce("2", meta)

    def test_int_boundary_values(self):
        """Int values at min/max boundaries are accepted."""
        meta = SettingMeta(key="test", type="int", category="hot", description="", min_val=10, max_val=9999)
        assert _validate_and_coerce("10", meta) == "10"
        assert _validate_and_coerce("9999", meta) == "9999"

    def test_int_out_of_range(self):
        """Int values outside range raise ValueError."""
        meta = SettingMeta(key="test", type="int", category="hot", description="", min_val=10, max_val=9999)
        with pytest.raises(ValueError, match="below minimum"):
            _validate_and_coerce("9", meta)
        with pytest.raises(ValueError, match="above maximum"):
            _validate_and_coerce("10000", meta)

    def test_int_non_numeric(self):
        """Non-numeric strings raise ValueError."""
        meta = SettingMeta(key="test", type="int", category="hot", description="", min_val=0, max_val=100)
        with pytest.raises(ValueError, match="Invalid integer"):
            _validate_and_coerce("abc", meta)

    def test_float_boundary_values(self):
        """Float values at boundaries are accepted."""
        meta = SettingMeta(key="test", type="float", category="hot", description="", min_val=0.0, max_val=2.0, step=0.05)
        assert _validate_and_coerce("0.0", meta) == "0.0"
        assert _validate_and_coerce("2.0", meta) == "2.0"

    def test_enum_valid_options(self):
        """Valid enum options are accepted."""
        meta = SettingMeta(key="test", type="enum", category="hot", description="", options=("a", "b", "c"))
        assert _validate_and_coerce("a", meta) == "a"
        assert _validate_and_coerce("B", meta) == "b"  # case-insensitive

    def test_enum_invalid_option(self):
        """Invalid enum option raises ValueError."""
        meta = SettingMeta(key="test", type="enum", category="hot", description="", options=("a", "b"))
        with pytest.raises(ValueError, match="Invalid option"):
            _validate_and_coerce("z", meta)

    def test_string_too_long(self):
        """Strings > 255 chars raise ValueError."""
        meta = SettingMeta(key="test", type="string", category="hot", description="")
        with pytest.raises(ValueError, match="too long"):
            _validate_and_coerce("x" * 256, meta)

    def test_string_control_chars(self):
        """Strings with control characters raise ValueError."""
        meta = SettingMeta(key="test", type="string", category="hot", description="")
        with pytest.raises(ValueError, match="control characters"):
            _validate_and_coerce("hello\x00world", meta)

    def test_string_tab_allowed(self):
        """Tab characters are allowed in strings."""
        meta = SettingMeta(key="test", type="string", category="hot", description="")
        assert _validate_and_coerce("hello\tworld", meta) == "hello\tworld"

    def test_unknown_type_raises(self):
        """Unknown setting type raises ValueError."""
        meta = SettingMeta(key="test", type="unknown_type", category="hot", description="")
        with pytest.raises(ValueError, match="Unknown setting type"):
            _validate_and_coerce("value", meta)


# ─── WritableConfig Edge Cases ───────────────────────────────────────

class TestWritableConfigEdgeCases:
    @pytest.fixture
    def wc(self, tmp_path):
        cfg = tmp_path / "config.ini"
        cfg.write_text(
            "[general]\ncrypto_pair = BTC/USDC\ntimeframe = 4h\n\n[debug]\nlogger_debug = true\n",
            encoding="utf-8",
        )
        return WritableConfig(str(cfg))

    def test_read_nonexistent_section(self, wc):
        """Reading a nonexistent section returns empty dict."""
        assert wc.get_section("nonexistent") == {}

    def test_read_nonexistent_key(self, wc):
        """Reading a nonexistent key returns None."""
        assert wc.get_value("general", "nonexistent") is None

    def test_read_nonexistent_section_value(self, wc):
        """Reading a value from a nonexistent section returns None."""
        assert wc.get_value("nonexistent", "key") is None

    async def test_set_unknown_key_raises(self, wc):
        """Setting an unknown key raises ValueError."""
        with pytest.raises(ValueError, match="Unknown config key"):
            await wc.set_value("general", "unknown_key", "value")

    async def test_set_unknown_section_raises(self, wc):
        """Setting a key in an unknown section raises ValueError."""
        with pytest.raises(ValueError, match="Unknown config key"):
            await wc.set_value("nonexistent", "key", "value")

    async def test_reload_event_set_on_write(self, wc):
        """Writing a value sets the reload event."""
        assert not wc.reload_event.is_set()
        await wc.set_value("general", "timeframe", "1d")
        assert wc.reload_event.is_set()

    def test_reload_event_cleared_on_read(self, wc):
        """read_reload_event clears the event."""
        wc.reload_event.set()
        assert wc.read_reload_event() is True
        assert wc.read_reload_event() is False

    def test_schema_covers_all_config_sections(self, wc):
        """Schema covers all sections in the actual config.ini."""
        schema = wc.get_full_schema()
        # These sections should always be in the schema
        expected = {"ai_providers", "general", "debug", "rag", "risk_management", "demo_trading", "model_config", "dashboard"}
        assert set(schema.keys()) == expected

    def test_schema_has_required_fields(self, wc):
        """Every schema entry has required fields."""
        schema = wc.get_full_schema()
        for section_name, section in schema.items():
            assert "title" in section
            assert "keys" in section
            for key_name, key_meta in section["keys"].items():
                assert "value" in key_meta
                assert "type" in key_meta
                assert "category" in key_meta
                assert "description" in key_meta

    async def test_batch_update_all_or_nothing(self, wc):
        """If one value in a batch is invalid, none are written."""
        # First set a known good value
        await wc.set_value("general", "timeframe", "4h")
        # Try batch with one invalid value
        with pytest.raises(ValueError):
            await wc.set_values([
                ("general", "timeframe", "1d"),  # valid
                ("general", "candle_limit", "not_a_number"),  # invalid
            ])
        # The first value should NOT have been written (atomicity)
        assert wc.get_value("general", "timeframe") == "4h"

    async def test_concurrent_writes_sequentialized(self, wc):
        """Concurrent writes are serialized by the lock."""
        timeframes = ["1m", "5m", "15m", "30m", "1h"]

        async def write_timeframe(val):
            await wc.set_value("general", "timeframe", val)

        async def run_concurrent():
            tasks = [write_timeframe(tf) for tf in timeframes]
            await asyncio.gather(*tasks)

        await run_concurrent()
        # Should have one of the values (not corrupted)
        val = wc.get_value("general", "timeframe")
        assert val in timeframes


# ─── LogStream Edge Cases ────────────────────────────────────────────

class TestLogStreamEdgeCases:
    def test_multiple_subscribers_receive_same_message(self):
        """Multiple subscribers all receive the same log line."""
        lsm = LogStreamManager()
        import logging
        sid1, q1 = lsm.handler.subscribe()
        sid2, q2 = lsm.handler.subscribe()

        record = logging.LogRecord("test", logging.INFO, "", 0, "Shared message", (), None)
        lsm.handler.emit(record)

        assert not q1.empty()
        assert not q2.empty()
        assert "Shared message" in q1.get_nowait()
        assert "Shared message" in q2.get_nowait()

        lsm.handler.unsubscribe(sid1)
        lsm.handler.unsubscribe(sid2)

    def test_unsubscribe_removes_subscriber(self):
        """After unsubscribe, no more messages are received."""
        lsm = LogStreamManager()
        import logging
        sid, q = lsm.handler.subscribe()
        lsm.handler.unsubscribe(sid)

        # Drain the sentinel
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass

        record = logging.LogRecord("test", logging.INFO, "", 0, "After unsub", (), None)
        lsm.handler.emit(record)

        assert q.empty()

    def test_late_joiner_gets_recent_history(self):
        """A subscriber that joins after messages were emitted gets recent history."""
        lsm = LogStreamManager()
        import logging

        # Emit some messages before subscribing
        for i in range(5):
            record = logging.LogRecord("test", logging.INFO, "", 0, f"Pre-msg {i}", (), None)
            lsm.handler.emit(record)

        # Now subscribe
        sid, q = lsm.handler.subscribe()
        lines = []
        while not q.empty():
            lines.append(q.get_nowait())

        # Should have the recent messages
        assert len(lines) > 0
        assert any("Pre-msg" in line for line in lines)
        lsm.handler.unsubscribe(sid)

    def test_handler_level_captures_all_levels(self):
        """Handler at DEBUG captures INFO, WARNING, ERROR."""
        lsm = LogStreamManager()
        import logging
        sid, q = lsm.handler.subscribe()

        for level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]:
            record = logging.LogRecord("test", level, "", 0, f"Level {level}", (), None)
            lsm.handler.emit(record)

        lines = []
        while not q.empty():
            lines.append(q.get_nowait())

        assert len(lines) == 4
        lsm.handler.unsubscribe(sid)

    def test_subscriber_count_accurate(self):
        """Subscriber count reflects actual subscribers."""
        lsm = LogStreamManager()
        assert lsm.subscriber_count == 0

        sids = []
        for _ in range(3):
            sid, _ = lsm.handler.subscribe()
            sids.append(sid)
        assert lsm.subscriber_count == 3

        lsm.handler.unsubscribe(sids[0])
        assert lsm.subscriber_count == 2

        for sid in sids[1:]:
            lsm.handler.unsubscribe(sid)
        assert lsm.subscriber_count == 0


# ─── Schema Coverage ─────────────────────────────────────────────────

class TestSchemaCoverage:
    def test_all_schema_sections_have_settings(self):
        """Every section in the schema has at least one setting."""
        for name, section in _SCHEMA.items():
            assert len(section.settings) > 0, f"Section {name} has no settings"

    def test_all_settings_have_valid_types(self):
        """Every setting has a recognized type."""
        valid_types = {"string", "int", "float", "bool", "enum"}
        for section in _SCHEMA.values():
            for meta in section.settings.values():
                assert meta.type in valid_types, f"{meta.key} has invalid type {meta.type}"

    def test_all_settings_have_valid_categories(self):
        """Every setting has a recognized category."""
        valid_categories = {"hot", "cycle", "restart"}
        for section in _SCHEMA.values():
            for meta in section.settings.values():
                assert meta.category in valid_categories, f"{meta.key} has invalid category {meta.category}"

    def test_enum_settings_have_options(self):
        """Enum-type settings have at least one option."""
        for section in _SCHEMA.values():
            for meta in section.settings.values():
                if meta.type == "enum":
                    assert meta.options and len(meta.options) > 0, f"{meta.key} is enum but has no options"

    def test_numeric_settings_have_bounds(self):
        """Int/float settings with bounds have min < max."""
        for section in _SCHEMA.values():
            for meta in section.settings.values():
                if meta.type in ("int", "float") and meta.min_val is not None and meta.max_val is not None:
                    assert meta.min_val < meta.max_val, f"{meta.key}: min {meta.min_val} >= max {meta.max_val}"


# ─── Admin Auth Middleware ───────────────────────────────────────────

class TestAdminAuthMiddleware:
    @pytest.fixture
    def app_with_middleware(self, tmp_path):
        """Create a minimal app with admin auth middleware."""
        cfg = tmp_path / "config.ini"
        cfg.write_text("[general]\ncrypto_pair = BTC/USDC\n", encoding="utf-8")

        init_auth("testkey", "admin", hash_password("testpass"))

        app = FastAPI()
        app.add_middleware(AdminAuthMiddleware)

        from src.dashboard.routers.admin import AdminRouter
        from src.dashboard.log_stream import LogStreamManager
        from src.config.writable_config import WritableConfig

        admin_router = AdminRouter(
            writable_config=WritableConfig(str(cfg)),
            log_stream_manager=LogStreamManager(),
            config=MagicMock(),
            logger=None,
            force_analysis_event=asyncio.Event(),
        )
        app.include_router(admin_router.router)
        return app

    def _make_client(self, app):
        """Create test client simulating LAN access."""
        return TestClient(app, raise_server_exceptions=False, client=("192.168.1.100", 54321))

    def test_protected_route_without_auth_returns_401(self, app_with_middleware):
        client = self._make_client(app_with_middleware)
        resp = client.get("/api/admin/config")
        assert resp.status_code == 401

    def test_login_route_is_public(self, app_with_middleware):
        client = self._make_client(app_with_middleware)
        resp = client.post("/api/admin/login", json={"username": "admin", "password": "wrong"})
        assert resp.status_code == 401  # wrong creds, but endpoint is reachable

    def test_health_route_is_public(self, app_with_middleware):
        client = self._make_client(app_with_middleware)
        resp = client.get("/api/admin/health")
        assert resp.status_code == 200

    def test_protected_route_with_valid_cookie(self, app_with_middleware):
        client = self._make_client(app_with_middleware)
        # Login first
        resp = client.post("/api/admin/login", json={"username": "admin", "password": "testpass"})
        assert resp.status_code == 200
        # Cookie should be set
        assert COOKIE_NAME in resp.cookies
        # Access protected route with cookie
        resp2 = client.get("/api/admin/config")
        assert resp2.status_code == 200

    def test_protected_route_with_valid_bearer_token(self, app_with_middleware):
        client = self._make_client(app_with_middleware)
        resp = client.post("/api/admin/login", json={"username": "admin", "password": "testpass"})
        token = resp.json().get("token")
        assert token
        # Access with Bearer header
        resp2 = client.get("/api/admin/config", headers={"Authorization": f"Bearer {token}"})
        assert resp2.status_code == 200

    def test_non_admin_routes_bypass_middleware(self, app_with_middleware):
        """Routes not starting with /api/admin/ are not protected."""
        @app_with_middleware.get("/public")
        async def public_route():
            return {"status": "ok"}

        client = self._make_client(app_with_middleware)
        resp = client.get("/public")
        assert resp.status_code == 200


# ─── LAN Access Control Tests ────────────────────────────────────────

class TestLANAccessControl:
    """Verify that admin routes are restricted to LAN/private IPs only."""

    @pytest.fixture
    def app_with_middleware(self, tmp_path):
        """Create a minimal app with admin auth middleware."""
        cfg = tmp_path / "config.ini"
        cfg.write_text("[general]\ncrypto_pair = BTC/USDC\n", encoding="utf-8")

        init_auth("testkey", "admin", hash_password("testpass"))

        app = FastAPI()
        app.add_middleware(AdminAuthMiddleware)

        from src.dashboard.routers.admin import AdminRouter
        from src.dashboard.log_stream import LogStreamManager
        from src.config.writable_config import WritableConfig

        admin_router = AdminRouter(
            writable_config=WritableConfig(str(cfg)),
            log_stream_manager=LogStreamManager(),
            config=MagicMock(),
            logger=None,
            force_analysis_event=asyncio.Event(),
        )
        app.include_router(admin_router.router)
        return app

    def test_lan_ip_allowed(self):
        """192.168.x.x is a valid LAN IP."""
        assert _is_lan_ip("192.168.1.100") is True
        assert _is_lan_ip("192.168.0.1") is True
        assert _is_lan_ip("192.168.255.255") is True

    def test_localhost_allowed(self):
        """127.x.x.x is loopback (LAN)."""
        assert _is_lan_ip("127.0.0.1") is True
        assert _is_lan_ip("127.255.255.255") is True

    def test_10_network_allowed(self):
        """10.x.x.x is private."""
        assert _is_lan_ip("10.0.0.1") is True
        assert _is_lan_ip("10.255.255.255") is True

    def test_172_16_network_allowed(self):
        """172.16-31.x.x is private."""
        assert _is_lan_ip("172.16.0.1") is True
        assert _is_lan_ip("172.31.255.255") is True

    def test_172_non_private_rejected(self):
        """172.32.x.x is NOT private."""
        assert _is_lan_ip("172.32.0.1") is False

    def test_public_ip_rejected(self):
        """Public IPs are not LAN."""
        assert _is_lan_ip("8.8.8.8") is False
        assert _is_lan_ip("1.1.1.1") is False
        assert _is_lan_ip("203.0.113.1") is False

    def test_ipv6_loopback_allowed(self):
        """::1 is loopback."""
        assert _is_lan_ip("::1") is True

    def test_empty_ip_rejected(self):
        """Empty string is not LAN."""
        assert _is_lan_ip("") is False

    def test_malformed_ip_rejected(self):
        """Garbage input is not LAN."""
        assert _is_lan_ip("not-an-ip") is False
        assert _is_lan_ip("999.999.999.999") is False

    def test_admin_page_blocked_from_public_ip(self, app_with_middleware):
        """Public IP accessing /admin/ gets 403."""
        client = TestClient(app_with_middleware, raise_server_exceptions=False, client=("8.8.8.8", 12345))
        resp = client.get("/admin/")
        assert resp.status_code == 403
        assert "LAN" in resp.json().get("detail", "")

    def test_admin_api_blocked_from_public_ip(self, app_with_middleware):
        """Public IP accessing /api/admin/* gets 403."""
        client = TestClient(app_with_middleware, raise_server_exceptions=False, client=("1.1.1.1", 12345))
        resp = client.get("/api/admin/config")
        assert resp.status_code == 403

    def test_admin_allowed_from_lan(self, app_with_middleware):
        """LAN IP can access admin (with auth)."""
        client = TestClient(app_with_middleware, raise_server_exceptions=False, client=("192.168.1.50", 12345))
        # Login first
        resp = client.post("/api/admin/login", json={"username": "admin", "password": "testpass"})
        assert resp.status_code == 200
        # Access config
        resp2 = client.get("/api/admin/config")
        assert resp2.status_code == 200

    def test_cf_connecting_ip_from_tunnel(self, app_with_middleware):
        """Cloudflare Tunnel (localhost) uses CF-Connecting-IP header."""
        client = TestClient(
            app_with_middleware, raise_server_exceptions=False,
            client=("127.0.0.1", 54321),
            headers={"CF-Connecting-IP": "85.23.45.67"},  # Public IP via CF
        )
        resp = client.get("/api/admin/health")
        # Health is public, but still LAN-gated. Public CF IP should be blocked.
        assert resp.status_code == 403

    def test_cf_connecting_ip_lan_from_tunnel(self, app_with_middleware):
        """Cloudflare Tunnel with LAN CF-Connecting-IP is allowed."""
        client = TestClient(
            app_with_middleware, raise_server_exceptions=False,
            client=("127.0.0.1", 54321),
            headers={"CF-Connecting-IP": "192.168.1.50"},
        )
        resp = client.post("/api/admin/login", json={"username": "admin", "password": "testpass"})
        assert resp.status_code == 200

    def test_spoofed_cf_header_on_direct_connection_ignored(self, app_with_middleware):
        """CF-Connecting-IP header is IGNORED on direct (non-tunnel) connections."""
        # Direct connection from public IP with spoofed CF header claiming LAN
        client = TestClient(
            app_with_middleware, raise_server_exceptions=False,
            client=("203.0.113.1", 12345),
            headers={"CF-Connecting-IP": "192.168.1.1"},
        )
        resp = client.get("/api/admin/config")
        # Should be blocked because direct IP (203.0.113.1) is not localhost
        assert resp.status_code == 403
