"""Dense domain tests for the FastAPI admin dashboard.

Covers the admin REST + WebSocket API (src/dashboard/routers/admin.py), the HMAC
cookie auth and LAN middleware (src/dashboard/auth.py), the writable config
schema/coercion layer (src/config/writable_config.py) and log streaming
(src/dashboard/log_stream.py).
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.config.writable_config import (
    _SCHEMA,
    SettingMeta,
    WritableConfig,
    _validate_and_coerce,
)
from src.dashboard import auth as auth_module
from src.dashboard.auth import (
    COOKIE_NAME,
    AdminAuthMiddleware,
    _is_lan_ip,
    _sign_token,
    _verify_password,
    _verify_token,
    check_credentials,
    hash_password,
    init_auth,
    reset_login_rate_limit,
)
from src.dashboard.log_stream import LogStreamHandler, LogStreamManager
from src.dashboard.routers.admin import AdminRouter

SIGNING_KEY = "test-signing-key-secret"
ADMIN_HASH = hash_password("testpass123")
LAN_CLIENT = ("192.168.1.100", 54321)
RATE_LIMITED_IPS = ("192.168.1.100", "192.168.11.7")

CONFIG_INI = """[general]
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


class RecordingSocket:
    """Websocket double that records every send_json payload."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)


class FailingSocket:
    """Websocket double that dies with a disconnect on the first send."""

    def __init__(self) -> None:
        self.attempts = 0

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.attempts += 1
        raise WebSocketDisconnect(1006)


def bot_config() -> MagicMock:
    """Bot config double carrying the admin/dashboard attributes the routers read."""
    config = MagicMock()
    config.ADMIN_USERNAME = "admin"
    config.ADMIN_PASSWORD_HASH = ADMIN_HASH
    config.ADMIN_SIGNING_KEY = SIGNING_KEY
    config.DASHBOARD_ENABLE_CORS = False
    config.DASHBOARD_CORS_ORIGINS = []
    return config


def build_admin_app(
    writable_config: WritableConfig,
    log_manager: LogStreamManager,
    dashboard_state: Any = None,
    force_analysis_event: asyncio.Event | None = None,
    logger: Any = None,
) -> FastAPI:
    """Assemble the FastAPI app around one AdminRouter, as src/dashboard/server.py does."""
    app = FastAPI()
    app.include_router(
        AdminRouter(
            writable_config=writable_config,
            log_stream_manager=log_manager,
            config=bot_config(),
            logger=MagicMock() if logger is None else logger,
            dashboard_state=dashboard_state,
            force_analysis_event=force_analysis_event,
        ).router
    )
    app.add_middleware(AdminAuthMiddleware)
    return app


def lan_client(app: FastAPI, address: tuple[str, int] = LAN_CLIENT, **kwargs: Any) -> TestClient:
    """TestClient simulating a LAN connection, with server exceptions surfaced as 500."""
    return TestClient(app, raise_server_exceptions=False, client=address, **kwargs)


@pytest.fixture(autouse=True)
def auth_state():
    """Keep the module-global signing key and login-attempt table deterministic."""
    init_auth(SIGNING_KEY, "admin", ADMIN_HASH)
    for ip in RATE_LIMITED_IPS:
        reset_login_rate_limit(ip)
    yield
    for ip in RATE_LIMITED_IPS:
        reset_login_rate_limit(ip)


@pytest.fixture
def tmp_config(tmp_path) -> Path:
    path = tmp_path / "config.ini"
    path.write_text(CONFIG_INI, encoding="utf-8")
    return path


@pytest.fixture
def writable_config(tmp_config) -> WritableConfig:
    return WritableConfig(tmp_config)


@pytest.fixture
def log_manager() -> LogStreamManager:
    return LogStreamManager(max_queue_size=100)


@pytest.fixture
def dashboard_state() -> MagicMock:
    state = MagicMock()
    state.broadcast = AsyncMock()
    return state


@pytest.fixture
def force_analysis_event() -> asyncio.Event:
    return asyncio.Event()


@pytest.fixture
def router_logger() -> MagicMock:
    return MagicMock()


@pytest.fixture
def admin_app(writable_config, log_manager, dashboard_state, force_analysis_event, router_logger) -> FastAPI:
    return build_admin_app(
        writable_config,
        log_manager,
        dashboard_state=dashboard_state,
        force_analysis_event=force_analysis_event,
        logger=router_logger,
    )


@pytest.fixture
def client(admin_app) -> TestClient:
    return lan_client(admin_app)


@pytest.fixture
def logged_in_client(client) -> TestClient:
    login = client.post("/api/admin/login", json={"username": "admin", "password": "testpass123"})
    assert login.status_code == 200
    return client


def test_password_hash_format_roundtrip_and_corrupt_records():
    """hash_password emits salt:hash, verifies both ways and rejects unparsable records."""
    stored = hash_password("my_secure_password_123!@#")
    salt_hex, hash_hex = stored.split(":")
    assert len(salt_hex) == 32
    assert len(hash_hex) == 64
    assert set(salt_hex + hash_hex) <= set("0123456789abcdef")
    assert _verify_password("my_secure_password_123!@#", stored) is True
    assert _verify_password("wrong", stored) is False

    repeated = hash_password("my_secure_password_123!@#")
    assert repeated != stored
    assert _verify_password("my_secure_password_123!@#", repeated) is True

    for corrupt in ("not-hex:garbage", "", "no-colon"):
        assert _verify_password("pass", corrupt) is False


@pytest.mark.parametrize(
    ("username", "password", "expected"),
    [
        ("admin", "testpass123", True),
        ("admin", "wrongpassword", False),
        ("wronguser", "testpass123", False),
        ("", "testpass123", False),
        ("admin", "", False),
        ("", "", False),
    ],
    ids=["valid", "bad-password", "bad-user", "empty-user", "empty-password", "both-empty"],
)
def test_check_credentials_matrix(username, password, expected):
    assert check_credentials(username, password) is expected


def test_check_credentials_rejects_everything_before_init(monkeypatch):
    """Without init_auth() every credential pair is refused."""
    monkeypatch.setattr(auth_module, "_initialized", False)
    assert check_credentials("admin", "testpass123") is False


def test_session_token_signature_age_and_tampering():
    """Tokens are username:timestamp:hmac, bound to the 8h window and the payload."""
    fresh = _sign_token("admin", time.time())
    assert _verify_token(fresh) == "admin"
    assert _verify_token(_sign_token("admin", time.time() - 3600)) == "admin"
    assert _verify_token(_sign_token("admin", time.time() - 100000)) is None
    assert _verify_token(_sign_token("admin", time.time() - 9 * 3600)) is None

    username, stamp, signature = fresh.split(":")
    assert _verify_token(f"{username}:{stamp}:tampered") is None
    assert _verify_token(f"{username}:{stamp}:{'0' * 64}") is None
    assert _verify_token(f"hacker:{stamp}:{signature}") is None
    assert _verify_token(f"{username}:{int(stamp) - 5000}:{signature}") is None


@pytest.mark.parametrize("token", ["", "invalid", "not-a-token", "a:b", "a:b:c:d", ":::"])
def test_verify_token_rejects_malformed_payloads(token):
    assert _verify_token(token) is None


def test_tokens_are_invalidated_by_signing_key_rotation():
    """A token signed under a previous signing key stops verifying after init_auth()."""
    rotated = _sign_token("admin", time.time())
    assert _verify_token(rotated) == "admin"

    init_auth("rotated-signing-key", "admin", ADMIN_HASH)
    assert _verify_token(rotated) is None
    assert _verify_token(_sign_token("admin", time.time())) == "admin"


def test_middleware_requires_session_and_leaves_public_paths_open(client):
    """Protected API needs a cookie; login/health stay public; logout drops the session."""
    assert client.get("/api/admin/config").status_code == 401

    health = client.get("/api/admin/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert type(health.json()["uptime"]) is int

    bad_login = client.post("/api/admin/login", json={"username": "admin", "password": "wrong"})
    assert bad_login.status_code == 401
    assert bad_login.json() == {"error": "Invalid credentials"}
    assert client.post("/api/admin/login", json={"username": "admin"}).status_code == 422

    login = client.post("/api/admin/login", json={"username": "admin", "password": "testpass123"})
    assert login.status_code == 200
    assert login.json()["status"] == "ok"
    assert client.get("/api/admin/config").status_code == 200

    assert client.post("/api/admin/logout").status_code == 200
    assert client.get("/api/admin/config").status_code == 401


def test_ws_token_endpoint_requires_session_and_signs_for_that_user(client):
    assert client.get("/api/admin/ws-token").status_code == 401

    client.post("/api/admin/login", json={"username": "admin", "password": "testpass123"})
    response = client.get("/api/admin/ws-token")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["username"] == "admin"
    assert _verify_token(payload["token"]) == "admin"


def test_middleware_rejects_missing_invalid_and_expired_credentials(client):
    """Cookie, bearer header and query token are each validated, and expiry is enforced."""
    valid = _sign_token("admin", time.time())
    expired = _sign_token("admin", time.time() - 9 * 3600)

    assert client.get("/api/admin/config").status_code == 401

    client.cookies.set(COOKIE_NAME, "garbage")
    assert client.get("/api/admin/config").status_code == 401
    client.cookies.set(COOKIE_NAME, expired)
    assert client.get("/api/admin/config").status_code == 401
    client.cookies.set(COOKIE_NAME, valid)
    assert client.get("/api/admin/config").status_code == 200
    client.cookies.clear()

    assert client.get("/api/admin/config", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/api/admin/config", headers={"Authorization": "Basic abc"}).status_code == 401
    assert client.get("/api/admin/config", headers={"Authorization": valid}).status_code == 401
    assert client.get(f"/api/admin/config?token={expired}").status_code == 401
    assert client.get(f"/api/admin/config?token={valid}").status_code == 200


def test_middleware_accepts_valid_cookie_and_bearer_token(client):
    login = client.post("/api/admin/login", json={"username": "admin", "password": "testpass123"})
    assert login.status_code == 200
    assert COOKIE_NAME in login.cookies
    assert client.get("/api/admin/config").status_code == 200

    client.cookies.clear()
    token = login.json()["token"]
    authorized = client.get("/api/admin/config", headers={"Authorization": f"Bearer {token}"})
    assert authorized.status_code == 200


def test_middleware_bypasses_non_admin_paths(admin_app):
    @admin_app.get("/public")
    async def public_route():
        return {"status": "ok"}

    response = lan_client(admin_app).get("/public")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_lan_gate_blocks_public_clients_and_spoofed_cf_header(admin_app):
    """Public peers are refused everywhere, and CF-Connecting-IP cannot spoof a direct peer."""
    page = lan_client(admin_app, ("8.8.8.8", 12345)).get("/admin/")
    assert page.status_code == 403
    assert page.json() == {"error": "Forbidden", "detail": "Admin access restricted to LAN"}

    api = lan_client(admin_app, ("1.1.1.1", 12345)).get("/api/admin/config")
    assert api.status_code == 403

    spoofed = lan_client(
        admin_app, ("203.0.113.1", 12345), headers={"CF-Connecting-IP": "192.168.1.1"}
    ).get("/api/admin/config")
    assert spoofed.status_code == 403


def test_lan_gate_trusts_cf_header_only_from_localhost(admin_app):
    """A tunnel (localhost peer) is judged by CF-Connecting-IP; a public claim is refused."""
    tunnel_public = lan_client(
        admin_app, ("127.0.0.1", 54321), headers={"CF-Connecting-IP": "85.23.45.67"}
    ).get("/api/admin/health")
    assert tunnel_public.status_code == 403

    tunnel_client = lan_client(
        admin_app, ("127.0.0.1", 54321), headers={"CF-Connecting-IP": "192.168.1.50"}
    )
    login = tunnel_client.post("/api/admin/login", json={"username": "admin", "password": "testpass123"})
    assert login.status_code == 200


def test_lan_gate_allows_lan_client_with_session(admin_app):
    client = lan_client(admin_app, ("192.168.1.50", 12345))
    assert client.post("/api/admin/login", json={"username": "admin", "password": "testpass123"}).status_code == 200
    assert client.get("/api/admin/config").status_code == 200


@pytest.mark.parametrize(
    ("ip", "expected"),
    [
        ("192.168.1.100", True),
        ("192.168.0.1", True),
        ("192.168.255.255", True),
        ("127.0.0.1", True),
        ("127.255.255.255", True),
        ("10.0.0.1", True),
        ("10.255.255.255", True),
        ("172.16.0.1", True),
        ("172.31.255.255", True),
        ("::1", True),
        ("172.32.0.1", False),
        ("8.8.8.8", False),
        ("1.1.1.1", False),
        ("203.0.113.1", False),
        ("", False),
        ("not-an-ip", False),
        ("999.999.999.999", False),
    ],
    ids=["lan", "lan-low", "lan-high", "loopback", "loopback-high", "ten", "ten-high",
         "172-low", "172-high", "ipv6-loopback", "172-public", "google-dns", "cloudflare",
         "documentation", "empty", "garbage", "out-of-range"],
)
def test_is_lan_ip_classifies_private_and_public_ranges(ip, expected):
    assert _is_lan_ip(ip) is expected


def test_config_schema_exposes_every_section_with_live_values(writable_config):
    schema = writable_config.get_full_schema()
    assert set(schema) == {
        "ai_providers",
        "general",
        "debug",
        "rag",
        "risk_management",
        "demo_trading",
        "model_config",
        "dashboard",
    }
    assert schema["general"]["title"] == "General"

    general = writable_config.get_section_schema("general")
    assert "crypto_pair" in general["keys"]
    assert general["keys"]["crypto_pair"]["value"] == "BTC/USDC"
    assert writable_config.get_section_schema("nonexistent") is None


def test_schema_entries_declare_typed_metadata(writable_config):
    """Every section renders keys with value/type/category/description, schema-wide valid."""
    for section_name, section in writable_config.get_full_schema().items():
        assert section["title"]
        assert section["keys"], section_name
        for key, entry in section["keys"].items():
            assert {"value", "type", "category", "description"} <= set(entry), key

    for section_name, section in _SCHEMA.items():
        assert section.settings, section_name
        for key, meta in section.settings.items():
            assert meta.type in {"string", "int", "float", "bool", "enum"}, key
            assert meta.category in {"hot", "cycle", "restart"}, key
            assert meta.description, key
            if meta.type == "enum":
                assert meta.options, key
            if meta.type in ("int", "float") and meta.min_val is not None and meta.max_val is not None:
                assert meta.min_val < meta.max_val, key


async def test_set_value_returns_category_and_persists_to_disk(writable_config, tmp_config):
    assert await writable_config.set_value("debug", "logger_debug", "false") == "hot"
    assert writable_config.get_value("debug", "logger_debug") == "false"
    assert await writable_config.set_value("general", "timeframe", "1h") == "cycle"
    assert writable_config.get_value("general", "timeframe") == "1h"
    assert await writable_config.set_value("general", "crypto_pair", "ETH/USDC") == "cycle"
    assert WritableConfig(tmp_config).get_value("general", "crypto_pair") == "ETH/USDC"


async def test_set_value_rejects_bad_enum_unknown_key_and_unknown_section(writable_config):
    with pytest.raises(ValueError, match="Invalid option"):
        await writable_config.set_value("general", "timeframe", "invalid")
    with pytest.raises(ValueError, match="Unknown config key"):
        await writable_config.set_value("general", "nonexistent", "value")
    with pytest.raises(ValueError, match="Unknown config key"):
        await writable_config.set_value("nonexistent", "key", "value")
    assert writable_config.get_value("general", "timeframe") == "4h"


async def test_batch_update_reports_categories_and_writes_nothing_on_error(writable_config):
    results = await writable_config.set_values(
        [
            ("general", "timeframe", "1d"),
            ("debug", "logger_debug", "false"),
        ]
    )
    assert len(results) == 2
    assert results[0] == {"section": "general", "key": "timeframe", "category": "cycle"}
    assert results[1] == {"section": "debug", "key": "logger_debug", "category": "hot"}

    await writable_config.set_value("general", "timeframe", "4h")
    with pytest.raises(ValueError, match="Invalid integer"):
        await writable_config.set_values(
            [
                ("general", "timeframe", "1d"),
                ("general", "candle_limit", "not_a_number"),
            ]
        )
    assert writable_config.get_value("general", "timeframe") == "4h"
    assert writable_config.get_value("general", "candle_limit") == "999"


async def test_reload_from_disk_picks_up_external_edits(writable_config, tmp_config):
    tmp_config.write_text("[general]\ncrypto_pair = SOL/USDC\n", encoding="utf-8")
    await writable_config.reload_from_disk()
    assert writable_config.get_value("general", "crypto_pair") == "SOL/USDC"


def test_get_value_returns_none_for_unknown_key_and_section(writable_config):
    assert writable_config.get_value("general", "nonexistent") is None
    assert writable_config.get_value("nonexistent", "key") is None


async def test_concurrent_writes_are_serialized_by_the_lock(writable_config, tmp_config):
    timeframes = ["1m", "5m", "15m", "30m", "1h"]
    await asyncio.gather(
        *(writable_config.set_value("general", "timeframe", value) for value in timeframes)
    )
    value = writable_config.get_value("general", "timeframe")
    assert value in timeframes
    assert WritableConfig(tmp_config).get_value("general", "timeframe") == value


def test_bool_coercion_normalises_truthy_and_falsy_variants():
    meta = SettingMeta(key="test", type="bool", category="hot", description="test")
    for raw in ("true", "1", "yes", "on", "True", "YES", "ON", 1, True):
        assert _validate_and_coerce(raw, meta) == "true"
    for raw in ("false", "0", "no", "off", "False", "NO", "OFF", 0):
        assert _validate_and_coerce(raw, meta) == "false"


def test_bool_coercion_rejects_non_boolean_text():
    meta = SettingMeta(key="test", type="bool", category="hot", description="test")
    for raw in ("maybe", "2"):
        with pytest.raises(ValueError, match="Invalid boolean"):
            _validate_and_coerce(raw, meta)


def test_int_coercion_bounds_and_type_errors():
    bounded = SettingMeta(key="test", type="int", category="hot", description="test", min_val=1, max_val=100)
    assert _validate_and_coerce(50, bounded) == "50"
    assert _validate_and_coerce("50", bounded) == "50"
    assert _validate_and_coerce("1", bounded) == "1"
    assert _validate_and_coerce("100", bounded) == "100"
    assert type(_validate_and_coerce("50", bounded)) is str
    with pytest.raises(ValueError, match="below minimum"):
        _validate_and_coerce(0, bounded)
    with pytest.raises(ValueError, match="above maximum"):
        _validate_and_coerce(101, bounded)

    wide = SettingMeta(key="test", type="int", category="hot", description="test", min_val=10, max_val=9999)
    assert _validate_and_coerce("10", wide) == "10"
    assert _validate_and_coerce("9999", wide) == "9999"
    with pytest.raises(ValueError, match="below minimum"):
        _validate_and_coerce("9", wide)
    with pytest.raises(ValueError, match="above maximum"):
        _validate_and_coerce("10000", wide)
    with pytest.raises(ValueError, match="Invalid integer"):
        _validate_and_coerce("abc", SettingMeta(key="test", type="int", category="hot", description="test"))


def test_float_coercion_bounds_and_out_of_range_values():
    meta = SettingMeta(
        key="test", type="float", category="hot", description="test", min_val=0.0, max_val=1.0, step=0.1
    )
    assert _validate_and_coerce(0.5, meta) == "0.5"
    with pytest.raises(ValueError, match="below minimum"):
        _validate_and_coerce(-0.1, meta)
    with pytest.raises(ValueError, match="above maximum"):
        _validate_and_coerce(1.5, meta)

    wide = SettingMeta(
        key="test", type="float", category="hot", description="test", min_val=0.0, max_val=2.0, step=0.05
    )
    assert _validate_and_coerce("0.0", wide) == "0.0"
    assert _validate_and_coerce("2.0", wide) == "2.0"


def test_float_coercion_lets_nan_through_bounds_but_catches_infinity():
    """NaN compares false against both bounds so it is stored verbatim; +/-inf is refused."""
    meta = SettingMeta(key="test", type="float", category="hot", description="test", min_val=0.0, max_val=2.0)
    assert _validate_and_coerce("nan", meta) == "nan"
    assert _validate_and_coerce("NaN", meta) == "nan"
    with pytest.raises(ValueError, match="below minimum"):
        _validate_and_coerce("-inf", meta)
    with pytest.raises(ValueError, match="above maximum"):
        _validate_and_coerce("inf", meta)


def test_enum_coercion_lowercases_and_rejects_unknown_options():
    meta = SettingMeta(key="test", type="enum", category="hot", description="test", options=("a", "b", "c"))
    assert _validate_and_coerce("a", meta) == "a"
    assert _validate_and_coerce("B", meta) == "b"
    with pytest.raises(ValueError, match="Invalid option"):
        _validate_and_coerce("d", meta)
    with pytest.raises(ValueError, match="Invalid option"):
        _validate_and_coerce("z", SettingMeta(key="t", type="enum", category="hot", description="", options=("a", "b")))


def test_string_coercion_trims_bounds_and_checks_control_chars():
    meta = SettingMeta(key="test", type="string", category="hot", description="test")
    assert _validate_and_coerce("hello", meta) == "hello"
    assert _validate_and_coerce("  padded  ", meta) == "padded"
    assert _validate_and_coerce("", meta) == ""
    assert _validate_and_coerce("hello\tworld", meta) == "hello\tworld"
    with pytest.raises(ValueError, match="too long"):
        _validate_and_coerce("x" * 256, meta)
    with pytest.raises(ValueError, match="control characters"):
        _validate_and_coerce("hello\x00world", meta)
    with pytest.raises(ValueError, match="Unknown setting type"):
        _validate_and_coerce("value", SettingMeta(key="t", type="unknown_type", category="hot", description=""))


def test_config_api_serves_schema_and_404s_unknown_section(logged_in_client):
    response = logged_in_client.get("/api/admin/config")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == set(_SCHEMA)
    assert "general" in payload
    assert "ai_providers" in payload

    section = logged_in_client.get("/api/admin/config/general")
    assert section.status_code == 200
    assert "crypto_pair" in section.json()["keys"]

    assert logged_in_client.get("/api/admin/config/nonexistent").status_code == 404


def test_config_api_patch_updates_disk_and_rejects_invalid_value(logged_in_client, writable_config):
    response = logged_in_client.patch("/api/admin/config/general/timeframe", json={"value": "1d"})
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "category": "cycle",
        "section": "general",
        "key": "timeframe",
    }
    assert writable_config.get_value("general", "timeframe") == "1d"

    rejected = logged_in_client.patch("/api/admin/config/general/timeframe", json={"value": "invalid_timeframe"})
    assert rejected.status_code == 400
    assert writable_config.get_value("general", "timeframe") == "1d"


def test_config_api_rejects_unknown_key_and_out_of_range_values(logged_in_client, writable_config):
    """A write the schema cannot resolve is a 400 payload, never a 500."""
    unknown = logged_in_client.patch("/api/admin/config/general/nonexistent", json={"value": "x"})
    assert unknown.status_code == 400
    assert "Unknown config key" in unknown.json()["error"]

    for extreme in (5, 99999):
        response = logged_in_client.patch("/api/admin/config/general/candle_limit", json={"value": extreme})
        assert response.status_code == 400
    assert writable_config.get_value("general", "candle_limit") == "999"

    accepted = logged_in_client.patch("/api/admin/config/general/candle_limit", json={"value": 500})
    assert accepted.status_code == 200
    assert accepted.json()["category"] == "cycle"
    assert writable_config.get_value("general", "candle_limit") == "500"


def test_config_api_batch_update_and_payload_failures(logged_in_client, writable_config):
    response = logged_in_client.post(
        "/api/admin/config/batch",
        json={
            "updates": [
                {"section": "general", "key": "timeframe", "value": "1h"},
                {"section": "debug", "key": "logger_debug", "value": "false"},
            ]
        },
    )
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2

    incomplete = logged_in_client.post("/api/admin/config/batch", json={"updates": [{"section": "general"}]})
    assert incomplete.status_code == 400
    assert "error" in incomplete.json()
    assert writable_config.get_value("general", "timeframe") == "1h"

    wrong_key = logged_in_client.post(
        "/api/admin/config/batch",
        json={"updates": [{"section": "general", "key": "nonexistent", "value": "1d"}]},
    )
    assert wrong_key.status_code == 400
    assert "Unknown config key" in wrong_key.json()["error"]


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/api/admin/login"),
        ("post", "/api/admin/config/batch"),
        ("post", "/api/admin/system/human-input"),
        ("patch", "/api/admin/config/general/timeframe"),
    ],
)
def test_malformed_json_body_rejected_with_422(logged_in_client, method, path):
    request = logged_in_client.post if method == "post" else logged_in_client.patch
    response = request(path, content=b"{not json", headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_body_field_limits_are_enforced_before_any_write(logged_in_client, writable_config):
    assert logged_in_client.post("/api/admin/system/human-input", json={"text": ""}).status_code == 422
    assert logged_in_client.post("/api/admin/system/human-input", json={"text": "x" * 2001}).status_code == 422
    assert logged_in_client.patch("/api/admin/config/general/timeframe", json={}).status_code == 422
    oversized = logged_in_client.patch("/api/admin/config/general/crypto_pair", json={"value": "x" * 4001})
    assert oversized.status_code == 422
    assert writable_config.get_value("general", "crypto_pair") == "BTC/USDC"


def test_trigger_analysis_sets_event_and_degrades_without_one(
    logged_in_client, force_analysis_event, writable_config, log_manager
):
    response = logged_in_client.post("/api/admin/system/trigger-analysis")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Analysis triggered"}
    assert force_analysis_event.is_set()

    bare = build_admin_app(writable_config, log_manager, force_analysis_event=None)
    client = lan_client(bare)
    client.post("/api/admin/login", json={"username": "admin", "password": "testpass123"})
    degraded = client.post("/api/admin/system/trigger-analysis")
    assert degraded.status_code == 503
    assert degraded.json() == {"error": "Force analysis event not configured"}
    assert client.get("/api/admin/system/status").json()["force_analysis_available"] is False


def test_toggle_feed_and_system_status_share_state(logged_in_client, log_manager, dashboard_state):
    status = logged_in_client.get("/api/admin/system/status")
    assert status.status_code == 200
    assert status.json()["feed_enabled"] is True
    assert status.json()["force_analysis_available"] is True
    assert status.json()["log_subscribers"] == 0
    assert type(status.json()["uptime_seconds"]) is int

    sid, _queue = log_manager.handler.subscribe()
    assert logged_in_client.get("/api/admin/system/status").json()["log_subscribers"] == 1
    log_manager.handler.unsubscribe(sid)

    toggled = logged_in_client.post("/api/admin/system/toggle-feed")
    assert toggled.status_code == 200
    assert toggled.json() == {"status": "ok", "feed_enabled": False}
    dashboard_state.broadcast.assert_awaited_once_with({"type": "feed_toggle", "enabled": False})
    assert logged_in_client.get("/api/admin/system/status").json()["feed_enabled"] is False
    assert logged_in_client.post("/api/admin/system/toggle-feed").json()["feed_enabled"] is True


def test_toggle_feed_survives_a_failing_broadcast(logged_in_client, dashboard_state, router_logger):
    """A dead dashboard state is a logged warning and a normal payload, never a 500."""
    dashboard_state.broadcast = AsyncMock(side_effect=RuntimeError("broadcast down"))
    response = logged_in_client.post("/api/admin/system/toggle-feed")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "feed_enabled": False}
    assert router_logger.warning.call_args.args[0] == "Failed to broadcast feed toggle state: %s"


def test_human_input_roundtrip_set_get_clear(logged_in_client):
    created = logged_in_client.post("/api/admin/system/human-input", json={"text": "Focus on RSI"})
    assert created.status_code == 200
    assert created.json() == {"status": "ok", "text": "Focus on RSI"}

    fetched = logged_in_client.get("/api/admin/system/human-input")
    assert fetched.status_code == 200
    assert fetched.json() == {"text": "Focus on RSI"}

    assert logged_in_client.delete("/api/admin/system/human-input").status_code == 200
    assert logged_in_client.get("/api/admin/system/human-input").json()["text"] == ""


def test_recent_logs_endpoint_serves_the_ring_buffer(logged_in_client, log_manager):
    empty = logged_in_client.get("/api/admin/logs/recent")
    assert empty.status_code == 200
    assert empty.json() == {"lines": [], "count": 0}

    for index in range(5):
        log_manager.handler.emit(logging.LogRecord("test", logging.INFO, "", 0, f"Line {index}", (), None))
    payload = logged_in_client.get("/api/admin/logs/recent").json()
    assert "lines" in payload
    assert payload["count"] == 5
    assert "Line 4" in payload["lines"][-1]


def test_recent_logs_count_zero_and_negative_expose_buffer_slicing(log_manager):
    """count=0 slices [0:] so it returns the whole buffer; a negative count drops the head."""
    for index in range(5):
        log_manager.handler.emit(logging.LogRecord("test", logging.INFO, "", 0, f"Line {index}", (), None))
    assert len(log_manager.get_recent_logs(count=0)) == 5
    assert log_manager.get_recent_logs(count=-3) == log_manager.get_recent_logs()[-2:]


def test_log_stream_backpressure_drops_oldest_line_and_caps_history():
    """A full queue sheds its oldest entry, and the history ring buffer stops at 200 lines."""
    handler = LogStreamHandler(max_queue_size=2)
    sid, queue = handler.subscribe()
    for index in range(3):
        handler.emit(logging.LogRecord("test", logging.INFO, "", 0, f"burst {index}", (), None))
    assert queue.qsize() == 2
    assert "burst 1" in queue.get_nowait()
    assert "burst 2" in queue.get_nowait()
    handler.unsubscribe(sid)

    ring = LogStreamHandler(max_queue_size=10)
    for index in range(250):
        ring.emit(logging.LogRecord("test", logging.INFO, "", 0, f"ring {index}", (), None))
    recent = ring.get_recent(1000)
    assert len(recent) == 200
    assert all(f"ring {index}" in recent[index - 50] for index in range(50, 250))


def test_handler_subscribe_emit_unsubscribe_lifecycle(log_manager):
    assert log_manager.subscriber_count == 0
    sid, queue = log_manager.handler.subscribe()
    assert type(sid) is str and sid
    assert type(queue) is asyncio.Queue
    assert log_manager.subscriber_count == 1

    log_manager.handler.emit(logging.LogRecord("test", logging.INFO, "", 0, "Hello world", (), None))
    assert not queue.empty()
    line = queue.get_nowait()
    assert type(line) is str
    assert "Hello world" in line

    log_manager.handler.unsubscribe(sid)
    assert log_manager.subscriber_count == 0
    while not queue.empty():
        queue.get_nowait()
    log_manager.handler.emit(logging.LogRecord("test", logging.INFO, "", 0, "After unsub", (), None))
    assert queue.empty()


def test_subscriber_count_tracks_every_subscriber(log_manager):
    assert log_manager.subscriber_count == 0
    sids = [log_manager.handler.subscribe()[0] for _ in range(3)]
    assert log_manager.subscriber_count == 3
    log_manager.handler.unsubscribe(sids[0])
    assert log_manager.subscriber_count == 2
    for sid in sids[1:]:
        log_manager.handler.unsubscribe(sid)
    assert log_manager.subscriber_count == 0


def test_emit_reaches_every_subscriber_and_late_joiners_get_history(log_manager):
    for index in range(5):
        log_manager.handler.emit(logging.LogRecord("test", logging.INFO, "", 0, f"Pre-msg {index}", (), None))

    first_id, first_queue = log_manager.handler.subscribe()
    second_id, second_queue = log_manager.handler.subscribe()
    assert not first_queue.empty()
    assert not second_queue.empty()

    log_manager.handler.emit(logging.LogRecord("test", logging.INFO, "", 0, "Shared message", (), None))
    first_lines = [first_queue.get_nowait() for _ in range(first_queue.qsize())]
    second_lines = [second_queue.get_nowait() for _ in range(second_queue.qsize())]
    assert first_lines == second_lines
    assert any("Pre-msg 0" in line for line in first_lines)
    assert "Shared message" in first_lines[-1]

    recent = log_manager.get_recent_logs(count=3)
    assert len(recent) == 3
    assert "Shared message" in recent[-1]
    log_manager.handler.unsubscribe(first_id)
    log_manager.handler.unsubscribe(second_id)


def test_handler_formats_and_keeps_every_level(log_manager):
    sid, queue = log_manager.handler.subscribe()
    for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR):
        log_manager.handler.emit(logging.LogRecord("test", level, "", 0, f"Level {level}", (), None))

    lines = [queue.get_nowait() for _ in range(queue.qsize())]
    assert len(lines) == 4
    assert "[DEBUG]" in lines[0]
    assert lines[0].endswith("test: Level 10")
    assert "[ERROR]" in lines[3]
    assert lines[3].endswith("test: Level 40")
    log_manager.handler.unsubscribe(sid)


async def test_stream_to_forwards_lines_until_the_none_sentinel():
    """Shared by the admin and console routers: lines stream until unsubscribe sends None."""
    handler = LogStreamHandler(max_queue_size=10)
    sid, queue = handler.subscribe()
    handler.emit(logging.LogRecord("test", logging.INFO, "", 0, "one", (), None))
    handler.emit(logging.LogRecord("test", logging.INFO, "", 0, "two", (), None))
    handler.unsubscribe(sid)

    socket = RecordingSocket()
    await handler.stream_to(queue, socket)
    assert [message["type"] for message in socket.sent] == ["log", "log"]
    assert "one" in socket.sent[0]["line"]
    assert "two" in socket.sent[1]["line"]


async def test_stream_to_raises_disconnect_when_the_socket_dies_mid_stream():
    """A closed socket surfaces as WebSocketDisconnect so the caller can unsubscribe."""
    handler = LogStreamHandler(max_queue_size=10)
    sid, queue = handler.subscribe()
    handler.emit(logging.LogRecord("test", logging.INFO, "", 0, "doomed", (), None))

    socket = FailingSocket()
    with pytest.raises(WebSocketDisconnect) as excinfo:
        await handler.stream_to(queue, socket)
    assert excinfo.value.code == 1006
    assert socket.attempts == 1
    handler.unsubscribe(sid)
    assert handler.subscriber_count == 0


@pytest.mark.parametrize("path", ["/api/admin/logs/stream", "/api/admin/console"])
def test_admin_websockets_require_a_valid_token(client, path):
    with pytest.raises(WebSocketDisconnect) as missing, client.websocket_connect(path):
        pass
    assert (missing.value.code, missing.value.reason) == (1008, "Authentication required")

    with pytest.raises(WebSocketDisconnect) as invalid, client.websocket_connect(f"{path}?token=garbage"):
        pass
    assert (invalid.value.code, invalid.value.reason) == (1008, "Invalid token")


def test_logs_stream_ws_streams_lines_and_unsubscribes_on_close(logged_in_client, log_manager):
    token = logged_in_client.get("/api/admin/ws-token").json()["token"]
    with logged_in_client.websocket_connect(f"/api/admin/logs/stream?token={token}") as websocket:
        log_manager.handler.emit(logging.LogRecord("test", logging.INFO, "", 0, "hello ws", (), None))
        assert websocket.receive_json() == {"type": "log", "line": log_manager.handler.get_recent(1)[0]}
        assert log_manager.subscriber_count == 1
    assert log_manager.subscriber_count == 0


def test_console_ws_dispatches_every_action(logged_in_client, force_analysis_event):
    token = logged_in_client.get("/api/admin/ws-token").json()["token"]
    with logged_in_client.websocket_connect(f"/api/admin/console?token={token}") as websocket:
        websocket.send_json({"action": "ping"})
        assert websocket.receive_json() == {"type": "pong"}

        websocket.send_json({"action": "human_input", "text": "  focus  "})
        assert websocket.receive_json() == {"type": "ack", "action": "human_input", "text": "focus"}

        websocket.send_json({"action": "human_input", "text": "   "})
        assert websocket.receive_json() == {"type": "error", "detail": "Empty input"}

        websocket.send_json({"action": "get_status"})
        status = websocket.receive_json()
        assert status["type"] == "status"
        assert status["human_input"] == "focus"
        assert status["feed_enabled"] is True

        websocket.send_json({"action": "toggle_feed"})
        assert websocket.receive_json() == {"type": "ack", "action": "toggle_feed", "enabled": False}

        websocket.send_json({"action": "force_analysis"})
        assert websocket.receive_json() == {
            "type": "ack",
            "action": "force_analysis",
            "status": "triggered",
        }
        assert force_analysis_event.is_set()

        websocket.send_json({"action": "explode"})
        assert websocket.receive_json() == {"type": "error", "detail": "Unknown action: explode"}


def test_console_ws_reports_when_force_analysis_is_unavailable(writable_config, log_manager):
    app = build_admin_app(writable_config, log_manager, force_analysis_event=None)
    client = lan_client(app)
    client.post("/api/admin/login", json={"username": "admin", "password": "testpass123"})
    token = client.get("/api/admin/ws-token").json()["token"]
    with client.websocket_connect(f"/api/admin/console?token={token}") as websocket:
        websocket.send_json({"action": "force_analysis"})
        assert websocket.receive_json() == {"type": "error", "detail": "Force analysis not available"}


def test_dashboard_server_wires_the_admin_router(tmp_path):
    from src.dashboard.server import DashboardServer

    server = DashboardServer(
        brain_service=None,
        vector_memory=None,
        analysis_engine=None,
        config=bot_config(),
        logger=MagicMock(),
        config_path=str(tmp_path / "config.ini"),
        admin_credentials={
            "username": "admin",
            "password_hash": ADMIN_HASH,
            "signing_key": "key",
        },
    )

    assert type(server.writable_config) is WritableConfig
    assert type(server.log_stream_manager) is LogStreamManager
    assert type(server.app.state.admin_router) is AdminRouter
    assert server.app.state.writable_config is server.writable_config
    assert check_credentials("admin", "testpass123") is True
