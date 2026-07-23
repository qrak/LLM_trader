"""HMAC cookie-based authentication for the admin dashboard.

Uses HMAC-SHA256 signed cookies for session management.
Credentials are loaded from keys.env (ADMIN_USERNAME, ADMIN_PASSWORD_HASH).
Password hash uses hashlib.pbkdf2_hmac (SHA256, 100k iterations) — no external deps.

All admin routes and admin WebSocket connections must pass through
verify_admin_session() or the HTTP middleware.
"""

import hashlib
import hmac
import os
import time
from typing import Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

# Cookie configuration
COOKIE_NAME = "admin_session"
COOKIE_MAX_AGE = 8 * 3600  # 8 hours
COOKIE_PATH = "/"

# Timing-safe comparison module-level constant
_SIGNING_KEY: bytes = b""
_ADMIN_USERNAME: str = ""
_ADMIN_PASSWORD_HASH: str = ""  # pbkdf2_hmac hex digest
_initialized: bool = False


def _derive_password_hash(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    """Derive a PBKDF2-SHA256 hash from a plaintext password.

    Returns (hash_hex_bytes, salt_bytes).
    """
    if salt is None:
        salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations=100_000)
    return dk.hex().encode("utf-8"), salt


def hash_password(password: str) -> str:
    """One-shot helper: returns 'salt_hex:hash_hex' for storage in keys.env."""
    h, salt = _derive_password_hash(password)
    return f"{salt.hex()}:{h.decode()}"


def _verify_password(password: str, stored: str) -> bool:
    """Verify a plaintext password against 'salt_hex:hash_hex' stored hash."""
    try:
        salt_hex, hash_hex = stored.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        expected, _ = _derive_password_hash(password, salt)
        return hmac.compare_digest(expected, hash_hex.encode("utf-8"))
    except (ValueError, AttributeError):
        return False


def init_auth(signing_key: str, admin_username: str, admin_password_hash: str) -> None:
    """Initialize the auth module with credentials from keys.env.

    Args:
        signing_key: Secret key for HMAC cookie signing (auto-generated if empty).
        admin_username: Expected admin username.
        admin_password_hash: Stored password hash in 'salt_hex:hash_hex' format.
    """
    global _SIGNING_KEY, _ADMIN_USERNAME, _ADMIN_PASSWORD_HASH, _initialized
    _SIGNING_KEY = signing_key.encode("utf-8") if signing_key else os.urandom(32)
    _ADMIN_USERNAME = admin_username
    _ADMIN_PASSWORD_HASH = admin_password_hash
    _initialized = True


def _sign_token(username: str, timestamp: float) -> str:
    """Create an HMAC-signed session token."""
    payload = f"{username}:{int(timestamp)}"
    sig = hmac.new(_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}:{sig}"


def _verify_token(token: str) -> Optional[str]:
    """Verify an HMAC-signed session token. Returns username or None."""
    try:
        parts = token.split(":")
        if len(parts) != 3:
            return None
        username, ts_str, sig = parts
        payload = f"{username}:{ts_str}"
        expected = hmac.new(_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        ts = int(ts_str)
        if time.time() - ts > COOKIE_MAX_AGE:
            return None
        return username
    except (ValueError, AttributeError):
        return None


def create_session(username: str, response: Response, secure: bool = True) -> None:
    """Set an authenticated session cookie on the response.

    Args:
        username: The authenticated username.
        response: FastAPI response object.
        secure: If True, set Secure flag (required for HTTPS). Set False for local HTTP testing.
    """
    token = _sign_token(username, time.time())
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=COOKIE_MAX_AGE,
        path=COOKIE_PATH,
        httponly=True,
        samesite="lax",
        secure=secure,
    )


def verify_admin_session(request: Request) -> Optional[str]:
    """Check the request for a valid admin session.

    Checks cookie first, then Authorization header (Bearer token).
    Returns the username if valid, None otherwise.
    """
    # Check cookie
    token = request.cookies.get(COOKIE_NAME)
    if token:
        username = _verify_token(token)
        if username:
            return username

    # Check Authorization header (for WebSocket and API clients)
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        username = _verify_token(token)
        if username:
            return username

    # Check query param (for WebSocket upgrade)
    token = request.query_params.get("token")
    if token:
        username = _verify_token(token)
        if username:
            return username

    return None


_DUMMY_HASH = "00000000000000000000000000000000:0000000000000000000000000000000000000000000000000000000000000000"


def check_credentials(username: str, password: str) -> bool:
    """Validate login credentials against stored hash.

    Executes PBKDF2 verification even when username is invalid to prevent
    username enumeration via timing side-channels.
    """
    if not _initialized:
        return False
    user_ok = hmac.compare_digest(username, _ADMIN_USERNAME)
    target_hash = _ADMIN_PASSWORD_HASH if (user_ok and _ADMIN_PASSWORD_HASH) else _DUMMY_HASH
    pass_ok = _verify_password(password, target_hash)
    return user_ok and pass_ok


def _is_lan_ip(ip: str) -> bool:
    """Check if an IP address is a LAN/private address."""
    if not ip:
        return False
    try:
        parts = ip.split(".")
        if len(parts) != 4:
            # IPv6 loopback
            return ip in ("::1", "fe80::1")
        a, b = int(parts[0]), int(parts[1])
        if a == 127:  # Loopback
            return True
        if a == 10:  # 10.0.0.0/8
            return True
        if a == 192 and b == 168:  # 192.168.0.0/16
            return True
        if a == 172 and 16 <= b <= 31:  # 172.16.0.0/12
            return True
        if a == 169 and b == 254:  # Link-local
            return True
        return False
    except (ValueError, IndexError):
        return False


def _get_real_client_ip(request: Request) -> str:
    """Extract real client IP for LAN checks.

    Security model:
    - Direct LAN connection: use request.client.host directly.
    - Cloudflare Tunnel (request from localhost): trust CF-Connecting-IP header
      because only Cloudflare's edge sets it and the tunnel is a local process.
    - Direct internet connection: use request.client.host (will be public IP → blocked).
    - Spoofed CF-Connecting-IP on direct connection: IGNORED because direct_ip
      is not localhost, so the header is never consulted.
    """
    direct_ip = request.client.host if request.client else ""

    # If request came through Cloudflare Tunnel (localhost connection),
    # trust CF-Connecting-IP for the real visitor IP.
    if direct_ip in ("127.0.0.1", "::1"):
        cf_ip = request.headers.get("cf-connecting-ip", "").strip()
        if cf_ip:
            return cf_ip
        # Tunnel but no CF header — still treat as local
        return direct_ip

    # Direct connection — use the actual client IP (header is NOT trusted)
    return direct_ip


class AdminAuthMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that protects all /api/admin/* routes.

    LAN-only access: rejects requests from non-private IPs on ALL admin paths
    (both API and static HTML). Static admin files (/admin/*) are served without
    authentication — the login form lives there and must be accessible.
    Only /api/admin/* endpoints require a valid session.
    """

    # API routes that don't require authentication (but still require LAN)
    _PUBLIC_PATHS: set[str] = {
        "/api/admin/login",
        "/api/admin/health",
    }

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ):
        path = request.url.path

        # Only process admin routes (API + static HTML)
        if not (path.startswith("/api/admin/") or path.startswith("/admin")):
            return await call_next(request)

        # LAN-only check: block non-private IPs from ALL admin paths
        client_ip = _get_real_client_ip(request)
        if not _is_lan_ip(client_ip):
            return JSONResponse(
                status_code=403,
                content={"error": "Forbidden", "detail": "Admin access restricted to LAN"},
            )

        # Static admin files (HTML, CSS, JS) — no auth required, just LAN gate.
        # The login form and SPA shell live here and must be loadable without a session.
        if not path.startswith("/api/admin/"):
            return await call_next(request)

        # Allow public API endpoints without auth
        if path in self._PUBLIC_PATHS:
            return await call_next(request)

        # Verify session for all other /api/admin/* endpoints
        username = verify_admin_session(request)
        if username is None:
            return JSONResponse(
                status_code=401,
                content={"error": "Authentication required", "detail": "Valid admin session needed"},
            )

        # Attach username to request state for downstream handlers
        request.state.admin_user = username
        return await call_next(request)
