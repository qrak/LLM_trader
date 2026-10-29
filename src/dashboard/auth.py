

import hashlib
import hmac
import os
import time

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

COOKIE_NAME = "admin_session"
COOKIE_MAX_AGE = 8 * 3600
COOKIE_PATH = "/"

_SIGNING_KEY: bytes = b""
_ADMIN_USERNAME: str = ""
_ADMIN_PASSWORD_HASH: str = ""
_initialized: bool = False


def _derive_password_hash(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:

    if salt is None:
        salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations=100_000)
    return dk.hex().encode("utf-8"), salt


def hash_password(password: str) -> str:

    h, salt = _derive_password_hash(password)
    return f"{salt.hex()}:{h.decode()}"


def _verify_password(password: str, stored: str) -> bool:

    try:
        salt_hex, hash_hex = stored.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        expected, _ = _derive_password_hash(password, salt)
        return hmac.compare_digest(expected, hash_hex.encode("utf-8"))
    except (ValueError, AttributeError):
        return False


def init_auth(signing_key: str, admin_username: str, admin_password_hash: str) -> None:

    global _SIGNING_KEY, _ADMIN_USERNAME, _ADMIN_PASSWORD_HASH, _initialized
    _SIGNING_KEY = signing_key.encode("utf-8") if signing_key else os.urandom(32)
    _ADMIN_USERNAME = admin_username
    _ADMIN_PASSWORD_HASH = admin_password_hash
    _initialized = True


def _sign_token(username: str, timestamp: float) -> str:

    payload = f"{username}:{int(timestamp)}"
    sig = hmac.new(_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}:{sig}"


def _verify_token(token: str) -> str | None:

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


def verify_admin_session(request: Request) -> str | None:

    token = request.cookies.get(COOKIE_NAME)
    if token:
        username = _verify_token(token)
        if username:
            return username

    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        username = _verify_token(token)
        if username:
            return username

    token = request.query_params.get("token")
    if token:
        username = _verify_token(token)
        if username:
            return username

    return None


_DUMMY_HASH = "00000000000000000000000000000000:0000000000000000000000000000000000000000000000000000000000000000"


def check_credentials(username: str, password: str) -> bool:

    if not _initialized:
        return False
    user_ok = hmac.compare_digest(username, _ADMIN_USERNAME)
    target_hash = _ADMIN_PASSWORD_HASH if (user_ok and _ADMIN_PASSWORD_HASH) else _DUMMY_HASH
    pass_ok = _verify_password(password, target_hash)
    return user_ok and pass_ok


def resolve_client_ip(direct_ip: str, cf_connecting_ip: str | None) -> str:

    if direct_ip in ("127.0.0.1", "::1"):
        cf_ip = (cf_connecting_ip or "").strip()
        if cf_ip:
            return cf_ip
    return direct_ip


_LOGIN_ATTEMPTS: dict[str, list[float]] = {}


def check_login_rate_limit(ip: str, max_attempts: int = 5, window_seconds: float = 60.0) -> tuple[bool, int]:

    now = time.time()
    stale_ips = [
        client_ip for client_ip, timestamps in _LOGIN_ATTEMPTS.items()
        if not any(now - t < window_seconds for t in timestamps)
    ]
    for client_ip in stale_ips:
        _LOGIN_ATTEMPTS.pop(client_ip, None)

    attempts = [t for t in _LOGIN_ATTEMPTS.get(ip, []) if now - t < window_seconds]
    if attempts:
        _LOGIN_ATTEMPTS[ip] = attempts
    else:
        _LOGIN_ATTEMPTS.pop(ip, None)

    if len(attempts) >= max_attempts:
        oldest = attempts[0]
        retry_after = int(window_seconds - (now - oldest)) + 1
        return False, max(1, retry_after)
    return True, 0


def record_login_attempt(ip: str) -> None:

    if ip not in _LOGIN_ATTEMPTS:
        _LOGIN_ATTEMPTS[ip] = []
    _LOGIN_ATTEMPTS[ip].append(time.time())


def reset_login_rate_limit(ip: str) -> None:

    _LOGIN_ATTEMPTS.pop(ip, None)


class AdminAuthMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that protects all /api/admin/* routes.

    LAN-only access: rejects requests from non-private IPs on ALL admin paths
    (both API and static HTML). Static admin files (/admin/*) are served without
    authentication — the login form lives there and must be accessible.
    Only /api/admin/* endpoints require a valid session.
    """

    _PUBLIC_PATHS: set[str] = {  # noqa: RUF012
        "/api/admin/login",
        "/api/admin/health",
    }

    @staticmethod
    def _is_lan_ip(ip: str) -> bool:

        if not ip:
            return False
        try:
            parts = ip.split(".")
            if len(parts) != 4:
                return ip in ("::1", "fe80::1")
            a, b = int(parts[0]), int(parts[1])
            if a == 127:
                return True
            if a == 10:
                return True
            if a == 192 and b == 168:
                return True
            if a == 172 and 16 <= b <= 31:
                return True
            if a == 169 and b == 254:  # noqa: SIM103
                return True
            return False
        except (ValueError, IndexError):
            return False

    @staticmethod
    def _get_real_client_ip(request: Request) -> str:

        direct_ip = request.client.host if request.client else ""
        return resolve_client_ip(direct_ip, request.headers.get("cf-connecting-ip"))

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ):
        path = request.url.path

        if not (path.startswith(("/api/admin/", "/admin"))):
            return await call_next(request)

        client_ip = self._get_real_client_ip(request)
        if not self._is_lan_ip(client_ip):
            return JSONResponse(
                status_code=403,
                content={"error": "Forbidden", "detail": "Admin access restricted to LAN"},
            )

        if not path.startswith("/api/admin/"):
            return await call_next(request)

        if path in self._PUBLIC_PATHS:
            return await call_next(request)

        username = verify_admin_session(request)
        if username is None:
            return JSONResponse(
                status_code=401,
                content={"error": "Authentication required", "detail": "Valid admin session needed"},
            )

        request.state.admin_user = username
        return await call_next(request)

