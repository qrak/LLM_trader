"""Admin router for the control console.

Provides REST endpoints for:
- Login / session management
- Config CRUD (read schema, update individual settings, batch update)
- System control (force analysis, toggle feed)
- Log streaming (WebSocket + REST recent logs)

All endpoints except /login and /health require authentication.
"""

import asyncio
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..auth import (
    check_credentials,
    create_session,
    verify_admin_session,
    COOKIE_NAME,
)
from ..log_stream import LogStreamManager
from ...config.writable_config import WritableConfig


# ─── Pydantic request models ────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=256)


class ConfigUpdateRequest(BaseModel):
    value: Any


class BatchConfigUpdateRequest(BaseModel):
    updates: list[dict[str, Any]] = Field(
        ...,
        description="List of {section, key, value} dicts",
    )


class HumanInputRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="User's iteration goal or instruction")


# ─── Admin Router ────────────────────────────────────────────────────

class AdminRouter:
    """Admin control console router.

    Integrates with:
    - WritableConfig for config.ini read/write
    - LogStreamManager for live log streaming
    - Bot's force_analysis event (asyncio.Event) for triggering analyses
    - Dashboard state for feed toggling
    """

    def __init__(
        self,
        writable_config: WritableConfig,
        log_stream_manager: LogStreamManager,
        config: Any,           # Bot's Config object (read-only reference)
        logger: Any,
        brain_service: Any = None,
        analysis_engine: Any = None,
        exchange_manager: Any = None,
        dashboard_state: Any = None,
        force_analysis_event: asyncio.Event | None = None,
    ):
        self.router = APIRouter(prefix="/api/admin", tags=["admin"])
        self.writable_config = writable_config
        self.log_stream_manager = log_stream_manager
        self.config = config
        self.logger = logger
        self.brain_service = brain_service
        self.analysis_engine = analysis_engine
        self.exchange_manager = exchange_manager
        self.dashboard_state = dashboard_state
        self._force_analysis = force_analysis_event
        self._dashboard_feed_enabled = True
        self._human_input: str = ""
        self._bot_start_time = time.time()

        self._register_routes()

    def _register_routes(self) -> None:
        """Register all admin routes."""

        # ── Auth ──────────────────────────────────────────────────────

        @self.router.post("/login")
        async def login(body: LoginRequest, response: Response, request: Request):
            if check_credentials(body.username, body.password):
                # Detect HTTPS (direct or via Cloudflare proxy)
                is_https = (
                    request.url.scheme == "https"
                    or request.headers.get("x-forwarded-proto") == "https"
                )
                create_session(body.username, response, secure=is_https)
                # Generate a WS token for the frontend
                from ..auth import _sign_token
                import time as _time
                ws_token = _sign_token(body.username, _time.time())
                return {"status": "ok", "username": body.username, "token": ws_token}
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid credentials"},
            )

        @self.router.get("/health")
        async def health():
            return {"status": "ok", "uptime": int(time.time() - self._bot_start_time)}

        @self.router.post("/logout")
        async def logout(response: Response):
            response.delete_cookie(COOKIE_NAME, path="/")
            return {"status": "ok"}

        @self.router.get("/ws-token")
        async def ws_token(request: Request):
            """Issue a fresh WebSocket token for an already-authenticated session.

            Auth is enforced by AdminAuthMiddleware (httponly cookie or bearer),
            so the SPA never has to persist the token in localStorage: it can
            obtain a short-lived token in memory after a page reload.
            """
            username = getattr(request.state, "admin_user", None)
            if not username:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Authentication required"},
                )
            from ..auth import _sign_token
            token = _sign_token(username, time.time())
            return {"status": "ok", "username": username, "token": token}

        # ── Config ────────────────────────────────────────────────────

        @self.router.get("/config")
        async def get_config():
            """Return full config schema with current values."""
            return self.writable_config.get_full_schema()

        @self.router.get("/config/{section}")
        async def get_config_section(section: str):
            """Return one config section with current values."""
            result = self.writable_config.get_section_schema(section)
            if result is None:
                return JSONResponse(status_code=404, content={"error": f"Unknown section: {section}"})
            return result

        @self.router.patch("/config/{section}/{key}")
        async def update_config_value(section: str, key: str, body: ConfigUpdateRequest):
            """Update a single config value."""
            try:
                category = await self.writable_config.set_value(section, key, body.value)
                return {"status": "ok", "category": category, "section": section, "key": key}
            except ValueError as exc:
                return JSONResponse(status_code=400, content={"error": str(exc)})

        @self.router.post("/config/batch")
        async def batch_update_config(body: BatchConfigUpdateRequest):
            """Update multiple config values atomically."""
            try:
                updates = [(u["section"], u["key"], u["value"]) for u in body.updates]
                results = await self.writable_config.set_values(updates)
                return {"status": "ok", "results": results}
            except (ValueError, KeyError) as exc:
                return JSONResponse(status_code=400, content={"error": str(exc)})

        @self.router.post("/config/reload")
        async def reload_config():
            """Reload config.ini from disk (for external edits)."""
            await self.writable_config.reload_from_disk()
            return {"status": "ok"}

        # ── System Control ────────────────────────────────────────────

        @self.router.post("/system/trigger-analysis")
        async def trigger_analysis():
            """Force immediate analysis (equivalent to pressing 'a')."""
            if self._force_analysis:
                self._force_analysis.set()
                if self.logger:
                    self.logger.info("Admin: Force analysis triggered via web console")
                return {"status": "ok", "message": "Analysis triggered"}
            return JSONResponse(
                status_code=503,
                content={"error": "Force analysis event not configured"},
            )

        @self.router.post("/system/toggle-feed")
        async def toggle_feed():
            """Toggle dashboard feed (equivalent to pressing 'd')."""
            self._dashboard_feed_enabled = not self._dashboard_feed_enabled
            state = "enabled" if self._dashboard_feed_enabled else "disabled"
            if self.logger:
                self.logger.info("Admin: Dashboard feed %s via web console", state)
            # Broadcast state change to all WS clients
            if self.dashboard_state:
                try:
                    await self.dashboard_state.broadcast({
                        "type": "feed_toggle",
                        "enabled": self._dashboard_feed_enabled,
                    })
                except Exception:
                    pass
            return {"status": "ok", "feed_enabled": self._dashboard_feed_enabled}

        @self.router.get("/system/status")
        async def system_status():
            """Return bot status overview."""
            uptime = int(time.time() - self._bot_start_time)
            return {
                "uptime_seconds": uptime,
                "feed_enabled": self._dashboard_feed_enabled,
                "log_subscribers": self.log_stream_manager.subscriber_count,
                "force_analysis_available": self._force_analysis is not None,
            }

        # ── Human-in-the-Loop Input ──────────────────────────────────

        @self.router.post("/system/human-input")
        async def set_human_input(body: HumanInputRequest):
            """Set a manual iteration goal / instruction for the bot."""
            self._human_input = body.text
            if self.logger:
                self.logger.info("Admin: Human input set: %s", body.text[:100])
            return {"status": "ok", "text": body.text}

        @self.router.get("/system/human-input")
        async def get_human_input():
            """Get the current human input (consumed by the bot on next cycle)."""
            return {"text": self._human_input}

        @self.router.delete("/system/human-input")
        async def clear_human_input():
            """Clear the human input after the bot has consumed it."""
            self._human_input = ""
            return {"status": "ok"}

        # ── Log Streaming (REST) ─────────────────────────────────────

        @self.router.get("/logs/recent")
        async def recent_logs(count: int = 200):
            """Return the last N log lines."""
            lines = self.log_stream_manager.get_recent_logs(count=min(count, 1000))
            return {"lines": lines, "count": len(lines)}

        # ── Log Streaming (WebSocket) ────────────────────────────────

        @self.router.websocket("/logs/stream")
        async def log_stream_ws(websocket: WebSocket):
            """WebSocket endpoint for live log streaming.

            Auth: Requires ?token=<session_token> query param.
            """
            # Verify auth via query param
            token = websocket.query_params.get("token")
            if not token:
                await websocket.close(code=1008, reason="Authentication required")
                return

            from ..auth import _verify_token
            username = _verify_token(token)
            if not username:
                await websocket.close(code=1008, reason="Invalid token")
                return

            # Accept and subscribe
            await websocket.accept()
            sid, queue = self.log_stream_manager.handler.subscribe()
            try:
                while True:
                    try:
                        line = await asyncio.wait_for(queue.get(), timeout=30.0)
                        if line is None:
                            break
                        await websocket.send_json({"type": "log", "line": line})
                    except asyncio.TimeoutError:
                        # Send keepalive ping
                        await websocket.send_json({"type": "ping"})
            except WebSocketDisconnect:
                pass
            except Exception:
                pass
            finally:
                self.log_stream_manager.handler.unsubscribe(sid)

        # ── Control Console WS (commands + state) ────────────────────

        @self.router.websocket("/console")
        async def console_ws(websocket: WebSocket):
            """Bidirectional WebSocket for the control console.

            Accepts commands from the frontend:
              {"action": "force_analysis"}
              {"action": "toggle_feed"}
              {"action": "human_input", "text": "..."}
              {"action": "get_status"}

            Auth: Requires ?token=<session_token> query param.
            """
            token = websocket.query_params.get("token")
            if not token:
                await websocket.close(code=1008, reason="Authentication required")
                return

            from ..auth import _verify_token
            username = _verify_token(token)
            if not username:
                await websocket.close(code=1008, reason="Invalid token")
                return

            await websocket.accept()
            try:
                while True:
                    raw = await websocket.receive_json()
                    action = raw.get("action", "")

                    if action == "force_analysis":
                        if self._force_analysis:
                            self._force_analysis.set()
                            await websocket.send_json({"type": "ack", "action": "force_analysis", "status": "triggered"})
                        else:
                            await websocket.send_json({"type": "error", "detail": "Force analysis not available"})

                    elif action == "toggle_feed":
                        self._dashboard_feed_enabled = not self._dashboard_feed_enabled
                        state = "enabled" if self._dashboard_feed_enabled else "disabled"
                        await websocket.send_json({"type": "ack", "action": "toggle_feed", "enabled": self._dashboard_feed_enabled})
                        if self.logger:
                            self.logger.info("Admin: Dashboard feed %s via console WS", state)

                    elif action == "human_input":
                        text = raw.get("text", "").strip()
                        if text:
                            self._human_input = text
                            await websocket.send_json({"type": "ack", "action": "human_input", "text": text})
                        else:
                            await websocket.send_json({"type": "error", "detail": "Empty input"})

                    elif action == "get_status":
                        uptime = int(time.time() - self._bot_start_time)
                        await websocket.send_json({
                            "type": "status",
                            "uptime_seconds": uptime,
                            "feed_enabled": self._dashboard_feed_enabled,
                            "human_input": self._human_input,
                        })

                    elif action == "ping":
                        await websocket.send_json({"type": "pong"})

                    else:
                        await websocket.send_json({"type": "error", "detail": f"Unknown action: {action}"})

            except WebSocketDisconnect:
                pass
            except Exception:
                pass

    @property
    def dashboard_feed_enabled(self) -> bool:
        """Check if dashboard feed is enabled (read by app loop)."""
        return self._dashboard_feed_enabled

    @property
    def human_input(self) -> str:
        """Get current human input text."""
        return self._human_input

    def consume_human_input(self) -> str:
        """Get and clear the human input (called by bot on each cycle)."""
        text = self._human_input
        self._human_input = ""
        return text
