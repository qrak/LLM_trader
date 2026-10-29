

import asyncio
import time
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from ...config.writable_config import WritableConfig
from ...trading.pending_approval import PendingApprovalStore
from ..auth import (
    COOKIE_NAME,
    AdminAuthMiddleware,
    _sign_token,
    _verify_token,
    check_credentials,
    check_login_rate_limit,
    create_session,
    record_login_attempt,
    reset_login_rate_limit,
)
from ..log_stream import LogStreamManager


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=256)


class ConfigUpdateRequest(BaseModel):
    value: Any

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: Any) -> Any:
        if isinstance(v, str) and len(v) > 4000:
            raise ValueError("Config value text cannot exceed 4000 characters.")
        return v


class BatchConfigUpdateRequest(BaseModel):
    updates: list[dict[str, Any]] = Field(
        ...,
        description="List of {section, key, value} dicts",
    )


class HumanInputRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="User's iteration goal or instruction")


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
        config: Any,
        logger: Any,
        brain_service: Any = None,
        analysis_engine: Any = None,
        exchange_manager: Any = None,
        dashboard_state: Any = None,
        force_analysis_event: asyncio.Event | None = None,
        approvals: PendingApprovalStore | None = None,
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
        self.approvals = approvals
        self._force_analysis = force_analysis_event
        self._dashboard_feed_enabled = True
        self._human_input: str = ""
        self._bot_start_time = time.time()

        self._register_routes()

    async def _authenticate_ws(self, websocket: WebSocket) -> bool:

        token = websocket.query_params.get("token")
        if not token:
            await websocket.close(code=1008, reason="Authentication required")
            return False
        if not _verify_token(token):
            await websocket.close(code=1008, reason="Invalid token")
            return False
        return True

    def _register_routes(self) -> None:



        @self.router.post("/login")
        async def login(body: LoginRequest, response: Response, request: Request):
            client_ip = AdminAuthMiddleware._get_real_client_ip(request)
            allowed, retry_after = check_login_rate_limit(client_ip)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"error": f"Too many login attempts. Please retry in {retry_after} seconds."},
                    headers={"Retry-After": str(retry_after)},
                )

            if check_credentials(body.username, body.password):
                reset_login_rate_limit(client_ip)
                is_https = (
                    request.url.scheme == "https"
                    or request.headers.get("x-forwarded-proto") == "https"
                )
                create_session(body.username, response, secure=is_https)
                ws_token = _sign_token(body.username, time.time())
                return {"status": "ok", "username": body.username, "token": ws_token}

            record_login_attempt(client_ip)
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

            username = getattr(request.state, "admin_user", None)
            if not username:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Authentication required"},
                )
            from ..auth import _sign_token
            token = _sign_token(username, time.time())
            return {"status": "ok", "username": username, "token": token}


        @self.router.get("/config")
        async def get_config():

            return self.writable_config.get_full_schema()

        @self.router.get("/config/{section}")
        async def get_config_section(section: str):

            result = self.writable_config.get_section_schema(section)
            if result is None:
                return JSONResponse(status_code=404, content={"error": f"Unknown section: {section}"})
            return result

        @self.router.patch("/config/{section}/{key}")
        async def update_config_value(section: str, key: str, body: ConfigUpdateRequest):

            try:
                category = await self.writable_config.set_value(section, key, body.value)
                return {"status": "ok", "category": category, "section": section, "key": key}
            except ValueError as exc:
                return JSONResponse(status_code=400, content={"error": str(exc)})

        @self.router.post("/config/batch")
        async def batch_update_config(body: BatchConfigUpdateRequest):

            try:
                updates = [(u["section"], u["key"], u["value"]) for u in body.updates]
                results = await self.writable_config.set_values(updates)
                return {"status": "ok", "results": results}
            except (ValueError, KeyError) as exc:
                return JSONResponse(status_code=400, content={"error": str(exc)})

        @self.router.post("/config/reload")
        async def reload_config():

            await self.writable_config.reload_from_disk()
            return {"status": "ok", "note": "form refreshed; the trading process still needs a restart"}


        @self.router.post("/system/trigger-analysis")
        async def trigger_analysis():

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

            self._dashboard_feed_enabled = not self._dashboard_feed_enabled
            state = "enabled" if self._dashboard_feed_enabled else "disabled"
            if self.logger:
                self.logger.info("Admin: Dashboard feed %s via web console", state)
            if self.dashboard_state:
                try:
                    await self.dashboard_state.broadcast({
                        "type": "feed_toggle",
                        "enabled": self._dashboard_feed_enabled,
                    })
                except Exception as exc:  # noqa: BLE001
                    if self.logger:
                        self.logger.warning("Failed to broadcast feed toggle state: %s", exc)
            return {"status": "ok", "feed_enabled": self._dashboard_feed_enabled}

        @self.router.get("/system/status")
        async def system_status():

            uptime = int(time.time() - self._bot_start_time)
            return {
                "uptime_seconds": uptime,
                "feed_enabled": self._dashboard_feed_enabled,
                "log_subscribers": self.log_stream_manager.subscriber_count,
                "force_analysis_available": self._force_analysis is not None,
            }


        @self.router.get("/approvals")
        async def list_approvals():

            if self.approvals is None:
                return {"approvals": []}
            return {"approvals": [asdict(approval) for approval in self.approvals.list()]}

        @self.router.post("/approvals/{approval_id}/approve")
        async def approve_decision(approval_id: str):

            if self.approvals is None:
                return JSONResponse(status_code=404, content={"status": "disabled"})
            approval = self.approvals.approve(approval_id)
            if approval is None:
                return JSONResponse(status_code=404, content={"status": "not_found"})
            if self.logger:
                self.logger.warning("Admin: approval %s -> %s", approval_id, approval.status)
            return {"status": approval.status, "approval": asdict(approval)}

        @self.router.post("/approvals/{approval_id}/reject")
        async def reject_decision(approval_id: str):

            if self.approvals is None:
                return JSONResponse(status_code=404, content={"status": "disabled"})
            approval = self.approvals.reject(approval_id)
            if approval is None:
                return JSONResponse(status_code=404, content={"status": "not_found"})
            if self.logger:
                self.logger.warning("Admin: approval %s -> %s", approval_id, approval.status)
            return {"status": approval.status, "approval": asdict(approval)}

        @self.router.post("/system/human-input")
        async def set_human_input(body: HumanInputRequest):

            self._human_input = body.text
            if self.logger:
                self.logger.info("Admin: Human input set: %s", body.text[:100])
            return {"status": "ok", "text": body.text}

        @self.router.get("/system/human-input")
        async def get_human_input():

            return {"text": self._human_input}

        @self.router.delete("/system/human-input")
        async def clear_human_input():

            self._human_input = ""
            return {"status": "ok"}


        @self.router.get("/logs/recent")
        async def recent_logs(count: int = 200):

            lines = self.log_stream_manager.get_recent_logs(count=min(count, 1000))
            return {"lines": lines, "count": len(lines)}


        @self.router.websocket("/logs/stream")
        async def log_stream_ws(websocket: WebSocket):

            if not await self._authenticate_ws(websocket):
                return

            await websocket.accept()
            sid, queue = self.log_stream_manager.handler.subscribe()
            try:
                await self.log_stream_manager.handler.stream_to(queue, websocket)
            except WebSocketDisconnect:
                pass
            except Exception as exc:  # noqa: BLE001
                if self.logger:
                    self.logger.warning("Admin logs WebSocket stream error: %s", exc)
            finally:
                self.log_stream_manager.handler.unsubscribe(sid)


        @self.router.websocket("/console")
        async def console_ws(websocket: WebSocket):

            if not await self._authenticate_ws(websocket):
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
            except Exception as exc:  # noqa: BLE001
                if self.logger:
                    self.logger.warning("Admin console WebSocket error: %s", exc)

