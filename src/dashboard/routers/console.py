"""Public read-only Live Console router.

Provides:
- WebSocket /ws/console — live log streaming (no auth, Cloudflare-protected)
- REST /api/console/recent — latest log lines
- REST /api/console/pages — day-page index
- REST /api/console/page/{n} — specific day page
"""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from ..console_buffer import ConsoleBuffer
from ..log_stream import LogStreamHandler


class ConsoleRouter:
    """Public console view router — read-only, no authentication required."""

    def __init__(self, buffer: ConsoleBuffer, log_stream_handler: LogStreamHandler):
        self.router = APIRouter(prefix="/api/console", tags=["console"])
        self.buffer = buffer
        self._handler = log_stream_handler
        self._register_routes()

    def _register_routes(self) -> None:
        @self.router.get("/recent")
        async def recent_logs(count: int = 200):
            lines = self.buffer.get_recent(count=min(count, 2000))
            return {"lines": lines, "count": len(lines)}

        @self.router.get("/pages")
        async def page_index():
            pages = []
            for p in range(self.buffer.page_count):
                info = self.buffer.page_info(p)
                if info:
                    pages.append(info)
            return {"pages": pages, "total": self.buffer.page_count}

        @self.router.get("/page/{page}")
        async def day_page(page: int):
            if page < 0 or page >= self.buffer.page_count:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Page {page} out of range (0–{self.buffer.page_count - 1})"},
                )
            info = self.buffer.page_info(page)
            lines = self.buffer.get_page(page) or []
            return {
                "page": page,
                "date": info["date"] if info else "unknown",
                "label": info["label"] if info else f"Day {page}",
                "lines": lines,
                "count": len(lines),
            }

        @self.router.websocket("/live")
        async def console_live(websocket: WebSocket):
            """Public WebSocket — streams log lines in real-time.

            No authentication required. Protected by Cloudflare in production.
            """
            await websocket.accept()
            sid, queue = self._handler.subscribe()

            try:
                while True:
                    try:
                        line = await asyncio.wait_for(queue.get(), timeout=30.0)
                        if line is None:
                            break
                        await websocket.send_json({"type": "log", "line": line})
                    except asyncio.TimeoutError:
                        await websocket.send_json({"type": "ping"})
            except WebSocketDisconnect:
                pass
            except Exception:  # noqa: BLE001, S110
                pass
            finally:
                self._handler.unsubscribe(sid)
