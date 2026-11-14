

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from ..console_buffer import ConsoleBuffer
from ..log_stream import LogStreamHandler
from .ws_router import ConnectionManager


class ConsoleRouter:
    def __init__(
        self,
        buffer: ConsoleBuffer,
        log_stream_handler: LogStreamHandler,
        logger,
        connection_manager: ConnectionManager,
    ):
        self.router = APIRouter(prefix="/api/console", tags=["console"])
        self.buffer = buffer
        self._handler = log_stream_handler
        self.logger = logger
        self._connections = connection_manager
        self._register_routes()

    def _register_routes(self) -> None:
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

            if not await self._connections.connect(websocket):
                return
            sid, queue = self._handler.subscribe()

            try:
                await self._handler.stream_to(queue, websocket)
            except WebSocketDisconnect:
                pass
            except Exception as error:
                self.logger.debug("Console stream ended: %s", error, exc_info=True)
            finally:
                self._handler.unsubscribe(sid)
                self._connections.disconnect(websocket)
