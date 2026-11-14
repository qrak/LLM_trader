

import asyncio
import logging
from collections import deque
from typing import Any
from uuid import uuid4


class LogStreamHandler(logging.Handler):
    def __init__(self, max_queue_size: int = 500):
        super().__init__()
        self.setLevel(logging.DEBUG)
        self.max_queue_size = max_queue_size
        self._subscribers: dict[str, asyncio.Queue[str | None]] = {}
        self._recent: deque[str] = deque(maxlen=200)
        self._loop: asyncio.AbstractEventLoop | None = None

    def emit(self, record: logging.LogRecord) -> None:

        try:
            msg = self.format(record)
            self._recent.append(msg)
            for queue in list(self._subscribers.values()):
                try:
                    queue.put_nowait(msg)
                except asyncio.QueueFull:
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        queue.put_nowait(msg)
                    except asyncio.QueueFull:
                        pass
        except Exception:  # noqa: BLE001
            self.handleError(record)

    def subscribe(self) -> tuple[str, asyncio.Queue[str | None]]:

        sid = uuid4().hex
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=self.max_queue_size)
        self._subscribers[sid] = queue

        for line in self._recent:
            try:
                queue.put_nowait(line)
            except asyncio.QueueFull:
                break

        return sid, queue

    def unsubscribe(self, subscriber_id: str) -> None:

        queue = self._subscribers.pop(subscriber_id, None)
        if queue is not None:
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    async def stream_to(self, queue: asyncio.Queue[str | None], websocket: Any) -> None:

        while True:
            try:
                line = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue
            if line is None:
                return
            await websocket.send_json({"type": "log", "line": line})


class LogStreamManager:
    def __init__(self, max_queue_size: int = 500):
        self.handler = LogStreamHandler(max_queue_size=max_queue_size)
        self.handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
        )

    def attach_to_logger(self, logger: logging.Logger) -> None:

        logger.addHandler(self.handler)

