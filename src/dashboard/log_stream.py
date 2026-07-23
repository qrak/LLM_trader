"""Live log streaming for the admin dashboard.

Provides a custom logging handler that pushes formatted log records
to subscriber asyncio.Queues. WebSocket endpoints consume these queues
to stream logs to the browser in real-time.
"""

import asyncio
import logging
from collections import deque
from uuid import uuid4


class LogStreamHandler(logging.Handler):
    """Custom logging handler that broadcasts records to subscriber queues.

    Each subscriber gets its own asyncio.Queue with a configurable max size.
    Slow consumers have their oldest entries dropped (not blocking the logger).
    """

    def __init__(self, max_queue_size: int = 500):
        super().__init__()
        self.setLevel(logging.DEBUG)  # Capture all levels regardless of logger level
        self.max_queue_size = max_queue_size
        self._subscribers: dict[str, asyncio.Queue[str | None]] = {}
        # Ring buffer for recent log lines (for late-joining subscribers)
        self._recent: deque[str] = deque(maxlen=200)
        self._loop: asyncio.AbstractEventLoop | None = None

    def emit(self, record: logging.LogRecord) -> None:
        """Called by the logging framework for each record."""
        try:
            msg = self.format(record)
            self._recent.append(msg)
            # Push to all subscriber queues
            for queue in list(self._subscribers.values()):
                try:
                    queue.put_nowait(msg)
                except asyncio.QueueFull:
                    # Drop oldest entry to make room
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        queue.put_nowait(msg)
                    except asyncio.QueueFull:
                        pass
        except Exception:
            self.handleError(record)

    def subscribe(self) -> tuple[str, asyncio.Queue[str | None]]:
        """Create a new subscriber queue.

        Returns (subscriber_id, queue). Caller reads from the queue.
        Send None as sentinel to signal unsubscription.
        """
        sid = uuid4().hex
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=self.max_queue_size)
        self._subscribers[sid] = queue

        # Send recent history so late joiners see context
        for line in self._recent:
            try:
                queue.put_nowait(line)
            except asyncio.QueueFull:
                break

        return sid, queue

    def unsubscribe(self, subscriber_id: str) -> None:
        """Remove a subscriber queue."""
        queue = self._subscribers.pop(subscriber_id, None)
        if queue is not None:
            try:
                queue.put_nowait(None)  # sentinel
            except asyncio.QueueFull:
                pass

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def get_recent(self, count: int = 200) -> list[str]:
        """Return the last N log lines from the ring buffer."""
        return list(self._recent)[-count:]


class LogStreamManager:
    """Manages the LogStreamHandler lifecycle and provides convenience methods.

    Usage in server startup:
        log_manager = LogStreamManager(log_dir="logs")
        log_manager.attach_to_logger(app_logger)
    """

    def __init__(self, max_queue_size: int = 500):
        self.handler = LogStreamHandler(max_queue_size=max_queue_size)
        self.handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
        )

    def attach_to_logger(self, logger: logging.Logger) -> None:
        """Attach the stream handler to a specific logger."""
        logger.addHandler(self.handler)

    def attach_to_root_logger(self) -> None:
        """Attach the stream handler to the root logger (captures everything)."""
        logging.getLogger().addHandler(self.handler)

    def get_recent_logs(self, count: int = 200) -> list[str]:
        """Return the last N formatted log lines."""
        return self.handler.get_recent(count)

    @property
    def subscriber_count(self) -> int:
        return self.handler.subscriber_count
