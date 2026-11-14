

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
from starlette.websockets import WebSocketDisconnect

from src.dashboard.log_stream import LogStreamHandler, LogStreamManager


class RecordingSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)


class FailingSocket:
    def __init__(self) -> None:
        self.attempts = 0

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.attempts += 1
        raise WebSocketDisconnect(1006)


@pytest.fixture
def log_manager() -> LogStreamManager:
    return LogStreamManager(max_queue_size=100)


def test_log_stream_backpressure_drops_oldest_line_and_caps_history():

    handler = LogStreamHandler(max_queue_size=2)
    sid, queue = handler.subscribe()
    for index in range(3):
        handler.emit(logging.LogRecord("test", logging.INFO, "", 0, f"burst {index}", (), None))
    assert queue.qsize() == 2
    assert "burst 1" in queue.get_nowait()
    assert "burst 2" in queue.get_nowait()
    handler.unsubscribe(sid)

    ring = LogStreamHandler(max_queue_size=250)
    for index in range(250):
        ring.emit(logging.LogRecord("test", logging.INFO, "", 0, f"ring {index}", (), None))
    late_id, late_queue = ring.subscribe()
    assert late_queue.qsize() == 200
    replayed = [late_queue.get_nowait() for _ in range(late_queue.qsize())]
    assert "ring 50" in replayed[0]
    assert "ring 249" in replayed[-1]
    ring.unsubscribe(late_id)


def test_handler_subscribe_emit_unsubscribe_lifecycle(log_manager):

    sid, queue = log_manager.handler.subscribe()
    assert type(sid) is str and sid
    assert type(queue) is asyncio.Queue
    assert queue.empty()

    log_manager.handler.emit(logging.LogRecord("test", logging.INFO, "", 0, "Hello world", (), None))
    line = queue.get_nowait()
    assert type(line) is str
    assert "Hello world" in line

    log_manager.handler.unsubscribe(sid)
    while not queue.empty():
        queue.get_nowait()
    log_manager.handler.emit(logging.LogRecord("test", logging.INFO, "", 0, "After unsub", (), None))
    assert queue.empty()


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

    late_id, late_queue = log_manager.handler.subscribe()
    late_lines = [late_queue.get_nowait() for _ in range(late_queue.qsize())]
    assert "Shared message" in late_lines[-1]
    log_manager.handler.unsubscribe(first_id)
    log_manager.handler.unsubscribe(second_id)
    log_manager.handler.unsubscribe(late_id)


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

    handler = LogStreamHandler(max_queue_size=10)
    sid, queue = handler.subscribe()
    handler.emit(logging.LogRecord("test", logging.INFO, "", 0, "doomed", (), None))

    socket = FailingSocket()
    with pytest.raises(WebSocketDisconnect) as excinfo:
        await handler.stream_to(queue, socket)
    assert excinfo.value.code == 1006
    assert socket.attempts == 1
    handler.unsubscribe(sid)
