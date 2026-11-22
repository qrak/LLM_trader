

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

import pytest
from rich.console import Console
from starlette.websockets import WebSocketDisconnect

from src.dashboard.log_stream import LogStreamHandler, LogStreamManager
from src.logger.logger import Logger


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


def test_debug_is_file_only_and_other_levels_still_reach_public_stream(log_manager, tmp_path):

    terminal = io.StringIO()
    logger = Logger("test", log_dir=str(tmp_path), logger_debug=True, console=Console(file=terminal))
    log_manager.attach_to_logger(logger)
    sid, queue = log_manager.handler.subscribe()
    try:
        for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR):
            logger.log(level, "Level %d", level)
    finally:
        logger.close()

    main_log = next(tmp_path.glob("test/*/test.log")).read_text(encoding="utf-8")
    error_log = next(tmp_path.glob("test/*/errors.log")).read_text(encoding="utf-8")
    assert "[DEBUG]" in main_log and "Level 10" in main_log
    assert "Level 10" not in error_log and "Level 40" in error_log
    assert "Level 10" not in terminal.getvalue() and "Level 20" in terminal.getvalue()
    lines = [queue.get_nowait() for _ in range(queue.qsize())]
    assert [line.split("] ", 1)[1] for line in lines] == ["test: Level 20", "test: Level 30", "test: Level 40"]
    late_id, late_queue = log_manager.handler.subscribe()
    assert [late_queue.get_nowait() for _ in range(late_queue.qsize())] == lines
    log_manager.handler.unsubscribe(late_id)
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


def test_records_marked_skip_live_stream_stay_out_of_the_stream(log_manager):

    sid, queue = log_manager.handler.subscribe()
    hidden = logging.LogRecord("test", logging.INFO, "", 0, "Full prompt content: secret", (), None)
    hidden.skip_live_stream = True
    log_manager.handler.emit(hidden)
    assert queue.empty()

    log_manager.handler.emit(logging.LogRecord("test", logging.INFO, "", 0, "visible line", (), None))
    assert "visible line" in queue.get_nowait()

    late_id, late_queue = log_manager.handler.subscribe()
    replayed = [late_queue.get_nowait() for _ in range(late_queue.qsize())]
    assert not any("secret" in line for line in replayed)
    log_manager.handler.unsubscribe(sid)
    log_manager.handler.unsubscribe(late_id)


def test_a_display_message_replaces_the_streamed_body(log_manager):

    sid, queue = log_manager.handler.subscribe()
    record = logging.LogRecord("test", logging.INFO, "", 0, "Full prompt content: %s", ("x" * 5000,), None)
    record.display_message = "Full prompt content: <truncated>"
    log_manager.handler.emit(record)

    line = queue.get_nowait()
    assert line.endswith("test: Full prompt content: <truncated>")
    assert "xxxx" not in line
    log_manager.handler.unsubscribe(sid)


def test_a_stream_message_outranks_the_console_message(log_manager):

    sid, queue = log_manager.handler.subscribe()
    record = logging.LogRecord("test", logging.INFO, "", 0, "Full prompt content: %s", ("x" * 5000,), None)
    record.display_message = "Full prompt content: ## Market Data\n4h: +1%"
    record.stream_message = "Full prompt content: <truncated>"
    log_manager.handler.emit(record)

    line = queue.get_nowait()
    assert line.endswith("test: Full prompt content: <truncated>")
    assert "Market Data" not in line
    log_manager.handler.unsubscribe(sid)


def test_a_display_message_reaches_the_console_but_not_the_file(tmp_path):

    terminal = io.StringIO()
    logger = Logger("test", log_dir=str(tmp_path), console=Console(file=terminal))
    try:
        logger.info(
            "Full prompt content: %s",
            "x" * 5000,
            extra={"display_message": "Full prompt content: <truncated>"},
        )
    finally:
        logger.close()

    assert "Full prompt content: <truncated>" in terminal.getvalue()
    assert "xxxx" not in terminal.getvalue()
    assert "x" * 5000 in next(tmp_path.glob("test/*/test.log")).read_text(encoding="utf-8")
