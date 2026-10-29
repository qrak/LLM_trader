

import pytest

from src.utils import profiler


class NoLogger:
    def sync_work(self, value: int) -> int:
        return value * 2

    async def async_work(self, value: int) -> int:
        return value * 3


NoLogger.sync_work = profiler.profile_performance(NoLogger.sync_work)
NoLogger.async_work = profiler.profile_performance(NoLogger.async_work)


@pytest.fixture(autouse=True)
def debug_enabled(monkeypatch):

    monkeypatch.setattr(profiler, "_logger_debug_enabled", lambda: True)


def test_sync_profiled_method_without_logger_returns_result():
    assert NoLogger().sync_work(21) == 42


@pytest.mark.asyncio
async def test_async_profiled_method_without_logger_returns_result():
    assert await NoLogger().async_work(14) == 42


def test_logger_still_receives_the_timing_line(monkeypatch):
    class Recorder:
        def __init__(self) -> None:
            self.messages: list[str] = []
            self.logger = self

        def debug(self, message: str) -> None:
            self.messages.append(message)

        def work(self, value: int) -> int:
            return value

    Recorder.work = profiler.profile_performance(Recorder.work)
    recorder = Recorder()

    assert recorder.work(1) == 1
    assert recorder.messages and recorder.messages[0].startswith("Performance: Recorder.work")
