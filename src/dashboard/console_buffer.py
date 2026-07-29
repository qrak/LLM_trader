"""In-memory ring buffer for console log lines — day-grouped, max 7 days.

Provides historical log access for the public read-only Live Console view.
Clears on restart (in-memory only). Subscribes to the existing LogStreamHandler
for zero-overhead log capture.
"""

import asyncio
from collections import deque
from datetime import datetime, timezone


class DayPage:
    """One day of log lines with a UTC date label."""

    __slots__ = ("date_str", "lines", "max_lines")

    def __init__(self, date_str: str, max_lines: int = 10000):
        """Create a day page for the given date string."""
        self.date_str = date_str
        self.lines: deque[str] = deque(maxlen=max_lines)
        self.max_lines = max_lines

    def append(self, line: str) -> None:
        """Append a log line to this day's page."""
        self.lines.append(line)

    def get_recent(self, count: int) -> list[str]:
        """Return the most recent N lines from this day."""
        return list(self.lines)[-count:]


class ConsoleBuffer:
    """Captures formatted log lines into a day-grouped ring buffer.

    Subscribes to LogStreamHandler and stores lines for up to max_days.
    Each day is a separate page accessible by offset (0=today, 1=yesterday, ...).
    """

    def __init__(self, max_days: int = 7, max_lines_per_day: int = 10000):
        self._max_days = max_days
        self._max_lines_per_day = max_lines_per_day
        self._days: deque[DayPage] = deque(maxlen=max_days)
        self._lock = asyncio.Lock()

    def _today_str(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def append(self, line: str) -> None:
        """Thread-safe: called from logging handler (non-async context)."""
        today = self._today_str()
        if not self._days or self._days[-1].date_str != today:
            self._days.append(DayPage(today, self._max_lines_per_day))
        self._days[-1].append(line)

    async def append_async(self, line: str) -> None:
        """Async-safe variant for use in asyncio contexts."""
        async with self._lock:
            self.append(line)

    async def consume_queue(self, queue: asyncio.Queue[str | None]) -> None:
        """Consume lines from a LogStreamHandler subscriber queue indefinitely."""
        while True:
            line = await queue.get()
            if line is None:
                break
            await self.append_async(line)

    def get_page(self, page: int) -> list[str] | None:
        """Return lines for day page offset (0=today, 1=yesterday, ...).

        Returns None if page is out of range.
        """
        if page < 0 or page >= len(self._days):
            return None
        idx = len(self._days) - 1 - page
        return list(self._days[idx].lines)

    def get_recent(self, count: int = 200) -> list[str]:
        """Return the most recent N lines across all days."""
        result: list[str] = []
        for day in reversed(self._days):
            needed = count - len(result)
            if needed <= 0:
                break
            result = day.get_recent(needed) + result
        return result[-count:]

    @property
    def page_count(self) -> int:
        """Number of available day pages."""
        return len(self._days)

    def page_info(self, page: int) -> dict | None:
        """Return metadata for a day page (date and line count)."""
        lines = self.get_page(page)
        if lines is None:
            return None
        idx = len(self._days) - 1 - page
        return {
            "page": page,
            "date": self._days[idx].date_str,
            "lines": len(lines),
            "label": "Today" if page == 0 else f"{self._days[idx].date_str}",
        }
