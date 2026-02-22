"""Verify that async_save_* methods in PersistenceManager are non-blocking.

These tests mirror the pattern in test_chroma_async_nonblocking.py:
  - Simulate a slow blocking write (via time.sleep in a thread)
  - Confirm the asyncio event loop remains free for other coroutines
  - Confirm the async wrapper produces the same file output as the sync version
"""
import asyncio
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from src.managers.persistence_manager import PersistenceManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(tmp_dir: str) -> PersistenceManager:
    """Return a PersistenceManager writing to a temp directory."""
    logger = MagicMock()
    return PersistenceManager(logger=logger, data_dir=tmp_dir)


def _make_position() -> MagicMock:
    """Return a minimal Position-like mock."""
    pos = MagicMock()
    pos.entry_price = 50_000.0
    pos.stop_loss = 48_000.0
    pos.take_profit = 55_000.0
    pos.size = 0.01
    pos.entry_time.isoformat.return_value = "2026-01-01T00:00:00+00:00"
    pos.confidence = "HIGH"
    pos.direction = "LONG"
    pos.symbol = "BTC/USDC"
    pos.confluence_factors = []
    pos.entry_fee = 1.5
    pos.size_pct = 0.1
    pos.quote_amount = 500.0
    pos.atr_at_entry = 1200.0
    pos.volatility_level = "MEDIUM"
    pos.sl_distance_pct = 0.04
    pos.tp_distance_pct = 0.1
    pos.rr_ratio_at_entry = 2.5
    pos.adx_at_entry = 28.0
    pos.rsi_at_entry = 55.0
    pos.max_drawdown_pct = 0.0
    pos.max_profit_pct = 0.0
    return pos


# ---------------------------------------------------------------------------
# Non-blocking behaviour
# ---------------------------------------------------------------------------

class TestPersistenceAsyncNonBlocking:
    """Verify async_save_position does not block the asyncio event loop."""

    DELAY_MS = 100

    @pytest.mark.asyncio
    async def test_event_loop_not_blocked_during_save(self, tmp_path):
        """Other coroutines should progress while async_save_position runs.

        We patch the underlying open() call to sleep for DELAY_MS, simulating
        slow disk I/O. A concurrent coroutine increments a counter every 10 ms.
        If the event loop is blocked the counter stays at 0; if non-blocking
        it will increment multiple times.
        """
        manager = _make_manager(str(tmp_path))
        position = _make_position()

        progress = {"count": 0}
        delay_s = self.DELAY_MS / 1000

        def slow_open(*args, **kwargs):
            time.sleep(delay_s)  # simulate blocking disk write
            return open(*args, **kwargs)  # noqa (real open for cleanup)

        async def concurrent_task():
            for _ in range(20):
                await asyncio.sleep(0.01)  # 10 ms
                progress["count"] += 1

        with patch("builtins.open", side_effect=slow_open):
            await asyncio.gather(
                manager.async_save_position(position),
                concurrent_task(),
            )

        assert progress["count"] >= 3, (
            f"Event loop was blocked: concurrent task only progressed "
            f"{progress['count']} times during {self.DELAY_MS}ms write. "
            "Expected >= 3."
        )

    @pytest.mark.asyncio
    async def test_parallel_faster_than_sequential(self, tmp_path):
        """async_save_position + asyncio.sleep in parallel < 2× single operation.

        We slow down json.dump inside persistence_manager (targeted patch so
        pytest's own file I/O is not affected), then run a same-duration async
        coroutine concurrently.  Sequential would take ~2×DELAY_MS; parallel
        should take ~1×DELAY_MS.
        """
        manager = _make_manager(str(tmp_path))
        position = _make_position()
        delay_s = self.DELAY_MS / 1000
        real_json_dump = json.dump

        def slow_json_dump(obj, fp, **kwargs):
            time.sleep(delay_s)
            return real_json_dump(obj, fp, **kwargs)

        async def other_io():
            await asyncio.sleep(delay_s)

        with patch("src.managers.persistence_manager.json.dump", side_effect=slow_json_dump):
            start = time.perf_counter()
            await asyncio.gather(
                manager.async_save_position(position),
                other_io(),
            )
        elapsed = time.perf_counter() - start

        # Allow 80 % of 2× as upper bound (generous for CI)
        max_expected = (delay_s * 2) * 0.80
        assert elapsed < max_expected, (
            f"Parallel execution took {elapsed:.3f}s, expected < {max_expected:.3f}s. "
            "The event loop may still be blocked."
        )


# ---------------------------------------------------------------------------
# Correctness: same output as sync version
# ---------------------------------------------------------------------------

class TestPersistenceAsyncCorrectness:
    """Verify async wrappers produce identical file output to the sync methods."""

    @pytest.mark.asyncio
    async def test_async_save_position_same_output_as_sync(self, tmp_path):
        """async_save_position should write the same JSON as save_position."""
        sync_dir = tmp_path / "sync"
        async_dir = tmp_path / "async"
        sync_dir.mkdir()
        async_dir.mkdir()

        sync_mgr = _make_manager(str(sync_dir))
        async_mgr = _make_manager(str(async_dir))
        position = _make_position()

        sync_mgr.save_position(position)
        await async_mgr.async_save_position(position)

        sync_data = json.loads((sync_dir / "positions.json").read_text(encoding="utf-8"))
        async_data = json.loads((async_dir / "positions.json").read_text(encoding="utf-8"))

        assert sync_data == async_data

    @pytest.mark.asyncio
    async def test_async_save_position_none_deletes_file(self, tmp_path):
        """Saving None should delete the positions file, same as sync."""
        manager = _make_manager(str(tmp_path))
        position = _make_position()

        # Create file first
        await manager.async_save_position(position)
        assert (tmp_path / "positions.json").exists()

        # Now delete by saving None
        await manager.async_save_position(None)
        assert not (tmp_path / "positions.json").exists()

    @pytest.mark.asyncio
    async def test_async_save_last_analysis_time_same_output_as_sync(self, tmp_path):
        """async_save_last_analysis_time writes the same JSON as the sync version."""
        from datetime import datetime, timezone
        ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        sync_dir = tmp_path / "sync"
        async_dir = tmp_path / "async"
        sync_dir.mkdir()
        async_dir.mkdir()

        sync_mgr = _make_manager(str(sync_dir))
        async_mgr = _make_manager(str(async_dir))

        sync_mgr.save_last_analysis_time(ts)
        await async_mgr.async_save_last_analysis_time(ts)

        sync_data = json.loads((sync_dir / "last_analysis.json").read_text(encoding="utf-8"))
        async_data = json.loads((async_dir / "last_analysis.json").read_text(encoding="utf-8"))

        assert sync_data == async_data
