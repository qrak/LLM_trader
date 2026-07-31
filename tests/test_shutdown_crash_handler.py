"""Shutdown crash handler test — verifies CancelledError is captured by errors.log.

Simulates the uvicorn/starlette shutdown scenario:
  1. A background task raises CancelledError during cancellation
  2. The event loop's exception handler should NOT log it as an error
  3. If it escapes, sys.excepthook should capture it in errors.log
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is on sys.path
_proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj_root))

from src.logger.logger import _CRASH_LOG_PATH, Logger, _write_fallback_crash


def test_fallback_writer_works():
    """Test 1: basic _write_fallback_crash writes to errors.log."""
    logger = Logger(logger_name="TestBot", log_dir="logs")
    current_date = datetime.now(timezone.utc).strftime("%Y_%m_%d")
    error_dir = os.path.join(logger.log_dir, logger.name, current_date)
    error_path = os.path.join(error_dir, "errors.log")
    # Set the global path
    import src.logger.logger as lg_mod

    lg_mod._CRASH_LOG_PATH = error_path
    # Write a test crash
    try:
        raise RuntimeError("TEST: fallback writer")
    except RuntimeError:
        _write_fallback_crash(*sys.exc_info(), "TEST: fallback writer")  # type: ignore[arg-type]
    # Read back
    if os.path.exists(error_path):
        with open(error_path) as f:
            content = f.read()
        assert "TEST: fallback writer" in content, f"FAIL: content={content[:200]}"
        assert "RuntimeError" in content, "FAIL: no RuntimeError in content"
        print("✅ Test 1 PASSED: _write_fallback_crash writes to errors.log")
    else:
        print(f"❌ Test 1 FAILED: {error_path} does not exist")


def test_excepthook_catches_cancelled_error():
    """Test 2: sys.excepthook writes CancelledError to errors.log."""
    logger = Logger(logger_name="TestBot", log_dir="logs")
    logger.install_crash_handler()
    path = _CRASH_LOG_PATH
    # Clear the file first
    if os.path.exists(path):
        os.remove(path)

    # Simulate CancelledError escaping to sys.excepthook
    old_hook = sys.excepthook
    try:
        exc = asyncio.CancelledError("TEST: cancelled from task")
        exc.__traceback__ = None
        old_hook(asyncio.CancelledError, exc, exc.__traceback__)
    finally:
        pass

    if os.path.exists(path):
        with open(path) as f:
            content = f.read()
        if "TEST: cancelled from task" in content:
            print("✅ Test 2 PASSED: excepthook writes CancelledError")
        else:
            print(f"❌ Test 2 FAILED: content={content[:200]}")
    else:
        print(f"❌ Test 2 FAILED: {path} does not exist")


def test_asyncio_exception_handler_suppresses_cancelled():
    """Test 3: Verify asyncio loop exception handler pattern works.

    Simulates: a background task is cancelled → CancelledError is sent
    to loop's exception handler → we check correct isinstance behavior.
    """
    logger = Logger(logger_name="TestBot", log_dir="logs")
    logger.install_crash_handler()

    cancelled_handled = False
    unknown_handled = False

    def handler(_loop, context):
        nonlocal cancelled_handled, unknown_handled
        exc = context.get("exception")
        if exc is not None:
            if isinstance(exc, asyncio.CancelledError):
                cancelled_handled = True
                return  # silently swallowed
            unknown_handled = True

    async def task_that_gets_cancelled():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise  # re-raise so the task sees it

    async def main():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(handler)
        t = asyncio.create_task(task_that_gets_cancelled())
        t.cancel()
        # Give the loop time to process the cancellation
        await asyncio.sleep(0.1)
        return cancelled_handled, unknown_handled, t

    loop = asyncio.new_event_loop()
    try:
        handled, unknown, task = loop.run_until_complete(main())
        if handled and not unknown:
            print("✅ Test 3 PASSED: asyncio handler Catches CancelledError")
        else:
            print(
                f"❌ Test 3 FAILED: handled={handled} unknown={unknown} task_cancelled={task.cancelled()}"
            )
    finally:
        loop.close()


def test_cancelled_error_in_shutdown_gracefully_style():
    """Test 4: Simulate shutdown_gracefully() task cancellation pattern.

    Multiple tasks, some raise CancelledError during cancellation,
    verify asyncio.wait() behavior.
    """
    logger = Logger(logger_name="TestBot", log_dir="logs")
    logger.install_crash_handler()

    async def lifespan_task():
        """Simulates starlette lifespan handler."""
        fake_queue = asyncio.Queue()
        try:
            await fake_queue.get()
        except asyncio.CancelledError:
            raise

    async def normal_task():
        """A task that just sleeps."""
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise

    async def graceful_shutdown_sim():
        loop = asyncio.get_running_loop()
        wait_raised = False

        # Create tasks
        t1 = asyncio.create_task(lifespan_task(), name="starlette-lifespan")
        t2 = asyncio.create_task(normal_task(), name="normal-task")
        await asyncio.sleep(0.01)  # let tasks start

        # Cancel them
        pending = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
        for t in pending:
            t.cancel()

        # Wait for them — this is what graceful_shutdown does
        try:
            await asyncio.wait_for(asyncio.wait(pending), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            wait_raised = True

        return wait_raised, t1.cancelled(), t2.cancelled()

    loop = asyncio.new_event_loop()
    try:
        wait_raised, t1_canc, t2_canc = loop.run_until_complete(graceful_shutdown_sim())
        if not wait_raised and t1_canc and t2_canc:
            print("✅ Test 4 PASSED: asyncio.wait() handles cancelled tasks cleanly")
        else:
            print(
                f"❌ Test 4 FAILED: wait_raised={wait_raised} t1_cancelled={t1_canc} t2_cancelled={t2_canc}"
            )
    finally:
        loop.close()


def test_loop_exception_handler_during_shutdown():
    """Test 5: Full simulation — cancelled task → handler catches CancelledError.

    This is the exact scenario happening in the bot:
    - Background task raises CancelledError from Queue.get()
    - Loop exception handler receives it
    - Handler should silently return (not log as error)
    """
    logger = Logger(logger_name="TestBot", log_dir="logs")
    logger.install_crash_handler()

    handler_calls = []

    def exception_handler(_loop, context):
        exc = context.get("exception")
        msg = context.get("message", "")
        handler_calls.append((type(exc).__name__ if exc else "None", msg[:80]))

        if exc is not None:
            if isinstance(exc, asyncio.CancelledError):
                return  # silently handle
            # Would log.error here in production

    async def starlette_lifespan_sim():
        """Simulate starlette lifespan → Queue.get() raises CancelledError."""
        q = asyncio.Queue()
        # This will raise CancelledError when the task is cancelled
        return await q.get()

    async def main():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(exception_handler)

        t = asyncio.create_task(starlette_lifespan_sim(), name="lifespan")
        await asyncio.sleep(0.01)
        t.cancel()

        # Wait for cancellation to propagate
        await asyncio.sleep(0.1)
        return handler_calls, t

    loop = asyncio.new_event_loop()
    try:
        calls, task = loop.run_until_complete(main())
        # If handler got CancelledError and our check worked, no error printed
        had_cancelled = any("CancelledError" in msg for _, msg in calls)
        if had_cancelled:
            print(
                "✅ Test 5 PASSED: loop handler receives CancelledError from Queue.get() and suppresses it"
            )
        else:
            print(f"❌ Test 5 FAILED: handler calls={calls}  task_cancelled={task.cancelled()}")

        # Also check: has any unhandled exception leaked?
        # It shouldn't, because the task machinery catches CancelledError
        if task.cancelled():
            print("   (task properly marked as cancelled)")
        else:
            print(f"   ⚠️ task state unexpected: {task._state}")
    finally:
        loop.close()


if __name__ == "__main__":
    print("=" * 60)
    print("SHUTDOWN CRASH HANDLER TESTS")
    print("=" * 60)
    test_fallback_writer_works()
    test_excepthook_catches_cancelled_error()
    test_asyncio_exception_handler_suppresses_cancelled()
    test_cancelled_error_in_shutdown_gracefully_style()
    test_loop_exception_handler_during_shutdown()
    print("=" * 60)
    print("All tests complete. Check errors.log if any used it.")
