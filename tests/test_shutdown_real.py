"""Integration test: reproduce the exact shutdown crash.

Starts a minimal uvicorn server with signal capture (like the real dashboard),
then cancels it during shutdown to trigger the capture_signals → KeyboardInterrupt
→ starlette CancelledError chain.
"""

import asyncio
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

_proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj_root))

from src.logger.logger import Logger, _write_fallback_crash, _CRASH_LOG_PATH


async def test_real_scenario():
    """Reproduce the exact shutdown chain:
    1. Uvicorn server running with capture_signals()
    2. SIGINT received → capture_signals records it
    3. Task cancelled → __exit__ raises KeyboardInterrupt
    4. KeyboardInterrupt chains with CancelledError from lifespan
    5. Everything should be silently handled
    """
    logger = Logger(logger_name="TestBot", log_dir="logs")
    logger.install_crash_handler()

    errors_before = _count_error_entries()

    # Simulate uvicorn's capture_signals context manager
    captured_signal = None
    old_sigint = signal.signal(signal.SIGINT, lambda s, f: None)

    async def capture_signals_ctx():
        """Mimics uvicorn.server.capture_signals()."""
        nonlocal captured_signal

        def _handler(sig, frame):
            nonlocal captured_signal
            captured_signal = sig

        old = signal.signal(signal.SIGINT, _handler)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, old)
            if captured_signal is not None:
                signal.raise_signal(captured_signal)

    async def starlette_lifespan():
        """Mimics starlette lifespan handler → Queue.get()."""
        q = asyncio.Queue()
        return await q.get()

    async def uvicorn_serve():
        """Mimics uvicorn.server.serve() with capture_signals."""
        async def ctx():
            yield

        g = capture_signals_ctx()
        await g.__anext__()
        try:
            # Inside the context manager, run a lifespan-like task
            lifespan_task = asyncio.create_task(starlette_lifespan())
            await asyncio.sleep(0.02)
            # Now simulate the dashboard server being cancelled
            # (like what happens during graceful shutdown)
            return lifespan_task
        finally:
            try:
                await g.__anext__()
            except StopAsyncIteration:
                pass
            except Exception as e:
                print(f"[TEST] capture_signals __exit__ raised: {type(e).__name__}: {e}")

    # --- Run the simulation ---
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Simulate Ctrl+C
        signal.raise_signal(signal.SIGINT)

        # Run the uvicorn-like server
        lifespan_task = loop.run_until_complete(uvicorn_serve())

        # Now simulate what shutdown_gracefully does:
        # The server task (lifespan_task) is cancelled
        remaining = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
        print(f"[TEST] Remaining tasks before cancel: {len(remaining)}")

        # Cancel the lifespan task (like the server task)
        for t in remaining:
            t.cancel()

        # Wait for them to finish
        try:
            loop.run_until_complete(
                asyncio.wait_for(asyncio.wait(remaining), timeout=2.0)
            )
        except asyncio.TimeoutError:
            print("[TEST] task wait timed out (expected)")
        except asyncio.CancelledError:
            print("[TEST] CancelledError from wait (expected with capture_signals?)")
        except KeyboardInterrupt:
            print("[TEST] KeyboardInterrupt from wait (THIS is the bug source!)")
        except Exception as e:
            print(f"[TEST] Unexpected exception from wait: {type(e).__name__}: {e}")

        print(f"[TEST] Shutdown complete. Checking errors.log...")
    finally:
        try:
            loop.close()
        except Exception:
            pass

    # Restore signal handler
    signal.signal(signal.SIGINT, old_sigint)

    # Check errors.log for any new entries
    errors_after = _count_error_entries()
    new_errors = errors_after - errors_before
    if new_errors > 0:
        print(f"❌ {new_errors} NEW error entries in errors.log after test")
        _print_last_errors(new_errors)
    else:
        print("✅ No new errors in errors.log")


def _count_error_entries():
    path = _CRASH_LOG_PATH or "logs/TestBot/2026_07_29/errors.log"
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return f.read().count("Traceback")


def _print_last_errors(n):
    path = _CRASH_LOG_PATH or "logs/TestBot/2026_07_29/errors.log"
    if not os.path.exists(path):
        return
    with open(path) as f:
        lines = f.readlines()
    # Find the last N entries
    traces = []
    for line in lines:
        if "Traceback" in line:
            traces.append(line.strip())
    for t in traces[-n:]:
        print(f"  {t}")


if __name__ == "__main__":
    print("=" * 60)
    print("REAL SHUTDOWN CRASH SIMULATION")
    print("=" * 60)
    asyncio.run(test_real_scenario())
