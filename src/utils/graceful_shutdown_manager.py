"""Graceful Shutdown Manager module.

Provides functionality for utils.graceful_shutdown_manager.py.
"""

import asyncio
import signal
import sys
import warnings
from collections.abc import Callable

try:
    import tkinter as tk
    from tkinter import messagebox

    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False


class GracefulShutdownManager:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        logger=None,
        confirmation_callback: Callable[[], bool] | None = None,
    ):
        self.loop = loop
        self.logger = logger
        self.confirmation_callback = confirmation_callback
        self._callbacks = []
        self._shutting_down = False

    @property
    def is_shutting_down(self) -> bool:
        """Return whether graceful shutdown is already in progress."""
        return self._shutting_down

    def setup_signal_handlers(self):
        if sys.platform == "win32":
            # On Windows, let Ctrl+C propagate as KeyboardInterrupt so start.py can
            # await shutdown synchronously before the event loop is closed.
            return

        for sig in (signal.SIGINT, signal.SIGTERM):
            self.loop.add_signal_handler(sig, lambda s=sig: self.handle_signal(s))

    def _request_shutdown(self):
        if self.loop.is_running() and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(
                lambda: self.loop.create_task(self.shutdown_gracefully())
            )

    def _confirm_shutdown(self) -> bool:
        if not self.confirmation_callback:
            return True
        try:
            return bool(self.confirmation_callback())
        except Exception as exc:  # noqa: BLE001
            if self.logger:
                self.logger.warning("Confirmation callback failed: %s", exc)
            return True

    def register_shutdown_callback(self, callback):
        """Register a callback to be executed during graceful shutdown."""
        if asyncio.iscoroutinefunction(callback):
            self._callbacks.append(callback)
        else:
            self._callbacks.append(callback)

    def handle_signal(self, sig: int):
        if self._shutting_down:
            return

        if self.logger:
            self.logger.info("Signal %s received. Asking for confirmation...", sig)
        else:
            print(f"Received signal {sig}, asking for confirmation...")

        if self._confirm_shutdown():
            if self.logger:
                self.logger.info(
                    "User confirmed shutdown. Initiating graceful shutdown..."
                )
            else:
                print("User confirmed shutdown, initiating...")
            self._request_shutdown()
        elif self.logger:
            self.logger.info("User cancelled shutdown. Continuing operation...")
        else:
            print("User cancelled shutdown. Continuing operation...")

    async def shutdown_gracefully(self):
        """Execute all registered shutdown callbacks and drain the event loop."""
        if self._shutting_down:
            return
        self._shutting_down = True

        if self.logger:
            self.logger.info("Performing graceful shutdown...")
        else:
            print("Performing graceful shutdown...")

        # Execute registered callbacks first (each cleans up its own tasks)
        if self._callbacks:
            if self.logger:
                self.logger.info(
                    "Executing %s shutdown callbacks...", len(self._callbacks)
                )
            else:
                print(f"Executing {len(self._callbacks)} shutdown callbacks...")

            for callback in self._callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:  # noqa: BLE001
                    error_msg = f"Error in shutdown callback {callback}: {e}"
                    if self.logger:
                        self.logger.error(error_msg)
                    else:
                        print(error_msg)

        # Let remaining tasks (dashboard server, uvicorn) drain naturally
        # instead of forcefully cancelling them. Cancelling the uvicorn
        # server task triggers capture_signals().__exit__ which raises
        # KeyboardInterrupt → chained CancelledError in Py3.13.
        # The event loop will clean them up on close.
        remaining = [
            t
            for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()
        ]
        if remaining:
            if self.logger:
                self.logger.info(
                    "Draining %s remaining tasks (no cancellation)...", len(remaining)
                )
            # Give server tasks a moment to respond to the shutdown signal
            # (sent via the callbacks above)
            await asyncio.sleep(0.2)

        # Shut down async generators
        try:
            await asyncio.wait_for(self.loop.shutdown_asyncgens(), timeout=2.0)
        except (asyncio.TimeoutError, Exception) as e:  # noqa: BLE001
            err_msg = f"Error shutting down async generators: {e}"
            if self.logger:
                self.logger.error(err_msg)
            else:
                print(err_msg)

        # ------------------------------------------------------------------
        # Transport & kaleido cleanup
        # ------------------------------------------------------------------
        # 1) Shut down kaleido sync server if it was ever started (Plotly
        #    chart export spawns a subprocess that leaks a BaseSubprocessTransport
        #    on Python 3.13+).
        try:
            import kaleido as _kl  # type: ignore[import-untyped]

            _kl.stop_sync_server(silence_warnings=True)
            if self.logger:
                self.logger.debug("Kaleido sync server stopped")
        except (ImportError, AttributeError, Exception):  # noqa: S110, BLE001
            pass

        # 2) Run the event loop a few extra iterations so pending transport
        #    __del__ callbacks (aiohttp, kaleido subprocess, chromadb) are
        #    drained before the loop is closed. This avoids
        #    _ProactorBasePipeTransport / BaseSubprocessTransport ResourceWarning.
        try:
            for _ in range(5):
                self.loop.call_soon(lambda: None)
                await asyncio.sleep(0.01)
        except Exception:  # noqa: S110, BLE001
            pass

        # 3) Suppress asyncio transport ResourceWarning noise from
        #    dependencies (kaleido, chromadb, discord.py) that may still
        #    have unclosed transports after explicit cleanup.
        warnings.filterwarnings(
            "ignore",
            category=ResourceWarning,
            message="unclosed transport",
        )

        # Final pause to allow background threads (e.g., Discord keep-alive
        # handler) to fully terminate before the event loop is closed.
        await asyncio.sleep(0.5)

    @staticmethod
    def _prompt_exit_confirmation() -> bool:
        try:
            response = input("\nAre you sure you want to exit? (y/n): ").strip().lower()
            return response in ["y", "yes"]
        except (EOFError, KeyboardInterrupt):
            return True

    @staticmethod
    def _is_headless() -> bool:
        """Detect headless environments: no display server + no interactive terminal.

        Returns True when there is no user to interact with — the process is
        running under systemd, Wired, Docker, or an unattended terminal
        where a blocking prompt would stall the shutdown indefinitely.
        """
        import os as _os

        return not _os.environ.get("DISPLAY") and not sys.stdin.isatty()

    @staticmethod
    def show_exit_confirmation() -> bool:
        """Show a confirmation dialog before closing the application.

        In headless environments (no DISPLAY + non-TTY stdin) skips all prompts
        and proceeds with shutdown immediately so systemd / Wired / Docker can
        recycle the process without blocking.

        In interactive environments, falls back to a terminal prompt when a GUI
        dialog is unavailable.

        Returns:
            True if user confirmed exit or environment is unattended, False if cancelled.
        """
        if GracefulShutdownManager._is_headless():
            return True

        if not TKINTER_AVAILABLE:
            return GracefulShutdownManager._prompt_exit_confirmation()

        root = None
        try:
            root = tk.Tk()  # type: ignore[reportPossiblyUnboundVariable]
            root.withdraw()
            root.attributes("-topmost", True)
            result = messagebox.askyesno(  # type: ignore[reportPossiblyUnboundVariable]
                "Exit Confirmation",
                "Are you sure you want to close the Crypto Trading Bot application?",
                parent=root,
            )
            return bool(result)
        except Exception as e:  # noqa: BLE001
            print(
                f"Warning: Could not show confirmation dialog: {e}. Falling back to terminal prompt."
            )
            return GracefulShutdownManager._prompt_exit_confirmation()
        finally:
            if root is not None:
                try:
                    root.destroy()
                except Exception:  # noqa: BLE001, S110 # best-effort cleanup
                    pass
