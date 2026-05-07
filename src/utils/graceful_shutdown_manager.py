import asyncio
import signal
import sys
from typing import Optional, Callable

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
        confirmation_callback: Optional[Callable[[], bool]] = None
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
        if sys.platform == 'win32':
            # On Windows, let Ctrl+C propagate as KeyboardInterrupt so start.py can
            # await shutdown synchronously before the event loop is closed.
            return

        for sig in (signal.SIGINT, signal.SIGTERM):
            self.loop.add_signal_handler(sig, lambda s=sig, *args: self.handle_signal(s))

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
        except Exception as exc:
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
                self.logger.info("User confirmed shutdown. Initiating graceful shutdown...")
            else:
                print("User confirmed shutdown, initiating...")
            self._request_shutdown()
        else:
            if self.logger:
                self.logger.info("User cancelled shutdown. Continuing operation...")
            else:
                print("User cancelled shutdown. Continuing operation...")

    async def shutdown_gracefully(self):
        """Execute all registered shutdown callbacks and cancel pending tasks."""
        if self._shutting_down:
            return
        self._shutting_down = True

        if self.logger:
            self.logger.info("Performing graceful shutdown...")
        else:
            print("Performing graceful shutdown...")

        # Execute registered callbacks first
        if self._callbacks:
            if self.logger:
                self.logger.info("Executing %s shutdown callbacks...", len(self._callbacks))
            else:
                print(f"Executing {len(self._callbacks)} shutdown callbacks...")

            for callback in self._callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    error_msg = f"Error in shutdown callback {callback}: {e}"
                    if self.logger:
                        self.logger.error(error_msg)
                    else:
                        print(error_msg)

        pending_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
        if pending_tasks:
            msg = f"Cancelling {len(pending_tasks)} tasks..."
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)

            for task in pending_tasks:
                task.cancel()
            try:
                await asyncio.wait_for(asyncio.wait(pending_tasks), timeout=10.0)
            except asyncio.TimeoutError:
                task_details = []
                for t in pending_tasks:
                    if not t.done():
                        try:
                            # asyncio.Task has get_name() since Python 3.8
                            name = t.get_name()
                        except AttributeError:
                            name = str(t)
                        task_details.append(name)

                timeout_msg = f"Some tasks didn't complete in time: {task_details}"
                if self.logger:
                    self.logger.warning(timeout_msg)
                else:
                    print(timeout_msg)
        try:
            await asyncio.wait_for(self.loop.shutdown_asyncgens(), timeout=2.0)
        except (asyncio.TimeoutError, Exception) as e:
            err_msg = f"Error shutting down async generators: {e}"
            if self.logger:
                self.logger.error(err_msg)
            else:
                print(err_msg)
        
        # Final pause to allow background threads (e.g., Discord keep-alive handler) to fully terminate
        # before the event loop is closed. This prevents RuntimeError: Event loop is closed
        await asyncio.sleep(0.5)

    @staticmethod
    def show_exit_confirmation() -> bool:
        """
        Show a confirmation dialog before closing the application.

        Returns:
            True if user confirmed exit, False if they cancelled.
        """
        if not TKINTER_AVAILABLE:
            try:
                response = input("\nAre you sure you want to exit? (y/n): ").strip().lower()
                return response in ['y', 'yes']
            except (EOFError, KeyboardInterrupt):
                return True

        root = None
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            result = messagebox.askyesno(
                "Exit Confirmation",
                "Are you sure you want to close the Crypto Trading Bot application?",
                parent=root
            )
            return bool(result)
        except Exception as e:
            print(f"Warning: Could not show confirmation dialog: {e}. Proceeding with shutdown.")
            return True
        finally:
            if root is not None:
                try:
                    root.destroy()
                except Exception:
                    pass
