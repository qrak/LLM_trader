"""
Crypto Trading Bot - Entry Point
Automated trading with AI-powered decisions.
"""

import asyncio
import atexit
import hashlib
import logging
import os
import sys
import time
import warnings
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.analyzer.sentiment_analyst import RedditSentimentAnalyst
from src.app import BotServices, CryptoTradingBot
from src.composition.provisioners import ProvisioningMixin
from src.composition.startup_support import (
    build_startup_banner,
    show_error_dialog,
)
from src.config.loader import config
from src.logger.logger import Logger
from src.trading import (
    ExecutorHandler,
    PositionStatusMonitor,
)
from src.utils.graceful_shutdown_manager import GracefulShutdownManager

# pylint: disable=wrong-import-position

warnings.filterwarnings("ignore", category=SyntaxWarning, module="docopt")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="discord")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.genai")


class SingleInstanceLock:
    """Manages a single instance lock file to prevent multiple application instances."""

    def __init__(
        self, app_name: str = ".llm_trader.lock", logger: Logger | None = None
    ):
        self.lock_file_path = Path.home() / app_name
        self._lock_handle: int | None = None
        self._mutex_handle = None
        self.logger = logger or logging.getLogger(__name__)

    def _acquire_windows_mutex(self) -> bool:
        """Use a named mutex on Windows to guarantee single process instance."""
        try:
            import ctypes

            lock_key = hashlib.sha1(
                str(self.lock_file_path).encode("utf-8")
            ).hexdigest()[:16]
            mutex_name = f"Local\\LLMTraderSingleInstance_{lock_key}"
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[reportAttributeAccessIssue]
            handle = kernel32.CreateMutexW(None, False, mutex_name)
            if not handle:
                return True

            self._mutex_handle = handle
            ERROR_ALREADY_EXISTS = 183
            if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:  # type: ignore[reportAttributeAccessIssue]
                kernel32.CloseHandle(handle)
                self._mutex_handle = None
                return False
            return True
        except Exception:  # noqa: BLE001
            return True

    def _release_windows_mutex(self) -> None:
        if self._mutex_handle:
            try:
                import ctypes

                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[reportAttributeAccessIssue]
                kernel32.CloseHandle(self._mutex_handle)
            except Exception:
                self.logger.exception("Failed to release Windows mutex handle")
            self._mutex_handle = None

    def acquire(self) -> bool:
        """Attempt to acquire the lock. Returns True if successful."""
        try:
            if sys.platform == "win32" and not self._acquire_windows_mutex():
                return False

            self._lock_handle = os.open(
                str(self.lock_file_path), os.O_CREAT | os.O_RDWR
            )

            if sys.platform == "win32":
                import msvcrt

                try:
                    msvcrt.locking(self._lock_handle, msvcrt.LK_NBLCK, 1)
                except OSError:
                    self._release_windows_mutex()
                    return False
            else:
                import fcntl  # pylint: disable=import-error

                try:
                    fcntl.flock(self._lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return False

            atexit.register(self.release)
            return True

        except Exception as e:  # noqa: BLE001
            self._release_windows_mutex()
            print(f"Warning: Could not create lock file: {e}")
            return True

    def release(self) -> None:
        """Release the lock and cleanup."""
        if self._lock_handle is not None:
            try:
                if sys.platform == "win32":
                    import msvcrt

                    msvcrt.locking(self._lock_handle, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl  # pylint: disable=import-error

                    fcntl.flock(self._lock_handle, fcntl.LOCK_UN)
                os.close(self._lock_handle)
            except Exception:  # noqa: BLE001
                self.logger.warning("Failed to release lock")
            self._lock_handle = None

            try:
                self.lock_file_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                self.logger.warning("Failed to unlink lock file")

        self._release_windows_mutex()


RELOAD_EXIT_CODE = 42


class CompositionRoot(ProvisioningMixin):
    """Composition Root for the trading bot application.

    Responsible for building and wiring all dependencies following the
    Dependency Injection pattern before injecting them into CryptoTradingBot.
    """

    def __init__(self):
        self.config = config
        terminal_width = os.get_terminal_size().columns
        self.console = Console(width=terminal_width)
        self.logger = Logger(
            logger_name="Bot", logger_debug=config.LOGGER_DEBUG, console=self.console
        )
        self.logger.install_crash_handler()
        self.loop = None
        self.shutdown_manager: GracefulShutdownManager | None = None

    # pylint: disable=too-many-statements
    async def build_dependencies(self) -> dict:
        """Build all dependencies for the trading bot via segmented provisions."""
        start_time = time.perf_counter()

        self.console.clear()
        project_root = Path(__file__).parent.resolve()
        self.console.print(build_startup_banner(project_root))
        self.console.print()
        self.console.rule("[dim]Initializing...[/]")
        self.console.print()

        progress = Progress(
            SpinnerColumn(spinner_name="dots12", style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None, style="blue", complete_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(elapsed_when_finished=True),
            console=self.console,
            expand=True,
        )

        with progress:
            task = progress.add_task(
                "[cyan]Stage 1/9 — Core infrastructure (exchanges, sessions)...",
                total=9,
            )

            self._init_directories()
            infra = await self._provision_infrastructure()
            progress.update(task, completed=1)

            progress.update(
                task, description="[cyan]Stage 2/9 — Helper utilities & parsers..."
            )
            utils = self._provision_utilities()
            progress.update(task, completed=2)

            progress.update(
                task,
                description="[cyan]Stage 3/9 — Market data platforms (CoinGecko, DeFiLlama, CCXT)...",
            )
            apis = await self._provision_platforms(infra)
            progress.update(task, completed=3)

            progress.update(
                task, description="[cyan]Stage 4/9 — RAG news engine & taxonomy..."
            )
            rag = await self._provision_rag_layer(infra, apis, utils)
            progress.update(task, completed=4)

            progress.update(
                task,
                description="[cyan]Stage 5/9 — AI provider models & orchestrator...",
            )
            models = self._provision_model_layer(utils)
            progress.update(task, completed=5)

            progress.update(
                task,
                description="[cyan]Stage 6/9 — Technical analysis engine & patterns...",
            )
            analyzer = await self._provision_analyzer_layer(
                infra, apis, utils, rag, models
            )
            progress.update(task, completed=6)

            progress.update(
                task,
                description="[cyan]Stage 7/9 — Trading brain, memory, risk manager & guards...",
            )
            trading = self._provision_trading_layer(utils, models)
            progress.update(task, completed=7)

            progress.update(
                task,
                description="[cyan]Stage 8/9 — Notification channels (Discord) & maintenance...",
            )
            notifiers = await self._provision_notifiers(utils)
            await self._run_maintenance_tasks(trading["brain_service"])
            progress.update(task, completed=8)

            progress.update(
                task, description="[green]Stage 9/9 — Dashboard server & web admin..."
            )
            dashboard = self._provision_dashboard_layer(infra, utils, analyzer, trading)
            progress.update(task, completed=9)

        end_time = time.perf_counter()
        init_duration = end_time - start_time
        self.logger.info(
            "All 9 provisioning stages initialized successfully in %.2f seconds",
            init_duration,
        )

        self._display_startup_summary(apis, rag, trading, init_duration)

        deps = {
            "exchange_manager": infra["exchange_manager"],
            "market_analyzer": analyzer["engine"],
            "trading_strategy": trading["strategy"],
            "discord_notifier": notifiers["notifier"],
            "discord_task": notifiers["task"],
            "keyboard_handler": infra["keyboard_handler"],
            "rag_engine": rag,
            "coingecko_api": apis["coingecko"],
            "market_api": apis["market"],
            "alternative_me_api": apis["alternative_me"],
            "http_session": infra["session"],
            "persistence": trading["persistence"],
            "model_manager": models["manager"],
            "brain_service": trading["brain_service"],
            "statistics_service": trading["statistics_service"],
            "memory_service": trading["memory_service"],
            "exit_monitor": trading["exit_monitor"],
            "sentiment_analyst": RedditSentimentAnalyst(
                logger=self.logger,
            ),
            "ev_formatter": analyzer["ev_formatter"],
            "executor_handler": ExecutorHandler(
                persistence=trading["persistence"],
                config=self.config,
                logger=self.logger,
            ),
            "dashboard_server": dashboard["dashboard_server"],
            "dashboard_state": dashboard["dashboard_state"],
            "force_analysis_event": dashboard["force_analysis_event"],
        }

        return deps


    async def run_async(self):
        """Async entry point for the application."""

        def _asyncio_exception_handler(_loop, context):
            if self.shutdown_manager and self.shutdown_manager.is_shutting_down:
                return
            if _loop.is_closed():
                return

            exc = context.get("exception")
            msg = context.get("message", "Unknown asyncio error")
            if exc is not None:
                if isinstance(exc, KeyboardInterrupt):
                    self.logger.debug("Asyncio task KeyboardInterrupt: %s", msg)
                    return
                self.logger.error("Asyncio unhandled exception: %s", msg, exc_info=exc)
            else:
                self.logger.error("Asyncio error: %s", msg)

        if self.loop:
            self.loop.set_exception_handler(_asyncio_exception_handler)

        dependencies = await self.build_dependencies()

        dashboard_server = dependencies.pop("dashboard_server", None)
        force_analysis_event = dependencies.pop("force_analysis_event", None)

        def _create_position_monitor(bot: CryptoTradingBot) -> PositionStatusMonitor:
            return PositionStatusMonitor(
                logger=self.logger,
                config=self.config,
                persistence=dependencies["persistence"],
                trading_strategy=dependencies["trading_strategy"],
                exit_monitor=dependencies["exit_monitor"],
                notifier=dependencies["discord_notifier"],
                active_tasks=bot.active_tasks,
                is_running=lambda: bot.running,
                fetch_current_ticker=bot.fetch_current_ticker,
                interruptible_sleep=bot.interruptible_sleep,
                get_symbol=lambda: bot.current_symbol,
            )

        bot = CryptoTradingBot(
            BotServices(
                logger=self.logger,
                config=self.config,
                shutdown_manager=self.shutdown_manager,
                position_monitor_factory=_create_position_monitor,
                force_analysis_event=force_analysis_event,
                **dependencies,
            )
        )

        try:
            await bot.initialize()
            symbol = self.config.CRYPTO_PAIR
            timeframe = self.config.TIMEFRAME

            dashboard_running = False

            async def _toggle_dashboard():
                nonlocal dashboard_running
                if not dashboard_server:
                    return
                if dashboard_running:
                    self.logger.info("Dashboard: stopping (kill switch)...")
                    await dashboard_server.stop()
                    dashboard_running = False
                    self.logger.info("Dashboard stopped. Press 'd' to restart.")
                else:
                    self.logger.info("Dashboard: starting...")
                    await dashboard_server.start()
                    dashboard_running = True
                    self.logger.info(
                        "Dashboard live at http://localhost:%s",
                        self.config.DASHBOARD_PORT,
                    )

            bot.keyboard_handler.register_command(
                "d", _toggle_dashboard, "Toggle dashboard on/off"
            )

            if dashboard_server and self.config.DASHBOARD_ENABLED:
                await dashboard_server.start()
                dashboard_running = True
            elif not self.config.DASHBOARD_ENABLED:
                self.logger.info("Dashboard disabled (config). Press 'd' to start it.")

            if dashboard_server and self.shutdown_manager:
                self.shutdown_manager.register_shutdown_callback(dashboard_server.stop)

            await bot.run(symbol, timeframe)

        except asyncio.CancelledError:
            self.logger.info("Trading cancelled, shutting down...")
        finally:
            if dashboard_server:
                await dashboard_server.stop()

    def start(self) -> int:
        """Main entry point with clean shutdown delegation.

        Returns the process exit code: RELOAD_EXIT_CODE when the user requested
        an in-place reload (SHIFT+R), 0 otherwise.
        """
        single_instance_lock = SingleInstanceLock(logger=self.logger)

        if not single_instance_lock.acquire():
            shown = show_error_dialog(
                "Crypto Trading Bot",
                "Another instance of Crypto Trading Bot is already running.",
            )
            if not shown:
                print("Another instance of Crypto Trading Bot is already running.")
            sys.exit(1)

        if sys.platform == "win32":
            try:
                sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[reportAttributeAccessIssue]
                sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[reportAttributeAccessIssue]
            except Exception:  # noqa: S110, BLE001
                pass
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.shutdown_manager = GracefulShutdownManager(
            self.loop,
            logger=self.logger,
            confirmation_callback=GracefulShutdownManager.show_exit_confirmation,
        )
        self.shutdown_manager.setup_signal_handlers()

        exit_code = 0
        try:
            while True:
                try:
                    self.loop.run_until_complete(self.run_async())
                    break
                except KeyboardInterrupt:
                    print("\nKeyboardInterrupt received.")
                    if not GracefulShutdownManager.show_exit_confirmation():
                        print("Shutdown cancelled. Continuing operation...")
                        continue
                    self.loop.run_until_complete(
                        self.shutdown_manager.shutdown_gracefully()
                    )
                    break
                except Exception:
                    self.logger.exception(
                        "Unhandled exception in main loop — shutting down"
                    )
                    self.loop.run_until_complete(
                        self.shutdown_manager.shutdown_gracefully()
                    )
                    break

            if self.shutdown_manager.reload_requested:
                self.logger.info(
                    "Reload requested - completing graceful shutdown for in-place restart..."
                )
                try:
                    self.loop.run_until_complete(
                        self.shutdown_manager.shutdown_gracefully()
                    )
                except Exception as exc:  # noqa: BLE001
                    self.logger.error("Error during reload shutdown: %s", exc)
                exit_code = RELOAD_EXIT_CODE
        finally:
            try:
                if not self.loop.is_closed():
                    self.loop.close()
            except Exception as e:  # noqa: BLE001
                self.logger.error("Error closing event loop: %s", e)

        return exit_code


if __name__ == "__main__":
    try:
        exit_code = CompositionRoot().start()
    except BaseException:
        import traceback

        print("\n" + "=" * 60)
        print("FATAL: Unhandled exception — copy this error and press Enter")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        try:
            input("Press Enter to close...")
        except (EOFError, KeyboardInterrupt):
            pass
        raise
    sys.exit(exit_code)
