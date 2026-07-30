"""Logger module.

Provides functionality for logger.logger.py.
"""

import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

install_rich_traceback()

# Module-level fallback crash log path (set by install_crash_handler).
# Written to directly when the logging system may be unavailable (shutdown/unraisable).
_CRASH_LOG_PATH: str = ""


def _write_fallback_crash(
    exc_type: type,
    exc_value: BaseException,
    exc_tb,
    tag: str = "Unhandled exception",
) -> None:
    """Write a crash directly to errors.log — bypasses the logging system entirely.

    Safe to call during interpreter shutdown (opens/append/closes each call).
    """
    path = _CRASH_LOG_PATH
    if not path:
        return
    try:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            f.write(f"[{now}] {tag}\n")
            traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
            f.write("\n")
    except Exception:  # noqa: BLE001, S110  # best-effort
        pass


def _resolve_default_log_dir() -> str:
    try:
        from src.config.loader import config

        return config.LOG_DIR
    except Exception:  # noqa: BLE001
        return "logs"


class DailyRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(
        self,
        filename,
        log_dir,
        log_filename_prefix,
        logger_name,
        *args,
        is_error_handler=False,
        **kwargs,
    ):
        self.log_dir = log_dir
        self.log_filename_prefix = log_filename_prefix
        self.logger_name = logger_name
        self.is_error_handler = is_error_handler
        super().__init__(filename, *args, **kwargs)

    def emit(self, record):
        current_date = datetime.now(timezone.utc).strftime("%Y_%m_%d")

        # Always use the main logger directory, even for errors
        current_log_dir = os.path.join(self.log_dir, self.logger_name, current_date)

        if self.is_error_handler:
            # Force filename to be errors.log for error handler
            current_filename = os.path.join(current_log_dir, "errors.log")
        else:
            current_filename = os.path.join(
                current_log_dir, f"{self.log_filename_prefix}{self.logger_name}.log"
            )

        # Normalize paths for consistent comparison across platforms
        current_filename_norm = os.path.normpath(current_filename)
        try:
            basefilename_norm = os.path.normpath(self.baseFilename)
        except AttributeError:
            basefilename_norm = None

        if basefilename_norm != current_filename_norm:
            # Close previous stream if it exists before opening a new one
            try:
                if hasattr(self, "stream") and self.stream:
                    self.stream.close()
            except Exception:  # noqa: BLE001, S110
                # best-effort stream cleanup during daily log file rollover
                pass

            self.baseFilename = current_filename_norm
            if not os.path.exists(current_log_dir):
                os.makedirs(current_log_dir, exist_ok=True)
            self.stream = self._open()

        super().emit(record)


class Logger(logging.Logger):
    def __init__(
        self,
        logger_name: str = "",
        log_filename_prefix: str = "",
        log_dir: str | None = None,  # type: ignore[arg-type]
        logger_debug: bool = False,
        console: Console | None = None,
    ) -> None:
        sanitized_name = logger_name.replace("/", "_").replace("\\", "_")

        level = logging.DEBUG if logger_debug else logging.INFO
        super().__init__(sanitized_name, level)
        self.propagate = False

        self.log_filename_prefix = log_filename_prefix

        if log_dir is None:
            self.log_dir = _resolve_default_log_dir()
        else:
            self.log_dir = log_dir

        self.date_format = "%d.%m.%Y %H:%M:%S"

        self._setup_logger(console=console)
        self.debug(
            "Logger %s initialized with log directory: %s", sanitized_name, self.log_dir
        )

    def _get_log_dir(self, current_date: str) -> str:
        # Simplified: no separate error directory logic needed
        log_dir = os.path.join(self.log_dir, self.name, current_date)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def _get_log_filename(self, log_dir: str, suffix: str = "") -> str:
        # Ensure we have a valid filename even if prefix or name are empty
        prefix = self.log_filename_prefix if self.log_filename_prefix else ""
        name = self.name if self.name else "default"
        return os.path.join(log_dir, f"{prefix}{name}{suffix}.log")

    def _plain_formatter(self) -> logging.Formatter:
        format_string = (
            "[{asctime}] {filename}.{funcName} - {message}"
            if self.level == logging.DEBUG
            else "[{asctime}] - {message}"
        )
        return logging.Formatter(format_string, datefmt=self.date_format, style="{")

    def _setup_logger(self, console: Console | None = None) -> None:
        current_date = datetime.now(timezone.utc).strftime("%Y_%m_%d")
        log_dir = self._get_log_dir(current_date)
        # Error log now lives in the same directory, so we reuse log_dir
        error_log_dir = log_dir

        if not self.handlers:
            self._add_console_handler(console=console)
            self._add_file_handler(log_dir)
            self._add_error_file_handler(error_log_dir)

    def _add_console_handler(self, console: Console | None = None):
        target_console = console or Console(color_system="auto", width=180)
        rich_handler = RichHandler(
            console=target_console, rich_tracebacks=False, omit_repeated_times=False
        )
        rich_handler.setLevel(self.level)
        self.addHandler(rich_handler)

    def _add_file_handler(self, log_dir):
        log_filename = self._get_log_filename(log_dir)
        file_handler = DailyRotatingFileHandler(
            log_filename,
            self.log_dir,
            self.log_filename_prefix,
            self.name,
            is_error_handler=False,
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8",
        )
        file_handler.setLevel(self.level)
        file_handler.setFormatter(self._plain_formatter())
        file_handler.namer = lambda name: name.replace(".log", "") + ".log"
        file_handler.rotator = lambda source, _dest: self._log_rotator(source)
        self.addHandler(file_handler)

    def _add_error_file_handler(self, error_log_dir):
        # We manually specify errors.log here, though the handler logic also enforces it
        error_log_filename = os.path.join(error_log_dir, "errors.log")
        error_file_handler = DailyRotatingFileHandler(
            error_log_filename,
            self.log_dir,
            self.log_filename_prefix,
            self.name,
            is_error_handler=True,
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8",
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(self._plain_formatter())
        error_file_handler.namer = lambda name: name.replace(".log", "") + ".log"
        error_file_handler.rotator = lambda source, _dest: self._log_rotator(source)
        self.addHandler(error_file_handler)

    def _log_rotator(self, source):
        new_date = datetime.now(timezone.utc).strftime("%Y_%m_%d")
        # _get_log_dir no longer accepts is_error, it returns the main directory
        new_dir = self._get_log_dir(new_date)
        new_file = os.path.join(new_dir, os.path.basename(source))
        open(new_file, "a", encoding="utf-8").close()

    def close(self) -> None:
        """Close all handlers and release resources."""
        for handler in self.handlers[:]:
            try:
                handler.close()
                self.removeHandler(handler)
            except Exception:  # noqa: BLE001, S110
                # best-effort handler close cleanup
                pass

    def install_crash_handler(self) -> None:
        """Install crash hooks to route ALL unhandled exceptions to errors.log.

        Covers three escape paths that bypass normal logging:

          1. sys.excepthook        — main-thread truly-unhandled exceptions
          2. threading.excepthook  — background thread crashes
          3. sys.unraisablehook    — __del__ / weakref exceptions at GC time
        """
        # Resolve the errors.log path this instance would write to, so the
        # fallback writer can open it directly when the logging system is
        # partially shut down or unavailable.
        current_date = datetime.now(timezone.utc).strftime("%Y_%m_%d")
        error_log_dir = self._get_log_dir(current_date)
        global _CRASH_LOG_PATH
        _CRASH_LOG_PATH = os.path.join(error_log_dir, "errors.log")

        logger_ref = self

        # ----------------------------------------------------------------
        # 1. sys.excepthook — main-thread truly-unhandled exceptions
        # ----------------------------------------------------------------
        def _handle_exception(exc_type, exc_value, exc_tb):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_tb)
                return
            _write_fallback_crash(
                exc_type, exc_value, exc_tb, "Unhandled exception (process crash)"
            )
            logger_ref.critical(
                "Unhandled exception (process crash)",
                exc_info=(exc_type, exc_value, exc_tb),
            )

        sys.excepthook = _handle_exception

        # ----------------------------------------------------------------
        # 2. threading.excepthook — background thread crashes
        # ----------------------------------------------------------------
        import threading

        def _handle_thread_exception(args):
            if args.exc_type is SystemExit:
                return
            if (
                args.exc_type is RuntimeError
                and "Event loop is closed" in str(args.exc_value)
                and "keep-alive-handler" in (args.thread.name if args.thread else "")
            ):
                return
            _write_fallback_crash(
                args.exc_type,
                args.exc_value,
                args.exc_traceback,
                f"Unhandled exception in thread '{args.thread.name if args.thread else 'unknown'}'",
            )
            logger_ref.critical(
                "Unhandled exception in thread '%s'",
                args.thread.name if args.thread else "unknown",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )

        threading.excepthook = _handle_thread_exception

        # ----------------------------------------------------------------
        # 3. sys.unraisablehook — __del__ / weakref at GC time
        #    Python prints these to stderr by default; we copy them to
        #    errors.log so they aren't lost when the terminal scrolls away.
        # ----------------------------------------------------------------
        _orig_unraisable = sys.unraisablehook

        def _handle_unraisable(args):
            _write_fallback_crash(
                args.exc_type,
                args.exc_value,
                args.exc_traceback,
                f"Unraisable exception — object={args.object!r}  msg={args.err_msg or ''}",
            )
            _orig_unraisable(args)

        sys.unraisablehook = _handle_unraisable
