

import logging
import os
import shutil
import sys
import traceback
from datetime import datetime, timedelta, timezone
from typing import ClassVar, cast

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from rich.traceback import install as install_rich_traceback

from src.utils.time_utils import (
    DISPLAY_DAY_FORMAT,
    day_stamp,
    format_offset,
    local_stamp,
    local_timezone,
)

install_rich_traceback()

_CRASH_LOG_PATH: str = ""


class _ZonedFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        moment = datetime.fromtimestamp(record.created, tz=local_timezone())
        rendered = moment.strftime(datefmt or self.datefmt or "%d.%m.%Y %H:%M:%S")
        return f"{rendered} {format_offset(moment)}"


class DailyRotatingFileHandler(logging.FileHandler):
    RETENTION_DAYS = 30

    def __init__(
        self,
        filename,
        log_dir,
        log_filename_prefix,
        logger_name,
        *,
        is_error_handler=False,
        encoding=None,
    ):
        self.log_dir = log_dir
        self.log_filename_prefix = log_filename_prefix
        self.logger_name = logger_name
        self.is_error_handler = is_error_handler
        super().__init__(filename, encoding=encoding)

    def emit(self, record):
        try:
            current_date = day_stamp()
            target = os.path.normpath(self._target_path(current_date))
            if target != os.path.normpath(self.baseFilename):
                self._switch_to(target)
                self._prune_old_days(current_date)
        except Exception:  # noqa: BLE001
            self.handleError(record)
            return
        super().emit(record)

    def _target_path(self, current_date: str) -> str:
        name = "errors.log" if self.is_error_handler else f"{self.log_filename_prefix}{self.logger_name}.log"
        return os.path.join(self.log_dir, self.logger_name, current_date, name)

    def _switch_to(self, target: str) -> None:
        if self.stream:
            try:
                self.stream.close()
            except (OSError, ValueError):
                pass
            self.stream = None
        self.baseFilename = target
        os.makedirs(os.path.dirname(target), exist_ok=True)
        self.stream = self._open()

    def _prune_old_days(self, current_date: str) -> None:
        try:
            today = datetime.strptime(current_date, DISPLAY_DAY_FORMAT).replace(tzinfo=timezone.utc).date()
        except ValueError:
            return
        cutoff = today - timedelta(days=self.RETENTION_DAYS)
        root = os.path.join(self.log_dir, self.logger_name)
        for entry in os.listdir(root):
            try:
                day = datetime.strptime(entry, DISPLAY_DAY_FORMAT).replace(tzinfo=timezone.utc).date()
            except ValueError:
                continue
            if day < cutoff:
                shutil.rmtree(os.path.join(root, entry), ignore_errors=True)


class _ConsoleHandler(RichHandler):
    _PREFIX_STYLES: ClassVar[dict[str, str]] = {
        "DEBUG": "dodger_blue1",
        "INFO": "green3",
        "WARNING": "orange3",
        "ERROR": "bright_red",
        "CRITICAL": "bold #c00000",
    }

    def render_message(self, record: logging.LogRecord, message: str) -> Text:
        message = getattr(record, "display_message", None) or message
        prefix, separator, body = message.partition(" - ")
        return Text.assemble(
            (
                prefix + separator,
                self._PREFIX_STYLES.get(record.levelname, self._PREFIX_STYLES["INFO"]),
            ),
            cast(Text, super().render_message(record, body)),
        )


class Logger(logging.Logger):
    def __init__(
        self,
        logger_name: str = "",
        log_filename_prefix: str = "",
        log_dir: str | None = None,
        logger_debug: bool = False,
        console: Console | None = None,
    ) -> None:
        sanitized_name = logger_name.replace("/", "_").replace("\\", "_")

        level = logging.DEBUG if logger_debug else logging.INFO
        super().__init__(sanitized_name, level)
        self.propagate = False

        self.log_filename_prefix = log_filename_prefix

        if log_dir is None:
            self.log_dir = self._resolve_default_log_dir()
        else:
            self.log_dir = log_dir

        self.date_format = "%d.%m.%Y %H:%M:%S"

        self._setup_logger(console=console)
        self.debug(
            "Logger %s initialized with log directory: %s", sanitized_name, self.log_dir
        )

    @staticmethod
    def _resolve_default_log_dir() -> str:
        try:
            from src.config.loader import config

            return config.LOG_DIR
        except Exception:  # noqa: BLE001
            return "logs"

    @staticmethod
    def _write_fallback_crash(
        exc_type: type,
        exc_value: BaseException,
        exc_tb,
        tag: str = "Unhandled exception",
    ) -> None:

        path = _CRASH_LOG_PATH
        if not path:
            return
        try:
            dirname = os.path.dirname(path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                now = local_stamp(datetime.now(timezone.utc))
                f.write(f"[{now}] {tag}\n")
                traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
                f.write("\n")
        except Exception:  # noqa: BLE001, S110  # best-effort
            pass

    def _get_log_dir(self, current_date: str) -> str:
        log_dir = os.path.join(self.log_dir, self.name, current_date)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def _get_log_filename(self, log_dir: str, suffix: str = "") -> str:
        prefix = self.log_filename_prefix if self.log_filename_prefix else ""
        name = self.name if self.name else "default"
        return os.path.join(log_dir, f"{prefix}{name}{suffix}.log")

    def _plain_formatter(self) -> logging.Formatter:
        format_string = (
            "[{asctime}] [{levelname}] {filename}.{funcName} - {message}"
            if self.level == logging.DEBUG
            else "[{asctime}] - {message}"
        )
        return _ZonedFormatter(format_string, datefmt=self.date_format, style="{")

    def _console_formatter(self) -> logging.Formatter:
        return _ZonedFormatter(
            "[{asctime}] {levelname} {filename}:{lineno} - {message}",
            datefmt="%H:%M:%S",
            style="{",
        )

    def _setup_logger(self, console: Console | None = None) -> None:
        current_date = day_stamp()
        log_dir = self._get_log_dir(current_date)
        error_log_dir = log_dir

        if not self.handlers:
            self._add_console_handler(console=console)
            self._add_file_handler(log_dir)
            self._add_error_file_handler(error_log_dir)

    def _add_console_handler(self, console: Console | None = None):
        target_console = console or Console(color_system="auto")
        rich_handler = _ConsoleHandler(
            console=target_console,
            rich_tracebacks=False,
            show_time=False,
            show_level=False,
            show_path=False,
        )
        rich_handler.setFormatter(self._console_formatter())
        rich_handler.setLevel(logging.INFO)
        self.addHandler(rich_handler)

    def _add_file_handler(self, log_dir):
        log_filename = self._get_log_filename(log_dir)
        file_handler = DailyRotatingFileHandler(
            log_filename,
            self.log_dir,
            self.log_filename_prefix,
            self.name,
            is_error_handler=False,
            encoding="utf-8",
        )
        file_handler.setLevel(self.level)
        file_handler.setFormatter(self._plain_formatter())
        self.addHandler(file_handler)

    def _add_error_file_handler(self, error_log_dir):
        error_log_filename = os.path.join(error_log_dir, "errors.log")
        error_file_handler = DailyRotatingFileHandler(
            error_log_filename,
            self.log_dir,
            self.log_filename_prefix,
            self.name,
            is_error_handler=True,
            encoding="utf-8",
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(self._plain_formatter())
        self.addHandler(error_file_handler)

    def close(self) -> None:

        for handler in self.handlers[:]:
            try:
                handler.close()
                self.removeHandler(handler)
            except Exception:  # noqa: BLE001, S110
                pass

    def install_crash_handler(self) -> None:

        current_date = day_stamp()
        error_log_dir = self._get_log_dir(current_date)
        global _CRASH_LOG_PATH
        _CRASH_LOG_PATH = os.path.join(error_log_dir, "errors.log")

        logger_ref = self

        def _handle_exception(exc_type, exc_value, exc_tb):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_tb)
                return
            logger_ref._write_fallback_crash(
                exc_type, exc_value, exc_tb, "Unhandled exception (process crash)"
            )
            logger_ref.critical(
                "Unhandled exception (process crash)",
                exc_info=(exc_type, exc_value, exc_tb),
            )

        sys.excepthook = _handle_exception

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
            logger_ref._write_fallback_crash(
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

        _orig_unraisable = sys.unraisablehook

        def _handle_unraisable(args):
            logger_ref._write_fallback_crash(
                args.exc_type,
                args.exc_value,
                args.exc_traceback,
                f"Unraisable exception — object={args.object!r}  msg={args.err_msg or ''}",
            )
            _orig_unraisable(args)

        sys.unraisablehook = _handle_unraisable
