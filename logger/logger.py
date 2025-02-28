import logging
import os
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback
from typing import Optional

install_rich_traceback()


class DailyRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, filename, log_dir, log_filename_prefix, logger_name, *args, **kwargs):
        self.log_dir = log_dir
        self.log_filename_prefix = log_filename_prefix
        self.logger_name = logger_name
        super().__init__(filename, *args, **kwargs)

    def emit(self, record):
        current_date = datetime.now().strftime("%Y_%m_%d")
        current_log_dir = os.path.join(self.log_dir, self.logger_name, current_date)
        current_filename = f"{current_log_dir}/{self.log_filename_prefix}{self.logger_name}.log"

        if self.baseFilename != current_filename:
            self.baseFilename = current_filename
            if not os.path.exists(current_log_dir):
                os.makedirs(current_log_dir, exist_ok=True)
            self.stream = self._open()

        super().emit(record)


class Logger(logging.Logger):
    def __init__(self, logger_name: str = '', log_filename_prefix: str = '', log_dir: str = 'logs',
                 logger_debug: bool = False) -> None:
        level = logging.DEBUG if logger_debug else logging.INFO
        super().__init__(logger_name, level)

        self.log_filename_prefix = log_filename_prefix
        main_project_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_dir = os.path.join(main_project_dir, log_dir)
        self.date_format = "%d.%m.%Y %H:%M:%S"
        self._logged_headers = set()
        self._last_header_time = datetime.now()
        self.thinking_mode = False
        self._setup_logger()
        self.debug(f"Logger {logger_name} initialized.")

    def custom_exception_hook(self, exctype, value, traceback):
        if exctype == KeyboardInterrupt:
            print("KeyboardInterrupt caught. Exiting gracefully.")
        else:
            self.error("Uncaught exception", exc_info=(exctype, value, traceback))
            sys.exit(1)

    def _get_log_dir(self, current_date: str, is_error: bool = False) -> str:
        base_dir = 'errors' if is_error else self.name
        log_dir = os.path.join(self.log_dir, base_dir, current_date)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def _get_log_filename(self, log_dir: str) -> str:
        return f"{log_dir}/{self.log_filename_prefix}{self.name}.log"

    def _create_rotating_handler(self, log_dir: str, level: int, is_error: bool = False) -> DailyRotatingFileHandler:
        log_filename = self._get_log_filename(log_dir)
        handler = DailyRotatingFileHandler(
            log_filename,
            self.log_dir,
            self.log_filename_prefix,
            self.name,
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        handler.setLevel(level)
        handler.setFormatter(self._plain_formatter())
        handler.namer = lambda name: name.replace(".log", "") + ".log"
        handler.rotator = lambda source, dest: self._log_rotator(source, is_error=is_error)
        return handler

    def _setup_logger(self) -> None:
        if self.handlers:
            return

        current_date = datetime.now().strftime("%Y_%m_%d")
        
        # Console handler
        self.console = Console(color_system="auto", width=180)
        rich_handler = RichHandler(console=self.console, rich_tracebacks=False, markup=True)
        rich_handler.setLevel(self.level)
        self.addHandler(rich_handler)

        # Regular file handler
        log_dir = self._get_log_dir(current_date)
        self.addHandler(self._create_rotating_handler(log_dir, self.level))

        # Error file handler
        error_log_dir = self._get_log_dir(current_date, is_error=True)
        self.addHandler(self._create_rotating_handler(error_log_dir, logging.ERROR, is_error=True))

    def stream_info(self, message: str, end: Optional[str] = None, flush: bool = False) -> None:
        if not message.strip():
            return

        # Add a minimum time threshold between headers (5 seconds)
        is_header = "=== " in message
        current_time = datetime.now()
        
        if is_header:
            time_diff = (current_time - self._last_header_time).total_seconds()
            if time_diff < 5 or message.strip() in self._logged_headers:
                return
            self._last_header_time = current_time
            self._logged_headers.add(message.strip())
            message = self._format_header_text(message)

        # Record for file logging
        record = logging.LogRecord(self.name, logging.INFO, "", 0, message, (), None)
        
        for handler in self.handlers:
            if isinstance(handler, RichHandler):
                if is_header:
                    self.console.print(message, end=end or "\n", markup=True)
                else:
                    print(message, end=end, flush=flush)
            elif isinstance(handler, DailyRotatingFileHandler):
                # Strip rich markup for file logging
                plain_message = message
                if "[bold cyan]" in plain_message:
                    plain_message = plain_message.replace("[bold cyan]", "").replace("[/bold cyan]", "")
                if "[bold green]" in plain_message:
                    plain_message = plain_message.replace("[bold green]", "").replace("[/bold green]", "")
                record.msg = plain_message
                handler.emit(record)

    def _plain_formatter(self) -> logging.Formatter:
        format_string = "[{asctime}] {filename}.{funcName} - {message}" if self.level == logging.DEBUG else "[{asctime}] - {message}"
        return logging.Formatter(format_string, datefmt=self.date_format, style="{")

    def _log_rotator(self, source, is_error=False):
        new_date = datetime.now().strftime("%Y_%m_%d")
        new_dir = self._get_log_dir(new_date, is_error=is_error)
        new_file = os.path.join(new_dir, os.path.basename(source))
        open(new_file, 'a').close()

    def _format_header_text(self, message: str) -> str:
        """Apply rich formatting to header text"""
        if "=== Thinking Process" in message:
            return message.replace("=== Thinking Process", "[bold cyan]=== Thinking Process[/bold cyan]")
        elif "=== Analysis Results" in message:
            return message.replace("=== Analysis Results", "[bold green]=== Analysis Results[/bold green]")
        return message

    def info(self, msg, *args, **kwargs):
        """Override info to support rich markup"""
        if "=== " in str(msg):
            msg = self._format_header_text(str(msg))
            kwargs["extra"] = kwargs.get("extra", {})
            kwargs["extra"]["markup"] = True
        super().info(msg, *args, **kwargs)
