import logging
import os
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

install_rich_traceback()


class DailyRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, filename, log_dir, log_filename_prefix, logger_name, is_error_handler=False, *args, **kwargs):
        self.log_dir = log_dir
        self.log_filename_prefix = log_filename_prefix
        self.logger_name = logger_name
        self.is_error_handler = is_error_handler
        super().__init__(filename, *args, **kwargs)

    def emit(self, record):
        current_date = datetime.now().strftime("%Y_%m_%d")
        
        if self.is_error_handler:
            current_log_dir = os.path.join(self.log_dir, "errors", current_date)
        else:
            current_log_dir = os.path.join(self.log_dir, self.logger_name, current_date)
            
        current_filename = os.path.join(current_log_dir, f"{self.log_filename_prefix}{self.logger_name}.log")

        # Normalize paths for consistent comparison across platforms
        current_filename_norm = os.path.normpath(current_filename)
        basefilename_norm = os.path.normpath(self.baseFilename) if getattr(self, 'baseFilename', None) else None

        if basefilename_norm != current_filename_norm:
            # Close previous stream if it exists before opening a new one
            try:
                if getattr(self, 'stream', None):
                    try:
                        self.stream.close()
                    except Exception:
                        pass
            except Exception:
                pass

            self.baseFilename = current_filename_norm
            if not os.path.exists(current_log_dir):
                os.makedirs(current_log_dir, exist_ok=True)
            self.stream = self._open()

        super().emit(record)


class Logger(logging.Logger):
    def __init__(self, logger_name: str = '', log_filename_prefix: str = '', log_dir: str = None,
                 logger_debug: bool = False) -> None:
        sanitized_name = logger_name.replace('/', '_').replace('\\', '_')
        
        level = logging.DEBUG if logger_debug else logging.INFO
        super().__init__(sanitized_name, level)

        self.log_filename_prefix = log_filename_prefix
        
        if log_dir is None:
            # Import config here to avoid circular imports
            from src.utils.loader import config
            self.log_dir = config.LOG_DIR
        else:
            self.log_dir = log_dir
            
        self.date_format = "%d.%m.%Y %H:%M:%S"

        self._setup_logger()
        self.debug(f"Logger {sanitized_name} initialized with log directory: {self.log_dir}")

    def custom_exception_hook(self, exctype, value, traceback):
        if exctype == KeyboardInterrupt:
            print("KeyboardInterrupt caught. Exiting gracefully.")
        else:
            self.error("Uncaught exception", exc_info=(exctype, value, traceback))
            sys.exit(1)

    def _get_log_dir(self, current_date: str, is_error: bool = False) -> str:
        if is_error:
            log_dir = os.path.join(self.log_dir, 'errors', current_date)
        else:
            log_dir = os.path.join(self.log_dir, self.name, current_date)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def _get_log_filename(self, log_dir: str, suffix: str = '') -> str:
        # Ensure we have a valid filename even if prefix or name are empty
        prefix = self.log_filename_prefix if self.log_filename_prefix else ""
        name = self.name if self.name else "default"
        return os.path.join(log_dir, f"{prefix}{name}{suffix}.log")

    def _plain_formatter(self) -> logging.Formatter:
        format_string = "[{asctime}] {filename}.{funcName} - {message}" if self.level == logging.DEBUG else "[{asctime}] - {message}"
        return logging.Formatter(format_string, datefmt=self.date_format, style="{")

    def _setup_logger(self) -> None:
        current_date = datetime.now().strftime("%Y_%m_%d")
        log_dir = self._get_log_dir(current_date)
        error_log_dir = self._get_log_dir(current_date, is_error=True)

        if not self.handlers:
            self._add_console_handler()
            self._add_file_handler(log_dir)
            self._add_error_file_handler(error_log_dir)

    def _add_console_handler(self):
        console = Console(color_system="auto", width=180)
        rich_handler = RichHandler(console=console, rich_tracebacks=False)
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
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        file_handler.setLevel(self.level)
        file_handler.setFormatter(self._plain_formatter())
        file_handler.namer = lambda name: name.replace(".log", "") + ".log"
        file_handler.rotator = lambda source, _dest: self._log_rotator(source, is_error=False)
        self.addHandler(file_handler)

    def _add_error_file_handler(self, error_log_dir):
        error_log_filename = self._get_log_filename(error_log_dir)
        error_file_handler = DailyRotatingFileHandler(
            error_log_filename,
            self.log_dir,
            self.log_filename_prefix,
            self.name,
            is_error_handler=True,
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(self._plain_formatter())
        error_file_handler.namer = lambda name: name.replace(".log", "") + ".log"
        error_file_handler.rotator = lambda source, _dest: self._log_rotator(source, is_error=True)
        self.addHandler(error_file_handler)

    def _log_rotator(self, source, is_error=False):
        new_date = datetime.now().strftime("%Y_%m_%d")
        new_dir = self._get_log_dir(new_date, is_error=is_error)
        new_file = os.path.join(new_dir, os.path.basename(source))
        open(new_file, 'a').close()
