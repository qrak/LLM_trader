import configparser
import hashlib
import io
import math
import os
import re
import stat
import tempfile
import threading
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from src.config.loader import (
    CONFIG_INI_PATH,
    KEYS_ENV_PATH,
    ROOT_DIR,
    VALID_EXIT_TYPES,
    VALID_MODEL_VERBOSITIES,
    VALID_PROVIDERS,
    Config,
)
from src.utils.timeframe_validator import TimeframeValidator

ASSIGNMENT = re.compile(r"^(\s*[^=\r\n#;]+?=\s*)(.*)$")
SECTION = re.compile(r"^\s*\[([^\]]+)\]\s*$")
ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
TRUE_VALUES = {"true", "yes", "on", "1"}
FALSE_VALUES = {"false", "no", "off", "0"}


class SettingsError(Exception):
    """A settings edit the file model refuses, with a user-facing message."""

    def __init__(
        self,
        message: str,
        section: str | None = None,
        key: str | None = None,
        status: int = 400,
    ):
        super().__init__(message)
        self.section = section
        self.key = key
        self.status = status


class SettingsEditor:
    """Reads, validates and atomically rewrites config.ini and keys.env."""

    SECTION_TITLES = {  # noqa: RUF012
        "general": "General",
        "ai_providers": "AI providers",
        "model_config": "Model configuration",
        "risk_management": "Risk management",
        "executor_api": "Executor API",
        "demo_trading": "Paper trading",
        "dashboard": "Dashboard",
        "exchanges": "Exchanges",
        "research_team": "Research team",
        "social_sentiment": "Social sentiment",
        "rag": "News and RAG",
        "prompt_budget": "Prompt budget",
        "model_context_windows": "Model context windows",
        "directories": "Directories",
        "debug": "Debug",
        "cooldowns": "Cooldowns",
        "circuit_breaker": "Circuit breaker",
        "decisions_gate": "Decisions gate",
    }

    ENUMS = {  # noqa: RUF012
        "ai_providers": {"provider": tuple(sorted(VALID_PROVIDERS))},
        "general": {
            "timeframe": tuple(TimeframeValidator.SUPPORTED_TIMEFRAMES),
            "market_type": ("spot", "futures"),
            "entry_order_type": ("market", "limit"),
        },
        "risk_management": {
            "stop_loss_type": tuple(sorted(VALID_EXIT_TYPES)),
            "take_profit_type": tuple(sorted(VALID_EXIT_TYPES)),
        },
        "model_config": {
            "model_verbosity": tuple(sorted(VALID_MODEL_VERBOSITIES)),
            "google_thinking_level": ("low", "medium", "high"),
        },
        "decisions_gate": {
            "mode": ("off", "shadow", "size"),
        },
    }

    REQUIRED = Config.REQUIRED_CONFIG_KEYS

    def __init__(
        self,
        config_path: str | Path | None = None,
        keys_path: str | Path | None = None,
        keys_example_path: str | Path | None = None,
        logger: Any = None,
    ):
        self.config_path = Path(config_path) if config_path is not None else CONFIG_INI_PATH
        self.keys_path = Path(keys_path) if keys_path is not None else KEYS_ENV_PATH
        self.keys_example_path = (
            Path(keys_example_path)
            if keys_example_path is not None
            else ROOT_DIR / "keys.env.example"
        )
        self.logger = logger
        self._write_lock = threading.Lock()

    def read_config(self) -> dict[str, Any]:
        text = self._read_text(self.config_path, "config/config.ini")
        return {
            "exists": True,
            "etag": self._etag(text),
            "sections": self._parse_config(text),
        }

    def update_config(self, etag: str, updates: list[dict[str, str]]) -> dict[str, Any]:
        with self._write_lock:
            text = self._read_text(self.config_path, "config/config.ini")
            if self._etag(text) != etag:
                raise SettingsError(
                    "config/config.ini changed on disk; reload before saving.",
                    status=409,
                )
            sections = self._parse_config(text)
            index = {
                (item["name"], entry["key"]): entry
                for item in sections
                for entry in item["keys"]
            }
            normalized: list[tuple[str, str, str, dict[str, Any]]] = []
            for update in updates:
                section = update["section"]
                key = update["key"]
                entry = index.get((section, key))
                if entry is None:
                    raise SettingsError(f"Unknown config key: [{section}] {key}", section, key)
                value = self._normalize_value(section, entry, update["value"])
                normalized.append((section, key, value, entry))
            staged, new_text = self._stage_config(text, normalized)
            self._replace_file(self.config_path, staged, "config/config.ini")
            if self.logger:
                self.logger.info("Settings desk: config.ini updated (%d keys)", len(normalized))
            return {
                "etag": self._etag(new_text),
                "updated": [
                    {"section": section, "key": key, "value": value}
                    for section, key, value, _entry in normalized
                ],
                "restart_required": True,
            }

    def read_keys(self) -> dict[str, Any]:
        exists = self.keys_path.exists()
        text = self._read_text(self.keys_path, "keys.env") if exists else ""
        file_values = self._dotenv(text)
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for entry in self._template_entries():
            rows.append(
                self._key_row(entry["key"], entry["help"], entry["inline_help"], file_values)
            )
            seen.add(entry["key"])
        for entry in self._env_entries(text):
            if entry["key"] in seen:
                continue
            rows.append(
                self._key_row(entry["key"], entry["help"], entry["inline_help"], file_values)
            )
            seen.add(entry["key"])
        return {
            "exists": exists,
            "etag": self._etag(text) if exists else None,
            "template_exists": self.keys_example_path.exists(),
            "keys": rows,
        }

    def update_key(self, etag: str | None, key: str, value: str) -> dict[str, Any]:
        with self._write_lock:
            if not self.keys_path.exists():
                raise SettingsError(
                    "keys.env does not exist; create it from the template first.",
                    key=key,
                    status=409,
                )
            text = self._read_text(self.keys_path, "keys.env")
            if etag is None or self._etag(text) != etag:
                raise SettingsError(
                    "keys.env changed on disk; reload before saving.",
                    key=key,
                    status=409,
                )
            allowed = {entry["key"] for entry in self._template_entries()}
            allowed.update(entry["key"] for entry in self._env_entries(text))
            if key not in allowed:
                raise SettingsError(
                    f"Unknown key {key}; add it to keys.env.example first.",
                    key=key,
                )
            normalized = self._normalize_secret(key, value)
            new_text = self._replace_env_value(text, key, normalized)
            self._write_atomic(self.keys_path, new_text, "keys.env")
            row = self._key_row(key, "", "", self._dotenv(new_text))
            if self.logger:
                self.logger.info("Settings desk: %s updated", key)
            return {
                "etag": self._etag(new_text),
                "key": key,
                "set": row["set"],
                "source": row["source"],
            }

    def create_keys_file(self) -> dict[str, bool]:
        if self.keys_path.exists():
            return {"created": False}
        if not self.keys_example_path.exists():
            raise SettingsError("keys.env.example is missing.", status=409)
        try:
            data = self.keys_example_path.read_bytes()
        except OSError as exc:
            raise SettingsError(f"Could not read keys.env.example: {exc}", status=500) from exc
        try:
            with self.keys_path.open("xb") as handle:
                handle.write(data)
        except FileExistsError:
            return {"created": False}
        except OSError as exc:
            raise SettingsError(f"Could not create keys.env: {exc}", status=500) from exc
        try:
            os.chmod(self.keys_path, 0o600)
        except OSError:
            pass
        if self.logger:
            self.logger.info("Settings desk: keys.env created from the template")
        return {"created": True}

    def _parse_config(self, text: str) -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        help_lines: list[str] = []
        seen: set[tuple[str, str]] = set()
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                help_lines = []
                continue
            if stripped.startswith(("#", ";")):
                help_lines.append(stripped[1:].strip())
                continue
            section_match = SECTION.match(line)
            if section_match:
                name = section_match.group(1).strip()
                current = {
                    "name": name,
                    "title": self.SECTION_TITLES.get(
                        name, name.replace("_", " ").title()
                    ),
                    "keys": [],
                }
                sections.append(current)
                help_lines = []
                continue
            assignment = ASSIGNMENT.match(line)
            if assignment and current is not None:
                left = assignment.group(1)
                key = left.split("=", 1)[0].strip()
                value, inline_help = self._split_inline(assignment.group(2))
                marker = (current["name"], key)
                if marker in seen:
                    raise SettingsError(
                        f"Duplicate option [{current['name']}] {key}",
                        current["name"],
                        key,
                    )
                seen.add(marker)
                current["keys"].append(
                    self._entry(current["name"], key, value, inline_help, help_lines)
                )
                help_lines = []
                continue
            help_lines = []
        return sections

    def _entry(
        self,
        section: str,
        key: str,
        value: str,
        inline_help: str,
        help_lines: list[str],
    ) -> dict[str, Any]:
        options = self.ENUMS.get(section, {}).get(key)
        kind = "enum" if options is not None else self._value_type(value)
        return {
            "key": key,
            "value": value,
            "type": kind,
            "options": list(options) if options is not None else None,
            "required": key in self.REQUIRED.get(section, ()),
            "help": "\n".join(help_lines).strip(),
            "inline_help": inline_help,
        }

    @staticmethod
    def _value_type(value: str) -> str:
        lowered = value.lower()
        if lowered in TRUE_VALUES or lowered in FALSE_VALUES:
            return "bool"
        if re.fullmatch(r"[+-]?\d+", value):
            return "int"
        try:
            number = float(value)
        except ValueError:
            number = None
        if number is not None and math.isfinite(number):
            return "float"
        if "," in value:
            return "list"
        return "string"

    @staticmethod
    def _split_inline(raw: str) -> tuple[str, str]:
        for index, character in enumerate(raw):
            if character in "#;" and (index == 0 or raw[index - 1].isspace()):
                return raw[:index].strip(), raw[index:].strip()
        return raw.strip(), ""

    @staticmethod
    def _split_raw(raw: str) -> tuple[str, str]:
        for index, character in enumerate(raw):
            if character in "#;" and (index == 0 or raw[index - 1].isspace()):
                return raw[:index], raw[index:]
        return raw, ""

    def _normalize_value(self, section: str, entry: dict[str, Any], raw: str) -> str:
        key = entry["key"]
        value = str(raw).strip()
        if "\n" in value or "\r" in value:
            raise SettingsError("A value cannot contain a newline.", section, key)
        if any(ord(character) < 32 for character in value):
            raise SettingsError("A value cannot contain control characters.", section, key)
        if len(value) > 4000:
            raise SettingsError("A value cannot exceed 4000 characters.", section, key)
        kind = entry["type"]
        if kind == "bool":
            lowered = value.lower()
            if lowered in TRUE_VALUES:
                return "true"
            if lowered in FALSE_VALUES:
                return "false"
            raise SettingsError("Expected true or false.", section, key)
        if kind == "int":
            try:
                return str(int(value))
            except ValueError as exc:
                raise SettingsError("Expected an integer.", section, key) from exc
        if kind == "float":
            try:
                number = float(value)
            except ValueError as exc:
                raise SettingsError("Expected a number.", section, key) from exc
            if not math.isfinite(number):
                raise SettingsError("Expected a finite number.", section, key)
            return str(number)
        if kind == "enum":
            lowered = value.lower()
            options = entry.get("options") or ()
            if lowered not in options:
                raise SettingsError(
                    f"Expected one of: {', '.join(options)}.",
                    section,
                    key,
                )
            return lowered
        if entry.get("required") and not value:
            raise SettingsError("This setting is required.", section, key)
        return value

    def _stage_config(
        self,
        text: str,
        normalized: list[tuple[str, str, str, dict[str, Any]]],
    ) -> tuple[Path, str]:
        new_text = self._replace_values(text, normalized)
        descriptor, temp_name = tempfile.mkstemp(
            dir=str(self.config_path.parent),
            prefix=".config_",
            suffix=".tmp",
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(new_text)
            parser = configparser.ConfigParser(
                interpolation=None,
                inline_comment_prefixes=("#", ";"),
            )
            parser.optionxform = lambda optionstr: optionstr
            parser.read(temp_path, encoding="utf-8")
            for section, key, value, _entry in normalized:
                if not parser.has_option(section, key):
                    raise SettingsError(
                        f"Unknown config key: [{section}] {key}", section, key
                    )
                if parser.get(section, key).strip() != value:
                    raise SettingsError(
                        "The value contains characters configparser reads as a comment.",
                        section,
                        key,
                    )
            Config(config_path=temp_path)
        except SettingsError:
            temp_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            message = str(exc).replace(str(temp_path), "config/config.ini")
            raise SettingsError(message or "The configuration is invalid.") from exc
        return temp_path, new_text

    def _replace_values(
        self,
        text: str,
        normalized: list[tuple[str, str, str, dict[str, Any]]],
    ) -> str:
        changes = {(section, key): value for section, key, value, _entry in normalized}
        applied: set[tuple[str, str]] = set()
        output: list[str] = []
        current: str | None = None
        for line in text.splitlines(keepends=True):
            section_match = SECTION.match(line.rstrip("\r\n"))
            if section_match:
                current = section_match.group(1).strip()
                output.append(line)
                continue
            body = line
            ending = ""
            if body.endswith("\r\n"):
                body, ending = body[:-2], "\r\n"
            elif body.endswith("\n"):
                body, ending = body[:-1], "\n"
            assignment = ASSIGNMENT.match(body)
            if assignment and current is not None:
                left = assignment.group(1)
                key = left.split("=", 1)[0].strip()
                change = changes.get((current, key))
                if change is not None:
                    raw_value, suffix = self._split_raw(assignment.group(2))
                    trailing = raw_value[len(raw_value.rstrip()) :]
                    output.append(left + change + trailing + suffix + ending)
                    applied.add((current, key))
                    continue
            output.append(line)
        missing = set(changes) - applied
        if missing:
            section, key = min(missing)
            raise SettingsError(f"Unknown config key: [{section}] {key}", section, key)
        return "".join(output)

    @staticmethod
    def _read_text(path: Path, label: str) -> str:
        if not path.exists():
            raise SettingsError(f"{label} not found.", status=409)
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                return handle.read()
        except UnicodeDecodeError as exc:
            raise SettingsError(f"{label} is not valid UTF-8.", status=409) from exc
        except OSError as exc:
            raise SettingsError(f"Could not read {label}: {exc}", status=500) from exc

    @staticmethod
    def _etag(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _dotenv(text: str) -> dict[str, str]:
        if not text:
            return {}
        try:
            values = dotenv_values(stream=io.StringIO(text))
        except Exception:  # noqa: BLE001
            return {}
        return {key: value for key, value in values.items() if value is not None}

    def _template_entries(self) -> list[dict[str, str]]:
        if not self.keys_example_path.exists():
            return []
        return self._env_entries(self._read_text(self.keys_example_path, "keys.env.example"))

    def _env_entries(self, text: str) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        help_lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                help_lines = []
                continue
            if stripped.startswith(("#", ";")):
                help_lines.append(stripped[1:].strip())
                continue
            match = ENV_LINE.match(line)
            if match:
                value, inline_help = self._split_inline(match.group(2))
                entries.append(
                    {
                        "key": match.group(1),
                        "value": value,
                        "help": "\n".join(help_lines).strip(),
                        "inline_help": inline_help,
                    }
                )
                help_lines = []
                continue
            help_lines = []
        return entries

    @staticmethod
    def _key_row(
        key: str,
        help_text: str,
        inline_help: str,
        file_values: dict[str, str],
    ) -> dict[str, Any]:
        environment = key in os.environ
        env_value = os.environ.get(key, "")
        file_value = file_values.get(key, "")
        file_set = bool(file_value)
        env_set = bool(env_value)
        if environment and file_set:
            source = "both"
        elif environment:
            source = "environment"
        elif file_set:
            source = "file"
        else:
            source = None
        return {
            "key": key,
            "help": help_text,
            "inline_help": inline_help,
            "set": file_set or env_set,
            "source": source,
            "empty_override": environment and not env_value,
        }

    @staticmethod
    def _normalize_secret(key: str, raw: str) -> str:
        value = str(raw).strip()
        if "\n" in value or "\r" in value:
            raise SettingsError("A key cannot contain a newline.", key=key)
        if any(ord(character) < 32 for character in value):
            raise SettingsError("A key cannot contain control characters.", key=key)
        if len(value) > 4096:
            raise SettingsError("A key cannot exceed 4096 characters.", key=key)
        return value

    @staticmethod
    def _replace_env_value(text: str, key: str, value: str) -> str:
        pattern = re.compile(
            rf"^(\s*(?:export\s+)?{re.escape(key)}\s*=\s*)(.*)$",
            re.MULTILINE,
        )
        if len(pattern.findall(text)) > 1:
            raise SettingsError(
                f"{key} appears more than once in keys.env; fix it by hand.",
                key=key,
            )
        output: list[str] = []
        applied = False
        for line in text.splitlines(keepends=True):
            body = line
            ending = ""
            if body.endswith("\r\n"):
                body, ending = body[:-2], "\r\n"
            elif body.endswith("\n"):
                body, ending = body[:-1], "\n"
            match = pattern.match(body)
            if match and not applied:
                raw_value, suffix = SettingsEditor._split_raw(match.group(2))
                trailing = raw_value[len(raw_value.rstrip()) :]
                output.append(match.group(1) + value + trailing + suffix + ending)
                applied = True
                continue
            output.append(line)
        if applied:
            return "".join(output)
        newline = "\r\n" if "\r\n" in text else "\n"
        prefix = "" if not text or text.endswith(("\n", "\r")) else newline
        return f"{text}{prefix}{key}={value}{newline}"

    def _write_atomic(self, target: Path, text: str, label: str) -> None:
        descriptor, temp_name = tempfile.mkstemp(
            dir=str(target.parent),
            prefix=f".{target.name}_",
            suffix=".tmp",
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
            mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else 0o600
            try:
                os.chmod(temp_path, mode)
            except OSError:
                pass
            os.replace(temp_path, target)
        except OSError as exc:
            temp_path.unlink(missing_ok=True)
            raise SettingsError(f"Could not write {label}: {exc}", status=500) from exc

    def _replace_file(self, target: Path, staged: Path, label: str) -> None:
        try:
            if target.exists():
                try:
                    os.chmod(staged, stat.S_IMODE(target.stat().st_mode))
                except OSError:
                    pass
            os.replace(staged, target)
        except OSError as exc:
            staged.unlink(missing_ok=True)
            raise SettingsError(f"Could not write {label}: {exc}", status=500) from exc
