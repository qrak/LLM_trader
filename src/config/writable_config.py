

import asyncio
import configparser
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.timeframe_validator import TimeframeValidator


@dataclass(frozen=True)
class SettingMeta:
    """Schema metadata for a single config key."""
    key: str
    type: str
    category: str
    description: str
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None
    options: tuple[str, ...] | None = None


@dataclass
class SectionMeta:
    """Schema metadata for a config section."""
    name: str
    title: str
    settings: dict[str, SettingMeta] = field(default_factory=dict)


_SCHEMA: dict[str, SectionMeta] = {
    "prompt_budget": SectionMeta(
        name="prompt_budget", title="Estimated-token safety budgets",
        settings={
            "brain_max_tokens": SettingMeta("brain_max_tokens", "int", "restart", "Brain estimated-token limit", min_val=1),
            "request_max_tokens": SettingMeta("request_max_tokens", "int", "restart", "Whole request estimated-token limit", min_val=1),
            "safety_margin_percent": SettingMeta("safety_margin_percent", "int", "restart", "Model context safety margin (%)", min_val=1, max_val=50),
            "image_reserve_tokens": SettingMeta("image_reserve_tokens", "int", "restart", "Conservative chart-image token reserve", min_val=1),
        },
    ),
    "ai_providers": SectionMeta(
        name="ai_providers",
        title="AI Providers",
        settings={
            "provider": SettingMeta(
                key="provider", type="enum", category="restart",
                description="Active AI provider",
                options=("local", "googleai", "openrouter", "deepseek", "all"),
            ),
            "lm_studio_base_url": SettingMeta(
                key="lm_studio_base_url", type="string", category="restart",
                description="LM Studio base URL",
            ),
            "lm_studio_model": SettingMeta(
                key="lm_studio_model", type="string", category="restart",
                description="LM Studio model name (empty = auto-detect)",
            ),
            "lm_studio_streaming": SettingMeta(
                key="lm_studio_streaming", type="bool", category="restart",
                description="Enable LM Studio streaming",
            ),
            "openrouter_base_url": SettingMeta(
                key="openrouter_base_url", type="string", category="restart",
                description="OpenRouter API base URL",
            ),
            "openrouter_base_model": SettingMeta(
                key="openrouter_base_model", type="string", category="restart",
                description="OpenRouter primary model",
            ),
            "openrouter_fallback_model": SettingMeta(
                key="openrouter_fallback_model", type="string", category="restart",
                description="OpenRouter fallback model",
            ),
            "google_studio_model": SettingMeta(
                key="google_studio_model", type="string", category="restart",
                description="Google AI Studio model name",
            ),
        },
    ),
    "general": SectionMeta(
        name="general", title="General",
        settings={
            "crypto_pair": SettingMeta(
                key="crypto_pair", type="string", category="restart",
                description="Trading pair (e.g. BTC/USDC)",
            ),
            "timeframe": SettingMeta(
                key="timeframe", type="enum", category="restart",
                description="Analysis timeframe",
                options=tuple(TimeframeValidator.SUPPORTED_TIMEFRAMES),
            ),
            "candle_limit": SettingMeta(
                key="candle_limit", type="int", category="restart",
                description="Number of candles to fetch",
                min_val=10, max_val=9999,
            ),
            "ai_chart_candle_limit": SettingMeta(
                key="ai_chart_candle_limit", type="int", category="restart",
                description="Candles rendered in AI chart image",
                min_val=10, max_val=500,
            ),
            "discord_bot": SettingMeta(
                key="discord_bot", type="bool", category="restart",
                description="Enable Discord notifications",
            ),
            "include_coin_description": SettingMeta(
                key="include_coin_description", type="bool", category="restart",
                description="Include coin description in analysis context",
            ),
        },
    ),
    "debug": SectionMeta(
        name="debug", title="Debug",
        settings={
            "save_chart_images": SettingMeta(
                key="save_chart_images", type="bool", category="restart",
                description="Save generated chart images to disk",
            ),
            "chart_save_path": SettingMeta(
                key="chart_save_path", type="string", category="restart",
                description="Directory to save chart images",
            ),
            "logger_debug": SettingMeta(
                key="logger_debug", type="bool", category="restart",
                description="Enable debug-level logging",
            ),
        },
    ),
    "rag": SectionMeta(
        name="rag", title="RAG / News",
        settings={
            "update_interval_hours": SettingMeta(
                key="update_interval_hours", type="float", category="restart",
                description="Hours between RAG rebuilds",
                min_val=0.1, max_val=168.0, step=0.1,
            ),
            "news_limit": SettingMeta(
                key="news_limit", type="int", category="restart",
                description="Max articles in AI context",
                min_val=1, max_val=50,
            ),
            "article_max_tokens": SettingMeta(
                key="article_max_tokens", type="int", category="restart",
                description="Max tokens per article",
                min_val=100, max_val=5000,
            ),
            "news_page_enrichment": SettingMeta(
                key="news_page_enrichment", type="bool", category="restart",
                description="Fetch full article body via Crawl4AI",
            ),
            "news_crawl4ai_enabled": SettingMeta(
                key="news_crawl4ai_enabled", type="bool", category="restart",
                description="Enable Crawl4AI for news enrichment",
            ),
        },
    ),
    "risk_management": SectionMeta(
        name="risk_management", title="Risk Management",
        settings={
            "stop_loss_type": SettingMeta(
                key="stop_loss_type", type="enum", category="restart",
                description="SL execution mode",
                options=("soft", "hard"),
            ),
            "take_profit_type": SettingMeta(
                key="take_profit_type", type="enum", category="restart",
                description="TP execution mode",
                options=("soft", "hard"),
            ),
            "max_position_size": SettingMeta(
                key="max_position_size", type="float", category="restart",
                description="Max fraction of capital per position",
                min_val=0.01, max_val=1.0, step=0.01,
            ),
            "position_size_fallback_low": SettingMeta(
                key="position_size_fallback_low", type="float", category="restart",
                description="Fallback position size for LOW confidence",
                min_val=0.001, max_val=0.5, step=0.001,
            ),
            "position_size_fallback_medium": SettingMeta(
                key="position_size_fallback_medium", type="float", category="restart",
                description="Fallback position size for MEDIUM confidence",
                min_val=0.001, max_val=0.5, step=0.001,
            ),
            "position_size_fallback_high": SettingMeta(
                key="position_size_fallback_high", type="float", category="restart",
                description="Fallback position size for HIGH confidence",
                min_val=0.001, max_val=0.5, step=0.001,
            ),
        },
    ),
    "demo_trading": SectionMeta(
        name="demo_trading", title="Paper Trading",
        settings={
            "demo_quote_capital": SettingMeta(
                key="demo_quote_capital", type="int", category="restart",
                description="Starting paper trading capital (USD)",
                min_val=100, max_val=100_000_000,
            ),
            "transaction_fee_percent": SettingMeta(
                key="transaction_fee_percent", type="float", category="restart",
                description="Transaction fee (decimal, e.g. 0.00075 = 0.075%)",
                min_val=0.0, max_val=0.01, step=0.00001,
            ),
        },
    ),
    "model_config": SectionMeta(
        name="model_config", title="Model Configuration",
        settings={
            "temperature": SettingMeta(
                key="temperature", type="float", category="restart",
                description="LLM temperature",
                min_val=0.0, max_val=2.0, step=0.05,
            ),
            "top_p": SettingMeta(
                key="top_p", type="float", category="restart",
                description="LLM top_p (nucleus sampling)",
                min_val=0.0, max_val=1.0, step=0.05,
            ),
            "max_tokens": SettingMeta(
                key="max_tokens", type="int", category="restart",
                description="Max output tokens",
                min_val=256, max_val=131072,
            ),
            "google_max_tokens": SettingMeta(
                key="google_max_tokens", type="int", category="restart",
                description="Max tokens for Google AI provider",
                min_val=256, max_val=131072,
            ),
            "google_thinking_level": SettingMeta(
                key="google_thinking_level", type="enum", category="restart",
                description="Google thinking/reasoning level",
                options=("off", "low", "medium", "high"),
            ),
            "google_code_execution": SettingMeta(
                key="google_code_execution", type="bool", category="restart",
                description="Enable Google code execution tool",
            ),
            "model_verbosity": SettingMeta(
                key="model_verbosity", type="enum", category="restart",
                description="New analysis detail: ~400/1000/2000/8000 narrative output tokens",
                options=("low", "medium", "high", "full"),
            ),
        },
    ),
    "dashboard": SectionMeta(
        name="dashboard", title="Dashboard",
        settings={
            "enabled": SettingMeta(
                key="enabled", type="bool", category="restart",
                description="Enable the dashboard server",
            ),
            "host": SettingMeta(
                key="host", type="string", category="restart",
                description="Dashboard bind host/IP",
            ),
            "port": SettingMeta(
                key="port", type="int", category="restart",
                description="Dashboard port",
                min_val=1, max_val=65535,
            ),
            "enable_cors": SettingMeta(
                key="enable_cors", type="bool", category="restart",
                description="Enable CORS for the dashboard",
            ),
        },
    ),
}


_BOOL_TRUTHY = {"true", "1", "yes", "on"}
_BOOL_FALSY = {"false", "0", "no", "off"}


class WritableConfig:
    """Async-safe read-write access to config.ini with atomic disk writes.

    - Reads config.ini via configparser.
    - Validates values against schema before writing.
    - Writes to a temp file then os.replace() for crash safety.
    """

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self._parser = configparser.ConfigParser(
            interpolation=None,
            inline_comment_prefixes=("#", ";"),
        )
        self._parser.optionxform = str  # type: ignore[assignment]  # preserve key case
        self._parser.read(self.config_path, encoding="utf-8")
        self._lock = asyncio.Lock()

    def get_value(self, section: str, key: str) -> str | None:

        try:
            return self._parser.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return None


    @staticmethod
    def _validate_and_coerce(value: Any, meta: SettingMeta) -> str:

        if meta.type == "bool":
            s = str(value).strip().lower()
            if s in _BOOL_TRUTHY:
                return "true"
            if s in _BOOL_FALSY:
                return "false"
            raise ValueError(f"Invalid boolean value: {value!r}")

        if meta.type == "int":
            try:
                v = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid integer: {value!r}") from exc
            if meta.min_val is not None and v < int(meta.min_val):
                raise ValueError(f"Value {v} below minimum {int(meta.min_val)}")
            if meta.max_val is not None and v > int(meta.max_val):
                raise ValueError(f"Value {v} above maximum {int(meta.max_val)}")
            return str(v)

        if meta.type == "float":
            try:
                v = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid float: {value!r}") from exc
            if meta.min_val is not None and v < meta.min_val:
                raise ValueError(f"Value {v} below minimum {meta.min_val}")
            if meta.max_val is not None and v > meta.max_val:
                raise ValueError(f"Value {v} above maximum {meta.max_val}")
            return str(v)

        if meta.type == "enum":
            s = str(value).strip().lower()
            if meta.options and s not in meta.options:
                raise ValueError(f"Invalid option {value!r}. Must be one of: {', '.join(meta.options)}")
            return s

        if meta.type == "string":
            s = str(value).strip()
            if len(s) > 255:
                raise ValueError("String too long (max 255 chars)")
            if any(ord(c) < 32 and c not in ("\t",) for c in s):
                raise ValueError("String contains control characters")
            return s

        raise ValueError(f"Unknown setting type: {meta.type}")

    def _resolve_meta(self, section: str, key: str, value: Any) -> tuple[SettingMeta, str]:

        meta = self._get_meta(section, key)
        if meta is None:
            raise ValueError(f"Unknown config key: [{section}] {key}")
        return meta, self._validate_and_coerce(value, meta)

    async def set_value(self, section: str, key: str, value: Any) -> str:

        meta, coerced = self._resolve_meta(section, key, value)

        async with self._lock:
            if not self._parser.has_section(section):
                self._parser.add_section(section)
            self._parser.set(section, key, coerced)

            await self._write_to_disk()

        return meta.category

    async def set_values(self, updates: list[tuple[str, str, Any]]) -> list[dict[str, str]]:

        results = []
        validated = []

        for section, key, value in updates:
            meta, coerced = self._resolve_meta(section, key, value)
            validated.append((section, key, coerced, meta))

        async with self._lock:
            for section, key, coerced, meta in validated:
                if not self._parser.has_section(section):
                    self._parser.add_section(section)
                self._parser.set(section, key, coerced)
                results.append({"section": section, "key": key, "category": meta.category})

            await self._write_to_disk()

        return results

    def get_full_schema(self) -> dict[str, Any]:

        result = {}
        for section_name, section_meta in _SCHEMA.items():
            keys = {}
            for key, meta in section_meta.settings.items():
                current = self.get_value(section_name, key)
                entry: dict[str, Any] = {
                    "value": current,
                    "type": meta.type,
                    "category": meta.category,
                    "description": meta.description,
                }
                if meta.min_val is not None:
                    entry["min"] = meta.min_val
                if meta.max_val is not None:
                    entry["max"] = meta.max_val
                if meta.step is not None:
                    entry["step"] = meta.step
                if meta.options is not None:
                    entry["options"] = list(meta.options)
                keys[key] = entry
            result[section_name] = {
                "title": section_meta.title,
                "keys": keys,
            }
        return result

    def get_section_schema(self, section: str) -> dict[str, Any] | None:

        if section not in _SCHEMA:
            return None
        full = self.get_full_schema()
        return full.get(section)

    def _get_meta(self, section: str, key: str) -> SettingMeta | None:

        section_meta = _SCHEMA.get(section)
        if section_meta is None:
            return None
        return section_meta.settings.get(key)

    async def _write_to_disk(self) -> None:

        config_dir = self.config_path.parent
        fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=str(config_dir), prefix=".config_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                self._parser.write(f)
            os.replace(tmp_path, str(self.config_path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    async def reload_from_disk(self) -> None:

        async with self._lock:
            self._parser.read(self.config_path, encoding="utf-8")

