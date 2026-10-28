"""
Configuration loader for LLM_Trader v2.
Loads secrets from the process environment or optional keys.env, and settings from config.ini.
"""

import configparser
import math
import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from src.utils.prompt_budget import PromptBudgetPolicy
from src.utils.time_utils import resolve_timezone
from src.utils.timeframe_validator import TimeframeValidator

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
CONFIG_DIR = ROOT_DIR / "config"
KEYS_ENV_PATH = ROOT_DIR / "keys.env"
CONFIG_INI_PATH = CONFIG_DIR / "config.ini"

VALID_PROVIDERS = {"local", "googleai", "openrouter", "deepseek", "all"}
VALID_EXIT_TYPES = {"soft", "hard"}
VALID_MODEL_VERBOSITIES = {"low", "medium", "high", "full"}
MIN_REFERENCE_EXIT_CHECK_SECONDS = 2.0
MAX_REFERENCE_EXIT_CHECK_SECONDS = 120.0

class Config:
    """Configuration class that loads settings from environment and INI files.
    """

    REQUIRED_CONFIG_KEYS: dict[str, tuple[str, ...]] = {  # noqa: RUF012
        "general": ("crypto_pair", "timeframe", "candle_limit", "ai_chart_candle_limit", "market_type"),
        "ai_providers": ("provider",),
        "model_config": ("max_tokens",),
    }

    def __init__(self):
        self._env_vars = {}
        self._config_data = {}
        self._load_environment()
        self._load_ini_config()
        self._validate_required_keys()
        self._validate_provider()
        self._validate_timezone()
        self._validate_exit_monitoring()
        self._build_model_configs()
        self._validate_model_verbosity()
        self._validate_prompt_budget()

    def _load_environment(self):
        """Merge optional keys.env with process variables; process values take precedence."""
        env_vars = dotenv_values(KEYS_ENV_PATH) if KEYS_ENV_PATH.exists() else {}
        env_vars.update(os.environ)
        for key, value in env_vars.items():
            if value is None:
                continue
            if key == "ADMIN_USER_IDS":
                try:
                    self._env_vars[key] = [int(uid.strip()) for uid in value.split(",") if uid.strip()]
                except ValueError as exc:
                    raise ValueError("Invalid ADMIN_USER_IDS format. Expected comma-separated integers.") from exc
            elif key == "MAIN_CHANNEL_ID" and value.isdigit():
                self._env_vars[key] = int(value)
            else:
                self._env_vars[key] = value

    def _load_ini_config(self):
        """Load configuration from config.ini file."""
        if not CONFIG_INI_PATH.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {CONFIG_INI_PATH}. "
                "Please create config.ini in the config directory."
            )

        try:
            config = configparser.ConfigParser(
                interpolation=None,
                inline_comment_prefixes=("#", ";"),
            )
            config.optionxform = lambda optionstr: optionstr
            config.read(CONFIG_INI_PATH, encoding="utf-8")

            for section_name in config.sections():
                section_data = {}
                for key, value in config.items(section_name):
                    if section_name != "model_context_windows":
                        key = key.lower()
                    if key in section_data:
                        raise ValueError(f"Duplicate option {key} in [{section_name}]")
                    converted_value = self._convert_value(value)
                    if section_name == "dashboard" and key == "cors_origins":
                        converted_value = ["*"] if value.strip() == "*" else [origin.strip() for origin in value.split(",") if origin.strip()]
                    elif section_name == "rag" and key == "news_sources":
                        converted_value = [source.strip() for source in value.split(",") if source.strip()]
                    section_data[key] = converted_value
                self._config_data[section_name] = section_data
        except Exception as e:
            raise RuntimeError(f"Error loading configuration file {CONFIG_INI_PATH}: {e}") from e

    def _validate_required_keys(self) -> None:
        """Refuse to start when a key the runtime cannot do without is absent.

        A missing ``timeframe`` used to fall back to a loader default, which is a silent
        second source of truth: the bot traded a frame nobody configured. Every key listed
        in ``REQUIRED_CONFIG_KEYS`` must be present in config.ini, with no default behind it.
        """
        missing = [
            f"[{section}] {key}"
            for section, keys in self.REQUIRED_CONFIG_KEYS.items()
            for key in keys
            if key not in self._config_data.get(section, {})
        ]
        if missing:
            raise RuntimeError(
                f"Missing required configuration keys in {CONFIG_INI_PATH}: {', '.join(missing)}"
            )

    def _prompt_budget_int(self, key: str, default: int) -> int:
        value = self.get_config("prompt_budget", key, default)
        if type(value) is not int or value <= 0:
            raise ValueError(f"[prompt_budget] {key} must be a positive integer")
        return value

    @property
    def BRAIN_MAX_TOKENS(self) -> int:
        return self._prompt_budget_int("brain_max_tokens", 50000)

    @property
    def PROMPT_MAX_TOKENS(self) -> int:
        return self._prompt_budget_int("request_max_tokens", 100000)

    @property
    def PROMPT_SAFETY_MARGIN_PERCENT(self) -> int:
        value = self._prompt_budget_int("safety_margin_percent", 10)
        if value > 50:
            raise ValueError("[prompt_budget] safety_margin_percent must not exceed 50")
        return value

    @property
    def PROMPT_IMAGE_RESERVE_TOKENS(self) -> int:
        return self._prompt_budget_int("image_reserve_tokens", 16384)

    @property
    def MODEL_CONTEXT_WINDOWS(self) -> dict[str, int]:
        windows = dict(PromptBudgetPolicy.DEFAULT_CONTEXT_WINDOWS)
        for key, value in self._config_data.get("model_context_windows", {}).items():
            provider, _, model = key.partition("/")
            if provider not in VALID_PROVIDERS | {"blockrun"} or not model or model != model.strip():
                raise ValueError(f"Invalid model context key: {key}")
            if type(value) is not int or value <= 0:
                raise ValueError(f"Context window for {key} must be a positive integer")
            windows[key] = value
        return windows

    def _validate_prompt_budget(self) -> None:
        _ = (self.BRAIN_MAX_TOKENS, self.PROMPT_MAX_TOKENS, self.PROMPT_SAFETY_MARGIN_PERCENT,
             self.PROMPT_IMAGE_RESERVE_TOKENS, self.MODEL_CONTEXT_WINDOWS)

    @property
    def CIRCUIT_BREAKER_ENABLED(self) -> bool:
        raw = self.get_config("circuit_breaker", "enabled", "true")
        return str(raw).strip().lower() in ("1", "true", "yes", "on")

    @property
    def CIRCUIT_BREAKER_FAILURE_THRESHOLD(self) -> int:
        try:
            return max(1, int(self.get_config("circuit_breaker", "failure_threshold", 3)))
        except (TypeError, ValueError):
            return 3

    @property
    def CIRCUIT_BREAKER_COOLDOWN_SECONDS(self) -> float:
        try:
            return max(1.0, float(self.get_config("circuit_breaker", "cooldown_seconds", 120)))
        except (TypeError, ValueError):
            return 120.0

    @property
    def REQUIRE_APPROVAL_ABOVE_USDC(self) -> float:
        """Quote value above which a BUY/SELL waits for a human. 0 disables the gate."""
        try:
            return max(0.0, float(self.get_config("risk_management", "require_approval_above_usdc", 0)))
        except (TypeError, ValueError):
            return 0.0

    @property
    def APPROVAL_TTL_MINUTES(self) -> float:
        try:
            return max(1.0, float(self.get_config("risk_management", "approval_ttl_minutes", 15)))
        except (TypeError, ValueError):
            return 15.0

    @property
    def APPROVAL_POLL_SECONDS(self) -> float:
        try:
            return max(1.0, float(self.get_config("risk_management", "approval_poll_seconds", 5)))
        except (TypeError, ValueError):
            return 5.0

    def _validate_provider(self):
        """Validate that the configured AI provider is supported."""
        provider = self.PROVIDER.lower()
        if provider not in VALID_PROVIDERS:
            valid_options = ", ".join(f'"{p}"' for p in sorted(VALID_PROVIDERS))
            error_msg = (
                f"Invalid AI provider '{provider}' in config.ini.\n"
                f"Supported values are: {valid_options}.\n"
                f"Please update the [ai_providers] -> provider setting."
            )
            raise ValueError(error_msg)

    def _validate_timezone(self) -> None:
        """Fail at startup, not on the first log line, when the zone name is wrong."""
        resolve_timezone(self.TIMEZONE or None)

    def _validate_exit_monitoring(self):
        """Validate configurable SL/TP execution settings against the timeframe."""
        timeframe = self.TIMEFRAME
        timeframe_minutes = self.TIMEFRAME_MINUTES

        for label, type_key, interval_key in (
            ("stop loss", "stop_loss_type", "stop_loss_check_interval"),
            ("take profit", "take_profit_type", "take_profit_check_interval"),
        ):
            self._normalize_exit_type(self.get_config("risk_management", type_key, "soft"), type_key)
            interval = self.get_config("risk_management", interval_key, timeframe)
            interval_minutes = self._parse_exit_interval_minutes(interval, interval_key)
            if interval_minutes > timeframe_minutes:
                error_msg = (
                    f"Invalid {label} check interval '{interval}' in config.ini. "
                    f"It must not be greater than timeframe '{timeframe}'."
                )
                raise ValueError(error_msg)

    @staticmethod
    def _normalize_exit_type(value: Any, key: str) -> str:
        """Normalize and validate an exit execution type."""
        normalized = str(value).strip().lower()
        if normalized not in VALID_EXIT_TYPES:
            valid_options = ", ".join(f'"{item}"' for item in sorted(VALID_EXIT_TYPES))
            raise ValueError(f"Invalid {key} '{value}' in config.ini. Supported values are: {valid_options}.")
        return normalized

    @staticmethod
    def _normalize_model_verbosity(value: Any, key: str) -> str:
        """Normalize and validate a model verbosity level."""
        normalized = str(value).strip().lower()
        if normalized not in VALID_MODEL_VERBOSITIES:
            valid_options = ", ".join(f'"{item}"' for item in sorted(VALID_MODEL_VERBOSITIES))
            raise ValueError(f"Invalid {key} '{value}' in config.ini. Supported values are: {valid_options}.")
        return normalized

    def _validate_model_verbosity(self) -> None:
        """Validate model_verbosity config value at startup."""
        self._normalize_model_verbosity(
            self.get_config("model_config", "model_verbosity", "full"),
            "model_verbosity",
        )

    @staticmethod
    def _parse_exit_interval_minutes(value: Any, key: str) -> int:
        """Parse an exit monitor interval and require a positive duration."""
        try:
            minutes = TimeframeValidator.parse_period_to_minutes(str(value).strip().lower())
        except ValueError as exc:
            raise ValueError(f"Invalid {key} '{value}' in config.ini: {exc}") from exc
        if minutes <= 0:
            raise ValueError(f"Invalid {key} '{value}' in config.ini. Interval must be positive.")
        return minutes

    @staticmethod
    def _convert_value(value: str) -> Any:
        """Convert string values to appropriate Python types.

        Conversion order (first match wins):
        1. Boolean literals (true/false/yes/no/on/off/1/0)
        2. Integers (pure digits, no sign)
        3. Floats (decimal point or scientific notation, e.g., 1.5, 1e5, 2.5e-3)
        4. Comma-separated lists (string list)
        5. Raw string fallback
        """
        value = value.strip()
        lowered = value.lower()
        if lowered in ("true", "yes", "on", "1"):
            return True
        if lowered in ("false", "no", "off", "0"):
            return False
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            pass
        if "," in value:
            return [item.strip() for item in value.split(",")]
        return value

    def _build_model_configs(self):
        """Build model configuration dictionaries as instance variables.

        Config key fallbacks (for backward compatibility):
          - ``frequency_penalty`` ← ``freq_penalty`` (prefer the full name)
          - ``presence_penalty`` ← ``pres_penalty`` (prefer the full name)
        """
        default_max_tokens = self.get_config("model_config", "max_tokens", None)
        if default_max_tokens is None:
            raise RuntimeError("`max_tokens` is required in [model_config] of config.ini")

        self._default_model_config = {
            "temperature": self.get_config("model_config", "temperature", None),
            "top_p": self.get_config("model_config", "top_p", None),
            "top_k": self.get_config("model_config", "top_k", None),
            "frequency_penalty": self.get_config("model_config", "frequency_penalty", self.get_config("model_config", "freq_penalty", None)),
            "presence_penalty": self.get_config("model_config", "presence_penalty", self.get_config("model_config", "pres_penalty", None)),
            "max_tokens": default_max_tokens,
            "openrouter_reasoning_effort": self.get_config("model_config", "openrouter_reasoning_effort", "max"),
        }

        google_max_tokens = self.get_config("model_config", "google_max_tokens", None)

        if google_max_tokens is None and self.PROVIDER in ("googleai", "all"):
            raise RuntimeError("`google_max_tokens` is required in [model_config] of config.ini when using Google models")

        self._google_model_config = {
            "max_tokens": google_max_tokens,
            "thinking_level": self.get_config("model_config", "google_thinking_level", "high"),
            "google_code_execution": self.get_config("model_config", "google_code_execution", False),
        }

        if self.PROVIDER.lower() == "deepseek" and not self.DEEPSEEK_API_KEY:
            raise RuntimeError("`DEEPSEEK_API_KEY` is required in the process environment or keys.env when using the DeepSeek provider")

        deepseek_max_tokens = self.get_config("model_config", "deepseek_max_tokens", default_max_tokens)
        self._deepseek_model_config = {
            "max_tokens": deepseek_max_tokens,
            "reasoning_effort": self.get_config("model_config", "deepseek_reasoning_effort", "max"),
            "response_format": {"type": "json_object"},
        }

    def get_env(self, key: str, default: Any = None) -> Any:
        """Get environment variable."""
        return self._env_vars.get(key, default)

    def get_config(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value from INI file."""
        return self._config_data.get(section, {}).get(key, default)

    @property
    def BOT_TOKEN_DISCORD(self):
        return self.get_env("BOT_TOKEN_DISCORD")

    @property
    def MAIN_CHANNEL_ID(self):
        return self.get_env("MAIN_CHANNEL_ID")

    @property
    def OPENROUTER_API_KEY(self):
        return self.get_env("OPENROUTER_API_KEY")

    @property
    def GOOGLE_STUDIO_API_KEY(self):
        return self.get_env("GOOGLE_STUDIO_API_KEY")

    @property
    def GOOGLE_STUDIO_PAID_API_KEY(self):
        return self.get_env("GOOGLE_STUDIO_PAID_API_KEY")

    @property
    def COINGECKO_API_KEY(self):
        return self.get_env("COINGECKO_API_KEY")

    @property
    def GITHUB_USER(self):
        """GitHub login whose public events feed the dashboard Activity panel."""
        return self.get_env("GITHUB_USER", "")

    @property
    def GITHUB_TOKEN(self):
        """Optional GitHub token: raises the public events quota from 60 to 5000 calls/hour."""
        return self.get_env("GITHUB_TOKEN", "")

    @property
    def ADMIN_USERNAME(self):
        """Admin console username from keys.env."""
        return self.get_env("ADMIN_USERNAME", "")

    @property
    def ADMIN_PASSWORD_HASH(self):
        """Admin console password hash from keys.env (salt_hex:hash_hex format)."""
        return self.get_env("ADMIN_PASSWORD_HASH", "")

    @property
    def ADMIN_SIGNING_KEY(self):
        """Admin session signing key from keys.env (auto-generated if empty)."""
        return self.get_env("ADMIN_SIGNING_KEY", "")

    @property
    def PROVIDER(self):
        return self.get_config("ai_providers", "provider", "googleai")

    @property
    def LM_STUDIO_BASE_URL(self):
        return self.get_config("ai_providers", "lm_studio_base_url", "http://localhost:1234/v1")

    @property
    def LM_STUDIO_MODEL(self):
        return self.get_config("ai_providers", "lm_studio_model", "local-model")

    @property
    def LM_STUDIO_STREAMING(self):
        return self.get_config("ai_providers", "lm_studio_streaming", True)

    @property
    def OPENROUTER_BASE_URL(self):
        return self.get_config("ai_providers", "openrouter_base_url", "https://openrouter.ai/api/v1")

    @property
    def OPENROUTER_BASE_MODEL(self):
        return self.get_config("ai_providers", "openrouter_base_model", "google/gemini-3-flash-preview")

    @property
    def OPENROUTER_FALLBACK_MODEL(self):
        return self.get_config("ai_providers", "openrouter_fallback_model", "deepseek/deepseek-v4.1-flash")

    @property
    def DEEPSEEK_API_KEY(self):
        return self.get_env("DEEPSEEK_API_KEY")

    @property
    def DEEPSEEK_BASE_URL(self):
        return self.get_config("ai_providers", "deepseek_base_url", "https://api.deepseek.com")

    @property
    def DEEPSEEK_MODEL(self):
        return self.get_config("ai_providers", "deepseek_model", "deepseek-flash")

    @property
    def GOOGLE_STUDIO_MODEL(self):
        return self.get_config("ai_providers", "google_studio_model", "gemini-3.8-flash")

    @property
    def BLOCKRUN_BASE_URL(self):
        return self.get_config("ai_providers", "blockrun_base_url", "https://blockrun.ai/api")

    @property
    def BLOCKRUN_MODEL(self):
        return self.get_config("ai_providers", "blockrun_model", "deepseek/deepseek-reasoner")

    @property
    def BLOCKRUN_WALLET_KEY(self):
        return self.get_env("BLOCKRUN_WALLET_KEY")

    @property
    def MODEL_VERBOSITY(self) -> str:
        return self._normalize_model_verbosity(
            self.get_config("model_config", "model_verbosity", "full"),
            "model_verbosity",
        )

    @property
    def LOGGER_DEBUG(self):
        return self.get_config("debug", "logger_debug", False)


    @property
    def CRYPTO_PAIR(self):
        return self.get_config("general", "crypto_pair", "BTC/USDT")

    @property
    def DISCORD_BOT_ENABLED(self):
        return self.get_config("general", "discord_bot", False)

    @property
    def TIMEFRAME(self) -> str:
        """Analysis timeframe from config.ini, validated and normalized.

        There is deliberately no default: a missing key is a configuration error, not an
        invitation to trade a frame the operator never chose.
        """
        raw = self.get_config("general", "timeframe", None)
        if raw is None:
            raise RuntimeError(f"`timeframe` is required in [general] of {CONFIG_INI_PATH}")
        return TimeframeValidator.validate_and_normalize(str(raw))

    @property
    def TIMEFRAME_MINUTES(self) -> int:
        """Configured timeframe expressed in minutes, derived in exactly one place."""
        return TimeframeValidator.to_minutes(self.TIMEFRAME)

    @property
    def CANDLE_LIMIT(self):
        return self.get_config("general", "candle_limit", 999)

    @property
    def AI_CHART_CANDLE_LIMIT(self):
        """Configured candle limit to use for AI chart images (must be present in config.ini)."""
        return int(self.get_config("general", "ai_chart_candle_limit", 200))

    @property
    def MARKET_TYPE(self) -> str:
        """spot = BUY/SELL signals, futures = LONG/SHORT signals with leverage."""
        return self.get_config("general", "market_type", "spot").lower().strip()

    @property
    def ENTRY_ORDER_TYPE(self) -> str:
        """market = instant execution, limit = conditional (infrastructure kept for future)."""
        return self.get_config("general", "entry_order_type", "market").lower().strip()

    @property
    def INCLUDE_COIN_DESCRIPTION(self) -> bool:
        """Whether to include project description in coin details section."""
        return self.get_config("general", "include_coin_description", False)

    @property
    def DEBUG_SAVE_CHARTS(self):
        return self.get_config("debug", "save_chart_images", False)

    @property
    def DEBUG_CHART_SAVE_PATH(self):
        return self.get_config("debug", "chart_save_path", "test_images")

    @property
    def TIMEZONE(self) -> str:
        """IANA zone used to render timestamps for humans; empty = the OS zone."""
        return str(self.get_config("general", "timezone", "") or "").strip()

    @property
    def LOG_DIR(self):
        return self.get_config("directories", "log_dir", "logs")

    @property
    def DATA_DIR(self):
        return self.get_config("directories", "data_dir", "data")

    @property
    def DASHBOARD_ENABLED(self):
        return self.get_config("dashboard", "enabled", True)

    @property
    def DASHBOARD_HOST(self):
        return self.get_config("dashboard", "host", "0.0.0.0")

    @property
    def DASHBOARD_PORT(self):
        return int(self.get_config("dashboard", "port", 8000))

    @property
    def DASHBOARD_ENABLE_CORS(self):
        return self.get_config("dashboard", "enable_cors", False)

    @property
    def DASHBOARD_CORS_ORIGINS(self):
        origins = self.get_config("dashboard", "cors_origins", [])
        return origins

    @property
    def FILE_MESSAGE_EXPIRY(self):
        """Get file message expiry time in seconds (configured in hours in config.ini)."""
        hours = self.get_config("cooldowns", "file_message_expiry", 168)
        return hours * 3600

    @property
    def RESEARCH_TEAM_ENABLED(self) -> bool:
        """Whether bull/bear debate is enabled in the LLM prompt."""
        return bool(self.get_config("research_team", "enabled", False))

    @property
    def SOCIAL_SENTIMENT_ENABLED(self) -> bool:
        """Whether Reddit social sentiment fetching is enabled."""
        return bool(self.get_config("social_sentiment", "enabled", False))

    @property
    def RAG_UPDATE_INTERVAL_HOURS(self):
        return self.get_config("rag", "update_interval_hours", 4)

    @property
    def RAG_COINGECKO_UPDATE_INTERVAL_HOURS(self):
        return self.get_config("rag", "coingecko_update_interval_hours", 24)

    @property
    def RAG_DEFILLAMA_UPDATE_INTERVAL_HOURS(self):
        return float(self.get_config("rag", "defillama_update_interval_hours", 0.25))

    @property
    def RAG_COINGECKO_GLOBAL_API_URL(self):
        return self.get_config("rag", "coingecko_global_api_url", "https://api.coingecko.com/api/v3/global")

    @property
    def RAG_NEWS_LIMIT(self):
        """Maximum number of news articles to include in context (configurable via [rag] news_limit)."""
        return int(self.get_config("rag", "news_limit", 5))

    @property
    def RAG_ARTICLE_MAX_TOKENS(self):
        """Maximum number of tokens per article (configurable via [rag] article_max_tokens)."""
        return int(self.get_config("rag", "article_max_tokens", 1000))

    @property
    def RAG_DENSITY_PENALTY_THRESHOLD(self):
        """Body length below which articles are penalized (default 300 chars)."""
        return int(self.get_config("rag", "density_penalty_threshold", 300))

    @property
    def RAG_DENSITY_BOOST_THRESHOLD(self):
        """Body length above which articles get a boost (default 1000 chars)."""
        return int(self.get_config("rag", "density_boost_threshold", 1000))

    @property
    def RAG_DENSITY_PENALTY_MULTIPLIER(self):
        """Score multiplier for short articles (default 0.5)."""
        return float(self.get_config("rag", "density_penalty_multiplier", 0.5))

    @property
    def RAG_DENSITY_BOOST_MULTIPLIER(self):
        """Score multiplier for long articles (default 1.2)."""
        return float(self.get_config("rag", "density_boost_multiplier", 1.2))

    @property
    def RAG_COOCCURRENCE_MULTIPLIER(self):
        """Score multiplier when all query keywords appear in article (default 1.5)."""
        return float(self.get_config("rag", "cooccurrence_multiplier", 1.5))


    @property
    def RAG_NEWS_SOURCES(self):
        """Enabled RSS source keys, or None to use all configured sources."""
        raw_sources = self.get_config("rag", "news_sources", None)
        if not raw_sources:
            return None
        return [source.strip() for source in raw_sources if source.strip()]

    @property
    def RAG_NEWS_SOURCE_URLS(self) -> dict[str, str]:
        """Configured RSS source URL mapping keyed by source name."""
        return {
            "coindesk": self.get_config("rag", "news_source_coindesk_url", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
            "cointelegraph": self.get_config("rag", "news_source_cointelegraph_url", "https://cointelegraph.com/rss"),
            "decrypt": self.get_config("rag", "news_source_decrypt_url", "https://decrypt.co/feed"),
            "cryptoslate": self.get_config("rag", "news_source_cryptoslate_url", "https://cryptoslate.com/feed/"),
        }

    @property
    def RAG_NEWS_PAGE_ENRICHMENT(self) -> bool:
        """Whether to enrich short RSS bodies by fetching article pages."""
        val = self.get_config("rag", "news_page_enrichment", True)
        return bool(val)

    @property
    def RAG_NEWS_ENRICH_MIN_CHARS(self) -> int:
        """Minimum body length before attempting page enrichment."""
        return int(self.get_config("rag", "news_min_body_chars", 400))

    @property
    def RAG_NEWS_FETCH_TIMEOUT(self) -> int:
        """Timeout in seconds for RSS feed and article-page requests."""
        return int(self.get_config("rag", "news_timeout_seconds", 20))

    @property
    def RAG_UPDATE_TIMEOUT(self) -> int:
        """Overall timeout in seconds for market knowledge refresh in a trading check."""
        return int(self.get_config("rag", "rag_update_timeout_seconds", 180))

    @property
    def RAG_NEWS_FETCH_TOTAL_TIMEOUT(self) -> int:
        """Outer timeout in seconds for full RSS fetch stage across all sources."""
        return int(self.get_config("rag", "news_fetch_total_timeout_seconds", 45))

    @property
    def RAG_NEWS_ENRICH_TIMEOUT(self) -> int:
        """Outer timeout in seconds for article-body enrichment batch."""
        return int(self.get_config("rag", "news_enrichment_timeout_seconds", 120))

    @property
    def RAG_NEWS_CRAWL_CONCURRENCY(self) -> int:
        """Max concurrent page-enrichment / Crawl4AI sessions."""
        return int(self.get_config("rag", "news_max_concurrency", 6))

    @property
    def RAG_NEWS_MAX_ITEMS_PER_SOURCE(self) -> int:
        """Maximum number of articles fetched per RSS source."""
        return int(self.get_config("rag", "news_max_items_per_source", 50))

    @property
    def RAG_NEWS_CRAWL4AI_ENABLED(self) -> bool:
        """Use Crawl4AI browser-based enrichment instead of plain HTTP.

        Defaults to *False* so the pipeline works without Playwright installed.
        Set ``news_crawl4ai_enabled = true`` in ``[rag]`` to opt in.
        """
        val = self.get_config("rag", "news_crawl4ai_enabled", False)
        return bool(val)

    @property
    def RAG_NEWS_CRAWL_TIMEOUT(self) -> int:
        """Per-page timeout in seconds for Crawl4AI (default: same as fetch timeout)."""
        return int(self.get_config("rag", "news_crawl_timeout", self.RAG_NEWS_FETCH_TIMEOUT))

    @property
    def SUPPORTED_EXCHANGES(self):
        """Returns list of supported exchanges in priority order."""
        return self.get_config("exchanges", "supported", ["binance", "kucoin", "gateio"])

    @property
    def MARKET_REFRESH_HOURS(self):
        return self.get_config("exchanges", "market_refresh_hours", 24)

    @property
    def EXCHANGE_SANDBOX(self) -> bool:
        """Whether market data comes from the exchange's testnet (ccxt sandbox URLs).

        Keep this in sync with the venue the executor trades on: with sandbox off the
        bot reads the REAL market while the executor's protective orders rest on the
        testnet book, so a stop or target can fill at a price the real chart never
        printed. See ``references/testnet-vs-real-market.md`` in the executor skill.
        """
        val = self.get_config("exchanges", "sandbox", False)
        return bool(val)

    @property
    def TRANSACTION_FEE_PERCENT(self):
        """Transaction fee percentage for limit orders (default 0.075%)."""
        return float(self.get_config("demo_trading", "transaction_fee_percent", 0.00075))

    @property
    def DEMO_QUOTE_CAPITAL(self):
        """Initial capital for demo trading (default 10000)."""
        return float(self.get_config("demo_trading", "demo_quote_capital", 10000.0))

    @property
    def MAX_POSITION_SIZE(self):
        """Maximum allowed position size as decimal (e.g. 0.10 = 10% of capital). Hard cap enforced in RiskManager."""
        return float(self.get_config("risk_management", "max_position_size", 0.10))

    @property
    def MIN_RR_ENTRY(self) -> float:
        """Configured floor for the risk/reward gate.

        The brain may raise the effective floor but cannot lower it below this value.
        Zero starts without a hard floor until the brain learns one. Default 0.0.
        """
        return float(self.get_config("risk_management", "min_rr_entry", 0.0))

    @property
    def POSITION_SIZE_FALLBACK_LOW(self) -> float:
        """Fallback position size for LOW confidence when AI size is missing or invalid."""
        return float(self.get_config("risk_management", "position_size_fallback_low", 0.01))

    @property
    def POSITION_SIZE_FALLBACK_MEDIUM(self) -> float:
        """Fallback position size for MEDIUM confidence when AI size is missing or invalid."""
        return float(self.get_config("risk_management", "position_size_fallback_medium", 0.02))

    @property
    def POSITION_SIZE_FALLBACK_HIGH(self) -> float:
        """Fallback position size for HIGH confidence when AI size is missing or invalid."""
        return float(self.get_config("risk_management", "position_size_fallback_high", 0.03))

    @property
    def SL_TIGHTENING_SCALPING(self) -> float:
        """Minimum progress fraction for SL tightening on sub-1h timeframes."""
        return float(self.get_config("risk_management", "sl_tightening_scalping", 0.25))

    @property
    def SL_TIGHTENING_INTRADAY(self) -> float:
        """Minimum progress fraction for SL tightening on 1h–4h timeframes."""
        return float(self.get_config("risk_management", "sl_tightening_intraday", 0.20))

    @property
    def SL_TIGHTENING_SWING(self) -> float:
        """Minimum progress fraction for SL tightening on 4h–1d timeframes."""
        return float(self.get_config("risk_management", "sl_tightening_swing", 0.15))

    @property
    def SL_TIGHTENING_POSITION(self) -> float:
        """Minimum progress fraction for SL tightening on daily+ timeframes."""
        return float(self.get_config("risk_management", "sl_tightening_position", 0.10))

    @property
    def SL_TIGHTENING_FLOOR(self) -> float:
        """Minimum clamp — brain cannot lower effective threshold below this."""
        return float(self.get_config("risk_management", "sl_tightening_floor", 0.05))

    @property
    def SL_TIGHTENING_CEILING(self) -> float:
        """Maximum clamp — brain cannot raise effective threshold above this."""
        return float(self.get_config("risk_management", "sl_tightening_ceiling", 0.40))

    @property
    def SL_TIGHTENING_MIN_SAMPLES(self) -> int:
        """Minimum paired update/outcome samples before trusting a brain override."""
        return int(self.get_config("risk_management", "sl_tightening_min_samples", 10))

    @property
    def STOP_LOSS_TYPE(self) -> str:
        """Stop-loss execution type: soft candle-close or hard ticker-polled."""
        return self._normalize_exit_type(self.get_config("risk_management", "stop_loss_type", "soft"), "stop_loss_type")

    @property
    def STOP_LOSS_CHECK_INTERVAL(self) -> str:
        """Configured stop-loss monitor interval."""
        return str(self.get_config("risk_management", "stop_loss_check_interval", self.TIMEFRAME)).strip().lower()

    @property
    def STOP_LOSS_CHECK_INTERVAL_MINUTES(self) -> int:
        """Stop-loss monitor interval in minutes."""
        return self._parse_exit_interval_minutes(self.STOP_LOSS_CHECK_INTERVAL, "stop_loss_check_interval")

    @property
    def STOP_LOSS_CHECK_INTERVAL_SECONDS(self) -> int:
        """Stop-loss monitor interval in seconds."""
        return self.STOP_LOSS_CHECK_INTERVAL_MINUTES * 60

    @property
    def TAKE_PROFIT_TYPE(self) -> str:
        """Take-profit execution type: soft candle-close or hard ticker-polled."""
        return self._normalize_exit_type(self.get_config("risk_management", "take_profit_type", "soft"), "take_profit_type")

    @property
    def TAKE_PROFIT_CHECK_INTERVAL(self) -> str:
        """Configured take-profit monitor interval."""
        return str(self.get_config("risk_management", "take_profit_check_interval", self.TIMEFRAME)).strip().lower()

    @property
    def TAKE_PROFIT_CHECK_INTERVAL_MINUTES(self) -> int:
        """Take-profit monitor interval in minutes."""
        return self._parse_exit_interval_minutes(self.TAKE_PROFIT_CHECK_INTERVAL, "take_profit_check_interval")

    @property
    def TAKE_PROFIT_CHECK_INTERVAL_SECONDS(self) -> int:
        """Take-profit monitor interval in seconds."""
        return self.TAKE_PROFIT_CHECK_INTERVAL_MINUTES * 60

    @property
    def REFERENCE_EXIT_CHECK_SECONDS(self) -> float:
        """How often the REAL-market stop is evaluated while a position is open.

        With the executor live the exchange's own stop-loss leg is only a
        catastrophe backstop (its ``PROTECTION_BACKSTOP_MULTIPLIER``): the real
        exit is this check, running against the real market the bot reads
        (``[exchanges] sandbox = false``). 0 disables it — and disabling it while
        the executor still places a widened leg leaves the position with a
        backstop only, so the two settings move together.
        """
        raw = self.get_config("risk_management", "reference_exit_check_seconds", 10)
        if isinstance(raw, bool):
            raise ValueError("reference_exit_check_seconds must be a number of seconds")  # noqa: TRY004
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("reference_exit_check_seconds must be a number of seconds") from exc
        if not math.isfinite(value) or value < 0 or value > MAX_REFERENCE_EXIT_CHECK_SECONDS:
            raise ValueError(
                "reference_exit_check_seconds must be 0 (disabled) or between "
                f"{MIN_REFERENCE_EXIT_CHECK_SECONDS:g} and "
                f"{MAX_REFERENCE_EXIT_CHECK_SECONDS:g}"
            )
        if 0 < value < MIN_REFERENCE_EXIT_CHECK_SECONDS:
            raise ValueError(
                "reference_exit_check_seconds must be 0 (disabled) or at least "
                f"{MIN_REFERENCE_EXIT_CHECK_SECONDS:g} — a faster poll only burns "
                f"the exchange's public rate limit"
            )
        return value


    @property
    def EXECUTOR_API_ENABLED(self) -> bool:
        """Forward decisions to llm_trader_executor via HTTP? (default: false)"""
        return self.get_config("executor_api", "enabled", False)

    @property
    def EXECUTOR_API_URL(self) -> str:
        """URL of the llm_trader_executor REST API."""
        return self.get_config("executor_api", "url", "http://127.0.0.1:9199/decision")

    @property
    def EXECUTOR_VERDICT_PATH(self) -> str:
        """Filesystem path of the executor's verdict journal (JSONL).

        The executor appends one outcome line per processed decision, keyed by
        the bot's order_id. The bot reads this to confirm what happened to its
        order instead of polling /position and guessing.
        """
        return self.get_config(
            "executor_api", "verdict_path", "data/trading/executor_verdicts.jsonl"
        )

    @property
    def BOT_INTENT_JOURNAL_PATH(self) -> str:
        """JSONL journal of position intents (one line per forwarded command).

        Lives next to the other trading state so a test or a second deployment can
        point the bot at an isolated directory instead of the shared default.
        """
        default = os.path.join(str(self.DATA_DIR), "trading", "bot_position_intents.jsonl")
        return str(self.get_config("executor_api", "intent_journal_path", default))

    @property
    def EXECUTOR_EXIT_PATH(self) -> str:
        """Filesystem path of the executor's exit journal (JSONL).

        The executor appends one line whenever a tracked position exits ON THE
        EXCHANGE (a stop-loss or take-profit order filled while the bot was
        between cycles). The bot reads it to book the CLOSE with the real fill
        price instead of keeping the entry open forever.
        """
        return self.get_config(
            "executor_api", "exit_path", "data/trading/executor_exits.jsonl"
        )

    @property
    def EXECUTOR_MAX_POSITION_USDC(self) -> float:
        """Hard cap on position notional forwarded to the executor (USDC).

        0/absent = no clamp (let the executor enforce its own limit).
        When set, entry sizing is scaled down so quantity × price never
        exceeds this value — prevents the executor rejecting every BUY
        with "Notional X exceeds max Y".
        """
        try:
            return float(self.get_config("executor_api", "max_position_usdc", 0.0))
        except (TypeError, ValueError):
            return 0.0

    @property
    def QUOTE_CURRENCY(self):
        """Extract quote currency from CRYPTO_PAIR (e.g., 'USDC' from 'BTC/USDC')."""
        pair = self.CRYPTO_PAIR
        if "/" in pair:
            return pair.split("/")[1]
        return "USDC"


    def get_model_config(self, model_name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Get configuration parameters for a specific model.
        Returns:
            A dictionary with configuration parameters
        """
        if self._is_google_model(model_name):
            base = self._google_model_config.copy()
        elif self._is_deepseek_model(model_name):
            base = self._deepseek_model_config.copy()
        else:
            base = self._default_model_config.copy()

        if overrides:
            base.update(overrides)

        cleaned = {k: v for k, v in base.items() if v is not None}
        return cleaned

    def _is_google_model(self, model_name: str) -> bool:
        """Determine if a model should use Google-specific configuration."""
        return model_name == self.GOOGLE_STUDIO_MODEL

    def _is_deepseek_model(self, model_name: str) -> bool:
        """Determine if a model should use DeepSeek-specific configuration."""
        return model_name == self.DEEPSEEK_MODEL


config = Config()
