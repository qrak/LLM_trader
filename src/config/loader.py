"""
Configuration loader for LLM_Trader v2.
Loads private keys from keys.env and public configuration from config.ini.
"""

import configparser
import logging
from pathlib import Path
from typing import Any, Dict
from dotenv import dotenv_values

# Get the root directory (where keys.env is located) and config directory (where config.ini is located)
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
CONFIG_DIR = ROOT_DIR / "config"
KEYS_ENV_PATH = ROOT_DIR / "keys.env"
CONFIG_INI_PATH = CONFIG_DIR / "config.ini"

VALID_PROVIDERS = {"local", "googleai", "openrouter", "blockrun", "all"}

class Config:
    """Configuration class that loads settings from environment and INI files.
    
    Implements ConfigProtocol for type safety and dependency injection.
    """
    
    def __init__(self):
        self._env_vars = {}
        self._config_data = {}
        self._load_environment()
        self._load_ini_config()
        self._validate_provider()
        self._build_dynamic_urls()
        self._build_model_configs()
    
    def _load_environment(self):
        """Load environment variables from keys.env file using python-dotenv."""
        if not KEYS_ENV_PATH.exists():
            raise FileNotFoundError(
                f"Private keys file not found: {KEYS_ENV_PATH}. "
                "Please create keys.env in the root directory with your API keys."
            )
        
        try:
            # Use dotenv_values to parse the .env file
            env_vars = dotenv_values(KEYS_ENV_PATH)
            
            # Convert values to appropriate types
            for key, value in env_vars.items():
                if value is not None:
                    # Convert numeric strings to integers
                    if value.isdigit():
                        value = int(value)
                    self._env_vars[key] = value
                    
        except Exception as e:
            raise RuntimeError(f"Error loading environment file {KEYS_ENV_PATH}: {e}") from e
    
    def _load_ini_config(self):
        """Load configuration from config.ini file."""
        if not CONFIG_INI_PATH.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {CONFIG_INI_PATH}. "
                "Please create config.ini in the config directory."
            )
        
        try:
            config = configparser.ConfigParser()
            config.read(CONFIG_INI_PATH, encoding='utf-8')
            
            for section_name in config.sections():
                section_data = {}
                for key, value in config.items(section_name):
                    # Type conversion
                    section_data[key] = self._convert_value(value)
                self._config_data[section_name] = section_data
        except Exception as e:
            raise RuntimeError(f"Error loading configuration file {CONFIG_INI_PATH}: {e}") from e
    
    def _validate_provider(self):
        """Validate that the configured AI provider is supported."""
        provider = self.PROVIDER.lower()
        if provider not in VALID_PROVIDERS:
            # Create a formatted list of valid options
            valid_options = ", ".join(f'"{p}"' for p in sorted(VALID_PROVIDERS))
            error_msg = (
                f"Invalid AI provider '{provider}' in config.ini.\n"
                f"Supported values are: {valid_options}.\n"
                f"Please update the [ai_providers] -> provider setting."
            )
            logging.critical(error_msg)
            raise ValueError(error_msg)

    @staticmethod
    def _convert_value(value: str) -> Any:
        """Convert string values to appropriate Python types."""
        if value.lower() in ('true', 'yes', 'on', '1'):
            return True
        elif value.lower() in ('false', 'no', 'off', '0'):
            return False
        if value.isdigit():
            return int(value)
        try:
            if '.' in value:
                return float(value)
        except ValueError:
            pass
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        return value
    
    def _build_dynamic_urls(self):
        """Build dynamic URLs that depend on API keys."""
        cryptocompare_key = self.get_env('CRYPTOCOMPARE_API_KEY')
        
        # Build base URLs
        news_url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&limit=200&extraParams=LLM_Trader_v2"
        categories_url = "https://min-api.cryptocompare.com/data/news/categories"
        price_url = "https://min-api.cryptocompare.com/data/pricemultifull?fsyms=BTC,ETH,BNB,SOL,XRP&tsyms=USD"
        
        # Append API key if available
        if cryptocompare_key:
            self.RAG_NEWS_API_URL = f"{news_url}&api_key={cryptocompare_key}"
            self.RAG_CATEGORIES_API_URL = f"{categories_url}?api_key={cryptocompare_key}"
            self.RAG_PRICE_API_URL = f"{price_url}&api_key={cryptocompare_key}"
        else:
            self.RAG_NEWS_API_URL = news_url
            self.RAG_CATEGORIES_API_URL = categories_url
            self.RAG_PRICE_API_URL = price_url
    
    def _build_model_configs(self):
        """Build model configuration dictionaries as instance variables."""
        default_max_tokens = self.get_config('model_config', 'max_tokens', None)
        if default_max_tokens is None:
            raise RuntimeError("`max_tokens` is required in [model_config] of config.ini")

        self._default_model_config = {
            "temperature": self.get_config('model_config', 'temperature', None),
            "top_p": self.get_config('model_config', 'top_p', None),
            "top_k": self.get_config('model_config', 'top_k', None),
            "freq_penalty": self.get_config('model_config', 'freq_penalty', None),
            "pres_penalty": self.get_config('model_config', 'pres_penalty', None),
            "max_tokens": default_max_tokens
        }

        google_max_tokens = self.get_config('model_config', 'google_max_tokens', None)
        
        # Only enforce Google config if we are actually using it
        if google_max_tokens is None and self.PROVIDER in ('googleai', 'all'):
            raise RuntimeError("`google_max_tokens` is required in [model_config] of config.ini when using Google models")

        self._google_model_config = {
            "temperature": self.get_config('model_config', 'google_temperature', None),
            "top_p": self.get_config('model_config', 'google_top_p', None),
            "top_k": self.get_config('model_config', 'google_top_k', None),
            "max_tokens": google_max_tokens,
            "thinking_level": self.get_config('model_config', 'google_thinking_level', 'high')
        }
    
    def get_env(self, key: str, default: Any = None) -> Any:
        """Get environment variable."""
        return self._env_vars.get(key, default)
    
    def get_config(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value from INI file."""
        return self._config_data.get(section, {}).get(key, default)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self._config_data.get(section, {})
    
    # Environment variables (private keys and sensitive data)
    @property
    def BOT_TOKEN_DISCORD(self):
        return self.get_env('BOT_TOKEN_DISCORD')
    
    @property
    def GUILD_ID_DISCORD(self):
        return self.get_env('GUILD_ID_DISCORD')
    
    @property
    def MAIN_CHANNEL_ID(self):
        return self.get_env('MAIN_CHANNEL_ID')
    
    @property
    def TEMPORARY_CHANNEL_ID_DISCORD(self):
        return self.get_env('TEMPORARY_CHANNEL_ID_DISCORD')
    
    @property
    def OPENROUTER_API_KEY(self):
        return self.get_env('OPENROUTER_API_KEY')
    
    @property
    def GOOGLE_STUDIO_API_KEY(self):
        return self.get_env('GOOGLE_STUDIO_API_KEY')
    
    @property
    def GOOGLE_STUDIO_PAID_API_KEY(self):
        return self.get_env('GOOGLE_STUDIO_PAID_API_KEY')
    
    @property
    def CRYPTOCOMPARE_API_KEY(self):
        return self.get_env('CRYPTOCOMPARE_API_KEY')

    @property
    def COINGECKO_API_KEY(self):
        return self.get_env('COINGECKO_API_KEY')
    
    @property
    def BLOCKRUN_WALLET_KEY(self):
        return self.get_env('BLOCKRUN_WALLET_KEY')
    
    @property
    def ADMIN_USER_IDS(self):
        """Get list of admin user IDs from environment."""
        admin_ids = self.get_env('ADMIN_USER_IDS', '')
        if not admin_ids:
            return []

        # Handle single integer values produced by automatic type conversion
        if isinstance(admin_ids, int):
            return [admin_ids]

        # Handle pre-parsed iterables (lists/tuples) defensively
        if isinstance(admin_ids, (list, tuple)):
            parsed_ids = []
            for raw_id in admin_ids:
                try:
                    parsed_ids.append(int(str(raw_id).strip()))
                except (TypeError, ValueError):
                    logging.warning("Invalid ADMIN_USER_IDS entry '%s' in keys.env. Expected integers.", raw_id)
                    return []
            return parsed_ids

        if isinstance(admin_ids, str):
            try:
                return [int(uid.strip()) for uid in admin_ids.split(',') if uid.strip()]
            except ValueError:
                logging.warning("Invalid ADMIN_USER_IDS format in keys.env. Expected comma-separated integers.")
                return []

        logging.warning(
            "Unsupported ADMIN_USER_IDS type %s encountered. Expected string, int, list, or tuple.",
            type(admin_ids).__name__
        )
        return []
    
    # AI Provider Configuration
    @property
    def PROVIDER(self):
        return self.get_config('ai_providers', 'provider', 'googleai')
    
    @property
    def LM_STUDIO_BASE_URL(self):
        return self.get_config('ai_providers', 'lm_studio_base_url', 'http://localhost:1234/v1')
    
    @property
    def LM_STUDIO_MODEL(self):
        return self.get_config('ai_providers', 'lm_studio_model', 'local-model')

    @property
    def LM_STUDIO_STREAMING(self):
        return self.get_config('ai_providers', 'lm_studio_streaming', True)
    
    @property
    def OPENROUTER_BASE_URL(self):
        return self.get_config('ai_providers', 'openrouter_base_url', 'https://openrouter.ai/api/v1')
    
    @property
    def OPENROUTER_BASE_MODEL(self):
        return self.get_config('ai_providers', 'openrouter_base_model', 'google/gemini-2.5-pro')
    
    @property
    def OPENROUTER_FALLBACK_MODEL(self):
        return self.get_config('ai_providers', 'openrouter_fallback_model', 'deepseek/deepseek-r1:free')
    
    @property
    def GOOGLE_STUDIO_MODEL(self):
        return self.get_config('ai_providers', 'google_studio_model', 'gemini-2.5-flash')
    
    @property
    def BLOCKRUN_BASE_URL(self):
        return self.get_config('ai_providers', 'blockrun_base_url', 'https://blockrun.ai/api')
    
    @property
    def BLOCKRUN_MODEL(self):
        return self.get_config('ai_providers', 'blockrun_model', 'openai/gpt-4o')
    
    # General Configuration
    @property
    def LOGGER_DEBUG(self):
        return self.get_config('debug', 'logger_debug', False)
    

    
    @property
    def CRYPTO_PAIR(self):
        return self.get_config('general', 'crypto_pair', 'BTC/USDT')

    @property
    def DISCORD_BOT_ENABLED(self):
        return self.get_config('general', 'discord_bot', False)
    
    @property
    def TIMEFRAME(self):
        return self.get_config('general', 'timeframe', '1h')
    
    @property
    def CANDLE_LIMIT(self):
        return self.get_config('general', 'candle_limit', 999)

    @property
    def AI_CHART_CANDLE_LIMIT(self):
        """Configured candle limit to use for AI chart images (must be present in config.ini)."""
        return int(self.get_config('general', 'ai_chart_candle_limit', 200))
    
    # Debug Configuration
    @property
    def DEBUG_SAVE_CHARTS(self):
        return self.get_config('debug', 'save_chart_images', False)
    
    @property
    def DEBUG_CHART_SAVE_PATH(self):
        return self.get_config('debug', 'chart_save_path', 'test_images')
    
    # Directory Configuration
    @property
    def LOG_DIR(self):
        return self.get_config('directories', 'log_dir', 'logs')
    
    @property
    def DATA_DIR(self):
        return self.get_config('directories', 'data_dir', 'data')
    
    # Cooldown Configuration

    
    @property
    def FILE_MESSAGE_EXPIRY(self):
        """Get file message expiry time in seconds (configured in hours in config.ini)."""
        hours = self.get_config('cooldowns', 'file_message_expiry', 168)
        return hours * 3600
    
    # RAG Configuration
    @property
    def RAG_UPDATE_INTERVAL_HOURS(self):
        return self.get_config('rag', 'update_interval_hours', 4)
    
    @property
    def RAG_CATEGORIES_UPDATE_INTERVAL_HOURS(self):
        return self.get_config('rag', 'categories_update_interval_hours', 24)
    
    @property
    def RAG_COINGECKO_UPDATE_INTERVAL_HOURS(self):
        return self.get_config('rag', 'coingecko_update_interval_hours', 24)
    
    @property
    def RAG_COINGECKO_GLOBAL_API_URL(self):
        return self.get_config('rag', 'coingecko_global_api_url', 'https://api.coingecko.com/api/v3/global')

    @property
    def RAG_NEWS_LIMIT(self):
        """Maximum number of news articles to include in context (configurable via [rag] news_limit)."""
        return int(self.get_config('rag', 'news_limit', 5))

    @property
    def RAG_ARTICLE_MAX_SENTENCES(self):
        """Maximum number of sentences per article (configurable via [rag] article_max_sentences)."""
        return int(self.get_config('rag', 'article_max_sentences', 6))

    @property
    def RAG_ARTICLE_MAX_TOKENS(self):
        """Maximum number of tokens per article (configurable via [rag] article_max_tokens)."""
        return int(self.get_config('rag', 'article_max_tokens', 256))
    @property
    def SUPPORTED_EXCHANGES(self):
        """Returns list of supported exchanges in priority order."""
        return self.get_config('exchanges', 'supported', ['binance', 'kucoin', 'gateio'])
    
    @property
    def MARKET_REFRESH_HOURS(self):
        return self.get_config('exchanges', 'market_refresh_hours', 24)

    # Demo Trading Configuration
    @property
    def TRANSACTION_FEE_PERCENT(self):
        """Transaction fee percentage for limit orders (default 0.075%)."""
        return float(self.get_config('demo_trading', 'transaction_fee_percent', 0.00075))

    @property
    def DEMO_QUOTE_CAPITAL(self):
        """Initial capital for demo trading (default 10000)."""
        return float(self.get_config('demo_trading', 'demo_quote_capital', 10000.0))

    # Risk Management Defaults
    @property
    def DEFAULT_POSITION_SIZE(self):
        """Default position size as decimal (e.g. 0.02) if AI doesn't specify."""
        return float(self.get_config('risk_management', 'default_position_size', 0.02))

    @property
    def DEFAULT_STOP_LOSS_PCT(self):
        """Default stop loss percentage as decimal (e.g. 0.02) if AI doesn't specify."""
        return float(self.get_config('risk_management', 'default_stop_loss_pct', 0.02))

    @property
    def DEFAULT_TAKE_PROFIT_PCT(self):
        """Default take profit percentage as decimal (e.g. 0.04) if AI doesn't specify."""
        return float(self.get_config('risk_management', 'default_take_profit_pct', 0.04))

    @property
    def QUOTE_CURRENCY(self):
        """Extract quote currency from CRYPTO_PAIR (e.g., 'USDC' from 'BTC/USDC')."""
        pair = self.CRYPTO_PAIR
        if '/' in pair:
            return pair.split('/')[1]
        return 'USDC'


    def get_model_config(self, model_name: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get configuration parameters for a specific model.
        
        Args:
            model_name: The name of the model
            overrides: Optional parameter overrides for this specific call
            
        Returns:
            A dictionary with configuration parameters
        """
        if self._is_google_model(model_name):
            base = self._google_model_config.copy()
        else:
            base = self._default_model_config.copy()

        if overrides:
            base.update(overrides)

        cleaned = {k: v for k, v in base.items() if v is not None}
        return cleaned

    def _is_google_model(self, model_name: str) -> bool:
        """Determine if a model should use Google-specific configuration."""
        return model_name == self.GOOGLE_STUDIO_MODEL

    def reload(self):
        """Reload both keys.env and config.ini files.
        
        This allows runtime configuration changes without restarting the application.
        """
        logging.info("Reloading configuration files...")
        try:
            self._env_vars = {}
            self._config_data = {}
            self._load_environment()
            self._load_ini_config()
            self._build_dynamic_urls()
            self._build_model_configs()
            logging.info("Configuration reloaded successfully")
        except Exception as e:
            logging.error(f"Error reloading configuration: {e}")
            raise


# Create global config instance
config = Config()
