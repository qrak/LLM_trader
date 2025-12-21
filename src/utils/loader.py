"""
Configuration loader for DiscordCryptoAnalyzer.
Loads private keys from keys.env and public configuration from config.ini.
"""

import configparser
import logging
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING
from dotenv import dotenv_values

if TYPE_CHECKING:
    from src.contracts.config import ConfigProtocol

# Get the root directory (where keys.env is located) and config directory (where config.ini is located)
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
CONFIG_DIR = ROOT_DIR / "config"
KEYS_ENV_PATH = ROOT_DIR / "keys.env"
CONFIG_INI_PATH = CONFIG_DIR / "config.ini"

class Config:
    """Configuration class that loads settings from environment and INI files.
    
    Implements ConfigProtocol for type safety and dependency injection.
    """
    
    def __init__(self):
        self._env_vars = {}
        self._config_data = {}
        self._load_environment()
        self._load_ini_config()
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
            raise RuntimeError(f"Error loading environment file {KEYS_ENV_PATH}: {e}")
    
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
            raise RuntimeError(f"Error loading configuration file {CONFIG_INI_PATH}: {e}")
    
    def _convert_value(self, value: str) -> Any:
        """Convert string values to appropriate Python types."""
        # Boolean conversion
        if value.lower() in ('true', 'yes', 'on', '1'):
            return True
        elif value.lower() in ('false', 'no', 'off', '0'):
            return False
        
        # Integer conversion
        if value.isdigit():
            return int(value)
        
        # Float conversion
        try:
            if '.' in value:
                return float(value)
        except ValueError:
            pass
        
        # List conversion (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        # Return as string
        return value
    
    def _build_dynamic_urls(self):
        """Build dynamic URLs that depend on API keys."""
        cryptocompare_key = self.get_env('CRYPTOCOMPARE_API_KEY')
        if cryptocompare_key:
            self.RAG_NEWS_API_URL = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&limit=200&extraParams=KurusDiscordCryptoBot&api_key={cryptocompare_key}"
            self.RAG_CATEGORIES_API_URL = f"https://min-api.cryptocompare.com/data/news/categories?api_key={cryptocompare_key}"
            self.RAG_PRICE_API_URL = f"https://min-api.cryptocompare.com/data/pricemultifull?fsyms=BTC,ETH,BNB,SOL,XRP&tsyms=USD&api_key={cryptocompare_key}"
    
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
        if google_max_tokens is None:
            raise RuntimeError("`google_max_tokens` is required in [model_config] of config.ini when using Google models")

        self._google_model_config = {
            "temperature": self.get_config('model_config', 'google_temperature', None),
            "top_p": self.get_config('model_config', 'google_top_p', None),
            "top_k": self.get_config('model_config', 'google_top_k', None),
            "max_tokens": google_max_tokens
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
    
    # General Configuration
    @property
    def LOGGER_DEBUG(self):
        return self.get_config('debug', 'logger_debug', False)
    
    @property
    def TEST_ENVIRONMENT(self):
        return self.get_config('debug', 'test_environment', False)
    
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
    def ANALYSIS_COOLDOWN_COIN(self):
        return self.get_config('cooldowns', 'analysis_cooldown_coin', 3600)
    
    @property
    def ANALYSIS_COOLDOWN_USER(self):
        return self.get_config('cooldowns', 'analysis_cooldown_user', 3600)
    
    @property
    def FILE_MESSAGE_EXPIRY(self):
        """Get file message expiry time in seconds (configured in hours in config.ini)."""
        hours = self.get_config('cooldowns', 'file_message_expiry', 168)
        return hours * 3600
    
    # RAG Configuration
    @property
    def RAG_UPDATE_INTERVAL_HOURS(self):
        return self.get_config('rag', 'update_interval_hours', 1)
    
    @property
    def RAG_CATEGORIES_UPDATE_INTERVAL_HOURS(self):
        return self.get_config('rag', 'categories_update_interval_hours', 24)
    
    @property
    def RAG_COINGECKO_UPDATE_INTERVAL_HOURS(self):
        return self.get_config('rag', 'coingecko_update_interval_hours', 24)
    
    @property
    def RAG_COINGECKO_GLOBAL_API_URL(self):
        return self.get_config('rag', 'coingecko_global_api_url', 'https://api.coingecko.com/api/v3/global')
    
    # Language Configuration
    @property
    def SUPPORTED_LANGUAGES(self):
        """Returns a dictionary mapping language names to codes."""
        names = self.get_config('languages', 'supported', ['English'])
        codes = self.get_config('languages', 'supported_codes', ['en'])
        
        if len(names) != len(codes):
            logging.warning("Mismatch between language names and codes, using defaults")
            return {"English": "en"}
        
        return dict(zip(names, codes))
    
    @property
    def DEFAULT_LANGUAGE(self):
        return self.get_config('languages', 'default', 'English')
    
    # Exchange Configuration
    @property
    def SUPPORTED_EXCHANGES(self):
        """Returns list of supported exchanges in priority order."""
        return self.get_config('exchanges', 'supported', ['binance', 'kucoin', 'gateio'])
    
    @property
    def MARKET_REFRESH_HOURS(self):
        return self.get_config('exchanges', 'market_refresh_hours', 24)
    
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
