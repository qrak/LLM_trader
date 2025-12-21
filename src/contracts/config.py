"""
Config Protocol - Interface for configuration management.

Defines the contract for configuration access without requiring concrete Config import.
Prevents circular dependencies by using typing.Protocol.
"""

from typing import Any, Dict, Protocol


class ConfigProtocol(Protocol):
    """Protocol defining the interface for configuration management.
    
    All configuration properties and methods must be defined here.
    Implementations must provide all these members to satisfy the Protocol.
    """
    
    # ===== Environment Variables (Private Keys) =====
    @property
    def BOT_TOKEN_DISCORD(self) -> str | None: ...
    
    @property
    def GUILD_ID_DISCORD(self) -> int | None: ...
    
    @property
    def MAIN_CHANNEL_ID(self) -> int | None: ...
    
    @property
    def TEMPORARY_CHANNEL_ID_DISCORD(self) -> int | None: ...
    
    @property
    def OPENROUTER_API_KEY(self) -> str | None: ...
    
    @property
    def GOOGLE_STUDIO_API_KEY(self) -> str | None: ...
    
    @property
    def GOOGLE_STUDIO_PAID_API_KEY(self) -> str | None: ...
    
    @property
    def CRYPTOCOMPARE_API_KEY(self) -> str | None: ...
    
    @property
    def ADMIN_USER_IDS(self) -> list[int]: ...
    
    # ===== AI Provider Configuration =====
    @property
    def PROVIDER(self) -> str: ...
    
    @property
    def LM_STUDIO_BASE_URL(self) -> str: ...
    
    @property
    def LM_STUDIO_MODEL(self) -> str: ...
    
    @property
    def OPENROUTER_BASE_URL(self) -> str: ...
    
    @property
    def OPENROUTER_BASE_MODEL(self) -> str: ...
    
    @property
    def OPENROUTER_FALLBACK_MODEL(self) -> str: ...
    
    @property
    def GOOGLE_STUDIO_MODEL(self) -> str: ...
    
    # ===== General Configuration =====
    @property
    def LOGGER_DEBUG(self) -> bool: ...
    
    @property
    def TEST_ENVIRONMENT(self) -> bool: ...
    
    @property
    def TIMEFRAME(self) -> str: ...
    
    @property
    def CANDLE_LIMIT(self) -> int: ...
    
    @property
    def AI_CHART_CANDLE_LIMIT(self) -> int: ...
    
    # ===== Debug Configuration =====
    @property
    def DEBUG_SAVE_CHARTS(self) -> bool: ...
    
    @property
    def DEBUG_CHART_SAVE_PATH(self) -> str: ...
    
    # ===== Directory Configuration =====
    @property
    def LOG_DIR(self) -> str: ...
    
    @property
    def DATA_DIR(self) -> str: ...
    
    # ===== Cooldown Configuration =====
    @property
    def ANALYSIS_COOLDOWN_COIN(self) -> int: ...
    
    @property
    def ANALYSIS_COOLDOWN_USER(self) -> int: ...
    
    @property
    def FILE_MESSAGE_EXPIRY(self) -> int: ...
    
    # ===== RAG Configuration =====
    @property
    def RAG_UPDATE_INTERVAL_HOURS(self) -> int: ...
    
    @property
    def RAG_CATEGORIES_UPDATE_INTERVAL_HOURS(self) -> int: ...
    
    @property
    def RAG_COINGECKO_UPDATE_INTERVAL_HOURS(self) -> int: ...
    
    @property
    def RAG_COINGECKO_GLOBAL_API_URL(self) -> str: ...
    
    RAG_NEWS_API_URL: str
    RAG_CATEGORIES_API_URL: str
    RAG_PRICE_API_URL: str
    
    # ===== Language Configuration =====
    @property
    def SUPPORTED_LANGUAGES(self) -> Dict[str, str]: ...
    
    @property
    def DEFAULT_LANGUAGE(self) -> str: ...
    
    # ===== Exchange Configuration =====
    @property
    def SUPPORTED_EXCHANGES(self) -> list[str]: ...
    
    @property
    def MARKET_REFRESH_HOURS(self) -> int: ...
    
    # ===== Methods =====
    def get_env(self, key: str, default: Any = None) -> Any: ...
    
    def get_config(self, section: str, key: str, default: Any = None) -> Any: ...
    
    def get_section(self, section: str) -> Dict[str, Any]: ...
    
    def get_model_config(self, model_name: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]: ...
    
    def reload(self) -> None: ...
