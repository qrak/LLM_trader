"""Factory for creating AI provider clients based on configuration."""
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.contracts.config import ConfigProtocol

from src.logger.logger import Logger
from src.platforms.ai_providers import OpenRouterClient, GoogleAIClient, LMStudioClient


class ProviderFactory:
    """
    Factory for creating AI provider client instances based on configuration.
    
    Centralizes provider instantiation logic and handles API key validation.
    Supports multiple providers: Google AI Studio (free/paid), OpenRouter, LM Studio (local).
    
    Usage:
        factory = ProviderFactory(logger, config)
        google_client, google_paid_client = factory.create_google_clients()
        openrouter_client = factory.create_openrouter_client()
        lmstudio_client = factory.create_lmstudio_client()
    """
    
    def __init__(self, logger: Logger, config: "ConfigProtocol"):
        """
        Initialize the provider factory.
        
        Args:
            logger: Logger instance for logging
            config: ConfigProtocol instance for configuration access
        """
        self.logger = logger
        self.config = config
    
    def create_google_clients(self) -> Tuple[Optional[GoogleAIClient], Optional[GoogleAIClient]]:
        """
        Create Google AI clients (free tier and optional paid tier).
        
        Returns:
            Tuple of (google_client, google_paid_client). Both can be None if API keys not configured.
        """
        google_client: Optional[GoogleAIClient] = None
        google_paid_client: Optional[GoogleAIClient] = None
        
        # Initialize Google AI client if API key is available
        if self.config.GOOGLE_STUDIO_API_KEY:
            google_client = GoogleAIClient(
                api_key=self.config.GOOGLE_STUDIO_API_KEY,
                model=self.config.GOOGLE_STUDIO_MODEL,
                logger=self.logger
            )
            self.logger.info("Google AI client initialized")
            
            # Initialize paid client if paid API key is available
            if self.config.GOOGLE_STUDIO_PAID_API_KEY:
                google_paid_client = GoogleAIClient(
                    api_key=self.config.GOOGLE_STUDIO_PAID_API_KEY,
                    model=self.config.GOOGLE_STUDIO_MODEL,
                    logger=self.logger
                )
                self.logger.info("Google AI paid client initialized as fallback for overloaded free tier")
        
        return google_client, google_paid_client
    
    def create_openrouter_client(self) -> Optional[OpenRouterClient]:
        """
        Create OpenRouter client if API key is configured.
        
        Returns:
            OpenRouterClient instance or None if API key not configured.
        """
        if not self.config.OPENROUTER_API_KEY:
            return None
        
        client = OpenRouterClient(
            api_key=self.config.OPENROUTER_API_KEY,
            base_url=self.config.OPENROUTER_BASE_URL,
            logger=self.logger
        )
        self.logger.info("OpenRouter client initialized")
        return client
    
    def create_lmstudio_client(self) -> Optional[LMStudioClient]:
        """
        Create LM Studio client for local inference.
        
        Returns:
            LMStudioClient instance or None if base URL not configured.
        """
        if not self.config.LM_STUDIO_BASE_URL:
            return None
        
        client = LMStudioClient(
            base_url=self.config.LM_STUDIO_BASE_URL,
            logger=self.logger
        )
        self.logger.info(f"LM Studio client initialized for URL: {self.config.LM_STUDIO_BASE_URL}")
        return client
    
    def create_all_clients(self) -> dict:
        """
        Create all available AI provider clients based on configuration.
        
        Returns:
            Dictionary with keys: 'google', 'google_paid', 'openrouter', 'lmstudio'.
            Values are client instances or None if not configured.
        """
        google_client, google_paid_client = self.create_google_clients()
        
        return {
            'google': google_client,
            'google_paid': google_paid_client,
            'openrouter': self.create_openrouter_client(),
            'lmstudio': self.create_lmstudio_client()
        }
