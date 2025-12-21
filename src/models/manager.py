import json
import io
from typing import Optional, Dict, Any, List, Union, cast, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.contracts.config import ConfigProtocol

from src.logger.logger import Logger
from src.parsing.unified_parser import UnifiedParser
from src.parsing.response_formatter import ResponseFormatter
from src.platforms.ai_providers.openrouter import ResponseDict
from src.platforms.ai_providers import OpenRouterClient, GoogleAIClient, LMStudioClient
from src.utils.token_counter import TokenCounter
from src.contracts.model_manager import ModelManagerProtocol
from src.factories import ProviderFactory


class ModelManager(ModelManagerProtocol):
    """Manages interactions with AI models"""

    def __init__(self, logger: Logger, config: "ConfigProtocol") -> None:
        """Initialize the ModelManager with its component parts.
        
        Args:
            logger: Logger instance for logging
            config: ConfigProtocol instance for configuration access
        
        Raises:
            ValueError: If config is None
        """
        if config is None:
            raise ValueError("config is a required parameter and cannot be None")
        
        self.logger = logger
        self.config = config
        self.provider = self.config.PROVIDER.lower()

        # Create API clients using factory pattern
        # Always initialize all available clients to support runtime provider overrides
        factory = ProviderFactory(logger, config)
        clients = factory.create_all_clients()
        
        self.openrouter_client: Optional[OpenRouterClient] = clients['openrouter']
        self.google_client: Optional[GoogleAIClient] = clients['google']
        self.google_paid_client: Optional[GoogleAIClient] = clients['google_paid']
        self.lm_studio_client: Optional[LMStudioClient] = clients['lmstudio']

        # Create helper components
        self.unified_parser = UnifiedParser(logger)
        self.token_counter = TokenCounter()
        self.response_formatter = ResponseFormatter()

        # Cache model names from config (DRY - avoid repeated config lookups)
        self.google_model = self.config.GOOGLE_STUDIO_MODEL
        self.openrouter_model = self.config.OPENROUTER_BASE_MODEL
        self.lmstudio_model = self.config.LM_STUDIO_MODEL

        # Create model configurations as instance variables
        self.model_config = self.config.get_model_config(self.lmstudio_model)
        self.google_config = self.config.get_model_config(self.google_model)
        self.openrouter_config = self.config.get_model_config(self.openrouter_model)

        # Provider metadata - single source of truth for provider information
        # This eliminates duplicate provider name mappings and model lookups throughout the class
        self.PROVIDER_METADATA = {
            'googleai': {
                'name': 'Google AI Studio',
                'client': self.google_client,
                'paid_client': self.google_paid_client,
                'default_model': self.google_model,
                'config': self.google_config,
                'supports_chart': True,
                'has_rate_limits': False  # Google handles internally
            },
            'openrouter': {
                'name': 'OpenRouter',
                'client': self.openrouter_client,
                'default_model': self.openrouter_model,
                'config': self.openrouter_config,
                'supports_chart': True,
                'has_rate_limits': True
            },
            'local': {
                'name': 'LM Studio',
                'client': self.lm_studio_client,
                'default_model': self.lmstudio_model,
                'config': self.model_config,
                'supports_chart': False,
                'has_rate_limits': False
            }
        }

        # Set up models and their configurations
        if self.provider == "local":
            self.model = self.lmstudio_model
        else:
            self.model = self.openrouter_model

    async def __aenter__(self):
        if self.openrouter_client:
            await self.openrouter_client.__aenter__()
        if self.google_client:
            await self.google_client.__aenter__()
        if self.google_paid_client:
            await self.google_paid_client.__aenter__()
        if self.lm_studio_client:
            await self.lm_studio_client.__aenter__()
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        await self.close()

    async def close(self) -> None:
        """Close all client connections"""
        try:
            if self.openrouter_client:
                await self.openrouter_client.close()

            if self.google_client:
                await self.google_client.close()
            
            if self.google_paid_client:
                await self.google_paid_client.close()

            if self.lm_studio_client:
                await self.lm_studio_client.close()

            self.logger.debug("All model clients closed successfully")
        except Exception as e:
            self.logger.error(f"Error during model manager cleanup: {e}")

    async def send_prompt(self, prompt: str, system_message: str = None, prepared_messages: List[Dict[str, str]] = None,
                         provider: Optional[str] = None, model: Optional[str] = None) -> str:
        """
        Send a prompt to the model and get a response.
        
        Args:
            prompt: User prompt
            system_message: Optional system instructions
            prepared_messages: Pre-prepared message list (if None, will be created from prompt)
            provider: Optional provider override (admin only)
            model: Optional model override (admin only)
            
        Returns:
            Response text from the AI model
        """
        messages = prepared_messages if prepared_messages is not None else self._prepare_messages(prompt, system_message)
        response_json = await self._get_model_response(messages, provider=provider, model=model)
        return self._process_response(response_json)

    async def send_prompt_streaming(self, prompt: str, system_message: str = None, 
                                    provider: Optional[str] = None, model: Optional[str] = None) -> str:
        """
        Send a prompt to the model and get a streaming response.
        
        Args:
            prompt: User prompt
            system_message: Optional system instructions
            provider: Optional provider override (admin only)
            model: Optional model override (admin only)
            
        Returns:
            Complete response text from the AI model
        """
        messages = self._prepare_messages(prompt, system_message)
        
        # Use admin-specified provider if provided
        effective_provider = provider if provider else self.provider
        
        # Only try streaming if using local provider (LM Studio) or all providers
        if (effective_provider == "local" or effective_provider == "all") and self.lm_studio_client:
            try:
                # Use admin-specified model if provided, otherwise use default
                effective_model = model if model else self.config.LM_STUDIO_MODEL
                response_json = await self.lm_studio_client.console_stream(effective_model, messages, self.model_config)
                if response_json is not None:  # Check for valid response before processing
                    return self._process_response(response_json)
                else:
                    self.logger.warning("LM Studio streaming returned None. Falling back to non-streaming mode.")
            except Exception as e:
                self.logger.warning(f"LM Studio streaming failed: {str(e)}. Falling back to non-streaming mode.")
        
        # Fallback to regular prompt for other providers or if streaming fails
        return await self.send_prompt(prompt, system_message, prepared_messages=messages, provider=provider, model=model)
    
    async def send_prompt_with_chart_analysis(self, prompt: str, chart_image: Union[io.BytesIO, bytes, str], 
                                            system_message: str = None, provider: Optional[str] = None, 
                                            model: Optional[str] = None) -> str:
        """
        Send a prompt with chart image for pattern analysis.
        
        Args:
            prompt: User prompt
            chart_image: Chart image data
            system_message: Optional system instructions
            provider: Optional provider override (admin only)
            model: Optional model override (admin only)
            
        Returns:
            Response text from the AI model
        """
        messages = self._prepare_messages(prompt, system_message)
        
        # Use admin-specified provider if provided
        effective_provider = provider if provider else self.provider
        
        # Use the fallback system for chart analysis based on provider configuration
        if effective_provider == "all":
            response_json = await self._get_chart_analysis_fallback_response(messages, chart_image, model)
        elif effective_provider == "googleai" and self.google_client:
            response_json = await self._try_single_provider_chart("googleai", messages, chart_image, model)
        elif effective_provider == "openrouter" and self.openrouter_client:
            response_json = await self._try_single_provider_chart("openrouter", messages, chart_image, model)
        elif effective_provider == "local" and self.lm_studio_client:
            # Local models typically don't support images - fall back to text-only
            self.logger.warning("Chart analysis requested but local provider doesn't support images. Falling back to text-only analysis.")
            raise ValueError("Chart analysis unavailable - local models don't support images")
        else:
            self.logger.error(f"Chart analysis not supported for provider '{effective_provider}' or client not available")
            self._log_provider_unavailable_guidance(effective_provider)
            raise ValueError(f"Chart analysis unavailable - provider '{effective_provider}' not supported")
        
        if self._is_valid_response(response_json):
            return self._process_response(response_json)
        else:
            # Let higher level handle logging - just propagate the error
            raise ValueError("Chart analysis failed - invalid response")

    # ----------------------------
    # Internal DRY helper methods
    # ----------------------------
    def _resolve_model(self, provider: str, model_override: Optional[str] = None) -> str:
        """
        Resolve the effective model name for a provider.
        
        Centralized model resolution eliminates repeated 'model if model else config.X' pattern.
        
        Args:
            provider: Provider ID (googleai, openrouter, local)
            model_override: Optional admin-specified model override
            
        Returns:
            Effective model name to use
        """
        if model_override:
            return model_override
        
        metadata = self.PROVIDER_METADATA.get(provider)
        if metadata:
            return metadata['default_model']
        
        # Fallback for unknown provider
        return "unknown-model"
    
    def _get_provider_name(self, provider: str) -> str:
        """
        Get the display name for a provider.
        
        Args:
            provider: Provider ID (googleai, openrouter, local)
            
        Returns:
            Display name for the provider
        """
        metadata = self.PROVIDER_METADATA.get(provider)
        return metadata['name'] if metadata else provider
    
    def _get_provider_client(self, provider: str):
        """
        Get the client instance for a provider.
        
        Args:
            provider: Provider ID (googleai, openrouter, local)
            
        Returns:
            Client instance or None if not available
        """
        metadata = self.PROVIDER_METADATA.get(provider)
        return metadata['client'] if metadata else None
    
    def _handle_provider_failure(self, provider: str, response_json: Optional[Dict[str, Any]], 
                                 is_final_fallback: bool = False) -> ResponseDict:
        """
        Handle provider failure and create error response.
        
        Args:
            provider: Provider name that failed
            response_json: Response from the provider (may contain error details)
            is_final_fallback: Whether this is the final fallback attempt
            
        Returns:
            Error response dictionary
        """
        provider_name = self._get_provider_name(provider)
        
        self.logger.error(f"{provider_name} request failed or returned invalid response")
        error_detail = response_json.get("error", f"Unknown {provider_name} failure") if response_json else f"No response from {provider_name}"
        
        if is_final_fallback:
            return cast(ResponseDict, self.response_formatter.format_final_fallback_error(provider_name, error_detail))
        else:
            return cast(ResponseDict, self.response_formatter.format_provider_error(provider_name, error_detail))
    
    def _provider_available(self, provider: str) -> bool:
        """Check if a provider's client is available."""
        return self._get_provider_client(provider) is not None

    def _log_provider_action(self, provider: str, *, action: str, chart: bool = False, model: Optional[str] = None) -> None:
        """
        Log provider action using metadata.
        
        Args:
            provider: Provider ID
            action: "attempting" for fallback path, "using" for single-provider path
            chart: Whether this is chart analysis
            model: Optional model override
        """
        noun = "chart analysis" if chart else "request"
        provider_name = self._get_provider_name(provider)
        effective_model = self._resolve_model(provider, model)
        
        if action == "attempting":
            self.logger.info(f"Attempting {noun} with {provider_name} model: {effective_model}")
        else:
            self.logger.info(f"Using {provider_name} model: {effective_model}")

    def _log_provider_unavailable_guidance(self, provider: str) -> None:
        """Log helpful guidance when a provider is unavailable."""
        provider_name = self._get_provider_name(provider)
        metadata = self.PROVIDER_METADATA.get(provider)
        
        if not metadata or not metadata['client']:
            if provider == "openrouter":
                self.logger.error(f"{provider_name} client not initialized. Check that OPENROUTER_API_KEY is set in keys.env")
            elif provider == "googleai":
                self.logger.error(f"{provider_name} client not initialized. Check that GOOGLE_STUDIO_API_KEY is set in keys.env")
            elif provider == "local":
                self.logger.error(f"{provider_name} client not initialized. Check that LM_STUDIO_BASE_URL is set in config.ini and server is running")
        elif provider == "local" and not metadata.get('supports_chart'):
            self.logger.error("Local models don't support image analysis")

    async def _invoke_provider(self, provider: str, messages: List[Dict[str, str]], *, chart: bool = False,
                               chart_image: Optional[Union[io.BytesIO, bytes, str]] = None, 
                               model: Optional[str] = None) -> Dict[str, Any]:
        """
        Invoke a provider for normal or chart analysis requests and return its raw response dict.
        
        Uses PROVIDER_METADATA for client/config lookup and _resolve_model for model resolution.
        
        Args:
            provider: Provider name (googleai, local, openrouter)
            messages: Message list for the AI model
            chart: Whether this is a chart analysis request
            chart_image: Optional chart image data
            model: Optional model override (admin only)
            
        Returns:
            Response dictionary from the provider
        """
        metadata = self.PROVIDER_METADATA.get(provider)
        if not metadata or not metadata['client']:
            return cast(ResponseDict, {"error": f"Provider '{provider}' is not available"})
        
        client = metadata['client']
        config = metadata['config']
        effective_model = self._resolve_model(provider, model)
        
        # Google AI has special free/paid tier logic
        if provider == "googleai":
            return await self._invoke_google_provider(client, messages, config, effective_model, chart, chart_image)
        
        # Local provider doesn't support chart analysis
        if provider == "local":
            if chart:
                return cast(ResponseDict, {"error": "Chart analysis unavailable - local models don't support images"})
            try:
                return await client.chat_completion(effective_model, messages, config)
            except Exception as e:
                return cast(ResponseDict, {"error": f"LM Studio connection failed: {str(e)}"})
        
        # OpenRouter standard flow
        if provider == "openrouter":
            if chart:
                return await client.chat_completion_with_chart_analysis(
                    effective_model, messages, cast(Any, chart_image), config
                )
            return await client.chat_completion(effective_model, messages, config)
        
        return cast(ResponseDict, {"error": f"Provider '{provider}' is not available"})
    
    async def _invoke_google_provider(self, client, messages: List[Dict[str, str]], config: Dict[str, Any],
                                      effective_model: str, chart: bool, 
                                      chart_image: Optional[Union[io.BytesIO, bytes, str]]) -> Dict[str, Any]:
        """
        Invoke Google AI provider with free/paid tier fallback logic.
        
        Args:
            client: Google AI client instance
            messages: Message list for the AI model
            config: Google AI config dict
            effective_model: Resolved model name
            chart: Whether this is chart analysis
            chart_image: Optional chart image data
            
        Returns:
            Response dictionary from Google AI
        """
        # Determine if model supports free tier (only Flash variants)
        is_free_tier_model = "flash" in effective_model.lower()
        tier_info = "free tier" if is_free_tier_model else "paid tier"
        self.logger.info(f"Attempting with Google AI {tier_info} API (model: {effective_model})")
        
        if chart:
            response = await client.chat_completion_with_chart_analysis(messages, cast(Any, chart_image), config, model=effective_model)
        else:
            response = await client.chat_completion(messages, config, model=effective_model)
        
        # If free tier is overloaded and paid client is available, retry with paid API
        if response and response.get("error") == "overloaded" and self.google_paid_client:
            self.logger.warning("Google AI free tier overloaded, retrying with paid API key")
            if chart:
                response = await self.google_paid_client.chat_completion_with_chart_analysis(messages, cast(Any, chart_image), config, model=effective_model)
            else:
                response = await self.google_paid_client.chat_completion(messages, config, model=effective_model)
            
            if self._is_valid_response(response):
                self.logger.info("Successfully used paid Google AI API after free tier overload")
            else:
                # Paid API also failed - log the specific error
                paid_error = response.get("error", "unknown") if response else "no response"
                self.logger.error(f"Paid Google AI API also failed: {paid_error}. Both free and paid tiers unavailable.")
        elif self._is_valid_response(response):
            tier_success = "free tier" if is_free_tier_model else "paid tier"
            self.logger.info(f"Successfully used {tier_success} Google AI API")
        
        return response

    def _valid_for_provider(self, provider: str, response: Optional[Dict[str, Any]]) -> bool:
        """Check validity and rate-limit conditions per provider."""
        if not self._is_valid_response(response):
            return False
        if provider == "openrouter" and response and self._rate_limited(response):
            return False
        return True

    async def _first_success(self, providers: List[str], messages: List[Dict[str, str]], *, chart: bool = False,
                              chart_image: Optional[Union[io.BytesIO, bytes, str]] = None,
                              model: Optional[str] = None, warn_on_fail: bool = True) -> Dict[str, Any]:
        """Try providers in order, returning the first valid response."""
        for idx, provider in enumerate(providers):
            if not self._provider_available(provider):
                continue
            self._log_provider_action(provider, action="attempting", chart=chart, model=model)
            response_json = await self._invoke_provider(provider, messages, chart=chart, chart_image=chart_image, model=model)
            if self._valid_for_provider(provider, response_json):
                return response_json
            if warn_on_fail:
                # Keep similar warning tone to existing code
                if provider == "googleai":
                    self.logger.warning("Google AI Studio model failed. Trying alternatives...")
                elif provider == "local":
                    self.logger.warning("LM Studio failed. Falling back to next provider.")
                elif provider == "openrouter":
                    self.logger.warning("OpenRouter failed or rate limited.")
        return cast(ResponseDict, {"error": "No providers available"})

    async def _get_chart_analysis_fallback_response(self, messages: List[Dict[str, str]], 
                                                    chart_image: Union[io.BytesIO, bytes, str],
                                                    model: Optional[str] = None) -> Dict[str, Any]:
        """Chart analysis fallback logic when provider is 'all'"""
        # Try Google first, then OpenRouter
        response = await self._first_success(["googleai", "openrouter"], messages, chart=True, chart_image=chart_image, model=model)
        if self._is_valid_response(response):
            return response
        return cast(ResponseDict, {"error": "No chart analysis providers available"})

    async def _try_single_provider_chart(self, provider: str, messages: List[Dict[str, str]], 
                                         chart_image: Union[io.BytesIO, bytes, str],
                                         model: Optional[str] = None) -> Dict[str, Any]:
        """
        Try a single provider for chart analysis.
        
        Consolidated method replacing _try_google_chart_analysis_only, _try_openrouter_chart_analysis_only.
        
        Args:
            provider: Provider ID (googleai, openrouter)
            messages: Message list for the AI model
            chart_image: Chart image data
            model: Optional model override
            
        Returns:
            Response dictionary from the provider
        """
        self._log_provider_action(provider, action="using", chart=True, model=model)
        response_json = await self._invoke_provider(provider, messages, chart=True, chart_image=chart_image, model=model)
        
        # Check validity with provider-specific rate limit handling
        metadata = self.PROVIDER_METADATA.get(provider, {})
        has_rate_limits = metadata.get('has_rate_limits', False)
        
        if not self._is_valid_response(response_json) or (has_rate_limits and self._rate_limited(response_json)):
            effective_model = self._resolve_model(provider, model)
            provider_name = self._get_provider_name(provider)
            error_detail = response_json.get("error", "Unknown error") if response_json else "No response"
            self.logger.error(
                f"{provider_name} chart analysis failed: model={effective_model}, error={error_detail}"
            )
            return cast(ResponseDict, {"error": f"{provider_name} chart analysis failed: {error_detail}"})
        return response_json
        
    def supports_image_analysis(self, provider_override: Optional[str] = None) -> bool:
        """Check if the selected provider supports image analysis."""
        provider_name = (provider_override or self.provider or "").lower()
        if provider_name == "all":
            return (self.google_client is not None or self.openrouter_client is not None)
        if provider_name == "googleai":
            return self.google_client is not None
        if provider_name == "openrouter":
            return self.openrouter_client is not None
        if provider_name == "local":
            return False
        return False

    def describe_provider_and_model(
        self,
        provider_override: Optional[str],
        model_override: Optional[str],
        *,
        chart: bool = False,
    ) -> Tuple[str, str]:
        """Return provider + model description for logging and telemetry."""
        provider_name = (provider_override or self.provider or "unknown").lower()

        if model_override:
            return provider_name, model_override

        if provider_name == "googleai":
            return provider_name, self.config.GOOGLE_STUDIO_MODEL
        if provider_name == "openrouter":
            return provider_name, self.config.OPENROUTER_BASE_MODEL
        if provider_name == "local":
            return provider_name, self.config.LM_STUDIO_MODEL
        if provider_name == "all":
            chain: List[str] = []
            # Always list configured defaults, even if client is temporarily unavailable.
            if self.config.GOOGLE_STUDIO_MODEL:
                chain.append(self.config.GOOGLE_STUDIO_MODEL)
            if chart:
                if self.config.OPENROUTER_BASE_MODEL:
                    chain.append(self.config.OPENROUTER_BASE_MODEL)
            else:
                if self.config.LM_STUDIO_MODEL:
                    chain.append(self.config.LM_STUDIO_MODEL)
                if self.config.OPENROUTER_BASE_MODEL:
                    chain.append(self.config.OPENROUTER_BASE_MODEL)

            model_chain = " -> ".join(chain) if chain else "fallback chain unavailable"
            return provider_name, model_chain

        return provider_name, "unspecified"
        
    def _prepare_messages(self, prompt: str, system_message: Optional[str] = None) -> List[Dict[str, str]]:
        """Prepare message structure and track tokens"""
        messages = []
    
        if system_message:
            combined_prompt = f"System instructions: {system_message}\n\nUser query: {prompt}"
            messages.append({"role": "user", "content": combined_prompt})
            
            system_tokens = self.token_counter.track_prompt_tokens(system_message, "system")
            prompt_tokens = self.token_counter.track_prompt_tokens(prompt, "prompt")
            self.logger.info(f"System message token count: {system_tokens}")
            self.logger.info(f"Prompt token count: {prompt_tokens}")
            self.logger.debug(f"Full prompt content: {combined_prompt}")
        else:
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = self.token_counter.track_prompt_tokens(prompt, "prompt")
            self.logger.info(f"Prompt token count: {prompt_tokens}")
            self.logger.debug(f"Full prompt content: {prompt}")
    
        return messages

    async def _get_model_response(self, messages: List[Dict[str, str]], provider: Optional[str] = None, 
                                  model: Optional[str] = None) -> Dict[str, Any]:
        """
        Get response from the selected provider(s).
        
        Args:
            messages: Message list for the AI model
            provider: Optional provider override (admin only)
            model: Optional model override (admin only)
            
        Returns:
            Response dictionary from the AI provider
        """
        # Use admin-specified provider if provided
        effective_provider = provider if provider else self.provider
        
        # If provider is "all", use fallback system (current behavior)
        if effective_provider == "all":
            return await self._get_fallback_response(messages, model)
        
        # Use single provider only
        if effective_provider in self.PROVIDER_METADATA and self._get_provider_client(effective_provider):
            return await self._try_single_provider(effective_provider, messages, model)
        else:
            # Fallback if provider is misconfigured or client not available
            self.logger.error(f"Provider '{effective_provider}' is not properly configured or client not available")
            self._log_provider_unavailable_guidance(effective_provider)
            return cast(ResponseDict, {"error": f"Provider '{effective_provider}' is not available"})

    async def _get_fallback_response(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> Dict[str, Any]:
        """Original fallback logic when provider is 'all'"""
        # Try Google, then LM Studio, then OpenRouter
        response = await self._first_success(["googleai", "local", "openrouter"], messages, model=model)
        if self._is_valid_response(response):
            return response
        # If still no success but OpenRouter is available, log the explicit fallback message then try once more
        if self.openrouter_client:
            return await self._try_openrouter(messages, model)
        return response

    async def _try_single_provider(self, provider: str, messages: List[Dict[str, str]], model: Optional[str] = None) -> Dict[str, Any]:
        """
        Try a single provider for text analysis.
        
        Consolidated method replacing _try_google_only, _try_lm_studio_only, _try_openrouter_only.
        
        Args:
            provider: Provider ID (googleai, local, openrouter)
            messages: Message list for the AI model
            model: Optional model override
            
        Returns:
            Response dictionary from the provider
        """
        self._log_provider_action(provider, action="using", model=model)
        response_json = await self._invoke_provider(provider, messages, model=model)

        # Check validity with provider-specific rate limit handling
        metadata = self.PROVIDER_METADATA.get(provider, {})
        has_rate_limits = metadata.get('has_rate_limits', False)
        
        if not self._is_valid_response(response_json) or (has_rate_limits and self._rate_limited(response_json)):
            return self._handle_provider_failure(provider, response_json)

        return response_json

    async def _try_openrouter(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> Optional[ResponseDict]:
        """Use OpenRouter as fallback"""
        self.logger.warning("Google AI Studio and LM Studio (if enabled) failed. Falling back to OpenRouter...")
        response_json = await self._invoke_provider("openrouter", messages, model=model)

        if not self._is_valid_response(response_json) or self._rate_limited(response_json):
            return self._handle_provider_failure("openrouter", response_json, is_final_fallback=True)

        return response_json

    def _rate_limited(self, response: Dict[str, Any]) -> bool:
        """Check if the response indicates rate limiting"""
        return (response and 
                isinstance(response, dict) and 
                response.get("error") == "rate_limit")

    def _is_valid_response(self, response: Optional[Dict[str, Any]]) -> bool:
        """Check if response contains valid choices with content"""
        # Check if response exists and is a dict with choices
        if not response or not isinstance(response, dict):
            return False
        
        if "choices" not in response or not response["choices"]:
            return False
        
        # Check first choice for errors or empty content
        first_choice = response["choices"][0]
        
        # Check if choice itself contains an error
        if "error" in first_choice:
            error_detail = first_choice['error']
            error_code = error_detail.get('code', 'unknown') if isinstance(error_detail, dict) else error_detail
            error_msg = error_detail.get('message', 'unknown') if isinstance(error_detail, dict) else str(error_detail)
            provider = error_detail.get('metadata', {}).get('provider_name', 'unknown') if isinstance(error_detail, dict) else 'unknown'
            
            self.logger.error(
                f"Error in API response choice from {provider}: [{error_code}] {error_msg}"
            )
            
            # Log full error details in debug mode
            self.logger.debug(f"Full error details: {error_detail}")
            return False
        
        # Check if message content exists and is not empty
        message = first_choice.get("message", {})
        content = message.get("content", "")
        
        if not content:
            self.logger.debug(f"Empty content in API response choice. Message: {message}")
            return False
        
        return True

    async def _try_google_api(self, messages: List[Dict[str, str]]) -> Optional[ResponseDict]:
        """Use Google Studio API as fallback"""
        self.logger.warning("OpenRouter rate limit hit or LM Studio/OpenRouter failed. Switching to Google Studio API...")
        response_json = await self.google_client.chat_completion(messages, self.google_config)

        if not response_json or not self._is_valid_response(response_json):
            self.logger.error("Google Studio API request failed or returned invalid response")
            error_detail = response_json.get("error", "Unknown Google API failure") if response_json else "No response from Google API"
            return cast(ResponseDict, {"error": f"All models failed. Last attempt (Google): {error_detail}"})

        return response_json

    def _process_response(self, response_json: Union[Dict[str, Any], ResponseDict]) -> str:
        """Extract and process content from the response"""
        try:
            if response_json is None:
                return self.response_formatter.format_error_response("Empty response from API")
                
            if "error" in response_json:
                return self.response_formatter.format_error_response(response_json["error"])

            if not self._is_valid_response(response_json):
                self.logger.error(f"Missing 'choices' key or invalid choices in API response: {response_json}")
                return self.response_formatter.format_error_response("Invalid API response format")

            # Extract content - validation already done by _is_valid_response
            content = response_json["choices"][0]["message"]["content"]

            formatted_content = content

            response_tokens = self.token_counter.track_prompt_tokens(formatted_content, "completion")
            self.logger.info(f"Response token count: {response_tokens}")

            stats = self.token_counter.get_usage_stats()
            self.logger.info(f"Total tokens used: {stats['total']}")

            self.logger.debug(f"Full response content: {formatted_content}")

            return formatted_content

        except Exception as e:
            self.logger.error(f"Error processing response: {e}")
            self.logger.debug(f"Response that caused error: {response_json}")
            return self.response_formatter.format_error_response(f"Error processing response: {str(e)}")
