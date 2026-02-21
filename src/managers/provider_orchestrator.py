"""Provider orchestration for AI model invocation with fallback logic."""
import io
from typing import Optional, Dict, Any, List, Union, cast, TYPE_CHECKING

from src.logger.logger import Logger
from src.platforms.ai_providers.response_models import ChatResponseModel
from src.managers.provider_types import ProviderMetadata, InvocationResult, ProviderClients

if TYPE_CHECKING:
    from src.config.protocol import ConfigProtocol


class ProviderOrchestrator:
    """
    Orchestrates AI provider invocation with fallback and retry logic.

    Responsibilities:
    - Provider metadata management
    - Single provider invocation
    - Multi-provider fallback chains
    - Google free/paid tier fallback
    - Response validation and rate limit detection
    """

    def __init__(
        self,
        logger: Logger,
        config: "ConfigProtocol",
        clients: ProviderClients
    ) -> None:
        """
        Initialize the provider orchestrator.

        Args:
            logger: Logger instance
            config: Configuration instance
            clients: Container with all AI provider clients
        """
        self.logger = logger
        self.config = config
        self.clients = clients
        self._providers = self._build_provider_metadata()

    def _build_provider_metadata(self) -> Dict[str, ProviderMetadata]:
        """Build provider metadata registry from clients and config."""
        return {
            'googleai': ProviderMetadata(
                name='Google AI Studio',
                client=self.clients.google,
                paid_client=self.clients.google_paid,
                default_model=self.config.GOOGLE_STUDIO_MODEL,
                config=self.config.get_model_config(self.config.GOOGLE_STUDIO_MODEL),
                supports_chart=True,
                has_rate_limits=True
            ),
            'openrouter': ProviderMetadata(
                name='OpenRouter',
                client=self.clients.openrouter,
                default_model=self.config.OPENROUTER_BASE_MODEL,
                config=self.config.get_model_config(self.config.OPENROUTER_BASE_MODEL),
                supports_chart=True,
                has_rate_limits=True
            ),
            'local': ProviderMetadata(
                name='LM Studio',
                client=self.clients.lmstudio,
                default_model=self.config.LM_STUDIO_MODEL,
                config=self.config.get_model_config(self.config.LM_STUDIO_MODEL),
                supports_chart=False,
                has_rate_limits=False
            ),
            'blockrun': ProviderMetadata(
                name='BlockRun.AI',
                client=self.clients.blockrun,
                default_model=self.config.BLOCKRUN_MODEL,
                config=self.config.get_model_config(self.config.BLOCKRUN_MODEL),
                supports_chart=True,
                has_rate_limits=False
            )
        }

    def get_metadata(self, provider: str) -> Optional[ProviderMetadata]:
        """Get metadata for a provider."""
        return self._providers.get(provider)

    def resolve_model(self, provider: str, model_override: Optional[str] = None) -> str:
        """Resolve effective model name for a provider."""
        if model_override:
            return model_override
        metadata = self._providers.get(provider)
        return metadata.default_model if metadata else "unknown-model"

    def is_available(self, provider: str) -> bool:
        """Check if a provider is available."""
        metadata = self._providers.get(provider)
        return metadata.is_available() if metadata else False

    def supports_chart(self, provider: str) -> bool:
        """Check if provider supports chart analysis."""
        if provider == "all":
            return self.is_available("googleai") or self.is_available("openrouter") or self.is_available("blockrun")
        metadata = self._providers.get(provider)
        return metadata.supports_chart if metadata and metadata.is_available() else False

    async def invoke(
        self,
        provider: str,
        messages: List[Dict[str, str]],
        *,
        chart: bool = False,
        chart_image: Optional[Union[io.BytesIO, bytes, str]] = None,
        model: Optional[str] = None
    ) -> InvocationResult:
        """
        Invoke a single provider and return structured result.

        Args:
            provider: Provider key (googleai, openrouter, local)
            messages: Chat messages
            chart: Whether this is a chart analysis request
            chart_image: Optional chart image for analysis
            model: Optional model override

        Returns:
            InvocationResult with success status, response, and metadata
        """
        metadata = self._providers.get(provider)
        if not metadata or not metadata.is_available():
            return InvocationResult(
                success=False,
                response=ChatResponseModel.from_error(f"Provider '{provider}' is not available"),
                provider=provider,
                model=self.resolve_model(provider, model)
            )
        effective_model = self.resolve_model(provider, model)
        if provider == "googleai":
            return await self._invoke_google(metadata, messages, effective_model, chart, chart_image)
        if provider == "local":
            return await self._invoke_local(metadata, messages, effective_model, chart)
        if provider == "openrouter":
            return await self._invoke_openrouter(metadata, messages, effective_model, chart, chart_image)
        if provider == "blockrun":
            return await self._invoke_blockrun(metadata, messages, effective_model, chart, chart_image)
        return InvocationResult(
            success=False,
            response=ChatResponseModel.from_error(f"Unknown provider '{provider}'"),
            provider=provider,
            model=effective_model
        )

    async def invoke_with_fallback(
        self,
        providers: List[str],
        messages: List[Dict[str, str]],
        *,
        chart: bool = False,
        chart_image: Optional[Union[io.BytesIO, bytes, str]] = None,
        model: Optional[str] = None
    ) -> InvocationResult:
        """
        Try providers in order, returning first successful result.

        Args:
            providers: List of provider keys to try in order
            messages: Chat messages
            chart: Whether this is a chart analysis request
            chart_image: Optional chart image
            model: Optional model override

        Returns:
            InvocationResult from first successful provider, or last failure
        """
        last_result: Optional[InvocationResult] = None
        for provider in providers:
            if not self.is_available(provider):
                continue
            effective_model = self.resolve_model(provider, model)
            self._log_attempt(provider, effective_model, chart)
            result = await self.invoke(provider, messages, chart=chart, chart_image=chart_image, model=model)
            if result.success and not self._is_rate_limited(result.response):
                return result
            self._log_failure(provider)
            last_result = result
        return last_result or InvocationResult(
            success=False,
            response=ChatResponseModel.from_error("No providers available"),
            provider="none",
            model="none"
        )

    async def get_text_response(
        self,
        effective_provider: str,
        messages: List[Dict[str, str]],
        model: Optional[str] = None
    ) -> InvocationResult:
        """
        Get text response using single provider or fallback chain.

        Args:
            effective_provider: Provider key or 'all' for fallback chain
            messages: Chat messages
            model: Optional model override

        Returns:
            InvocationResult with response
        """
        if effective_provider == "all":
            result = await self.invoke_with_fallback(
                ["googleai", "local", "openrouter"], messages, model=model
            )
            if not result.success and self.is_available("openrouter"):
                self.logger.warning("Google AI Studio and LM Studio failed. Final fallback to OpenRouter...")
                result = await self.invoke("openrouter", messages, model=model)
            return result
        if self.is_available(effective_provider):
            effective_model = self.resolve_model(effective_provider, model)
            self.logger.info(f"Using {self._providers[effective_provider].name} model: {effective_model}")
            return await self.invoke(effective_provider, messages, model=model)
        self._log_unavailable_guidance(effective_provider)
        return InvocationResult(
            success=False,
            response=ChatResponseModel.from_error(f"Provider '{effective_provider}' is not available"),
            provider=effective_provider,
            model=self.resolve_model(effective_provider, model)
        )

    async def get_chart_response(
        self,
        effective_provider: str,
        messages: List[Dict[str, str]],
        chart_image: Union[io.BytesIO, bytes, str],
        model: Optional[str] = None
    ) -> InvocationResult:
        """
        Get chart analysis response using single provider or fallback chain.

        Args:
            effective_provider: Provider key or 'all' for fallback chain
            messages: Chat messages
            chart_image: Chart image for analysis
            model: Optional model override

        Returns:
            InvocationResult with response
        """
        if effective_provider == "all":
            return await self.invoke_with_fallback(
                ["googleai", "openrouter"], messages, chart=True, chart_image=chart_image, model=model
            )
        if effective_provider == "local":
            return InvocationResult(
                success=False,
                response=ChatResponseModel.from_error("Chart analysis unavailable - local models don't support images"),
                provider="local",
                model=self.resolve_model("local", model)
            )
        if self.is_available(effective_provider):
            effective_model = self.resolve_model(effective_provider, model)
            self.logger.info(f"Using {self._providers[effective_provider].name} for chart analysis: {effective_model}")
            return await self.invoke(effective_provider, messages, chart=True, chart_image=chart_image, model=model)
        self._log_unavailable_guidance(effective_provider)
        return InvocationResult(
            success=False,
            response=ChatResponseModel.from_error(f"Provider '{effective_provider}' is not available for chart analysis"),
            provider=effective_provider,
            model=self.resolve_model(effective_provider, model)
        )

    async def _invoke_google(
        self,
        metadata: ProviderMetadata,
        messages: List[Dict[str, str]],
        effective_model: str,
        chart: bool,
        chart_image: Optional[Union[io.BytesIO, bytes, str]]
    ) -> InvocationResult:
        """Invoke Google AI with free/paid tier fallback logic."""
        is_free_tier_model = "flash" in effective_model.lower()
        tier_info = "free tier" if is_free_tier_model else "paid tier"
        self.logger.info(f"Attempting with Google AI {tier_info} API (model: {effective_model})")
        if chart and chart_image:
            response = await metadata.client.chat_completion_with_chart_analysis(
                effective_model, messages, cast(Any, chart_image), metadata.config
            )
        else:
            response = await metadata.client.chat_completion(effective_model, messages, metadata.config)
        error_type = response.error if response else None
        if error_type and ("overloaded" in error_type or "rate_limit" in error_type) and metadata.paid_client:
            error_reason = "rate limited" if error_type == "rate_limit" else "overloaded"
            self.logger.warning(f"Google AI free tier {error_reason}, retrying with paid API key")
            if chart and chart_image:
                response = await metadata.paid_client.chat_completion_with_chart_analysis(
                    effective_model, messages, cast(Any, chart_image), metadata.config
                )
            else:
                response = await metadata.paid_client.chat_completion(effective_model, messages, metadata.config)
            if self._is_valid_response(response):
                self.logger.info(f"Successfully used paid Google AI API after free tier {error_reason}")
                return InvocationResult(
                    success=True,
                    response=response,
                    provider="google",
                    model=effective_model,
                    used_paid_tier=True
                )
            paid_error = response.error if response else "no response"
            self.logger.error(f"Paid Google AI API also failed: {paid_error}")
            return InvocationResult(
                success=False,
                response=response,
                provider="google",
                model=effective_model,
                used_paid_tier=True
            )
        if self._is_valid_response(response):
            tier_success = "free tier" if is_free_tier_model else "paid tier"
            self.logger.info(f"Successfully used {tier_success} Google AI API")
            return InvocationResult(
                success=True,
                response=response,
                provider="google",
                model=effective_model,
                used_paid_tier=not is_free_tier_model
            )
        return InvocationResult(
            success=False,
            response=response,
            provider="google",
            model=effective_model
        )

    async def _invoke_local(
        self,
        metadata: ProviderMetadata,
        messages: List[Dict[str, str]],
        effective_model: str,
        chart: bool
    ) -> InvocationResult:
        """Invoke LM Studio local provider."""
        if chart:
            return InvocationResult(
                success=False,
                response=ChatResponseModel.from_error("Chart analysis unavailable - local models don't support images"),
                provider="lmstudio",
                model=effective_model
            )
        try:
            response = await metadata.client.chat_completion(effective_model, messages, metadata.config)
            success = self._is_valid_response(response)
            return InvocationResult(
                success=success,
                response=response,
                provider="lmstudio",
                model=effective_model
            )
        except Exception as e:
            return InvocationResult(
                success=False,
                response=ChatResponseModel.from_error(f"LM Studio connection failed: {str(e)}"),
                provider="lmstudio",
                model=effective_model
            )

    async def _invoke_openrouter(
        self,
        metadata: ProviderMetadata,
        messages: List[Dict[str, str]],
        effective_model: str,
        chart: bool,
        chart_image: Optional[Union[io.BytesIO, bytes, str]]
    ) -> InvocationResult:
        """Invoke OpenRouter provider."""
        if chart and chart_image:
            response = await metadata.client.chat_completion_with_chart_analysis(
                effective_model, messages, cast(Any, chart_image), metadata.config
            )
        else:
            response = await metadata.client.chat_completion(effective_model, messages, metadata.config)
        success = self._is_valid_response(response) and not self._is_rate_limited(response)
        return InvocationResult(
            success=success,
            response=response,
            provider="openrouter",
            model=effective_model
        )

    def _is_valid_response(self, response: Optional[ChatResponseModel]) -> bool:
        """Check if response contains valid choices with content."""
        if not response:
            return False
        if not response.choices:
            return False
        first_choice = response.choices[0]
        if first_choice.error:
            error_detail = first_choice.error
            error_code = error_detail.get('code', 'unknown') if isinstance(error_detail, dict) else 'unknown'
            error_msg = error_detail.get('message', 'unknown') if isinstance(error_detail, dict) else str(error_detail)
            provider = error_detail.get('metadata', {}).get('provider_name', 'unknown') if isinstance(error_detail, dict) else 'unknown'
            self.logger.error(f"Error in API response choice from {provider}: [{error_code}] {error_msg}")
            self.logger.debug(f"Full error details: {error_detail}")
            return False
        content = first_choice.message.content if first_choice.message else ""
        if not content:
            self.logger.debug(f"Empty content in API response choice. Message: {first_choice.message}")
            return False
        return True

    def _is_rate_limited(self, response: Optional[ChatResponseModel]) -> bool:
        """Check if response indicates rate limiting."""
        return bool(response and response.error and "rate_limit" in response.error)

    def _log_attempt(self, provider: str, model: str, chart: bool) -> None:
        """Log provider attempt."""
        metadata = self._providers.get(provider)
        if not metadata:
            return
        noun = "chart analysis" if chart else "request"
        self.logger.info(f"Attempting {noun} with {metadata.name} model: {model}")

    def _log_failure(self, provider: str) -> None:
        """Log provider failure with appropriate message."""
        if provider == "googleai":
            self.logger.warning("Google AI Studio model failed. Trying alternatives...")
        elif provider == "local":
            self.logger.warning("LM Studio failed. Falling back to next provider.")
        elif provider == "openrouter":
            self.logger.warning("OpenRouter failed or rate limited.")

    def _log_unavailable_guidance(self, provider: str) -> None:
        """Log guidance when provider is unavailable."""
        metadata = self._providers.get(provider)
        if not metadata or not metadata.client:
            if provider == "openrouter":
                self.logger.error("OpenRouter client not initialized. Check OPENROUTER_API_KEY in keys.env")
            elif provider == "googleai":
                self.logger.error("Google AI client not initialized. Check GOOGLE_STUDIO_API_KEY in keys.env")
            elif provider == "local":
                self.logger.error("LM Studio client not initialized. Check LM_STUDIO_BASE_URL in config.ini")
            elif provider == "blockrun":
                self.logger.error("BlockRun client not initialized. Check BLOCKRUN_WALLET_KEY in keys.env")
        elif provider == "local":
            self.logger.error("Local models don't support image analysis")

    async def _invoke_blockrun(
        self,
        metadata: ProviderMetadata,
        messages: List[Dict[str, str]],
        effective_model: str,
        chart: bool,
        chart_image: Optional[Union[io.BytesIO, bytes, str]]
    ) -> InvocationResult:
        """Invoke BlockRun provider."""
        if chart and chart_image:
            response = await metadata.client.chat_completion_with_chart_analysis(
                effective_model, messages, cast(Any, chart_image), metadata.config
            )
        else:
            response = await metadata.client.chat_completion(effective_model, messages, metadata.config)
        success = self._is_valid_response(response)
        return InvocationResult(
            success=success,
            response=response,
            provider="blockrun",
            model=effective_model
        )
