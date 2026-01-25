"""ModelManager - Public API for AI model interactions."""
import io
from typing import Optional, Dict, List, Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.protocol import ConfigProtocol

from src.logger.logger import Logger
from src.utils.token_counter import TokenCounter, CostStorage, ModelPricing
from src.contracts.model_contract import ModelManagerProtocol
from src.factories import ProviderFactory
from .provider_types import ProviderClients
from .provider_orchestrator import ProviderOrchestrator


class ModelManager(ModelManagerProtocol):
    """
    Public API for AI model interactions.
    
    Responsibilities:
    - Public send_prompt methods
    - Message preparation
    - Response processing and cost tracking
    - Async context management
    
    Provider orchestration is delegated to ProviderOrchestrator.
    """

    def __init__(
        self, 
        logger: Logger, 
        config: "ConfigProtocol", 
        unified_parser=None,
        token_counter: Optional[TokenCounter] = None,
        cost_storage: Optional[CostStorage] = None,
        model_pricing: Optional[ModelPricing] = None,
        orchestrator: Optional[ProviderOrchestrator] = None,
        provider_clients: Optional[ProviderClients] = None
    ) -> None:
        """
        Initialize the ModelManager.
        
        Args:
            logger: Logger instance for logging
            config: ConfigProtocol instance for configuration access
            unified_parser: UnifiedParser instance (must be injected from app.py)
            token_counter: TokenCounter instance
            cost_storage: CostStorage instance
            model_pricing: ModelPricing instance
            orchestrator: ProviderOrchestrator instance
            provider_clients: ProviderClients instance
        
        Raises:
            ValueError: If required dependencies are missing
        """
        self.logger = logger
        self.config = config
        self.provider = self.config.PROVIDER.lower()
        self.unified_parser = unified_parser
        self.token_counter = token_counter
        self.cost_storage = cost_storage
        self.model_pricing = model_pricing
        self._clients = provider_clients
        self._orchestrator = orchestrator

    async def __aenter__(self):
        """Async context manager entry."""
        if self._clients.openrouter:
            await self._clients.openrouter.__aenter__()
        if self._clients.google:
            await self._clients.google.__aenter__()
        if self._clients.google_paid:
            await self._clients.google_paid.__aenter__()
        if self._clients.lmstudio:
            await self._clients.lmstudio.__aenter__()
        if self._clients.blockrun:
            await self._clients.blockrun.__aenter__()
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close all client connections."""
        try:
            if self._clients.openrouter:
                await self._clients.openrouter.close()
            if self._clients.google:
                await self._clients.google.close()
            if self._clients.google_paid:
                await self._clients.google_paid.close()
            if self._clients.lmstudio:
                await self._clients.lmstudio.close()
            if self._clients.blockrun:
                await self._clients.blockrun.close()
            self.logger.debug("All model clients closed successfully")
        except Exception as e:
            self.logger.error(f"Error during model manager cleanup: {e}")

    async def send_prompt(
        self,
        prompt: str,
        system_message: str = None,
        prepared_messages: List[Dict[str, str]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Send a prompt to the model and get a response.
        
        Args:
            prompt: User prompt
            system_message: Optional system instructions
            prepared_messages: Pre-prepared message list (if None, will be created from prompt)
            provider: Optional provider override
            model: Optional model override
            
        Returns:
            Response text from the AI model
        """
        messages = prepared_messages if prepared_messages is not None else self._prepare_messages(prompt, system_message)
        effective_provider = provider if provider else self.provider
        result = await self._orchestrator.get_text_response(effective_provider, messages, model)
        return await self._process_result(result)

    async def send_prompt_streaming(
        self,
        prompt: str,
        system_message: str = None,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Send a prompt to the model and get a streaming response.
        
        Falls back to regular prompt if streaming is unavailable.
        """
        messages = self._prepare_messages(prompt, system_message)
        effective_provider = provider if provider else self.provider
        if (effective_provider in ("local", "all")) and self._clients.lmstudio and self.config.LM_STUDIO_STREAMING:
            try:
                effective_model = model if model else self.config.LM_STUDIO_MODEL
                async def print_stream_callback(chunk):
                    print(chunk, end='', flush=True)
                response_json = await self._clients.lmstudio.stream_chat_completion(
                    effective_model, messages, self._orchestrator.get_metadata("local").config, callback=print_stream_callback
                )
                if response_json is not None:
                    from .provider_types import InvocationResult
                    result = InvocationResult(
                        success=True,
                        response=response_json,
                        provider="lmstudio",
                        model=effective_model
                    )
                    return await self._process_result(result)
                self.logger.warning("LM Studio streaming returned None. Falling back to non-streaming mode.")
            except Exception as e:
                self.logger.warning(f"LM Studio streaming failed: {str(e)}. Falling back to non-streaming mode.")
        return await self.send_prompt(prompt, system_message, prepared_messages=messages, provider=provider, model=model)

    async def send_prompt_with_chart_analysis(
        self,
        prompt: str,
        chart_image: Union[io.BytesIO, bytes, str],
        system_message: str = None,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Send a prompt with chart image for pattern analysis.
        
        Args:
            prompt: User prompt
            chart_image: Chart image (BytesIO, bytes, or base64 string)
            system_message: Optional system instructions
            provider: Optional provider override
            model: Optional model override
            
        Returns:
            Response text from the AI model
            
        Raises:
            ValueError: If chart analysis is unavailable
        """
        messages = self._prepare_messages(prompt, system_message)
        effective_provider = provider if provider else self.provider
        result = await self._orchestrator.get_chart_response(effective_provider, messages, chart_image, model)
        if not result.success:
            raise ValueError(f"Chart analysis failed: {result.error or 'invalid response'}")
        return await self._process_result(result)

    def supports_image_analysis(self, provider_override: Optional[str] = None) -> bool:
        """Check if the selected provider supports image analysis."""
        provider_name = (provider_override or self.provider or "").lower()
        return self._orchestrator.supports_chart(provider_name)

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
        if provider_name in ("googleai", "openrouter", "local"):
            return provider_name, self._orchestrator.resolve_model(provider_name)
        if provider_name == "all":
            chain: List[str] = []
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
        """Prepare message structure for API call."""
        self.token_counter.reset_session_stats()
        messages = []
        if system_message:
            combined_prompt = f"System instructions: {system_message}\n\nUser query: {prompt}"
            messages.append({"role": "user", "content": combined_prompt})
            system_tokens = self.token_counter.count_tokens(system_message)
            prompt_tokens = self.token_counter.count_tokens(prompt)
            self.logger.debug(f"Pre-call estimate: system={system_tokens:,}, prompt={prompt_tokens:,}")
            self.logger.debug(f"Full prompt content: {combined_prompt}")
        else:
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = self.token_counter.count_tokens(prompt)
            self.logger.debug(f"Pre-call estimate: prompt={prompt_tokens:,}")
            self.logger.debug(f"Full prompt content: {prompt}")
        return messages

    async def _process_result(self, result) -> str:
        """
        Process invocation result: extract content, track costs, handle errors.
        
        Args:
            result: InvocationResult from orchestrator
            
        Returns:
            Response content string
        """
        from .provider_types import InvocationResult
        if not isinstance(result, InvocationResult):
            return self.unified_parser.format_error_response("Invalid result type")
        response = result.response
        if response is None:
            return self.unified_parser.format_error_response("Empty response from API")
        if response.error:
            return self.unified_parser.format_error_response(response.error)
        if not result.success:
            self.logger.error(f"API response invalid: {response}")
            return self.unified_parser.format_error_response("Invalid API response format")
        if not response.choices or not response.choices[0].message:
            return self.unified_parser.format_error_response("No content in response")
        content = response.choices[0].message.content
        await self._track_cost(result, response, content)
        self.logger.debug(f"Full response content: {content}")
        return content

    async def _track_cost(self, result, response, content: str) -> None:
        """Track token usage and costs for the response."""
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        cost: Optional[float] = None
        if result.provider == "openrouter" and self._clients.openrouter:
            generation_id = response.id
            if generation_id:
                cost_data = await self._clients.openrouter.get_generation_cost(generation_id, retry_delay=0.8)
                if cost_data:
                    cost = cost_data.get("total_cost", 0.0)
                    prompt_tokens = cost_data.get("native_prompt_tokens", prompt_tokens)
                    completion_tokens = cost_data.get("native_completion_tokens", completion_tokens)
        elif result.provider == "google" and result.model:
            is_free_tier = "flash" in result.model.lower() and not result.used_paid_tier
            if not is_free_tier:
                cost = self.model_pricing.get_cost("google", result.model, prompt_tokens, completion_tokens)
        self.token_counter.process_response_usage(
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "cost": cost},
            provider=result.provider,
            logger=self.logger,
            fallback_text=content
        )
        self.cost_storage.record_usage(result.provider, prompt_tokens, completion_tokens, cost)
