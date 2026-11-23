
from typing import TYPE_CHECKING

from src.logger.logger import Logger
from src.managers.provider_orchestrator import ProviderOrchestrator
from src.managers.provider_types import ProviderClients
from src.utils.token_counter import CostStorage, ModelPricing, TokenCounter

if TYPE_CHECKING:
    from src.config.loader import Config
    from src.parsing.unified_parser import UnifiedParser


class ModelManager:
    def __init__(
        self,
        logger: Logger,
        config: "Config",
        unified_parser: "UnifiedParser",
        token_counter: TokenCounter,
        cost_storage: CostStorage,
        model_pricing: ModelPricing,
        orchestrator: ProviderOrchestrator,
        provider_clients: ProviderClients
    ) -> None:

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

        for client in self._clients.all():
            await client.__aenter__()
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):

        await self.close()

    async def close(self) -> None:

        try:
            for client in self._clients.all():
                await client.close()
            self.logger.debug("All model clients closed successfully")
        except Exception as e:  # noqa: BLE001
            self.logger.error("Error during model manager cleanup: %s", e)

    async def send_prompt(
        self,
        prompt: str,
        system_message: str | None = None,
        prepared_messages: list[dict[str, str]] | None = None,
        provider: str | None = None,
        model: str | None = None
    ) -> str:

        messages = prepared_messages if prepared_messages is not None else self._prepare_messages(prompt, system_message)
        effective_provider = provider if provider else self.provider
        result = await self._orchestrator.get_text_response(effective_provider, messages, model)
        return await self._process_result(result)

    def describe_provider_and_model(
        self,
        provider_override: str | None,
        model_override: str | None,
    ) -> tuple[str, str]:

        provider_name = (provider_override or self.provider or "unknown").lower()
        if model_override:
            return provider_name, model_override
        if self._orchestrator.get_metadata(provider_name) is not None:
            return provider_name, self._orchestrator.resolve_model(provider_name)
        return provider_name, "unspecified"

    def _prepare_messages(self, prompt: str, system_message: str | None = None) -> list[dict[str, str]]:

        self.token_counter.reset_session_stats()
        if system_message:
            system_tokens = self.token_counter.count_tokens(system_message)
            prompt_tokens = self.token_counter.count_tokens(prompt)
            self.logger.debug("Pre-call estimate: system=%s, prompt=%s", f"{system_tokens:,}", f"{prompt_tokens:,}")
            return [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]
        prompt_tokens = self.token_counter.count_tokens(prompt)
        self.logger.debug("Pre-call estimate: prompt=%s", f"{prompt_tokens:,}")
        return [{"role": "user", "content": prompt}]

    async def _process_result(self, result) -> str:

        response = result.response
        if response is None:
            return self.unified_parser.format_error_response("Empty response from API")
        if response.error:
            return self.unified_parser.format_error_response(response.error)
        if not result.success:
            self.logger.error("API response invalid: %s", response)
            return self.unified_parser.format_error_response("Invalid API response format")
        if not response.choices or not response.choices[0].message:
            return self.unified_parser.format_error_response("No content in response")
        content = response.choices[0].message.content
        self.logger.info("Full response content: %s", content, extra={"skip_live_stream": True})
        await self._track_cost(result, response, content)
        return content

    async def _track_cost(self, result, response, content: str) -> None:

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        cost: float | None = None
        from_provider = False
        if result.provider == "openrouter" and self._clients.openrouter:
            generation_id = response.id
            if generation_id:
                cost_data = await self._clients.openrouter.get_generation_cost(generation_id, retry_delay=0.8)
                if cost_data:
                    cost = cost_data.get("total_cost", 0.0)
                    prompt_tokens = cost_data.get("native_prompt_tokens", prompt_tokens)
                    completion_tokens = cost_data.get("native_completion_tokens", completion_tokens)
        elif result.provider == "google" and result.model:
            is_free_tier = "flash" in result.model.lower()
            if not is_free_tier:
                cost = self.model_pricing.get_cost("google", result.model, prompt_tokens, completion_tokens)
        elif result.provider == "deepseek" and result.model:
            cost = self.model_pricing.get_cost("deepseek", result.model, prompt_tokens, completion_tokens)
        elif result.provider == "opencode" and result.model:
            reported = usage.reported_cost if usage else None
            from_provider = reported is not None
            cost = reported if from_provider else self.model_pricing.get_cost(
                "opencode", result.model, prompt_tokens, completion_tokens)
        elif result.provider == "commandcode" and result.model:
            cost = self.model_pricing.get_cost("commandcode", result.model, prompt_tokens, completion_tokens)
        cost_note = (
            self.model_pricing.cost_note(result.provider, result.model)
            if cost is not None and result.model and result.provider in ("google", "deepseek", "opencode", "commandcode")
            else None
        )
        if cost_note and result.provider == "opencode" and from_provider:
            cost_note = f"{cost_note}, from provider"
        self.token_counter.process_response_usage(
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "cost": cost},
            logger=self.logger,
            fallback_text=content,
            cost_note=cost_note
        )
        self.cost_storage.record_usage(result.provider, prompt_tokens, completion_tokens, cost)

