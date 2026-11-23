
from typing import TYPE_CHECKING

from src.analyzer.formatters.ohlcv_formatter import truncate_ohlcv_section
from src.logger.logger import Logger
from src.managers.provider_types import (
    InvocationResult,
    ProviderClients,
    ProviderMetadata,
)
from src.platforms.ai_providers.response_models import ChatResponseModel
from src.utils.prompt_budget import PromptBudgetExceeded, PromptBudgetPolicy

if TYPE_CHECKING:
    from src.config.loader import Config


class ProviderOrchestrator:
    RESULT_NAMES = {"googleai": "google"}  # noqa: RUF012
    MISSING_CREDENTIAL = {  # noqa: RUF012
        "googleai": "GOOGLE_STUDIO_API_KEY in keys.env",
        "openrouter": "OPENROUTER_API_KEY in keys.env",
        "opencode": "OPENCODE_API_KEY in keys.env",
        "commandcode": "COMMANDCODE_API_KEY in keys.env",
        "deepseek": "DEEPSEEK_API_KEY in keys.env",
        "custom": "custom_base_url in config.ini",
    }

    def __init__(
        self,
        logger: Logger,
        config: "Config",
        clients: ProviderClients,
        prompt_budget: PromptBudgetPolicy | None = None,
    ) -> None:

        self.logger = logger
        self.config = config
        self.clients = clients
        self.prompt_budget = prompt_budget
        self._providers = self._build_provider_metadata()

    def _build_provider_metadata(self) -> dict[str, ProviderMetadata]:

        entries = {
            "googleai": ("Google AI Studio", self.clients.google, self.config.GOOGLE_STUDIO_MODEL),
            "openrouter": ("OpenRouter", self.clients.openrouter, self.config.OPENROUTER_BASE_MODEL),
            "opencode": ("OpenCode Go", self.clients.opencode, self.config.OPENCODE_MODEL),
            "commandcode": ("Command Code", self.clients.commandcode, self.config.COMMANDCODE_MODEL),
            "custom": ("Custom endpoint", self.clients.custom, self.config.CUSTOM_MODEL),
            "deepseek": ("DeepSeek", self.clients.deepseek, self.config.DEEPSEEK_MODEL),
        }
        return {
            provider: ProviderMetadata(
                name=name, client=client, default_model=model, config=self.config.get_model_config(model),
            )
            for provider, (name, client, model) in entries.items()
        }

    def prepare_request_messages(
        self, provider: str, messages: list[dict[str, str]], model: str,
    ) -> list[dict[str, str]]:
        metadata = self._providers[provider]
        result = (
            self.prompt_budget.prepare_messages(messages, provider, model, metadata.config)
            if self.prompt_budget is not None else messages
        )
        parts = [
            ("System instructions: " if message["role"] == "system" else "User query: ") + message["content"]
            for message in result
        ]
        full_prompt = "\n\n".join(parts)
        self.logger.info(
            "Full prompt content: %s",
            full_prompt,
            extra={
                "display_message": f"Full prompt content: {truncate_ohlcv_section(full_prompt)}",
                "stream_message": "Full prompt content: <truncated>",
            },
        )
        return result

    def get_metadata(self, provider: str) -> ProviderMetadata | None:

        return self._providers.get(provider)

    def resolve_model(self, provider: str, model_override: str | None = None) -> str:

        if model_override:
            return model_override
        metadata = self._providers.get(provider)
        if not metadata:
            return "unknown-model"
        return metadata.default_model

    def is_available(self, provider: str) -> bool:

        metadata = self._providers.get(provider)
        return metadata.is_available() if metadata else False

    async def get_text_response(
        self,
        provider: str,
        messages: list[dict[str, str]],
        model: str | None = None
    ) -> InvocationResult:

        metadata = self._providers.get(provider)
        effective_model = self.resolve_model(provider, model)
        if metadata is None:
            return self._failure(provider, effective_model, f"Unknown provider '{provider}'")
        if not metadata.is_available():
            self.logger.error(
                "%s client not initialized. Check %s", metadata.name, self.MISSING_CREDENTIAL[provider]
            )
            return self._failure(provider, effective_model, f"Provider '{provider}' is not available")
        self.logger.info("Using %s model: %s", metadata.name, effective_model)
        try:
            prepared = self.prepare_request_messages(provider, messages, effective_model)
        except PromptBudgetExceeded as exc:
            self.logger.warning("Prompt budget refused %s/%s: %s", provider, effective_model, exc)
            return self._failure(provider, effective_model, str(exc))
        assert metadata.client is not None
        response = await metadata.client.chat_completion(effective_model, prepared, metadata.config)
        success = self._is_valid_response(response) and not self._is_rate_limited(response)
        return InvocationResult(
            success=success,
            response=response,
            provider=self.RESULT_NAMES.get(provider, provider),
            model=effective_model,
            error_message=None if success else (
                response.error if response and response.error
                else f"Empty or invalid response content from {metadata.name}"
            ),
        )

    def _failure(self, provider: str, model: str, error: str) -> InvocationResult:

        return InvocationResult(
            success=False, response=ChatResponseModel.from_error(error), provider=provider, model=model,
        )

    def _is_valid_response(self, response: ChatResponseModel | None) -> bool:

        if not response:
            return False
        if not response.choices:
            return False
        first_choice = response.choices[0]
        if first_choice.error:
            error_detail = first_choice.error
            error_code = error_detail.get("code", "unknown")
            error_msg = error_detail.get("message", "unknown")
            provider = error_detail.get("metadata", {}).get("provider_name", "unknown")
            self.logger.error("Error in API response choice from %s: [%s] %s", provider, error_code, error_msg)
            return False
        content = first_choice.message.content if first_choice.message else ""
        if not content:
            self.logger.debug("Empty content in API response choice (finish_reason=%s)", first_choice.finish_reason)
            return False
        return True

    def _is_rate_limited(self, response: ChatResponseModel | None) -> bool:

        return bool(response and response.error and "rate_limit" in response.error)
