
from typing import Any

from src.logger.logger import Logger
from src.platforms.ai_providers.base import BaseAIClient
from src.platforms.ai_providers.response_models import ChatResponseModel
from src.utils.decorators import retry_api_call_default

try:
    from blockrun_llm import AsyncLLMClient
except ImportError:
    AsyncLLMClient = None


class BlockRunClient(BaseAIClient):
    def __init__(self, wallet_key: str, base_url: str, logger: Logger) -> None:
        super().__init__(logger)
        self._wallet_key = wallet_key
        self._base_url = base_url
        self._client: Any | None = None

    async def _initialize_client(self) -> None:

        if AsyncLLMClient is None:
            raise ImportError("blockrun-llm SDK is required but not installed")
        self._client = AsyncLLMClient(
            private_key=self._wallet_key,
            api_url=self._base_url,
        )
        self.logger.debug("BlockRun SDK client initialized successfully")

    async def close(self) -> None:

        if self._client:
            self.logger.debug("Closing BlockRunClient SDK session")
            self._client = None

    async def _ensure_sdk_client(self) -> Any:

        if not self._client:
            await self._initialize_client()
        return self._client

    def _redact_private_key(self, message: str) -> str:

        if self._wallet_key and len(self._wallet_key) > 10:
            return message.replace(self._wallet_key, f"{self._wallet_key[:6]}...{self._wallet_key[-4:]}")
        return message

    @retry_api_call_default
    async def chat_completion(
        self, model: str, messages: list[dict[str, Any]], model_config: dict[str, Any]
    ) -> ChatResponseModel | None:

        client = await self._ensure_sdk_client()
        try:
            self.logger.debug("Sending request to BlockRun SDK with model: %s", model)
            effective_model = self._ensure_provider_prefix(model)

            kwargs = self._build_chat_kwargs(model_config)
            response = await client.chat_completion(
                model=effective_model,
                messages=messages,
                **kwargs,
            )
            return self._convert_response(response)
        except Exception as e:  # noqa: BLE001
            return self._handle_exception(e)

    def _build_chat_kwargs(self, model_config: dict[str, Any]) -> dict[str, Any]:

        kwargs: dict[str, Any] = {}
        if "temperature" in model_config and model_config["temperature"] is not None:
            kwargs["temperature"] = model_config["temperature"]
        if "max_tokens" in model_config and model_config["max_tokens"] is not None:
            kwargs["max_tokens"] = model_config["max_tokens"]
        if "top_p" in model_config and model_config["top_p"] is not None:
            kwargs["top_p"] = model_config["top_p"]
        if "top_k" in model_config and model_config["top_k"] is not None:
            kwargs["top_k"] = model_config["top_k"]
        return kwargs

    def _convert_response(self, response: Any) -> ChatResponseModel | None:

        if response is None:
            return None
        content = ""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and choice.message:
                content = choice.message.content or ""
        else:
            try:
                content = response["choices"][0]["message"]["content"]
            except (KeyError, TypeError, IndexError):
                content = ""

        if not content:
            self.logger.warning("BlockRun returned empty content")
            return ChatResponseModel.from_error("BlockRun returned empty content")

        return ChatResponseModel.from_content(
            content=content,
            model=getattr(response, "model", "unknown"),
        )

    def _ensure_provider_prefix(self, model: str) -> str:

        if "/" not in model:
            return f"openai/{model}"
        return model

    def _handle_exception(self, exception: Exception) -> ChatResponseModel | None:

        redacted_error = self._redact_private_key(str(exception))
        self.logger.error("BlockRun API error: %s", redacted_error)
        result = self.handle_common_errors(exception)
        if result:
            return result
        return ChatResponseModel.from_error(redacted_error)

