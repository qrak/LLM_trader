
import inspect
from typing import Any

from google import genai
from google.genai import errors, types

from src.logger.logger import Logger
from src.platforms.ai_providers.base import BaseAIClient
from src.platforms.ai_providers.response_models import ChatResponseModel, UsageModel
from src.utils.decorators import retry_api_call_default


class GoogleAIClient(BaseAIClient):
    def __init__(self, api_key: str, model: str, logger: Logger) -> None:

        super().__init__(logger)
        self.api_key = api_key
        self.model = model
        self.client: genai.Client | None = None

    async def _initialize_client(self) -> None:

        self.client = genai.Client(api_key=self.api_key)

    async def close(self) -> None:

        if self.client:
            try:
                aio_client = getattr(self.client, "aio", None)
                if aio_client is not None:
                    aclose = getattr(aio_client, "aclose", None)
                    if callable(aclose):
                        try:
                            close_result = aclose()  # pylint: disable=not-callable
                            if inspect.isawaitable(close_result):
                                await close_result
                        except Exception as exc:  # pylint: disable=broad-exception-caught  # noqa: BLE001
                            self.logger.warning("GoogleAIClient async cleanup failed: %s", exc)

                sync_close = getattr(self.client, "close", None)
                if callable(sync_close):
                    try:
                        sync_close()  # pylint: disable=not-callable
                    except Exception as exc:  # pylint: disable=broad-exception-caught  # noqa: BLE001
                        self.logger.warning("GoogleAIClient sync cleanup failed: %s", exc)
            finally:
                self.client = None
                self.logger.debug("GoogleAIClient closed successfully")

    def _ensure_client(self) -> genai.Client:

        if not self.client:
            self.client = genai.Client(api_key=self.api_key)
        return self.client

    def _extract_text_from_messages(self, messages: list[dict[str, Any]]) -> str:

        text_parts = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                text_parts.append(f"System: {content}")
            else:
                text_parts.append(content)
        return "\n\n".join(text_parts)

    def _extract_text_from_response(self, response) -> str:

        try:
            text_parts = []
            non_text_parts = []
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    try:
                        text = part.text
                        if text is not None:
                            text_parts.append(text)
                        else:
                            non_text_parts.append(type(part).__name__ + " (None)")
                    except AttributeError:
                        non_text_parts.append(type(part).__name__)
            if non_text_parts:
                self.logger.debug("Google AI response contains non-text parts: %s. Extracting text only.", non_text_parts)
            return "\n".join(text_parts)
        except Exception as e:  # noqa: BLE001
            self.logger.error("Failed to extract text from Google AI response: %s", e)
            return ""

    def _extract_usage_metadata(self, response) -> UsageModel | None:

        try:
            metadata = response.usage_metadata
            if not metadata:
                return None

            prompt = getattr(metadata, "prompt_token_count", 0) or 0
            completion = getattr(metadata, "candidates_token_count", 0) or 0
            thoughts = getattr(metadata, "thoughts_token_count", 0) or 0

            output_tokens = completion + thoughts

            total = getattr(metadata, "total_token_count", 0) or 0
            return UsageModel(
                prompt_tokens=prompt,
                completion_tokens=output_tokens,
                total_tokens=total or prompt + output_tokens,
                thoughts_token_count=thoughts,
            )
        except AttributeError:
            pass
        except Exception as e:  # noqa: BLE001
            self.logger.debug("Failed to extract usage metadata: %s", e)
        return None

    def _create_generation_config(
        self,
        model_config: dict[str, Any],
        include_thinking: bool = True,
    ) -> types.GenerateContentConfig:

        thinking_config = None
        if include_thinking:
            thinking_level = model_config.get("thinking_level", "high")
            thinking_levels = {
                "minimal": types.ThinkingLevel.MINIMAL,
                "low": types.ThinkingLevel.LOW,
                "medium": types.ThinkingLevel.MEDIUM,
                "high": types.ThinkingLevel.HIGH,
            }
            if thinking_level in thinking_levels:
                thinking_config = types.ThinkingConfig(thinking_level=thinking_levels[thinking_level])

        config = {
            "max_output_tokens": model_config.get("max_tokens", 32768),
            "thinking_config": thinking_config,
        }
        return types.GenerateContentConfig.model_validate(config)

    def _should_retry_without_thinking(self, exception: Exception) -> bool:

        return self._is_unsupported_feature_error(exception, "thinking")

    def _is_unsupported_feature_error(self, exception: Exception, feature: str) -> bool:

        code = None
        message = str(exception)
        if isinstance(exception, errors.APIError):
            code = getattr(exception, "code", None)
            sdk_message = getattr(exception, "message", None)
            status = getattr(exception, "status", None)
            message = " ".join(str(part) for part in (sdk_message, status, exception) if part)

        if code is not None and code != 400:
            return False

        error_text = message.lower()
        if feature not in error_text:
            return False
        return any(term in error_text for term in ("invalid", "unsupported", "unknown", "field", "400"))

    @retry_api_call_default
    async def chat_completion(
        self, model: str, messages: list[dict[str, Any]], model_config: dict[str, Any]
    ) -> ChatResponseModel | None:

        client = self._ensure_client()
        prompt = self._extract_text_from_messages(messages)
        effective_model = model if model else self.model
        for include_thinking in (True, False):
            try:
                generation_config = self._create_generation_config(
                    model_config,
                    include_thinking=include_thinking
                )
                self.logger.debug("Sending request to Google AI with model: %s (thinking=%s)", effective_model, include_thinking)
                response = await client.aio.models.generate_content(
                    model=effective_model,
                    contents=prompt,
                    config=generation_config
                )
                content_text = self._extract_text_from_response(response)
                usage = self._extract_usage_metadata(response)
                self.logger.debug("Received successful response from Google AI")
                return self.create_response(content_text, usage=usage)
            except Exception as e: # pylint: disable=broad-exception-caught  # noqa: BLE001
                if include_thinking and self._should_retry_without_thinking(e):
                    self.logger.warning("Model may not support thinking_config, retrying without it: %s", e)
                    continue
                self.logger.error("Error during Google AI request: %s", e)
                return self._handle_exception(e)
        return None

    def _handle_exception(self, exception: Exception) -> ChatResponseModel | None:

        result = self.handle_common_errors(exception)
        if result:
            return result
        sanitized_error = self._sanitize_error_message(str(exception))
        self.logger.error("Unexpected Google AI error: %s", sanitized_error)
        return None

