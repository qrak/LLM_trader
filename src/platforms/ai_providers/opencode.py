
import io
import uuid
from typing import Any

from openai import AsyncOpenAI

from src.logger.logger import Logger
from src.platforms.ai_providers.base import BaseAIClient
from src.platforms.ai_providers.response_models import ChatResponseModel
from src.utils.decorators import retry_api_call_default


class OpenCodeClient(BaseAIClient):
    """Client for OpenCode Go through the OpenAI-compatible inference endpoint."""

    USER_AGENT = "llm-trader/1.0"

    def __init__(self, api_key: str, base_url: str, logger: Logger, session_id: str | None = None) -> None:
        super().__init__(logger)
        self.api_key = api_key
        self.base_url = base_url
        self.session_id = session_id or f"llm-trader-{uuid.uuid4().hex[:16]}"

    def _build_client(self) -> AsyncOpenAI:

        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers={
                "User-Agent": self.USER_AGENT,
                "x-opencode-session": self.session_id,
            },
        )

    @retry_api_call_default
    async def chat_completion(
        self, model: str, messages: list, model_config: dict[str, Any]
    ) -> ChatResponseModel | None:

        client = self._ensure_client()
        try:
            self.logger.debug("Sending request to OpenCode API with model: %s", model)
            response = await self._execute_with_param_retry(
                client.chat.completions.create,
                model_config,
                model=model,
                messages=messages
            )
            return self.convert_pydantic_response(response)
        except Exception as e:  # noqa: BLE001
            return self._handle_exception(e)

    @retry_api_call_default
    async def chat_completion_with_chart_analysis(
        self,
        model: str,
        messages: list[dict[str, Any]],
        chart_image: io.BytesIO | bytes | str,
        model_config: dict[str, Any]
    ) -> ChatResponseModel | None:

        client = self._ensure_client()
        try:
            user_text = self._extract_user_text_from_messages(messages)
            img_data, multimodal_content = self._build_multimodal_content(user_text, chart_image)
            multimodal_messages = self._attach_multimodal_content(messages, multimodal_content)
            self.logger.debug("Sending chart analysis request to OpenCode API (%s bytes)", len(img_data))
            response = await self._execute_with_param_retry(
                client.chat.completions.create,
                model_config,
                model=model,
                messages=multimodal_messages
            )
            return self.convert_pydantic_response(response)
        except Exception as e:  # noqa: BLE001
            self.logger.error("Error during OpenCode chart analysis request: %s", str(e))
            return self._handle_exception(e)
