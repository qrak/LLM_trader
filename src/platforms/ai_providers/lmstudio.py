"""
LM Studio client implementation using the official LM Studio Python SDK.
Supports text-only and multimodal (text + image) requests for local inference.
"""
import io
import re
from typing import Optional, Dict, Any, List, Union

import lmstudio as lms

from src.logger.logger import Logger
from src.platforms.ai_providers.base import BaseAIClient
from src.platforms.ai_providers.response_models import ChatResponseModel, UsageModel
from src.utils.decorators import retry_api_call


class LMStudioClient(BaseAIClient):
    """Client for handling LM Studio API requests using the official SDK."""

    def __init__(self, base_url: str, logger: Logger) -> None:
        super().__init__(logger)
        self.base_url = base_url
        self._client: Optional[lms.AsyncClient] = None

    async def _initialize_client(self) -> None:
        """Initialize the LM Studio SDK client."""
        self._client = lms.AsyncClient(api_host=self._get_api_host())

    async def close(self) -> None:
        """Close the SDK client."""
        if self._client:
            self.logger.debug("LMStudioClient SDK session does not require explicit closing.")

    def _get_api_host(self) -> str:
        """Parse base_url to get valid api_host for SDK."""
        api_host = self.base_url
        if "://" in api_host:
            api_host = api_host.split("://")[1]
        if "/" in api_host:
            api_host = api_host.split("/")[0]
        return api_host

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def chat_completion(
        self, model: str, messages: list, model_config: Dict[str, Any]
    ) -> Optional[ChatResponseModel]:
        """Send a chat completion request to the LM Studio API using the SDK."""
        api_host = self._get_api_host()
        try:
            async with lms.AsyncClient(api_host=api_host) as client:
                self.logger.debug(f"Sending request to LM Studio SDK with model: {model} (host={api_host})")
                if not model:
                    loaded_models = await client.llm.list_loaded()
                    if loaded_models:
                        model = loaded_models[0].identifier
                        self.logger.info(f"Auto-selected loaded model: {model}")
                    else:
                        raise ValueError("No model specified and no models loaded in LM Studio")
                llm = await client.llm.model(model)
                chat = lms.Chat()
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        chat.add_user_message(f"System: {content}")
                    elif role == "user":
                        chat.add_user_message(content)
                    elif role == "assistant":
                        chat.add_assistant_response(content)
                config = self._build_prediction_config(model_config)
                response = await llm.respond(chat, config=config)
                self.logger.debug("Received successful response from LM Studio SDK")
                content = str(response) if response else ""
                return self.create_response(
                    content=content,
                    usage=UsageModel(prompt_tokens=0, completion_tokens=0, total_tokens=0)
                )
        except Exception as e:
            self.logger.error(f"Error during LM Studio request: {str(e)}")
            return self._handle_exception(e)

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def chat_completion_with_chart_analysis(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        chart_image: Union[io.BytesIO, bytes, str],
        model_config: Dict[str, Any]
    ) -> Optional[ChatResponseModel]:
        """
        Send a chat completion request with a chart image for pattern analysis.

        Args:
            model: Model name to use
            messages: List of OpenAI-style messages
            chart_image: Chart image as BytesIO, bytes, or file path string
            model_config: Configuration parameters for the model

        Returns:
            ChatResponseModel or None if failed
        """
        api_host = self._get_api_host()
        try:
            img_data = self.process_chart_image(chart_image)
            async with lms.AsyncClient(api_host=api_host) as client:
                self.logger.debug(f"Sending chart analysis request to LM Studio SDK with model: {model} (host={api_host})")
                if not model:
                    loaded_models = await client.llm.list_loaded()
                    if loaded_models:
                        model = loaded_models[0].identifier
                        self.logger.info(f"Auto-selected loaded model: {model}")
                    else:
                        raise ValueError("No model specified and no models loaded in LM Studio")
                image_handle = await client.files.prepare_image(img_data)
                llm = await client.llm.model(model)
                chat = lms.Chat()
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        chat.add_user_message(f"System: {content}")
                    elif role == "user":
                        chat.add_user_message(content, images=[image_handle])
                    elif role == "assistant":
                        chat.add_assistant_response(content)
                config = self._build_prediction_config(model_config)
                response = await llm.respond(chat, config=config)
                self.logger.debug("Received successful chart analysis response from LM Studio SDK")
                content = str(response) if response else ""
                return self.create_response(
                    content=content,
                    usage=UsageModel(prompt_tokens=0, completion_tokens=0, total_tokens=0)
                )
        except Exception as e:
            self.logger.error(f"Error during LM Studio chart analysis request: {str(e)}")
            return self._handle_exception(e)

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def stream_chat_completion(
        self,
        model: str,
        messages: list,
        model_config: Dict[str, Any],
        callback=None
    ) -> Optional[ChatResponseModel]:
        """Send a streaming chat completion request to the LM Studio API."""
        api_host = self._get_api_host()
        try:
            async with lms.AsyncClient(api_host=api_host) as client:
                self.logger.debug(f"Sending streaming request to LM Studio SDK with model: {model} (host={api_host})")
                if not model:
                    loaded_models = await client.llm.list_loaded()
                    if loaded_models:
                        model = loaded_models[0].identifier
                        self.logger.info(f"Auto-selected loaded model: {model}")
                    else:
                        raise ValueError("No model specified and no models loaded in LM Studio")
                llm = await client.llm.model(model)
                chat = lms.Chat()
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        chat.add_user_message(f"System: {content}")
                    elif role == "user":
                        chat.add_user_message(content)
                    elif role == "assistant":
                        chat.add_assistant_response(content)
                config = self._build_prediction_config(model_config)
                complete_content = ""
                stream = await llm.respond_stream(chat, config=config)
                try:
                    async for fragment in stream:
                        text = str(fragment.content)
                        complete_content += text
                        if callback:
                            await callback(text)
                except Exception:
                    if not complete_content:
                        raise
                self.logger.debug("Streaming response from LM Studio completed")
                return self.create_response(
                    content=complete_content,
                    usage=UsageModel(prompt_tokens=0, completion_tokens=0, total_tokens=0)
                )
        except Exception as e:
            self.logger.error(f"Error during LM Studio streaming request: {str(e)}")
            return self._handle_exception(e)

    def _build_prediction_config(self, model_config: Dict[str, Any]) -> Optional[lms.LlmPredictionConfig]:
        """Build LM Studio prediction config from model_config dict."""
        config_dict = {}
        if "temperature" in model_config:
            config_dict["temperature"] = model_config["temperature"]
        if "max_tokens" in model_config:
            config_dict["max_tokens"] = model_config["max_tokens"]
        if "top_p" in model_config:
            config_dict["top_p"] = model_config["top_p"]
        if "top_k" in model_config:
            config_dict["top_k"] = model_config["top_k"]
        if config_dict:
            while config_dict:
                try:
                    return lms.LlmPredictionConfig(**config_dict)
                except TypeError as e:
                    error_str = str(e)
                    if "unexpected keyword argument" in error_str.lower():
                        match = re.search(r"argument '([^']+)'", error_str)
                        if match:
                            bad_arg = match.group(1)
                            self.logger.warning(f"LM Studio SDK rejected parameter '{bad_arg}', retrying without it.")
                            del config_dict[bad_arg]
                            continue
                    self.logger.warning(f"Failed to build LlmPredictionConfig: {e}. Falling back to default config.")
                    break
        return None

    def _handle_exception(self, exception: Exception) -> Optional[ChatResponseModel]:
        """Handle LM Studio specific exceptions, falling back to common handler."""
        error_message = str(exception)
        if "ErrorDeviceLost" in error_message or "vk::Queue::submit" in error_message:
            friendly_msg = (
                "GPU Error detected on LM Studio server. "
                "Your GPU driver may have crashed or run out of memory. "
                "Try using a smaller model or reducing 'n_gpu_layers' in LM Studio."
            )
            self.logger.error(f"LM Studio GPU Crash: {friendly_msg}")
            return ChatResponseModel.from_error(f"gpu_crash: {friendly_msg}")
        result = self.handle_common_errors(exception)
        if result:
            return result
        self.logger.error(f"LM Studio Error: {error_message}")
        return None