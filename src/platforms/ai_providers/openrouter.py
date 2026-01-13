"""
OpenRouter client implementation using the official OpenRouter SDK.
Supports text-only and multimodal (text + image) requests with cost tracking.
"""
import io
import base64
from typing import Optional, Dict, Any, List, TypedDict, Union

from PIL import Image
from openrouter import OpenRouter

from src.logger.logger import Logger
from src.utils.decorators import retry_api_call


class UsageDict(TypedDict, total=False):
    """Token usage and cost information from API response."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float


class ResponseDict(TypedDict, total=False):
    """Type for API responses with usage tracking."""
    error: str
    choices: List[Dict[str, Any]]
    usage: UsageDict
    id: str
    model: str


class OpenRouterClient:
    """Client for handling OpenRouter API requests using the official SDK."""

    def __init__(self, api_key: str, base_url: str, logger: Logger) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.logger = logger
        self._client: Optional[OpenRouter] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._client = OpenRouter(api_key=self.api_key)
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close the SDK client."""
        if self._client:
            try:
                self.logger.debug("Closing OpenRouterClient SDK session")
                # SDK handles cleanup internally
                self._client = None
            except Exception as e:
                self.logger.error(f"Error closing OpenRouter SDK session: {e}")

    def _ensure_client(self) -> OpenRouter:
        """Ensure a client exists and return it."""
        if not self._client:
            self._client = OpenRouter(api_key=self.api_key)
        return self._client

    def _detect_unsupported_param(self, error_msg: str) -> Optional[str]:
        """Detect which parameter caused the error from error message."""
        import re
        match = re.search(r"unexpected keyword argument '(\w+)'", error_msg)
        if match:
            return match.group(1)
        match = re.search(r"unknown (parameter|argument)[:\s]+['\"]?(\w+)['\"]?", error_msg, re.IGNORECASE)
        if match:
            return match.group(2)
        return None

    def _filter_known_unsupported(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out known unsupported parameters (pre-filter before first attempt)."""
        known_unsupported = {'thinking_budget', 'thinking_config'}
        return {k: v for k, v in model_config.items() if k not in known_unsupported}

    async def _try_with_param_retry(
        self, client, model: str, messages: list, model_config: Dict[str, Any], is_chart: bool = False
    ) -> Any:
        """Try SDK call and retry without rejected parameters."""
        config = self._filter_known_unsupported(model_config)
        rejected_params = set()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return await client.chat.send_async(model=model, messages=messages, **config)
            except Exception as e:
                error_msg = str(e)
                bad_param = self._detect_unsupported_param(error_msg)
                if bad_param and bad_param in config:
                    self.logger.warning(f"Parameter '{bad_param}' not supported by model. Retrying without it.")
                    rejected_params.add(bad_param)
                    config = {k: v for k, v in config.items() if k not in rejected_params}
                    continue
                raise
        raise RuntimeError(f"Failed after {max_retries} retries with rejected params: {rejected_params}")

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def chat_completion(
        self, model: str, messages: list, model_config: Dict[str, Any]
    ) -> Optional[ResponseDict]:
        """Send a chat completion request to the OpenRouter API using the SDK."""
        client = self._ensure_client()
        try:
            self.logger.debug(f"Sending request to OpenRouter SDK with model: {model}")
            response = await self._try_with_param_retry(client, model, messages, model_config)
            return self._convert_sdk_response(response)
        except Exception as e:
            return self._handle_exception(e)

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def chat_completion_with_chart_analysis(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        chart_image: Union[io.BytesIO, bytes, str],
        model_config: Dict[str, Any]
    ) -> Optional[ResponseDict]:
        """
        Send a chat completion request with a chart image for pattern analysis.

        Args:
            model: Model name to use
            messages: List of OpenAI-style messages
            chart_image: Chart image as BytesIO, bytes, or file path string
            model_config: Configuration parameters for the model

        Returns:
            Response in OpenRouter-compatible format or None if failed
        """
        client = self._ensure_client()
        try:
            img_data = self._process_chart_image(chart_image)
            base64_image = base64.b64encode(img_data).decode('utf-8')
            user_text = self._extract_user_text_from_messages(messages)
            multimodal_content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
            multimodal_messages = self._prepare_multimodal_messages(
                messages, user_text, multimodal_content
            )
            self.logger.debug(f"Sending chart analysis request to OpenRouter SDK ({len(img_data)} bytes)")
            response = await self._try_with_param_retry(
                client, model, multimodal_messages, model_config, is_chart=True
            )
            if response:
                self.logger.debug("Received successful chart analysis response from OpenRouter SDK")
            return self._convert_sdk_response(response)
        except Exception as e:
            self.logger.error(f"Error during OpenRouter chart analysis request: {str(e)}")
            return self._handle_exception(e)

    async def get_generation_cost(self, generation_id: str, retry_delay: float = 0.5) -> Optional[Dict[str, Any]]:
        """
        Retrieve cost and stats for a specific generation.

        Args:
            generation_id: The generation ID from completion response
            retry_delay: Seconds to wait before querying (API may need time to index)

        Returns:
            Dictionary with token counts and costs
        """
        import asyncio
        await asyncio.sleep(retry_delay)
        client = self._ensure_client()
        try:
            generation = client.generations.get_generation(id=generation_id)
            if generation and generation.data:
                data = generation.data
                return {
                    "model": getattr(data, 'model', 'unknown'),
                    "total_cost": getattr(data, 'total_cost', 0),
                    "prompt_tokens": getattr(data, 'tokens_prompt', 0),
                    "completion_tokens": getattr(data, 'tokens_completion', 0),
                    "native_prompt_tokens": getattr(data, 'native_tokens_prompt', 0),
                    "native_completion_tokens": getattr(data, 'native_tokens_completion', 0),
                }
            return None
        except Exception as e:
            error_msg = str(e)
            if "not found" in error_msg.lower():
                self.logger.debug(f"Generation stats not yet available for {generation_id[:20]}... (will be indexed shortly)")
            else:
                self.logger.warning(f"Could not retrieve generation stats: {error_msg}")
            return None

    def _convert_sdk_response(self, response: Any) -> ResponseDict:
        """Convert SDK Pydantic response to ResponseDict format."""
        result: ResponseDict = {}
        if hasattr(response, 'id'):
            result['id'] = response.id
        if hasattr(response, 'model'):
            result['model'] = response.model
        if hasattr(response, 'choices') and response.choices:
            result['choices'] = []
            for choice in response.choices:
                choice_dict: Dict[str, Any] = {}
                if hasattr(choice, 'message') and choice.message:
                    choice_dict['message'] = {
                        'role': getattr(choice.message, 'role', 'assistant'),
                        'content': getattr(choice.message, 'content', '')
                    }
                if hasattr(choice, 'finish_reason'):
                    choice_dict['finish_reason'] = choice.finish_reason
                if hasattr(choice, 'error') and choice.error:
                    choice_dict['error'] = choice.error.model_dump() if hasattr(choice.error, 'model_dump') else str(choice.error)
                result['choices'].append(choice_dict)
        if hasattr(response, 'usage') and response.usage:
            result['usage'] = {
                'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                'total_tokens': getattr(response.usage, 'total_tokens', 0),
            }
        if hasattr(response, 'error') and response.error:
            result['error'] = response.error.model_dump() if hasattr(response.error, 'model_dump') else str(response.error)
        return result

    def _extract_user_text_from_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Extract text content from the last user message."""
        for message in reversed(messages):
            if message["role"] == "user":
                return message["content"]
        return ""

    def _prepare_multimodal_messages(
        self,
        messages: List[Dict[str, Any]],
        user_text: str,
        multimodal_content: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert messages to OpenRouter multimodal format."""
        multimodal_messages = []
        for message in messages:
            if message["role"] == "system":
                multimodal_messages.append({
                    "role": "user",
                    "content": f"System instructions: {message['content']}"
                })
            elif message["role"] == "user" and message == messages[-1]:
                multimodal_messages.append({
                    "role": "user",
                    "content": multimodal_content
                })
            else:
                multimodal_messages.append(message)
        return multimodal_messages

    def _process_chart_image(self, chart_image: Union[io.BytesIO, bytes, str]) -> bytes:
        """Process chart image and return as bytes."""
        if isinstance(chart_image, io.BytesIO):
            chart_image.seek(0)
            img_data = chart_image.read()
            chart_image.seek(0)
            return img_data
        elif isinstance(chart_image, str):
            with open(chart_image, 'rb') as f:
                return f.read()
        else:
            return chart_image

    def _process_image(self, image: Union[Image.Image, bytes, str]) -> bytes:
        """Process image and return as bytes."""
        if isinstance(image, Image.Image):
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            return img_buffer.getvalue()
        elif isinstance(image, bytes):
            return image
        elif isinstance(image, str):
            with open(image, 'rb') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _handle_exception(self, exception: Exception) -> Optional[ResponseDict]:
        """Handle exceptions from OpenRouter API."""
        error_message = str(exception)
        if "quota" in error_message.lower() or "rate limit" in error_message.lower():
            self.logger.error(f"Rate limit or quota exceeded: {error_message}")
            return {"error": "rate_limit", "details": error_message}  # type: ignore
        elif "authentication" in error_message.lower() or "api key" in error_message.lower():
            self.logger.error(f"Authentication error: {error_message}")
            return {"error": "authentication", "details": error_message}  # type: ignore
        elif "timeout" in error_message.lower():
            self.logger.error(f"Timeout error: {error_message}")
            return {"error": "timeout", "details": error_message}  # type: ignore
        else:
            self.logger.error(f"Unexpected error: {error_message}")
            return None