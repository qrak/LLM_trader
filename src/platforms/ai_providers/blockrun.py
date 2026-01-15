"""
BlockRun.AI client implementation using the official blockrun-llm SDK.
Supports text-only and multimodal (text + image) requests with x402 micropayments.
"""
import io
import base64
from typing import Optional, Dict,Any, List, Union

from src.logger.logger import Logger
from src.platforms.ai_providers.base import BaseAIClient, ResponseDict
from src.utils.decorators import retry_api_call


class BlockRunClient(BaseAIClient):
    """Client for handling BlockRun.AI API requests using the official SDK."""

    def __init__(self, wallet_key: str, base_url: str, logger: Logger) -> None:
        super().__init__(logger)
        self._wallet_key = wallet_key
        self.base_url = base_url
        self._client: Optional[Any] = None

    async def _initialize_client(self) -> None:
        """Initialize the BlockRun SDK client."""
        try:
            from blockrun_llm import AsyncLLMClient
            self._client = AsyncLLMClient(private_key=self._wallet_key, api_url=self.base_url)
            self.logger.debug("BlockRun SDK client initialized successfully")
        except ImportError as e:
            self.logger.error(f"BlockRun SDK not installed. Run: pip install blockrun-llm")
            raise ImportError("blockrun-llm SDK is required but not installed") from e

    async def close(self) -> None:
        """Close the SDK client."""
        if self._client:
            self.logger.debug("Closing BlockRunClient SDK session")
            self._client = None

    def _ensure_client(self) -> Any:
        """Ensure a client exists and return it."""
        if not self._client:
            raise RuntimeError("BlockRun client not initialized. Call __aenter__ or _initialize_client first.")
        return self._client

    def _redact_private_key(self, message: str) -> str:
        """Redact private key from error messages and logs."""
        if self._wallet_key and len(self._wallet_key) > 10:
            return message.replace(self._wallet_key, f"{self._wallet_key[:6]}...{self._wallet_key[-4:]}")
        return message

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def chat_completion(
        self, model: str, messages: List[Dict[str, Any]], model_config: Dict[str, Any]
    ) -> Optional[ResponseDict]:
        """
        Send a chat completion request to the BlockRun API using the SDK.
        
        Args:
            model: Model name in provider/model format (e.g., openai/gpt-4o, anthropic/claude-sonnet-4)
            messages: List of OpenAI-style messages
            model_config: Configuration parameters for the model
            
        Returns:
            Response in OpenAI-compatible format or None if failed
        """
        client = self._ensure_client()
        try:
            self.logger.debug(f"Sending request to BlockRun SDK with model: {model}")
            effective_model = self._ensure_provider_prefix(model)
            response = await client.chat_completion(
                model=effective_model,
                messages=messages,
                **model_config
            )
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
            model: Model name in provider/model format (e.g., openai/gpt-4o, anthropic/claude-sonnet-4)
            messages: List of OpenAI-style messages
            chart_image: Chart image as BytesIO, bytes, or file path string
            model_config: Configuration parameters for the model
            
        Returns:
            Response in OpenAI-compatible format or None if failed
        """
        client = self._ensure_client()
        try:
            img_data = self.process_chart_image(chart_image)
            base64_image = base64.b64encode(img_data).decode('utf-8')
            user_text = self._extract_all_user_text_from_messages(messages)
            multimodal_content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
            multimodal_messages = self._prepare_multimodal_messages(
                messages, multimodal_content
            )
            self.logger.debug(f"Sending chart analysis request to BlockRun SDK ({len(img_data)} bytes)")
            effective_model = self._ensure_provider_prefix(model)
            response = await client.chat_completion(
                model=effective_model,
                messages=multimodal_messages,
                **model_config
            )
            if response:
                self.logger.debug("Received successful chart analysis response from BlockRun SDK")
            return self._convert_sdk_response(response)
        except Exception as e:
            self.logger.error(f"Error during BlockRun chart analysis request: {self._redact_private_key(str(e))}")
            return self._handle_exception(e)

    def _ensure_provider_prefix(self, model: str) -> str:
        """
        Ensure model has provider/model format.
        If no slash is present, assume openai.
        """
        if "/" not in model:
            return f"openai/{model}"
        return model

    def _extract_all_user_text_from_messages(self, messages: List[Dict[str, Any]]) -> str:
        """
        Extract and concatenate text content from all user messages.
        
        This ensures multimodal requests preserve full conversation context,
        not just the last user message.
        """
        user_texts = []
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content", "")
                if content:
                    user_texts.append(content)
        return "\n\n".join(user_texts) if user_texts else ""

    def _prepare_multimodal_messages(
        self,
        messages: List[Dict[str, Any]],
        multimodal_content: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert messages to BlockRun multimodal format.
        
        Strategy: Replace the last user message with multimodal content,
        preserve system messages, and keep the conversation structure.
        """
        multimodal_messages = []
        for i, message in enumerate(messages):
            if message.get("role") == "system":
                multimodal_messages.append({
                    "role": "user",
                    "content": f"System instructions: {message['content']}"
                })
            elif message.get("role") == "user" and i == len(messages) - 1:
                multimodal_messages.append({
                    "role": "user",
                    "content": multimodal_content
                })
            else:
                multimodal_messages.append(message)
        return multimodal_messages

    def _convert_sdk_response(self, response: Any) -> ResponseDict:
        """Convert SDK response to ResponseDict format."""
        if response is None:
            return {"error": "Empty response from BlockRun SDK"}  # type: ignore
        
        try:
            result: ResponseDict = {
                "choices": [{
                    "message": {
                        "content": response.get("choices", [{}])[0].get("message", {}).get("content", ""),
                        "role": response.get("choices", [{}])[0].get("message", {}).get("role", "assistant")
                    }
                }]
            }
            
            if "usage" in response:
                result["usage"] = {
                    "prompt_tokens": response["usage"].get("prompt_tokens", 0),
                    "completion_tokens": response["usage"].get("completion_tokens", 0),
                    "total_tokens": response["usage"].get("total_tokens", 0)
                }
            
            if "id" in response:
                result["id"] = response["id"]
            if "model" in response:
                result["model"] = response["model"]
                
            return result
        except (KeyError, IndexError, AttributeError) as e:
            self.logger.error(f"Failed to parse BlockRun response: {e}")
            return {"error": f"Invalid response format: {str(e)}"}  # type: ignore

    def _handle_exception(self, exception: Exception) -> Optional[ResponseDict]:
        """Handle BlockRun specific exceptions, falling back to common handler."""
        redacted_error = self._redact_private_key(str(exception))
        self.logger.error(f"BlockRun API error: {redacted_error}")
        
        result = self.handle_common_errors(exception)
        if result:
            return result
        
        return {"error": redacted_error}  # type: ignore
