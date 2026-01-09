"""
BlockRun.AI provider using the official BlockRun LLM SDK.

BlockRun.AI is a pay-as-you-go AI gateway providing ChatGPT and all major LLMs
(OpenAI, Anthropic, Google, DeepSeek, xAI) via x402 micropayments on Base.

Website: https://blockrun.ai
Docs: https://blockrun.ai/docs

Requirements:
    pip install blockrun-llm

Configuration:
    Set BLOCKRUN_WALLET_KEY environment variable with your Base chain wallet private key.
    Your private key never leaves your machine - it's only used for local payment signing.
"""
import io
import base64
from typing import Optional, Dict, Any, List, Union

from PIL import Image

try:
    from blockrun_llm import AsyncLLMClient, APIError, PaymentError
except ImportError:
    raise ImportError(
        "blockrun-llm SDK not installed. Install with: pip install blockrun-llm"
    )

from src.logger.logger import Logger


class BlockRunClient:
    """
    Client for BlockRun.AI using the official SDK.

    The SDK automatically handles x402 micropayments on Base chain.
    Your private key is used for local signing only - it never leaves your machine.
    """

    def __init__(self, private_key: str, api_url: str, logger: Logger) -> None:
        """
        Initialize BlockRun client with the official SDK.

        Args:
            private_key: Base chain wallet private key (used for LOCAL signing only)
            api_url: API endpoint URL (default: https://blockrun.ai/api)
            logger: Logger instance for logging

        Security:
            Your private key NEVER leaves your machine. It is only used to sign
            EIP-712 typed data locally. Only the signature is sent to the server.
        """
        self.private_key = private_key
        self.api_url = api_url
        self.logger = logger
        self._client: Optional[AsyncLLMClient] = None

    async def __aenter__(self):
        """Initialize async SDK client."""
        self._client = AsyncLLMClient(
            private_key=self.private_key,
            api_url=self.api_url,
            timeout=600.0  # 10 minute timeout for long-running requests
        )
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        await self.close()

    async def close(self) -> None:
        """Close the SDK client."""
        if self._client:
            try:
                self.logger.debug("Closing BlockRun.AI SDK client")
                await self._client.close()
                self._client = None
            except Exception as e:
                self.logger.error(f"Error closing BlockRun.AI SDK client: {e}")

    def _ensure_client(self) -> AsyncLLMClient:
        """Ensure client is initialized."""
        if not self._client:
            self._client = AsyncLLMClient(
                private_key=self.private_key,
                api_url=self.api_url,
                timeout=600.0
            )
        return self._client

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        model_config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Send a chat completion request using the BlockRun SDK.

        Args:
            model: Model name (e.g., "gpt-4o", "claude-sonnet-4", "gemini-2.5-pro")
            messages: OpenAI-style messages
            model_config: Configuration parameters (max_tokens, temperature, top_p, etc.)

        Returns:
            Response in OpenAI-compatible format or None if failed
        """
        try:
            client = self._ensure_client()

            # Add provider prefix if not present (SDK expects "provider/model" format)
            if "/" not in model:
                # Default to openai for models without prefix
                model = f"openai/{model}"

            self.logger.debug(f"Sending BlockRun.AI SDK request with model: {model}")

            # Extract parameters from config
            max_tokens = model_config.get("max_tokens")
            temperature = model_config.get("temperature")
            top_p = model_config.get("top_p")

            # Use SDK's chat_completion method
            response = await client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )

            self.logger.debug("Received successful response from BlockRun.AI SDK")

            # Convert SDK response to dict format expected by the application
            return {
                "choices": [
                    {
                        "message": {
                            "role": response.choices[0].message.role,
                            "content": response.choices[0].message.content
                        },
                        "finish_reason": response.choices[0].finish_reason
                    }
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }

        except PaymentError as e:
            self.logger.error(f"BlockRun.AI payment error: {str(e)}")
            return {"error": "payment_failed", "details": str(e)}
        except APIError as e:
            self.logger.error(f"BlockRun.AI API error ({e.status_code}): {str(e)}")
            return {"error": f"api_error_{e.status_code}", "details": str(e)}
        except Exception as e:
            self.logger.error(f"Unexpected error with BlockRun.AI SDK: {str(e)}")
            return {"error": "unknown_error", "details": str(e)}

    async def chat_completion_with_chart_analysis(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        chart_image: Union[io.BytesIO, bytes, str],
        model_config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Send a chat completion request with a chart image for pattern analysis.

        Args:
            model: Model name (must support vision, e.g., "gpt-4o", "claude-sonnet-4")
            messages: OpenAI-style messages
            chart_image: Chart image as BytesIO, bytes, or file path string
            model_config: Configuration parameters

        Returns:
            Response in OpenAI-compatible format or None if failed
        """
        try:
            # Process chart image to base64
            img_data = self._process_chart_image(chart_image)
            base64_image = base64.b64encode(img_data).decode('utf-8')

            # Extract user text from last message
            user_text = ""
            for message in reversed(messages):
                if message["role"] == "user":
                    user_text = message["content"]
                    break

            # Convert messages to multimodal format
            multimodal_messages = []
            for message in messages:
                if message["role"] == "system":
                    # Keep system messages as-is
                    multimodal_messages.append(message)
                elif message["role"] == "user" and message == messages[-1]:
                    # Replace last user message with multimodal content
                    multimodal_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_text
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    })
                else:
                    # Keep other messages as-is
                    multimodal_messages.append(message)

            self.logger.debug(
                f"Sending chart analysis request to BlockRun.AI SDK "
                f"with chart image ({len(img_data)} bytes)"
            )

            # Use regular chat_completion with multimodal messages
            return await self.chat_completion(model, multimodal_messages, model_config)

        except Exception as e:
            self.logger.error(f"Error during BlockRun.AI chart analysis: {str(e)}")
            return {"error": "chart_analysis_failed", "details": str(e)}

    def _process_chart_image(self, chart_image: Union[io.BytesIO, bytes, str]) -> bytes:
        """
        Process chart image and return as bytes.

        Args:
            chart_image: Chart image as BytesIO, bytes, or file path string

        Returns:
            Image data as bytes
        """
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
