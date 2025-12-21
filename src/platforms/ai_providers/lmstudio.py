import asyncio
import json
from typing import Optional, Dict, Any, cast

import aiohttp

from src.logger.logger import Logger
from src.platforms.ai_providers.base import BaseApiClient
from src.platforms.ai_providers.openrouter import ResponseDict
from src.utils.decorators import retry_api_call


class LMStudioClient(BaseApiClient):
    """Client for handling LM Studio API requests."""

    def __init__(self, base_url: str, logger: Logger) -> None:
        # LM Studio often doesn't require an API key, pass a dummy one
        super().__init__(api_key="dummy-key", base_url=base_url, logger=logger)

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def chat_completion(self, model: str, messages: list, model_config: Dict[str, Any]) -> Optional[ResponseDict]:
        """Send a chat completion request to the LM Studio API."""
        headers = {
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": messages,
            **model_config
        }

        url = f"{self.base_url}/chat/completions"
        response = await self._make_post_request(url, headers, payload, model, timeout=500)
        
        return cast(ResponseDict, response) if response else None

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def stream_chat_completion(self, model: str, messages: list, model_config: Dict[str, Any], 
                                    callback=None) -> Optional[ResponseDict]:
        """Send a streaming chat completion request to the LM Studio API."""
        session = self._ensure_session()

        headers = {
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,  # Enable streaming
            **model_config
        }

        try:
            self.logger.debug(f"Sending streaming request to LM Studio API with model: {model}")
            complete_response = {"choices": [{"message": {"content": "", "role": "assistant"}}]}
            
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=500
            ) as response:
                if response.status != 200:
                    error_response = await self._handle_error_response(response, model)
                    return cast(ResponseDict, error_response)

                # Iterate through streaming response chunks
                async for chunk in response.content:
                    if not chunk:
                        continue
                        
                    try:
                        chunk_text = chunk.decode('utf-8').strip()
                        
                        # Skip empty chunks and non-data lines
                        if not chunk_text or not chunk_text.startswith('data: '):
                            continue
                            
                        # Remove 'data: ' prefix
                        chunk_text = chunk_text[6:].strip()
                        
                        # Check for the end of the stream
                        if chunk_text == '[DONE]':
                            break
                            
                        # Parse JSON chunk
                        chunk_data = json.loads(chunk_text)
                        delta_content = chunk_data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                        
                        if delta_content:
                            # Update complete response content
                            complete_response["choices"][0]["message"]["content"] += delta_content
                            
                            # Call callback with new content if provided
                            if callback:
                                await callback(delta_content)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Failed to decode chunk as JSON: {chunk_text}")
                    except Exception as e:
                        self.logger.error(f"Error processing streaming chunk: {e}")

            self.logger.debug("Streaming response from LM Studio completed")
            return cast(ResponseDict, complete_response)

        except asyncio.TimeoutError as e:
            self.logger.error(f"Timeout error when requesting streaming from LM Studio: {e}")
            return cast(ResponseDict, {"error": "timeout", "details": str(e)})
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error in streaming request: {type(e).__name__} - {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in streaming request: {type(e).__name__} - {e}")
            return None
            
    async def console_stream(self, model: str, messages: list, model_config: Dict[str, Any]) -> Optional[ResponseDict]:
        """Stream model output directly to console with real-time updates."""

        async def print_chunk(chunk):
            """Print chunk to console without line breaks."""
            print(chunk, end='', flush=True)
            
        print(f"\n[Streaming response from {model}]\n")
        response = await self.stream_chat_completion(
            model, messages, model_config, callback=print_chunk
        )
        print("\n\n[Stream completed]")
        return response