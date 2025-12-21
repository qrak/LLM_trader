"""
Google GenAI client implementation using the official Google GenAI SDK.
Replaces the previous custom HTTP-based implementation with the official SDK.
Supports both text-only and multimodal (text + image) requests for pattern analysis.
"""

import io
from typing import Optional, Dict, Any, List, Union, cast

from google import genai
from google.genai import types
from PIL import Image

from src.logger.logger import Logger
from src.platforms.ai_providers.openrouter import ResponseDict
from src.utils.decorators import retry_api_call


class GoogleAIClient:
    """Client for handling Google AI API requests using the official Google GenAI SDK."""
    
    def __init__(self, api_key: str, model: str, logger: Logger) -> None:
        """
        Initialize the GoogleAIClient.
        
        Args:
            api_key: Google AI API key
            model: Model name (e.g., 'gemini-2.5-flash')
            logger: Logger instance
        """
        self.api_key = api_key
        self.model = model
        self.logger = logger
        self.client: Optional[genai.Client] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.client = genai.Client(api_key=self.api_key)
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self) -> None:
        """Close the client."""
        try:
            if self.client:
                # The Google GenAI client doesn't require explicit closing
                # but we set it to None for consistency
                self.client = None
                self.logger.debug("GoogleAIClient closed successfully")
        except Exception as e:
            self.logger.error(f"Error during Google AI client cleanup: {e}")
    
    def _ensure_client(self) -> genai.Client:
        """Ensure a client exists and return it."""
        if not self.client:
            self.client = genai.Client(api_key=self.api_key)
        return self.client
    
    def _extract_text_from_messages(self, messages: List[Dict[str, Any]]) -> str:
        """
        Extract combined text content from OpenAI-style messages.
        The Google GenAI SDK works better with simple string prompts.
        """
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
        """
        Extract text content from Google AI response, handling non-text parts gracefully.
        
        Some models (like Gemini with thinking mode) return additional parts like 'thought_signature'
        that are not text. We extract only text parts and log info about non-text parts.
        
        Args:
            response: Response object from Google AI SDK
            
        Returns:
            Concatenated text content from all text parts
        """
        try:
            text_parts = []
            non_text_parts = []
            
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                    else:
                        part_type = type(part).__name__
                        non_text_parts.append(part_type)
            
            if non_text_parts:
                details = []
                for candidate in response.candidates:
                    for part in candidate.content.parts:
                        try:
                            details.append(repr(part))
                        except Exception:
                            details.append(type(part).__name__)

                self.logger.debug(
                    f"Google AI response contains non-text parts: {non_text_parts}. "
                    f"Part details: {details}. Extracting text content only."
                )
            
            return "\n".join(text_parts)
        except Exception as e:
            self.logger.error(f"Failed to extract text from Google AI response: {e}")
            return ""
    
    def _create_generation_config(self, model_config: Dict[str, Any]) -> types.GenerateContentConfig:
        """
        Create a generation config from model configuration dictionary.
        
        Args:
            model_config: Configuration parameters for the model
            
        Returns:
            GenerateContentConfig object
        """
        return types.GenerateContentConfig(
            temperature=model_config.get("temperature", 0.7),
            top_p=model_config.get("top_p", 0.9),
            top_k=model_config.get("top_k", 40),
            max_output_tokens=model_config.get("max_tokens", 32768),
        )
    
    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def chat_completion(self, messages: List[Dict[str, Any]], model_config: Dict[str, Any], model: Optional[str] = None) -> Optional[ResponseDict]:
        """
        Send a chat completion request to the Google AI API using the official SDK.
        
        Args:
            messages: List of OpenAI-style messages
            model_config: Configuration parameters for the model
            model: Optional model override (e.g., admin-specified model)
            
        Returns:
            Response in OpenRouter-compatible format or None if failed
        """
        try:
            client = self._ensure_client()
            
            # Convert messages to simple text prompt
            prompt = self._extract_text_from_messages(messages)
            
            generation_config = self._create_generation_config(model_config)
            
            effective_model = model if model else self.model
            self.logger.debug(f"Sending request to Google AI with model: {effective_model}")
            
            response = await client.aio.models.generate_content(
                model=effective_model,
                contents=prompt,
                config=generation_config
            )
            
            content_text = self._extract_text_from_response(response)
            
            self.logger.debug("Received successful response from Google AI")
            
            return cast(ResponseDict, {
                "choices": [{
                    "message": {
                        "content": content_text,
                        "role": "assistant"
                    }
                }]
            })
            
        except Exception as e:
            self.logger.error(f"Error during Google AI request: {str(e)}")
            return self._handle_exception(e)
    
    async def chat_completion_with_chart_analysis(self, 
                                                 messages: List[Dict[str, Any]], 
                                                 chart_image: Union[io.BytesIO, bytes, str],
                                                 model_config: Dict[str, Any],
                                                 model: Optional[str] = None) -> Optional[ResponseDict]:
        """
        Send a chat completion request with a chart image for pattern analysis.
        
        Args:
            messages: List of OpenAI-style messages
            chart_image: Chart image as BytesIO, bytes, or file path string
            model_config: Configuration parameters for the model
            model: Optional model override (e.g., admin-specified model)
            
        Returns:
            Response in OpenRouter-compatible format or None if failed
        """
        try:
            client = self._ensure_client()
            
            # Extract text prompt
            prompt = self._extract_text_from_messages(messages)
            
            if isinstance(chart_image, io.BytesIO):
                chart_image.seek(0)
                img_data = chart_image.read()
                chart_image.seek(0)
            elif isinstance(chart_image, str):
                with open(chart_image, 'rb') as f:
                    img_data = f.read()
            else:
                img_data = chart_image
            
            image_part = types.Part.from_bytes(
                data=img_data,
                mime_type='image/png'
            )
            
            contents = [prompt, image_part]
            
            generation_config = self._create_generation_config(model_config)
            
            effective_model = model if model else self.model
            self.logger.debug(f"Sending chart analysis request to Google AI with model: {effective_model} (chart image: {len(img_data)} bytes)")
            
            response = await client.aio.models.generate_content(
                model=effective_model,
                contents=contents,
                config=generation_config
            )
            
            content_text = self._extract_text_from_response(response)
            
            self.logger.debug("Received successful chart analysis response from Google AI")
            
            return cast(ResponseDict, {
                "choices": [{
                    "message": {
                        "content": content_text,
                        "role": "assistant"
                    }
                }]
            })
            
        except Exception as e:
            self.logger.error(f"Error during Google AI chart analysis request: {str(e)}")
            return self._handle_exception(e)
    
    async def chat_completion_with_images(self, 
                                        messages: List[Dict[str, Any]], 
                                        images: List[Union[Image.Image, bytes, str]], 
                                        model_config: Dict[str, Any]) -> Optional[ResponseDict]:
        """
        Send a chat completion request with image inputs.
        
        Args:
            messages: List of OpenAI-style messages
            images: List of images (PIL Images, bytes, or file paths)
            model_config: Configuration parameters for the model
            
        Returns:
            Response in OpenRouter-compatible format or None if failed
        """
        try:
            client = self._ensure_client()
            
            # Extract text prompt
            prompt = self._extract_text_from_messages(messages)
            
            image_parts = []
            for image in images:
                if isinstance(image, Image.Image):
                    img_buffer = io.BytesIO()
                    image.save(img_buffer, format='PNG')
                    img_data = img_buffer.getvalue()
                    image_parts.append(types.Part.from_bytes(
                        data=img_data,
                        mime_type='image/png'
                    ))
                elif isinstance(image, bytes):
                    image_parts.append(types.Part.from_bytes(
                        data=image,
                        mime_type='image/png'
                    ))
                elif isinstance(image, str):
                    if image.startswith(('http://', 'https://', 'gs://')):
                        image_parts.append(types.Part.from_uri(
                            file_uri=image,
                            mime_type='image/png'
                        ))
                    else:
                        with open(image, 'rb') as f:
                            img_data = f.read()
                        image_parts.append(types.Part.from_bytes(
                            data=img_data,
                            mime_type='image/png'
                        ))
            
            contents = [prompt] + image_parts
            
            generation_config = self._create_generation_config(model_config)
            
            self.logger.debug(f"Sending multimodal request to Google AI with {len(images)} images")
            
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=generation_config
            )
            
            content_text = self._extract_text_from_response(response)
            
            self.logger.debug("Received successful multimodal response from Google AI")
            
            return cast(ResponseDict, {
                "choices": [{
                    "message": {
                        "content": content_text,
                        "role": "assistant"
                    }
                }]
            })
            
        except Exception as e:
            self.logger.error(f"Error during Google AI multimodal request: {str(e)}")
            return self._handle_exception(e)
    
    def _handle_exception(self, exception: Exception) -> Optional[ResponseDict]:
        """
        Handle exceptions from Google AI API.
        
        Args:
            exception: The exception that occurred
            
        Returns:
            Error response dictionary or None
        """
        error_message = str(exception)
        
        if "503" in error_message or "overloaded" in error_message.lower() or "unavailable" in error_message.lower():
            self.logger.error(f"Google AI API overloaded (503): {error_message}")
            return cast(ResponseDict, {"error": "overloaded", "details": error_message})
        elif "quota" in error_message.lower() or "rate limit" in error_message.lower():
            self.logger.error(f"Rate limit or quota exceeded: {error_message}")
            return cast(ResponseDict, {"error": "rate_limit", "details": error_message})
        elif "authentication" in error_message.lower() or "api key" in error_message.lower():
            self.logger.error(f"Authentication error: {error_message}")
            return cast(ResponseDict, {"error": "authentication", "details": error_message})
        elif "timeout" in error_message.lower():
            self.logger.error(f"Timeout error: {error_message}")
            return cast(ResponseDict, {"error": "timeout", "details": error_message})
        else:
            self.logger.error(f"Unexpected error: {error_message}")
            return None
