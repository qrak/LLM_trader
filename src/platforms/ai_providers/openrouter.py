import io
import base64
from typing import Optional, Dict, Any, List, TypedDict, cast, Union
from PIL import Image

from src.platforms.ai_providers.base import BaseApiClient
from src.utils.decorators import retry_api_call


class ResponseDict(TypedDict, total=False):
    """Type for API responses."""
    error: str


class OpenRouterClient(BaseApiClient):
    """Client for handling OpenRouter API requests."""
    
    def _extract_user_text_from_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Extract text content from the last user message."""
        for message in reversed(messages):
            if message["role"] == "user":
                return message["content"]
        return ""
    
    def _prepare_multimodal_messages(self, 
                                     messages: List[Dict[str, Any]], 
                                     user_text: str,
                                     multimodal_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert messages to OpenRouter multimodal format.
        
        Args:
            messages: Original messages
            user_text: Extracted user text
            multimodal_content: Content parts for multimodal message (text + images)
            
        Returns:
            Converted messages with system messages as user messages
        """
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
    
    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def chat_completion(self, model: str, messages: list, model_config: Dict[str, Any]) -> Optional[ResponseDict]:
        """Send a chat completion request to the OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "Kuruś Crypto Analyzer",
            "X-Title": "Kuruś Crypto Analyzer"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            **model_config
        }
        
        url = f"{self.base_url}/chat/completions"
        response = await self._make_post_request(url, headers, payload, model, timeout=600)
        
        return cast(ResponseDict, response) if response else None

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def chat_completion_with_chart_analysis(self, 
                                                 model: str,
                                                 messages: List[Dict[str, Any]], 
                                                 chart_image: Union[io.BytesIO, bytes, str],
                                                 model_config: Dict[str, Any]) -> Optional[ResponseDict]:
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
        try:
            # Process chart image to base64
            img_data = self._process_chart_image(chart_image)
            base64_image = base64.b64encode(img_data).decode('utf-8')
            
            user_text = self._extract_user_text_from_messages(messages)
            
            # Create multimodal content with image
            multimodal_content = [
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
            
            multimodal_messages = self._prepare_multimodal_messages(
                messages, user_text, multimodal_content
            )
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "Kuruś Crypto Analyzer",
                "X-Title": "Kuruś Crypto Analyzer"
            }
            
            payload = {
                "model": model,
                "messages": multimodal_messages,
                **model_config
            }
            
            self.logger.debug(f"Sending chart analysis request to OpenRouter with chart image ({len(img_data)} bytes)")
            
            url = f"{self.base_url}/chat/completions"
            response = await self._make_post_request(url, headers, payload, model, timeout=600)
            
            if response:
                self.logger.debug("Received successful chart analysis response from OpenRouter")
            
            return cast(ResponseDict, response) if response else None
            
        except Exception as e:
            self.logger.error(f"Error during OpenRouter chart analysis request: {str(e)}")
            return self._handle_exception(e)

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def chat_completion_with_images(self, 
                                        model: str,
                                        messages: List[Dict[str, Any]], 
                                        images: List[Union[Image.Image, bytes, str]], 
                                        model_config: Dict[str, Any]) -> Optional[ResponseDict]:
        """
        Send a chat completion request with image inputs.
        
        Args:
            model: Model name to use
            messages: List of OpenAI-style messages
            images: List of images (PIL Images, bytes, or file paths)
            model_config: Configuration parameters for the model
            
        Returns:
            Response in OpenRouter-compatible format or None if failed
        """
        try:
            # Process images to base64
            image_parts = []
            for image in images:
                img_data = self._process_image(image)
                base64_image = base64.b64encode(img_data).decode('utf-8')
                image_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
            
            user_text = self._extract_user_text_from_messages(messages)
            
            # Create multimodal content with images
            multimodal_content = [{"type": "text", "text": user_text}] + image_parts
            
            multimodal_messages = self._prepare_multimodal_messages(
                messages, user_text, multimodal_content
            )
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "Kuruś Crypto Analyzer",
                "X-Title": "Kuruś Crypto Analyzer"
            }
            
            payload = {
                "model": model,
                "messages": multimodal_messages,
                **model_config
            }
            
            self.logger.debug(f"Sending multimodal request to OpenRouter with {len(images)} images")
            
            url = f"{self.base_url}/chat/completions"
            response = await self._make_post_request(url, headers, payload, model, timeout=600)
            
            if response:
                self.logger.debug("Received successful multimodal response from OpenRouter")
            
            return cast(ResponseDict, response) if response else None
            
        except Exception as e:
            self.logger.error(f"Error during OpenRouter multimodal request: {str(e)}")
            return self._handle_exception(e)

    def _process_chart_image(self, chart_image: Union[io.BytesIO, bytes, str]) -> bytes:
        """
        Process chart image and return as bytes.
        
        Args:
            chart_image: Chart image as BytesIO, bytes, or file path string
            
        Returns:
            Image data as bytes
        """
        if isinstance(chart_image, io.BytesIO):
            # Read from BytesIO
            chart_image.seek(0)
            img_data = chart_image.read()
            chart_image.seek(0)  # Reset for potential reuse
            return img_data
        elif isinstance(chart_image, str):
            # File path - read the file
            with open(chart_image, 'rb') as f:
                return f.read()
        else:
            # Assume it's already bytes
            return chart_image

    def _process_image(self, image: Union[Image.Image, bytes, str]) -> bytes:
        """
        Process image and return as bytes.
        
        Args:
            image: Image as PIL Image, bytes, or file path string
            
        Returns:
            Image data as bytes
        """
        if isinstance(image, Image.Image):
            # PIL Image - convert to bytes
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            return img_buffer.getvalue()
        elif isinstance(image, bytes):
            # Raw bytes
            return image
        elif isinstance(image, str):
            # File path - read the file
            with open(image, 'rb') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _handle_exception(self, exception: Exception) -> Optional[ResponseDict]:
        """
        Handle exceptions from OpenRouter API.
        
        Args:
            exception: The exception that occurred
            
        Returns:
            Error response dictionary or None
        """
        error_message = str(exception)
        
        if "quota" in error_message.lower() or "rate limit" in error_message.lower():
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