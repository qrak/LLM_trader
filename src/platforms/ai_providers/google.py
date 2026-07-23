"""
Google GenAI client implementation using the official Google GenAI SDK.
Supports both text-only and multimodal (text + image) requests for pattern analysis.
"""
import inspect
import io
import struct
from typing import Any, Union

from google import genai
from google.genai import errors, types

from src.logger.logger import Logger
from src.platforms.ai_providers.base import BaseAIClient
from src.platforms.ai_providers.response_models import ChatResponseModel, UsageModel
from src.utils.decorators import retry_api_call

# Gemini charges 258 tokens per 75x75 image tile for flash models.
# Source: ai.google.dev/gemini-api/docs/tokens and
# https://ai.google.dev/gemini-api/docs/tokens#image
_IMAGE_TILE_SIZE = 75
_IMAGE_TOKENS_PER_TILE = 258


class GoogleAIClient(BaseAIClient):
    """Client for handling Google AI API requests using the official Google GenAI SDK."""

    def __init__(self, api_key: str, model: str, logger: Logger) -> None:
        """
        Initialize the GoogleAIClient.

        Args:
            api_key: Google AI API key
            model: Model name (e.g., 'gemini-2.5-flash')
            logger: Logger instance
        """
        super().__init__(logger)
        self.api_key = api_key
        self.model = model
        self.client: genai.Client | None = None

    async def _initialize_client(self) -> None:
        """Initialize the Google GenAI client."""
        self.client = genai.Client(api_key=self.api_key)

    async def close(self) -> None:
        """Close the client."""
        if self.client:
            try:
                aio_client = getattr(self.client, "aio", None)
                aclose = getattr(aio_client, "aclose", None)
                if callable(aclose):
                    close_result = aclose()
                    if inspect.isawaitable(close_result):
                        await close_result
            finally:
                self.client = None
                self.logger.debug("GoogleAIClient closed successfully")

    def _ensure_client(self) -> genai.Client:
        """Ensure a client exists and return it."""
        if not self.client:
            self.client = genai.Client(api_key=self.api_key)
        return self.client

    def _extract_text_from_messages(self, messages: list[dict[str, Any]]) -> str:
        """Extract combined text content from OpenAI-style messages."""
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
        """Extract text content from Google AI response, handling non-text parts gracefully."""
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
        except Exception as e:
            self.logger.error("Failed to extract text from Google AI response: %s", e)
            return ""

    def _get_image_dimensions(self, img_bytes: bytes) -> tuple[int, int] | None:
        """Extract image dimensions from PNG/JPEG bytes without full decode.

        Returns (width, height) or None if format is unknown.
        """
        if img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            # PNG: IHDR chunk at offset 16 (4B width, 4B height, big-endian)
            if len(img_bytes) >= 33:
                w = struct.unpack('>I', img_bytes[16:20])[0]
                h = struct.unpack('>I', img_bytes[20:24])[0]
                return w, h
        elif img_bytes[:2] in (b'\xff\xd8',):
            # JPEG: scan for SOF0 (0xff 0xc0/0xc1/0xc2) marker
            i = 2
            while i < len(img_bytes) - 1:
                if img_bytes[i] == 0xff and img_bytes[i + 1] in (0xc0, 0xc1, 0xc2):
                    if i + 9 < len(img_bytes):
                        h = struct.unpack('>H', img_bytes[i + 5:i + 7])[0]
                        w = struct.unpack('>H', img_bytes[i + 7:i + 9])[0]
                        return w, h
                i += 1
        return None

    def _estimate_image_tokens(self, img_bytes: bytes | None) -> int:
        """Estimate the number of tokens Google charges for an image.

        Formula: tiles = ceil(w/75) * ceil(h/75), tokens = tiles * 258.
        Returns 0 if dimensions cannot be determined.
        """
        if not img_bytes:
            return 0
        dims = self._get_image_dimensions(img_bytes)
        if dims is None:
            return 0
        w, h = dims
        tiles_x = (w + _IMAGE_TILE_SIZE - 1) // _IMAGE_TILE_SIZE
        tiles_y = (h + _IMAGE_TILE_SIZE - 1) // _IMAGE_TILE_SIZE
        total_tiles = tiles_x * tiles_y
        image_tokens = total_tiles * _IMAGE_TOKENS_PER_TILE
        self.logger.debug(
            "Image %dx%d -> %d tiles (%dx%d) -> %d image tokens",
            w, h, total_tiles, tiles_x, tiles_y, image_tokens,
        )
        return image_tokens

    def _extract_usage_metadata(
        self, response, image_bytes: bytes | None = None
    ) -> UsageModel | None:
        """Extract token usage metadata from Google AI response.

        Uses prompt_tokens_details (modality breakdown) from the SDK when available
        for transparency in logging. Trusts prompt_token_count as the primary value
        since the SDK docs confirm it includes image tokens.

        Args:
            response: Google GenAI SDK response object
            image_bytes: Raw image bytes for diagnostic logging only
        """
        try:
            metadata = response.usage_metadata
            if not metadata:
                return None

            prompt = getattr(metadata, 'prompt_token_count', 0) or 0
            completion = getattr(metadata, 'candidates_token_count', 0) or 0
            thoughts = getattr(metadata, 'thoughts_token_count', 0) or 0

            # Google bills thinking tokens as output ("Output price including thinking tokens").
            output_tokens = completion + thoughts

            # Log per-modality breakdown from SDK for transparency
            text_tokens = None
            image_tokens_sdk = None
            prompt_details = getattr(metadata, 'prompt_tokens_details', None)
            if prompt_details:
                for detail in prompt_details:
                    mod = getattr(detail, 'modality', '')
                    count = getattr(detail, 'token_count', 0) or 0
                    if mod == 'TEXT':
                        text_tokens = count
                    elif mod == 'IMAGE':
                        image_tokens_sdk = count

            # Diagnostic: log the modality breakdown if available
            if image_tokens_sdk is not None:
                self.logger.info(
                    "Token breakdown: TEXT=%s, IMAGE=%s, prompt=%s, output=%s (incl. thoughts=%s)",
                    text_tokens, image_tokens_sdk, prompt, output_tokens, thoughts,
                )
            elif image_bytes:
                # SDK didn't return modality details — estimate for diagnostic
                est = self._estimate_image_tokens(image_bytes)
                self.logger.info(
                    "Token breakdown (estimated): text=%s, image(est)=%s, sdk_prompt=%s, output=%s",
                    text_tokens or prompt, est, prompt, output_tokens,
                )

            total = getattr(metadata, 'total_token_count', 0) or 0
            return UsageModel(
                prompt_tokens=prompt,
                completion_tokens=output_tokens,
                total_tokens=total or prompt + output_tokens,
                thoughts_token_count=thoughts,
            )
        except AttributeError:
            pass
        except Exception as e:
            self.logger.debug("Failed to extract usage metadata: %s", e)
        return None

    def _create_generation_config(
        self,
        model_config: dict[str, Any],
        include_thinking: bool = True,
        include_code_execution: bool = False
    ) -> types.GenerateContentConfig:
        """Create a generation config from model configuration dictionary."""
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

        tools = []
        if include_code_execution:
            tools.append(types.Tool(code_execution=types.ToolCodeExecution()))

        config = {
            "max_output_tokens": model_config.get("max_tokens", 32768),
            "thinking_config": thinking_config,
            "tools": tools if tools else None,
        }
        return types.GenerateContentConfig.model_validate(config)

    def _should_retry_without_thinking(self, exception: Exception) -> bool:
        """Return whether a Google SDK error indicates unsupported thinking config."""
        return self._is_unsupported_feature_error(exception, "thinking")

    def _should_retry_without_code_execution(self, exception: Exception) -> bool:
        """Return whether a Google SDK error indicates unsupported code_execution tool."""
        return self._is_unsupported_feature_error(exception, "code_execution")

    def _is_unsupported_feature_error(self, exception: Exception, feature: str) -> bool:
        """Return whether a Google SDK error indicates an unsupported feature."""
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

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def chat_completion(
        self, model: str, messages: list[dict[str, Any]], model_config: dict[str, Any]
    ) -> ChatResponseModel | None:
        """
        Send a chat completion request to the Google AI API.

        Args:
            model: Model name (overrides default if provided)
            messages: list of OpenAI-style messages
            model_config: Configuration parameters for the model

        Returns:
            ChatResponseModel or None if failed
        """
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
                usage = self._extract_usage_metadata(response, image_bytes=None)
                self.logger.debug("Received successful response from Google AI")
                return self.create_response(content_text, usage=usage)
            except Exception as e: # pylint: disable=broad-exception-caught
                if include_thinking and self._should_retry_without_thinking(e):
                    self.logger.warning("Model may not support thinking_config, retrying without it: %s", e)
                    continue
                self.logger.error("Error during Google AI request: %s", e)
                return self._handle_exception(e)
        return None

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def chat_completion_with_chart_analysis(
        self,
        model: str,
        messages: list[dict[str, Any]],
        chart_image: Union[io.BytesIO, bytes, str],
        model_config: dict[str, Any]
    ) -> ChatResponseModel | None:
        """
        Send a chat completion request with a chart image for pattern analysis.

        Args:
            model: Model name (overrides default if provided)
            messages: list of OpenAI-style messages
            chart_image: Chart image as BytesIO, bytes, or file path string
            model_config: Configuration parameters for the model

        Returns:
            ChatResponseModel or None if failed
        """
        client = self._ensure_client()
        prompt = self._extract_text_from_messages(messages)
        effective_model = model if model else self.model
        img_data = self.process_chart_image(chart_image)
        image_part = types.Part.from_bytes(data=img_data, mime_type='image/png')
        contents = [prompt, image_part]
        include_code_execution = model_config.get("google_code_execution", False)

        # Outer: try with thinking, then without
        # Inner: try with code_execution (if enabled), then without
        for include_thinking in (True, False):
            for include_ce in (include_code_execution, False):
                if not include_ce and not include_code_execution:
                    break  # code_execution not requested — skip second inner iteration

                try:
                    generation_config = self._create_generation_config(
                        model_config,
                        include_thinking=include_thinking,
                        include_code_execution=include_ce
                    )
                    self.logger.debug(
                        "Sending chart analysis to Google AI: %s (thinking=%s, code_execution=%s, %s bytes)",
                        effective_model, include_thinking, include_ce, len(img_data)
                    )
                    response = await client.aio.models.generate_content(
                        model=effective_model,
                        contents=contents,
                        config=generation_config
                    )
                    content_text = self._extract_text_from_response(response)
                    # Pass raw image bytes so _extract_usage_metadata can estimate image tokens
                    # when the SDK omits them (known bug googleapis/python-genai#470)
                    usage = self._extract_usage_metadata(response, image_bytes=img_data)
                    self.logger.debug("Received successful chart analysis response from Google AI")
                    return self.create_response(content_text, usage=usage)
                except Exception as e: # pylint: disable=broad-exception-caught
                    if include_ce and self._should_retry_without_code_execution(e):
                        self.logger.warning(
                            "Model may not support code_execution, retrying without it: %s", e
                        )
                        continue
                    if include_thinking and self._should_retry_without_thinking(e):
                        self.logger.warning(
                            "Model may not support thinking_config for chart analysis, retrying without it: %s", e
                        )
                        break  # break inner loop, go to next thinking iteration
                    self.logger.error("Error during Google AI chart analysis request: %s", e)
                    return self._handle_exception(e)
        return None

    def _handle_exception(self, exception: Exception) -> ChatResponseModel | None:
        """Handle Google AI specific exceptions, falling back to common handler."""
        result = self.handle_common_errors(exception)
        if result:
            return result
        sanitized_error = self._sanitize_error_message(str(exception))
        self.logger.error("Unexpected Google AI error: %s", sanitized_error)
        return None
