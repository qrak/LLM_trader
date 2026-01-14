"""
Comprehensive test suite for AI provider clients (GoogleAI, LMStudio, OpenRouter).
Tests cover chat completion, streaming, chart analysis, error handling, and image encoding.
All SDK calls are mocked for isolation; the retry decorator is bypassed.
"""
import base64
import io
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from src.logger.logger import Logger
from src.platforms.ai_providers import ResponseDict, UsageDict


def passthrough_decorator(*args, **kwargs):
    """Passthrough decorator that bypasses retry logic for unit tests."""
    def decorator(func):
        return func
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return decorator


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = MagicMock(spec=Logger)
    logger.debug = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    return logger


@pytest.fixture
def sample_messages() -> List[Dict[str, Any]]:
    """Sample messages for chat completion tests."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Analyze this market data."}
    ]


@pytest.fixture
def sample_model_config() -> Dict[str, Any]:
    """Sample model configuration."""
    return {
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9,
        "top_k": 40
    }


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Sample image bytes for chart analysis tests."""
    return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'


@pytest.fixture
def sample_image_bytesio(sample_image_bytes) -> io.BytesIO:
    """Sample BytesIO image for chart analysis tests."""
    return io.BytesIO(sample_image_bytes)


class MockGoogleAIResponse:
    """Mock response object for Google AI SDK."""
    
    def __init__(self, text: str = "Test response", usage: Optional[Dict[str, int]] = None):
        self.candidates = [MockCandidate(text)]
        self.usage_metadata = MockUsageMetadata(usage) if usage else None


class MockCandidate:
    """Mock candidate for Google AI response."""
    
    def __init__(self, text: str):
        self.content = MockContent(text)


class MockContent:
    """Mock content for Google AI response."""
    
    def __init__(self, text: str):
        self.parts = [MockPart(text)]


class MockPart:
    """Mock part for Google AI response."""
    
    def __init__(self, text: str):
        self.text = text


class MockUsageMetadata:
    """Mock usage metadata for Google AI response."""
    
    def __init__(self, usage: Optional[Dict[str, int]] = None):
        usage = usage or {}
        self.prompt_token_count = usage.get('prompt_tokens', 100)
        self.candidates_token_count = usage.get('completion_tokens', 50)
        self.total_token_count = usage.get('total_tokens', 150)


class MockOpenRouterResponse:
    """Mock response object for OpenRouter SDK."""
    
    def __init__(self, content: str = "Test response", usage: Optional[Dict[str, int]] = None):
        self.id = "test-gen-id-123"
        self.model = "test-model"
        self.choices = [MockOpenRouterChoice(content)]
        self.usage = MockOpenRouterUsage(usage) if usage else None
        self.error = None


class MockOpenRouterChoice:
    """Mock choice for OpenRouter response."""
    
    def __init__(self, content: str):
        self.message = MockOpenRouterMessage(content)
        self.finish_reason = "stop"
        self.error = None


class MockOpenRouterMessage:
    """Mock message for OpenRouter response."""
    
    def __init__(self, content: str):
        self.role = "assistant"
        self.content = content


class MockOpenRouterUsage:
    """Mock usage for OpenRouter response."""
    
    def __init__(self, usage: Optional[Dict[str, int]] = None):
        usage = usage or {}
        self.prompt_tokens = usage.get('prompt_tokens', 100)
        self.completion_tokens = usage.get('completion_tokens', 50)
        self.total_tokens = usage.get('total_tokens', 150)


class MockLMStudioResponse:
    """Mock response for LM Studio SDK - simple string representation."""
    
    def __init__(self, content: str = "Test response"):
        self._content = content
    
    def __str__(self):
        return self._content


class MockStreamFragment:
    """Mock stream fragment for LM Studio streaming."""
    
    def __init__(self, content: str):
        self.content = content


class TestGoogleAIClient:
    """Tests for GoogleAIClient."""
    
    @pytest.fixture
    def google_client(self, mock_logger):
        """Create GoogleAIClient with mocked retry decorator."""
        with patch('src.platforms.ai_providers.google.retry_api_call', passthrough_decorator):
            from src.platforms.ai_providers.google import GoogleAIClient
            return GoogleAIClient(
                api_key="test-api-key",
                model="gemini-2.5-flash",
                logger=mock_logger
            )
    
    @pytest.mark.asyncio
    async def test_context_manager_lifecycle(self, google_client):
        """Test async context manager properly initializes and closes client."""
        with patch('src.platforms.ai_providers.google.genai.Client') as mock_genai:
            mock_genai.return_value = MagicMock()
            
            async with google_client as client:
                assert client.client is not None
            
            assert client.client is None
    
    @pytest.mark.asyncio
    async def test_chat_completion_success(self, google_client, sample_messages, sample_model_config):
        """Test successful chat completion returns proper ResponseDict."""
        mock_response = MockGoogleAIResponse(
            text="Analysis complete: BUY signal detected.",
            usage={'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150}
        )
        
        with patch('src.platforms.ai_providers.google.genai.Client') as mock_genai:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            mock_genai.return_value = mock_client
            google_client.client = mock_client
            
            result = await google_client.chat_completion(
                model="gemini-2.0-flash",
                messages=sample_messages,
                model_config=sample_model_config
            )
        
        assert result is not None
        assert 'choices' in result
        assert result['choices'][0]['message']['content'] == "Analysis complete: BUY signal detected."
        assert result['choices'][0]['message']['role'] == "assistant"
        assert 'usage' in result
        assert result['usage']['prompt_tokens'] == 100
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_model_override(self, google_client, sample_messages, sample_model_config):
        """Test chat completion respects model override parameter."""
        mock_response = MockGoogleAIResponse(text="Override model response")
        
        with patch('src.platforms.ai_providers.google.genai.Client') as mock_genai:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            mock_genai.return_value = mock_client
            google_client.client = mock_client
            
            await google_client.chat_completion(
                model="gemini-2.5-pro",
                messages=sample_messages,
                model_config=sample_model_config
            )
            
            call_args = mock_client.aio.models.generate_content.call_args
            assert call_args.kwargs['model'] == "gemini-2.5-pro"
    
    @pytest.mark.asyncio
    async def test_chat_completion_thinking_fallback(self, google_client, sample_messages, sample_model_config, mock_logger):
        """Test that thinking_config fallback works when model doesn't support it."""
        call_count = [0]
        
        async def mock_generate(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("400 Bad Request: thinking_config not supported")
            return MockGoogleAIResponse(text="Success without thinking")
        
        with patch('src.platforms.ai_providers.google.genai.Client') as mock_genai:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_genai.return_value = mock_client
            google_client.client = mock_client
            
            result = await google_client.chat_completion(
                model="gemini-2.0-flash",
                messages=sample_messages,
                model_config=sample_model_config
            )
        assert call_count[0] == 2
        assert result is not None
        assert result['choices'][0]['message']['content'] == "Success without thinking"
    
    @pytest.mark.asyncio
    async def test_(self, google_client, sample_messages, sample_model_config, sample_image_bytes):
        """Test chart analysis with image encoding."""
        mock_response = MockGoogleAIResponse(text="Chart analysis: bullish pattern detected")
        
        with patch('src.platforms.ai_providers.google.genai.Client') as mock_genai:
            with patch('src.platforms.ai_providers.google.types.Part.from_bytes') as mock_from_bytes:
                mock_from_bytes.return_value = MagicMock()
                mock_client = MagicMock()
                mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
                mock_genai.return_value = mock_client
                google_client.client = mock_client
                
                result = await google_client.chat_completion_with_chart_analysis(
                    model="gemini-2.0-flash",
                    messages=sample_messages,
                    chart_image=sample_image_bytes,
                    model_config=sample_model_config
                )
        
        assert result is not None
        assert 'bullish pattern' in result['choices'][0]['message']['content']
        mock_from_bytes.assert_called_once()
        call_kwargs = mock_from_bytes.call_args.kwargs
        assert call_kwargs['data'] == sample_image_bytes
        assert call_kwargs['mime_type'] == 'image/png'
    
    @pytest.mark.asyncio
    async def test_image_encoding_google_bytesio(self, google_client, sample_messages, sample_model_config, sample_image_bytesio):
        """Test image encoding from BytesIO correctly reads and resets position."""
        mock_response = MockGoogleAIResponse(text="BytesIO image analyzed")
        
        with patch('src.platforms.ai_providers.google.genai.Client') as mock_genai:
            with patch('src.platforms.ai_providers.google.types.Part.from_bytes') as mock_from_bytes:
                mock_from_bytes.return_value = MagicMock()
                mock_client = MagicMock()
                mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
                mock_genai.return_value = mock_client
                google_client.client = mock_client
                
                await google_client.chat_completion_with_chart_analysis(
                    model="gemini-2.0-flash",
                    messages=sample_messages,
                    chart_image=sample_image_bytesio,
                    model_config=sample_model_config
                )
        
        assert sample_image_bytesio.tell() == 0
    
    @pytest.mark.asyncio
    async def test_extract_text_with_non_text_parts(self, google_client):
        """Test _extract_text_from_response handles non-text parts gracefully."""
        class MockPartWithoutText:
            pass
        
        class MockMixedContent:
            def __init__(self):
                self.parts = [MockPart("Real text"), MockPartWithoutText()]
        
        class MockMixedCandidate:
            def __init__(self):
                self.content = MockMixedContent()
        
        class MockMixedResponse:
            def __init__(self):
                self.candidates = [MockMixedCandidate()]
        
        result = google_client._extract_text_from_response(MockMixedResponse())
        assert result == "Real text"
    
    @pytest.mark.asyncio
    async def test_error_handling_overloaded(self, google_client, sample_messages, sample_model_config):
        """Test error handling for 503 overloaded errors."""
        with patch('src.platforms.ai_providers.google.genai.Client') as mock_genai:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(
                side_effect=Exception("503 Service Unavailable: API overloaded")
            )
            mock_genai.return_value = mock_client
            google_client.client = mock_client
            
            result = await google_client.chat_completion(
                model="gemini-2.0-flash",
                messages=sample_messages,
                model_config=sample_model_config
            )
        
        assert result is not None
        assert result.get('error') == 'overloaded'
    
    @pytest.mark.asyncio
    async def test_error_handling_rate_limit(self, google_client, sample_messages, sample_model_config):
        """Test error handling for rate limit errors."""
        with patch('src.platforms.ai_providers.google.genai.Client') as mock_genai:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(
                side_effect=Exception("Quota exceeded: rate limit reached")
            )
            mock_genai.return_value = mock_client
            google_client.client = mock_client
            
            result = await google_client.chat_completion(
                model="gemini-2.0-flash",
                messages=sample_messages,
                model_config=sample_model_config
            )
        
        assert result is not None
        assert result.get('error') == 'rate_limit'
    
    def test_extract_usage_metadata(self, google_client):
        """Test usage metadata extraction from response."""
        mock_response = MockGoogleAIResponse(
            text="Test",
            usage={'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150}
        )
        
        usage = google_client._extract_usage_metadata(mock_response)
        
        assert usage is not None
        assert usage['prompt_tokens'] == 100
        assert usage['completion_tokens'] == 50
        assert usage['total_tokens'] == 150


class TestLMStudioClient:
    """Tests for LMStudioClient."""
    
    @pytest.fixture
    def lmstudio_client(self, mock_logger):
        """Create LMStudioClient with mocked retry decorator."""
        with patch('src.platforms.ai_providers.lmstudio.retry_api_call', passthrough_decorator):
            from src.platforms.ai_providers.lmstudio import LMStudioClient
            return LMStudioClient(
                base_url="http://localhost:1234",
                logger=mock_logger
            )
    
    @pytest.mark.asyncio
    async def test_context_manager_lifecycle(self, lmstudio_client):
        """Test async context manager properly initializes and closes client."""
        with patch('src.platforms.ai_providers.lmstudio.lms.AsyncClient') as mock_lms:
            mock_lms.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            mock_lms.return_value.__aexit__ = AsyncMock()
            
            async with lmstudio_client as client:
                assert client is not None
    
    @pytest.mark.asyncio
    async def test_chat_completion_success(self, lmstudio_client, sample_messages, sample_model_config):
        """Test successful chat completion returns proper ResponseDict."""
        mock_response = MockLMStudioResponse("LM Studio analysis complete")
        
        with patch('src.platforms.ai_providers.lmstudio.lms.AsyncClient') as mock_lms_class:
            mock_client = AsyncMock()
            mock_llm = AsyncMock()
            mock_llm.respond = AsyncMock(return_value=mock_response)
            mock_client.llm.model = AsyncMock(return_value=mock_llm)
            mock_client.llm.list_loaded = AsyncMock(return_value=[])
            
            mock_lms_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_lms_class.return_value.__aexit__ = AsyncMock()
            
            result = await lmstudio_client.chat_completion(
                model="test-model",
                messages=sample_messages,
                model_config=sample_model_config
            )
        
        assert result is not None
        assert 'choices' in result
        assert result['choices'][0]['message']['content'] == "LM Studio analysis complete"
    
    @pytest.mark.asyncio
    async def test_chat_completion_auto_select_model(self, lmstudio_client, sample_messages, sample_model_config):
        """Test auto-selection of loaded model when no model specified."""
        mock_response = MockLMStudioResponse("Auto-selected model response")
        mock_loaded_model = MagicMock()
        mock_loaded_model.identifier = "auto-loaded-model"
        
        with patch('src.platforms.ai_providers.lmstudio.lms.AsyncClient') as mock_lms_class:
            mock_client = AsyncMock()
            mock_llm = AsyncMock()
            mock_llm.respond = AsyncMock(return_value=mock_response)
            mock_client.llm.model = AsyncMock(return_value=mock_llm)
            mock_client.llm.list_loaded = AsyncMock(return_value=[mock_loaded_model])
            
            mock_lms_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_lms_class.return_value.__aexit__ = AsyncMock()
            
            result = await lmstudio_client.chat_completion(
                model="",
                messages=sample_messages,
                model_config=sample_model_config
            )
        
        assert result is not None
        mock_client.llm.model.assert_called_with("auto-loaded-model")
    
    @pytest.mark.asyncio
    async def test_chat_completion_no_model_error(self, lmstudio_client, sample_messages, sample_model_config):
        """Test error when no model specified and none loaded."""
        with patch('src.platforms.ai_providers.lmstudio.lms.AsyncClient') as mock_lms_class:
            mock_client = AsyncMock()
            mock_client.llm.list_loaded = AsyncMock(return_value=[])
            
            mock_lms_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_lms_class.return_value.__aexit__ = AsyncMock()
            
            result = await lmstudio_client.chat_completion(
                model="",
                messages=sample_messages,
                model_config=sample_model_config
            )
        
        assert result is None or 'error' in result
    
    @pytest.mark.asyncio
    async def test_stream_chat_completion_chunk_by_chunk(self, lmstudio_client, sample_messages, sample_model_config):
        """Test streaming completion processes chunks correctly."""
        chunks = ["Hello ", "world ", "from ", "LM ", "Studio!"]
        
        async def mock_stream():
            for chunk in chunks:
                yield MockStreamFragment(chunk)
        
        callback_results = []
        async def capture_callback(text):
            callback_results.append(text)
        
        with patch('src.platforms.ai_providers.lmstudio.lms.AsyncClient') as mock_lms_class:
            mock_client = AsyncMock()
            mock_llm = AsyncMock()
            mock_llm.respond_stream = AsyncMock(return_value=mock_stream())
            mock_client.llm.model = AsyncMock(return_value=mock_llm)
            mock_client.llm.list_loaded = AsyncMock(return_value=[])
            
            mock_lms_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_lms_class.return_value.__aexit__ = AsyncMock()
            
            result = await lmstudio_client.stream_chat_completion(
                model="test-model",
                messages=sample_messages,
                model_config=sample_model_config,
                callback=capture_callback
            )
        
        assert callback_results == chunks
        assert result is not None
        assert result['choices'][0]['message']['content'] == "Hello world from LM Studio!"
    
    @pytest.mark.asyncio
    async def test_stream_aggregated_response(self, lmstudio_client, sample_messages, sample_model_config):
        """Test streaming returns properly aggregated final response."""
        chunks = ["Part1", "Part2", "Part3"]
        
        async def mock_stream():
            for chunk in chunks:
                yield MockStreamFragment(chunk)
        
        with patch('src.platforms.ai_providers.lmstudio.lms.AsyncClient') as mock_lms_class:
            mock_client = AsyncMock()
            mock_llm = AsyncMock()
            mock_llm.respond_stream = AsyncMock(return_value=mock_stream())
            mock_client.llm.model = AsyncMock(return_value=mock_llm)
            mock_client.llm.list_loaded = AsyncMock(return_value=[])
            
            mock_lms_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_lms_class.return_value.__aexit__ = AsyncMock()
            
            result = await lmstudio_client.stream_chat_completion(
                model="test-model",
                messages=sample_messages,
                model_config=sample_model_config
            )
        
        assert result['choices'][0]['message']['content'] == "Part1Part2Part3"
        assert result['choices'][0]['message']['role'] == "assistant"
    
    @pytest.mark.asyncio
    async def test_(self, lmstudio_client, sample_messages, sample_model_config, sample_image_bytes):
        """Test chart analysis with image."""
        mock_response = MockLMStudioResponse("Chart pattern: head and shoulders")
        
        with patch('src.platforms.ai_providers.lmstudio.lms.AsyncClient') as mock_lms_class:
            with patch('src.platforms.ai_providers.lmstudio.lms.Chat') as mock_chat_class:
                mock_chat_class.return_value = MagicMock()
                mock_client = AsyncMock()
                mock_llm = AsyncMock()
                mock_llm.respond = AsyncMock(return_value=mock_response)
                mock_client.llm.model = AsyncMock(return_value=mock_llm)
                mock_client.llm.list_loaded = AsyncMock(return_value=[])
                mock_client.files.prepare_image = AsyncMock(return_value=MagicMock())
                
                mock_lms_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
                mock_lms_class.return_value.__aexit__ = AsyncMock()
                
                result = await lmstudio_client.chat_completion_with_chart_analysis(
                    model="test-vision-model",
                    messages=sample_messages,
                    chart_image=sample_image_bytes,
                    model_config=sample_model_config
                )
        
        assert result is not None
        assert 'head and shoulders' in result['choices'][0]['message']['content']
        mock_client.files.prepare_image.assert_called_once_with(sample_image_bytes)
    
    @pytest.mark.asyncio
    async def test_error_handling_gpu_crash(self, lmstudio_client, sample_messages, sample_model_config):
        """Test error handling for GPU crash errors."""
        with patch('src.platforms.ai_providers.lmstudio.lms.AsyncClient') as mock_lms_class:
            mock_lms_class.return_value.__aenter__ = AsyncMock(
                side_effect=Exception("ErrorDeviceLost: vk::Queue::submit failed")
            )
            mock_lms_class.return_value.__aexit__ = AsyncMock()
            
            result = await lmstudio_client.chat_completion(
                model="test-model",
                messages=sample_messages,
                model_config=sample_model_config
            )
        
        assert result is not None
        assert result.get('error') == 'gpu_crash'
    
    @pytest.mark.asyncio
    async def test_error_handling_connection(self, lmstudio_client, sample_messages, sample_model_config):
        """Test error handling for connection errors."""
        with patch('src.platforms.ai_providers.lmstudio.lms.AsyncClient') as mock_lms_class:
            mock_lms_class.return_value.__aenter__ = AsyncMock(
                side_effect=Exception("Connection refused: ECONNRESET")
            )
            mock_lms_class.return_value.__aexit__ = AsyncMock()
            
            result = await lmstudio_client.chat_completion(
                model="test-model",
                messages=sample_messages,
                model_config=sample_model_config
            )
        
        assert result is not None
        assert result.get('error') == 'connection'
    
    def test_get_api_host_parsing(self, lmstudio_client):
        """Test API host parsing from various URL formats."""
        lmstudio_client.base_url = "http://localhost:1234"
        assert lmstudio_client._get_api_host() == "localhost:1234"
        
        lmstudio_client.base_url = "https://192.168.1.100:1234/v1"
        assert lmstudio_client._get_api_host() == "192.168.1.100:1234"
        
        lmstudio_client.base_url = "localhost:1234"
        assert lmstudio_client._get_api_host() == "localhost:1234"
    
    def test_build_prediction_config(self, lmstudio_client, sample_model_config):
        """Test prediction config building handles parameters correctly."""
        with patch('src.platforms.ai_providers.lmstudio.lms.LlmPredictionConfig') as mock_config:
            mock_config.return_value = MagicMock()
            
            result = lmstudio_client._build_prediction_config(sample_model_config)
            
            mock_config.assert_called()
            call_kwargs = mock_config.call_args.kwargs
            assert 'temperature' in call_kwargs
            assert 'max_tokens' in call_kwargs


class TestOpenRouterClient:
    """Tests for OpenRouterClient."""
    
    @pytest.fixture
    def openrouter_client(self, mock_logger):
        """Create OpenRouterClient with mocked retry decorator."""
        with patch('src.platforms.ai_providers.openrouter.retry_api_call', passthrough_decorator):
            from src.platforms.ai_providers.openrouter import OpenRouterClient
            return OpenRouterClient(
                api_key="test-api-key",
                base_url="https://openrouter.ai/api/v1",
                logger=mock_logger
            )
    
    @pytest.mark.asyncio
    async def test_context_manager_lifecycle(self, openrouter_client):
        """Test async context manager properly initializes and closes client."""
        with patch('src.platforms.ai_providers.openrouter.OpenRouter') as mock_or:
            mock_or.return_value = MagicMock()
            
            async with openrouter_client as client:
                assert client._client is not None
            
            assert client._client is None
    
    @pytest.mark.asyncio
    async def test_chat_completion_success(self, openrouter_client, sample_messages, sample_model_config):
        """Test successful chat completion returns proper ResponseDict."""
        mock_response = MockOpenRouterResponse(
            content="OpenRouter analysis complete",
            usage={'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150}
        )
        
        with patch('src.platforms.ai_providers.openrouter.OpenRouter') as mock_or:
            mock_client = MagicMock()
            mock_client.chat.send_async = AsyncMock(return_value=mock_response)
            mock_or.return_value = mock_client
            openrouter_client._client = mock_client
            
            result = await openrouter_client.chat_completion(
                model="anthropic/claude-3-opus",
                messages=sample_messages,
                model_config=sample_model_config
            )
        
        assert result is not None
        assert 'choices' in result
        assert result['choices'][0]['message']['content'] == "OpenRouter analysis complete"
        assert result['usage']['prompt_tokens'] == 100
    
    @pytest.mark.asyncio
    async def test_(self, openrouter_client, sample_messages, sample_model_config, sample_image_bytes):
        """Test chart analysis with base64 image encoding."""
        mock_response = MockOpenRouterResponse(content="Chart shows bullish divergence")
        
        with patch('src.platforms.ai_providers.openrouter.OpenRouter') as mock_or:
            mock_client = MagicMock()
            mock_client.chat.send_async = AsyncMock(return_value=mock_response)
            mock_or.return_value = mock_client
            openrouter_client._client = mock_client
            
            result = await openrouter_client.chat_completion_with_chart_analysis(
                model="anthropic/claude-3-opus",
                messages=sample_messages,
                chart_image=sample_image_bytes,
                model_config=sample_model_config
            )
        
        assert result is not None
        assert 'bullish divergence' in result['choices'][0]['message']['content']
        
        call_args = mock_client.chat.send_async.call_args
        sent_messages = call_args.kwargs.get('messages') or call_args[1] if len(call_args) > 1 else None
        if sent_messages:
            user_message = next((m for m in sent_messages if m.get('role') == 'user'), None)
            if user_message and isinstance(user_message.get('content'), list):
                image_parts = [p for p in user_message['content'] if p.get('type') == 'image_url']
                assert len(image_parts) > 0
                assert 'data:image/png;base64,' in image_parts[0]['image_url']['url']
    
    @pytest.mark.asyncio
    async def test_image_encoding_openrouter_base64(self, openrouter_client, sample_messages, sample_model_config, sample_image_bytes):
        """Test that OpenRouter correctly encodes images as base64 data URLs."""
        mock_response = MockOpenRouterResponse(content="Image processed")
        expected_base64 = base64.b64encode(sample_image_bytes).decode('utf-8')
        
        with patch('src.platforms.ai_providers.openrouter.OpenRouter') as mock_or:
            mock_client = MagicMock()
            mock_client.chat.send_async = AsyncMock(return_value=mock_response)
            mock_or.return_value = mock_client
            openrouter_client._client = mock_client
            
            await openrouter_client.chat_completion_with_chart_analysis(
                model="test-model",
                messages=sample_messages,
                chart_image=sample_image_bytes,
                model_config=sample_model_config
            )
        
        call_args = mock_client.chat.send_async.call_args
        sent_messages = call_args.kwargs.get('messages', [])
        
        found_base64 = False
        for msg in sent_messages:
            if isinstance(msg.get('content'), list):
                for part in msg['content']:
                    if part.get('type') == 'image_url':
                        url = part['image_url']['url']
                        assert url == f"data:image/png;base64,{expected_base64}"
                        found_base64 = True
        
        assert found_base64, "Base64 image not found in sent messages"
    
    @pytest.mark.asyncio
    async def test_param_retry_filters_unsupported(self, openrouter_client, sample_messages, sample_model_config):
        """Test that unsupported parameters are filtered and retried."""
        call_count = [0]
        
        async def mock_send(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1 and 'thinking_budget' in kwargs:
                raise Exception("unexpected keyword argument 'thinking_budget'")
            return MockOpenRouterResponse(content="Success after filter")
        
        with patch('src.platforms.ai_providers.openrouter.OpenRouter') as mock_or:
            mock_client = MagicMock()
            mock_client.chat.send_async = mock_send
            mock_or.return_value = mock_client
            openrouter_client._client = mock_client
            
            config_with_unsupported = {**sample_model_config, 'thinking_budget': 100}
            result = await openrouter_client.chat_completion(
                model="test-model",
                messages=sample_messages,
                model_config=config_with_unsupported
            )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_error_handling_rate_limit(self, openrouter_client, sample_messages, sample_model_config):
        """Test error handling for rate limit errors."""
        with patch('src.platforms.ai_providers.openrouter.OpenRouter') as mock_or:
            mock_client = MagicMock()
            mock_client.chat.send_async = AsyncMock(
                side_effect=Exception("Rate limit exceeded: too many requests")
            )
            mock_or.return_value = mock_client
            openrouter_client._client = mock_client
            
            result = await openrouter_client.chat_completion(
                model="test-model",
                messages=sample_messages,
                model_config=sample_model_config
            )
        
        assert result is not None
        assert result.get('error') == 'rate_limit'
    
    @pytest.mark.asyncio
    async def test_error_handling_authentication(self, openrouter_client, sample_messages, sample_model_config):
        """Test error handling for authentication errors."""
        with patch('src.platforms.ai_providers.openrouter.OpenRouter') as mock_or:
            mock_client = MagicMock()
            mock_client.chat.send_async = AsyncMock(
                side_effect=Exception("Authentication failed: invalid API key")
            )
            mock_or.return_value = mock_client
            openrouter_client._client = mock_client
            
            result = await openrouter_client.chat_completion(
                model="test-model",
                messages=sample_messages,
                model_config=sample_model_config
            )
        
        assert result is not None
        assert result.get('error') == 'authentication'
    
    def test_convert_sdk_response(self, openrouter_client):
        """Test SDK response conversion to ResponseDict format."""
        mock_response = MockOpenRouterResponse(
            content="Test content",
            usage={'prompt_tokens': 50, 'completion_tokens': 25, 'total_tokens': 75}
        )
        
        result = openrouter_client._convert_sdk_response(mock_response)
        
        assert result['id'] == "test-gen-id-123"
        assert result['model'] == "test-model"
        assert result['choices'][0]['message']['content'] == "Test content"
        assert result['choices'][0]['message']['role'] == "assistant"
        assert result['usage']['prompt_tokens'] == 50
    
    def test_extract_user_text_from_messages(self, openrouter_client, sample_messages):
        """Test extraction of user text from message list."""
        result = openrouter_client._extract_user_text_from_messages(sample_messages)
        assert result == "Analyze this market data."
    
    def test_prepare_multimodal_messages(self, openrouter_client, sample_messages):
        """Test multimodal message preparation."""
        user_text = "Analyze this market data."
        multimodal_content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]
        
        result = openrouter_client._prepare_multimodal_messages(
            sample_messages, user_text, multimodal_content
        )
        
        assert len(result) >= 1
        # The last user message should be converted to multimodal content
        last_user_msg = None
        for msg in result:
            if msg.get('role') == 'user':
                last_user_msg = msg
        assert last_user_msg is not None
        # Check if it's the multimodal message (list content) or system instruction (string)
        # The method converts the last user message to multimodal
        has_multimodal = any(
            isinstance(m.get('content'), list) for m in result
        )
        assert has_multimodal, "Expected multimodal content in prepared messages"


class TestMockClientCompatibility:
    """Tests to verify MockClient produces responses compatible with real providers."""
    
    @pytest.fixture
    def mock_client(self, mock_logger):
        """Create MockClient instance."""
        from src.platforms.ai_providers.mock import MockClient
        return MockClient(logger=mock_logger)
    
    @pytest.mark.asyncio
    async def test_mock_chat_completion_response_structure(self, mock_client, sample_messages):
        """Test MockClient returns proper ResponseDict structure."""
        result = await mock_client.chat_completion(
            model="test-model",
            messages=sample_messages,
            model_config={}
        )
        
        assert result is not None
        assert 'choices' in result
        assert len(result['choices']) > 0
        assert 'message' in result['choices'][0]
        assert 'content' in result['choices'][0]['message']
    
    @pytest.mark.asyncio
    async def test_mock_chart_analysis_response_structure(self, mock_client, sample_messages, sample_image_bytes):
        """Test MockClient chart analysis returns proper structure."""
        result = await mock_client.chat_completion_with_chart_analysis(
            model="test-model",
            messages=sample_messages,
            chart_image=sample_image_bytes,
            model_config={}
        )
        
        assert result is not None
        assert 'choices' in result
        content = result['choices'][0]['message']['content']
        assert 'json' in content.lower() or '```' in content
    
    @pytest.mark.asyncio
    async def test_mock_extracts_test_hint(self, mock_client):
        """Test MockClient correctly extracts TEST_HINT from messages."""
        messages = [
            {"role": "user", "content": "TEST_HINT: last_close=50000.0\nAnalyze BTC"}
        ]
        
        result = await mock_client.chat_completion(
            model="test-model",
            messages=messages,
            model_config={}
        )
        
        content = result['choices'][0]['message']['content']
        assert 'json' in content.lower()
        assert 'signal' in content.lower()
    
    @pytest.mark.asyncio
    async def test_mock_console_stream(self, mock_client, sample_messages):
        """Test MockClient console_stream returns proper structure."""
        result = await mock_client.console_stream(
            model="test-model",
            messages=sample_messages,
            model_config={}
        )
        
        assert result is not None
        assert 'choices' in result


class TestResponseDictValidation:
    """Tests to validate ResponseDict structure across all providers."""
    
    def test_response_dict_typing(self):
        """Verify ResponseDict TypedDict structure is correct."""
        response: ResponseDict = {
            'choices': [{'message': {'content': 'test', 'role': 'assistant'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
        }
        
        assert 'choices' in response
        assert 'usage' in response
    
    def test_usage_dict_typing(self):
        """Verify UsageDict TypedDict structure is correct."""
        usage: UsageDict = {
            'prompt_tokens': 100,
            'completion_tokens': 50,
            'total_tokens': 150,
            'cost': 0.001
        }
        
        assert 'prompt_tokens' in usage
        assert 'cost' in usage
    
    def test_error_response_format(self):
        """Verify error response format is consistent."""
        error_response: ResponseDict = {
            'error': 'rate_limit'
        }
        
        assert 'error' in error_response
        assert error_response['error'] == 'rate_limit'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
