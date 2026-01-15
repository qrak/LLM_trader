"""
Unit tests for BlockRun.AI provider client using mocks.
Tests security features, message handling, and API interactions.
"""
import base64
import io
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.logger.logger import Logger
from src.platforms.ai_providers.blockrun import BlockRunClient


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = Mock(spec=Logger)
    logger.debug = Mock()
    logger.error = Mock()
    logger.warning = Mock()
    return logger


@pytest.fixture
def blockrun_client(mock_logger):
    """Create a BlockRunClient with mock dependencies."""
    return BlockRunClient(
        wallet_key="0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        base_url="https://blockrun.ai/api",
        logger=mock_logger
    )


@pytest.fixture
def mock_blockrun_sdk():
    """Mock the BlockRun SDK."""
    with patch('src.platforms.ai_providers.blockrun.BlockRun') as mock:
        yield mock


class TestBlockRunClientInitialization:
    """Test BlockRun client initialization and lifecycle."""

    @pytest.mark.asyncio
    async def test_initialize_client(self, blockrun_client, mock_blockrun_sdk, mock_logger):
        """Test SDK client initialization."""
        mock_sdk_instance = AsyncMock()
        mock_blockrun_sdk.return_value = mock_sdk_instance
        
        await blockrun_client._initialize_client()
        
        mock_blockrun_sdk.assert_called_once_with(
            wallet_private_key="0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            base_url="https://blockrun.ai/api"
        )
        assert blockrun_client._client == mock_sdk_instance
        mock_logger.debug.assert_called_with("BlockRun SDK client initialized successfully")

    @pytest.mark.asyncio
    async def test_initialize_client_missing_sdk(self, blockrun_client, mock_logger):
        """Test initialization with missing SDK."""
        with patch('src.platforms.ai_providers.blockrun.BlockRun', side_effect=ImportError("No module")):
            with pytest.raises(ImportError, match="blockrun-llm SDK is required"):
                await blockrun_client._initialize_client()

    @pytest.mark.asyncio
    async def test_close(self, blockrun_client, mock_logger):
        """Test client closure."""
        blockrun_client._client = Mock()
        
        await blockrun_client.close()
        
        assert blockrun_client._client is None
        mock_logger.debug.assert_called_with("Closing BlockRunClient SDK session")


class TestSecurityFeatures:
    """Test security features like private key redaction."""

    def test_redact_private_key(self, blockrun_client):
        """Test private key is redacted from error messages."""
        error_message = "Connection failed with key 0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        
        redacted = blockrun_client._redact_private_key(error_message)
        
        assert "0x1234...cdef" in redacted
        assert "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef" not in redacted

    def test_redact_private_key_no_key(self):
        """Test redaction when key is not in message."""
        client = BlockRunClient(wallet_key="", base_url="", logger=Mock())
        message = "Some error occurred"
        
        redacted = client._redact_private_key(message)
        
        assert redacted == message


class TestMessageHandling:
    """Test message extraction and transformation."""

    def test_extract_all_user_text_single_message(self, blockrun_client):
        """Test extraction from single user message."""
        messages = [
            {"role": "user", "content": "Analyze BTC"}
        ]
        
        result = blockrun_client._extract_all_user_text_from_messages(messages)
        
        assert result == "Analyze BTC"

    def test_extract_all_user_text_multiple_messages(self, blockrun_client):
        """Test extraction from multiple user messages (conversation history)."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "Analyze BTC"},
        ]
        
        result = blockrun_client._extract_all_user_text_from_messages(messages)
        
        assert result == "Hello\n\nAnalyze BTC"

    def test_extract_all_user_text_empty_messages(self, blockrun_client):
        """Test extraction with no user messages."""
        messages = [
            {"role": "system", "content": "Instructions"}
        ]
        
        result = blockrun_client._extract_all_user_text_from_messages(messages)
        
        assert result == ""

    def test_prepare_multimodal_messages(self, blockrun_client):
        """Test multimodal message preparation."""
        messages = [
            {"role": "system", "content": "You are a trading assistant"},
            {"role": "user", "content": "Analyze this chart"}
        ]
        multimodal_content = [
            {"type": "text", "text": "Analyze this chart"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}}
        ]
        
        result = blockrun_client._prepare_multimodal_messages(messages, multimodal_content)
        
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert "System instructions:" in result[0]["content"]
        assert result[1]["role"] == "user"
        assert result[1]["content"] == multimodal_content


class TestModelFormatHandling:
    """Test provider/model format handling."""

    def test_ensure_provider_prefix_with_prefix(self, blockrun_client):
        """Test model already has provider prefix."""
        result = blockrun_client._ensure_provider_prefix("openai/gpt-4o")
        assert result == "openai/gpt-4o"

    def test_ensure_provider_prefix_without_prefix(self, blockrun_client):
        """Test model without provider prefix."""
        result = blockrun_client._ensure_provider_prefix("gpt-4o")
        assert result == "openai/gpt-4o"


class TestChatCompletion:
    """Test chat completion requests."""

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, blockrun_client, mock_blockrun_sdk):
        """Test successful chat completion."""
        mock_client = AsyncMock()
        mock_client.chat_completion = AsyncMock(return_value={
            "id": "test-id",
            "model": "openai/gpt-4o",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "BTC is bullish"
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        })
        
        blockrun_client._client = mock_client
        
        result = await blockrun_client.chat_completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Analyze BTC"}],
            model_config={"temperature": 0.7}
        )
        
        assert result is not None
        assert result["choices"][0]["message"]["content"] == "BTC is bullish"
        assert result["usage"]["prompt_tokens"] == 10
        mock_client.chat_completion.assert_called_once_with(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Analyze BTC"}],
            temperature=0.7
        )

    @pytest.mark.asyncio
    async def test_chat_completion_not_initialized(self, blockrun_client):
        """Test chat completion when client not initialized."""
        with pytest.raises(RuntimeError, match="BlockRun client not initialized"):
            await blockrun_client.chat_completion(
                model="gpt-4o",
                messages=[],
                model_config={}
            )


class TestChartAnalysis:
    """Test chart analysis with images."""

    @pytest.mark.asyncio
    async def test_chart_analysis_with_bytes(self, blockrun_client, mock_blockrun_sdk):
        """Test chart analysis with bytes image."""
        mock_client = AsyncMock()
        mock_client.chat_completion = AsyncMock(return_value={
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Pattern detected"
                }
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        })
        
        blockrun_client._client = mock_client
        chart_data = b"fake_image_data"
        
        result = await blockrun_client.chat_completion_with_chart_analysis(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Analyze"}],
            chart_image=chart_data,
            model_config={}
        )
        
        assert result is not None
        assert result["choices"][0]["message"]["content"] == "Pattern detected"
        
        call_args = mock_client.chat_completion.call_args
        assert call_args[1]["model"] == "openai/gpt-4o"
        assert len(call_args[1]["messages"]) > 0

    @pytest.mark.asyncio
    async def test_chart_analysis_with_bytesio(self, blockrun_client, mock_blockrun_sdk):
        """Test chart analysis with BytesIO image."""
        mock_client = AsyncMock()
        mock_client.chat_completion = AsyncMock(return_value={
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Chart analyzed"
                }
            }]
        })
        
        blockrun_client._client = mock_client
        chart_buffer = io.BytesIO(b"image_data")
        
        result = await blockrun_client.chat_completion_with_chart_analysis(
            model="anthropic/claude-sonnet-4",
            messages=[{"role": "user", "content": "Check chart"}],
            chart_image=chart_buffer,
            model_config={}
        )
        
        assert result is not None
        assert result["choices"][0]["message"]["content"] == "Chart analyzed"


class TestErrorHandling:
    """Test error handling and exception management."""

    def test_convert_sdk_response_none(self, blockrun_client):
        """Test conversion of None response."""
        result = blockrun_client._convert_sdk_response(None)
        assert "error" in result
        assert result["error"] == "Empty response from BlockRun SDK"

    def test_convert_sdk_response_invalid_format(self, blockrun_client, mock_logger):
        """Test conversion of invalid response format."""
        result = blockrun_client._convert_sdk_response({"invalid": "response"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_handle_exception_with_redaction(self, blockrun_client, mock_logger):
        """Test exception handling redacts private keys."""
        exception = Exception("Auth failed with key 0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef")
        
        result = blockrun_client._handle_exception(exception)
        
        assert result is not None
        mock_logger.error.assert_called()
        error_call_args = str(mock_logger.error.call_args)
        assert "0x1234...cdef" in error_call_args or "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef" not in error_call_args
