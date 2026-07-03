"""
Unit tests for BlockRun.AI provider client using mocks.
SDK v1.4.7+ — tests security features, message handling, and API interactions.
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
    logger.info = Mock()
    return logger


@pytest.fixture
def blockrun_client(mock_logger):
    """Create a BlockRunClient with a fake wallet key."""
    return BlockRunClient(
        wallet_key="0x0000000000000000000000000000000000000000000000000000000000000001",
        base_url="https://blockrun.ai/api",
        logger=mock_logger,
    )


# ─────────────────────────────────────────────────────────────────
# Security tests
# ─────────────────────────────────────────────────────────────────

class TestKeyRedaction:
    """Verify private key is never leaked in logs or error messages."""

    def test_redact_short_key(self, blockrun_client):
        result = blockrun_client._redact_private_key("short")
        assert result == "short"

    def test_redact_long_key(self, blockrun_client):
        msg = f"Error with key {blockrun_client._wallet_key} in request"
        result = blockrun_client._redact_private_key(msg)
        assert blockrun_client._wallet_key not in result
        assert "0x0000" in result
        assert "0001" in result

    def test_redact_no_key(self, mock_logger):
        client = BlockRunClient(wallet_key="", base_url="https://blockrun.ai/api", logger=mock_logger)
        msg = "Some error message"
        assert client._redact_private_key(msg) == msg


# ─────────────────────────────────────────────────────────────────
# Model prefix tests
# ─────────────────────────────────────────────────────────────────

class TestModelPrefix:
    """Verify model names get provider prefix when missing."""

    def test_add_openai_prefix(self, blockrun_client):
        assert blockrun_client._ensure_provider_prefix("gpt-4o") == "openai/gpt-4o"

    def test_keep_custom_prefix(self, blockrun_client):
        assert blockrun_client._ensure_provider_prefix("anthropic/claude-sonnet-4") == "anthropic/claude-sonnet-4"


# ─────────────────────────────────────────────────────────────────
# User message extraction
# ─────────────────────────────────────────────────────────────────

class TestUserTextExtraction:
    """Verify user text is preserved in multimodal requests."""

    def test_extract_user_texts(self, blockrun_client):
        messages = [
            {"role": "system", "content": "You are a trading bot"},
            {"role": "user", "content": "Analyze BTC chart"},
            {"role": "user", "content": "What about ETH?"},
        ]
        result = blockrun_client._extract_all_user_text_from_messages(messages)
        assert "Analyze BTC chart" in result
        assert "What about ETH?" in result

    def test_extract_empty(self, blockrun_client):
        messages = [{"role": "system", "content": "System prompt"}]
        result = blockrun_client._extract_all_user_text_from_messages(messages)
        assert result == ""


# ─────────────────────────────────────────────────────────────────
# Multimodal message preparation
# ─────────────────────────────────────────────────────────────────

class TestMultimodalMessages:
    """Verify messages are converted correctly for multimodal API calls."""

    def test_system_becomes_user_prefix(self, blockrun_client):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User prompt"},
        ]
        mm = [{"type": "text", "text": "Analyze"}]
        result = blockrun_client._prepare_multimodal_messages(messages, mm)
        assert result[0]["content"] == "System instructions: System prompt"
        assert result[0]["role"] == "user"

    def test_last_user_gets_multimodal(self, blockrun_client):
        messages = [
            {"role": "user", "content": "User prompt"},
        ]
        mm = [{"type": "text", "text": "Analyze"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}}]
        result = blockrun_client._prepare_multimodal_messages(messages, mm)
        assert result[0]["content"] == mm

    def test_intermediate_user_unchanged(self, blockrun_client):
        messages = [
            {"role": "user", "content": "First prompt"},
            {"role": "user", "content": "Second prompt"},
        ]
        mm = [{"type": "text", "text": "Analyze"}]
        result = blockrun_client._prepare_multimodal_messages(messages, mm)
        assert result[0]["content"] == "First prompt"
        assert result[1]["content"] == mm


# ─────────────────────────────────────────────────────────────────
# SDK integration tests (mocked)
# ─────────────────────────────────────────────────────────────────

class TestChatCompletion:
    """Verify chat_completion calls SDK with correct params."""

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, blockrun_client):
        """Full text completion returns ChatResponseModel."""
        from blockrun_llm.types import ChatResponse, ChatChoice, ChatMessage

        fake_response = ChatResponse(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="openai/gpt-4o",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="BUY signal detected"),
                    finish_reason="stop",
                )
            ],
            cost_usd=0.001,
        )

        with patch.object(blockrun_client, "_ensure_client", AsyncMock()) as mock_ensure:
            mock_client = AsyncMock()
            mock_client.chat_completion = AsyncMock(return_value=fake_response)
            mock_ensure.return_value = mock_client

            result = await blockrun_client.chat_completion(
                "gpt-4o",
                [{"role": "user", "content": "Analyze BTC"}],
                {"temperature": 0.5, "max_tokens": 1000},
            )

            assert result is not None
            assert result.choices[0].message.content == "BUY signal detected"
            assert result.model == "openai/gpt-4o"

    @pytest.mark.asyncio
    async def test_chat_completion_empty_content(self, blockrun_client):
        """Empty content returns error ChatResponseModel."""
        from blockrun_llm.types import ChatResponse, ChatChoice, ChatMessage

        fake_response = ChatResponse(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="openai/gpt-4o",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=""),
                    finish_reason="stop",
                )
            ],
        )

        with patch.object(blockrun_client, "_ensure_client", AsyncMock()) as mock_ensure:
            mock_client = AsyncMock()
            mock_client.chat_completion = AsyncMock(return_value=fake_response)
            mock_ensure.return_value = mock_client

            result = await blockrun_client.chat_completion(
                "gpt-4o",
                [{"role": "user", "content": "Analyze"}],
                {},
            )

            assert result is not None
            assert result.error is not None
            assert "empty content" in result.error

    @pytest.mark.asyncio
    async def test_chat_completion_none_response(self, blockrun_client):
        """None response from SDK returns None."""
        with patch.object(blockrun_client, "_ensure_client", AsyncMock()) as mock_ensure:
            mock_client = AsyncMock()
            mock_client.chat_completion = AsyncMock(return_value=None)
            mock_ensure.return_value = mock_client

            result = await blockrun_client.chat_completion(
                "gpt-4o",
                [{"role": "user", "content": "Analyze"}],
                {},
            )
            assert result is None


# ─────────────────────────────────────────────────────────────────
# Chart analysis tests (mocked)
# ─────────────────────────────────────────────────────────────────

class TestChartAnalysis:
    """Verify multimodal chart analysis passes images correctly."""

    @pytest.mark.asyncio
    async def test_chart_analysis_success(self, blockrun_client):
        """Chart image is base64-encoded and included in multimodal content."""
        from blockrun_llm.types import ChatResponse, ChatChoice, ChatMessage

        fake_response = ChatResponse(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="openai/gpt-4o",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Bullish engulfing pattern detected"),
                    finish_reason="stop",
                )
            ],
        )

        # Create a 1x1 PNG pixel
        fake_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )

        with patch.object(blockrun_client, "_ensure_client", AsyncMock()) as mock_ensure:
            mock_client = AsyncMock()
            mock_client.chat_completion = AsyncMock(return_value=fake_response)
            mock_ensure.return_value = mock_client

            result = await blockrun_client.chat_completion_with_chart_analysis(
                "gpt-4o",
                [{"role": "user", "content": "Analyze this chart"}],
                io.BytesIO(fake_png),
                {"temperature": 0.3},
            )

            assert result is not None
            assert result.choices[0].message.content == "Bullish engulfing pattern detected"
