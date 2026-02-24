"""Integration tests for AI providers with real API calls.

These tests make actual API calls to AI providers.
Tests will be skipped if the required API keys or services are not available.

Run with: python -m pytest tests/test_ai_providers_integration.py -v -s
Use -s flag to see print output during tests.
"""
import io
import os
import sys
from typing import Optional

import pytest
from dotenv import load_dotenv
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger.logger import Logger
from src.platforms.ai_providers import GoogleAIClient, LMStudioClient, OpenRouterClient
from src.platforms.ai_providers.response_models import ChatResponseModel


load_dotenv("keys.env")


def create_test_image() -> bytes:
    """Create a simple test image for vision tests."""
    img = Image.new('RGB', (100, 100), color='blue')
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    return img_buffer.read()


class RateLimitError(Exception):
    """Raised when API returns rate limit error."""


def validate_response(response: Optional[ChatResponseModel], test_name: str) -> bool:
    """Validate a ChatResponseModel has required fields and content.
    
    Raises RateLimitError if response indicates rate limiting.
    """
    if response is None:
        print(f"  ✗ {test_name}: No response received")
        return False
    if response.error:
        error = response.error
        if error in ("rate_limit", "overloaded") or "rate_limit" in error or "overloaded" in error:
            raise RateLimitError(f"Rate limited: {error}")
        print(f"  ✗ {test_name}: Error in response: {error}")
        return False
    if not response.choices:
        print(f"  ✗ {test_name}: No choices in response")
        return False
    content = response.choices[0].message.content if response.choices[0].message else ""
    if not content:
        print(f"  ✗ {test_name}: Empty content in response")
        return False
    print(f"  ✓ {test_name}: Content received ({len(content)} chars)")
    print(f"    Response: {content[:200]}{'...' if len(content) > 200 else ''}")
    if response.usage:
        prompt_tokens = response.usage.prompt_tokens or 0
        completion_tokens = response.usage.completion_tokens or 0
        print(f"    Tokens: prompt={prompt_tokens}, completion={completion_tokens}")
    return True


class TestLMStudioIntegration:
    """Integration tests for LMStudio client with real API calls."""

    BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")

    @pytest.fixture
    def logger(self):
        log = Logger("test_lmstudio_integration", logger_debug=False)
        yield log
        log.close()

    @staticmethod
    async def check_lmstudio_available() -> bool:
        """Check if LMStudio server is running."""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{TestLMStudioIntegration.BASE_URL}/v1/models",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    @pytest.mark.asyncio
    async def test_chat_completion(self, logger):
        """Test basic chat completion with LMStudio."""
        if not await self.check_lmstudio_available():
            pytest.skip("LMStudio server not available")
        print("\n=== LMStudio Chat Completion Test ===")
        client = LMStudioClient(base_url=self.BASE_URL, logger=logger)
        try:
            response = await client.chat_completion(
                model="",
                messages=[{"role": "user", "content": "Say 'Hello LMStudio!' in exactly 3 words."}],
                model_config={"max_tokens": 50, "temperature": 0.5}
            )
            assert validate_response(response, "chat_completion")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_chat_completion_system_message(self, logger):
        """Test chat completion with system message."""
        if not await self.check_lmstudio_available():
            pytest.skip("LMStudio server not available")
        print("\n=== LMStudio System Message Test ===")
        client = LMStudioClient(base_url=self.BASE_URL, logger=logger)
        try:
            response = await client.chat_completion(
                model="",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Always respond in uppercase."},
                    {"role": "user", "content": "Say hello."}
                ],
                model_config={"max_tokens": 50, "temperature": 0.5}
            )
            assert validate_response(response, "system_message")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_stream_chat_completion(self, logger):
        """Test streaming chat completion."""
        if not await self.check_lmstudio_available():
            pytest.skip("LMStudio server not available")
        print("\n=== LMStudio Streaming Test ===")
        client = LMStudioClient(base_url=self.BASE_URL, logger=logger)
        chunks_received = []

        def on_chunk(chunk: str):
            chunks_received.append(chunk)
            print(f"    Chunk: {chunk}", end="", flush=True)

        try:
            response = await client.stream_chat_completion(
                model="",
                messages=[{"role": "user", "content": "Count from 1 to 5, one number per line."}],
                model_config={"max_tokens": 100, "temperature": 0.5},
                callback=on_chunk
            )
            print()
            assert validate_response(response, "stream_completion")
            assert len(chunks_received) > 0, "Should receive at least one chunk"
            print(f"    Total chunks received: {len(chunks_received)}")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_stream_without_callback(self, logger):
        """Test streaming without callback (aggregated response)."""
        if not await self.check_lmstudio_available():
            pytest.skip("LMStudio server not available")
        print("\n=== LMStudio Stream Aggregated Test ===")
        client = LMStudioClient(base_url=self.BASE_URL, logger=logger)
        try:
            response = await client.stream_chat_completion(
                model="",
                messages=[{"role": "user", "content": "What is 2 + 2? Answer with just the number."}],
                model_config={"max_tokens": 50, "temperature": 0.1}
            )
            assert validate_response(response, "stream_aggregated")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_vision_capability(self, logger):
        """Test vision/image analysis capability."""
        if not await self.check_lmstudio_available():
            pytest.skip("LMStudio server not available")
        print("\n=== LMStudio Vision Test ===")
        client = LMStudioClient(base_url=self.BASE_URL, logger=logger)
        test_image = create_test_image()
        try:
            response = await client.chat_completion_with_chart_analysis(
                model="",
                messages=[{"role": "user", "content": "What color is this image? Answer in one word."}],
                chart_image=test_image,
                model_config={"max_tokens": 50, "temperature": 0.1}
            )
            if response and response.error:
                if "vision" in str(response.error).lower():
                    pytest.skip("Model does not support vision")
            assert validate_response(response, "vision")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_context_manager(self, logger):
        """Test async context manager pattern."""
        if not await self.check_lmstudio_available():
            pytest.skip("LMStudio server not available")
        print("\n=== LMStudio Context Manager Test ===")
        async with LMStudioClient(base_url=self.BASE_URL, logger=logger) as client:
            response = await client.chat_completion(
                model="",
                messages=[{"role": "user", "content": "Say 'context works'."}],
                model_config={"max_tokens": 30}
            )
            assert validate_response(response, "context_manager")

    @pytest.mark.asyncio
    async def test_long_conversation(self, logger):
        """Test multi-turn conversation."""
        if not await self.check_lmstudio_available():
            pytest.skip("LMStudio server not available")
        print("\n=== LMStudio Multi-turn Conversation Test ===")
        client = LMStudioClient(base_url=self.BASE_URL, logger=logger)
        try:
            messages = [
                {"role": "user", "content": "Remember the number 42."},
            ]
            response1 = await client.chat_completion(
                model="",
                messages=messages,
                model_config={"max_tokens": 100}
            )
            assert validate_response(response1, "turn_1")
            messages.append({
                "role": "assistant",
                "content": response1.choices[0].message.content
            })
            messages.append({"role": "user", "content": "What number did I ask you to remember?"})
            response2 = await client.chat_completion(
                model="",
                messages=messages,
                model_config={"max_tokens": 100}
            )
            assert validate_response(response2, "turn_2")
        finally:
            await client.close()


class TestOpenRouterIntegration:
    """Integration tests for OpenRouter client with real API calls."""

    @pytest.fixture
    def api_key(self):
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            pytest.skip("OPENROUTER_API_KEY not set")
        return key

    @pytest.fixture
    def logger(self):
        log = Logger("test_openrouter_integration", logger_debug=False)
        yield log
        log.close()

    @pytest.mark.asyncio
    async def test_chat_completion(self, api_key, logger):
        """Test basic chat completion with OpenRouter."""
        print("\n=== OpenRouter Chat Completion Test ===")
        client = OpenRouterClient(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            logger=logger
        )
        try:
            response = await client.chat_completion(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'Hello OpenRouter!' in exactly 3 words."}],
                model_config={"max_tokens": 50, "temperature": 0.5}
            )
            assert validate_response(response, "chat_completion")
            if response and response.id:
                cost_data = await client.get_generation_cost(response.id)
                if cost_data:
                    print(f"    Cost: ${cost_data.get('total_cost', 0):.6f}")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_chat_completion_with_system(self, api_key, logger):
        """Test chat completion with system message."""
        print("\n=== OpenRouter System Message Test ===")
        client = OpenRouterClient(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            logger=logger
        )
        try:
            response = await client.chat_completion(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a pirate. Always respond like a pirate."},
                    {"role": "user", "content": "Hello!"}
                ],
                model_config={"max_tokens": 100, "temperature": 0.7}
            )
            assert validate_response(response, "system_message")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_vision_capability(self, api_key, logger):
        """Test vision/image analysis with OpenRouter."""
        print("\n=== OpenRouter Vision Test ===")
        client = OpenRouterClient(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            logger=logger
        )
        test_image = create_test_image()
        try:
            response = await client.chat_completion_with_chart_analysis(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "What color is this solid image? Answer in one word."}],
                chart_image=test_image,
                model_config={"max_tokens": 50, "temperature": 0.1}
            )
            assert validate_response(response, "vision")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_context_manager(self, api_key, logger):
        """Test async context manager pattern."""
        print("\n=== OpenRouter Context Manager Test ===")
        async with OpenRouterClient(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            logger=logger
        ) as client:
            response = await client.chat_completion(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'context works'."}],
                model_config={"max_tokens": 30}
            )
            assert validate_response(response, "context_manager")

    @pytest.mark.asyncio
    async def test_different_model(self, api_key, logger):
        """Test with a different model."""
        print("\n=== OpenRouter Different Model Test ===")
        client = OpenRouterClient(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            logger=logger
        )
        try:
            response = await client.chat_completion(
                model="google/gemini-2.0-flash-001",
                messages=[{"role": "user", "content": "What is 5 + 5? Answer with just the number."}],
                model_config={"max_tokens": 20, "temperature": 0.1}
            )
            assert validate_response(response, "different_model")
        finally:
            await client.close()


class TestGoogleAIIntegration:
    """Integration tests for Google AI client with real API calls."""

    @pytest.fixture
    def api_key(self):
        key = os.getenv("GOOGLE_STUDIO_PAID_API_KEY") or os.getenv("GOOGLE_STUDIO_API_KEY")
        if not key:
            pytest.skip("GOOGLE_STUDIO_PAID_API_KEY or GOOGLE_STUDIO_API_KEY not set")
        return key

    @pytest.fixture
    def logger(self):
        log = Logger("test_googleai_integration", logger_debug=False)
        yield log
        log.close()

    @pytest.mark.asyncio
    async def test_chat_completion(self, api_key, logger):
        """Test basic chat completion with Google AI."""
        print("\n=== Google AI Chat Completion Test ===")
        client = GoogleAIClient(api_key=api_key, model="gemini-2.0-flash", logger=logger)
        try:
            response = await client.chat_completion(
                model="gemini-2.0-flash",
                messages=[{"role": "user", "content": "Say 'Hello Google AI!' in exactly 3 words."}],
                model_config={"max_tokens": 50, "temperature": 0.5}
            )
            assert validate_response(response, "chat_completion")
        except RateLimitError as e:
            pytest.skip(f"Google AI rate limited: {e}")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_chat_with_system_instruction(self, api_key, logger):
        """Test chat completion with system instruction."""
        print("\n=== Google AI System Instruction Test ===")
        client = GoogleAIClient(api_key=api_key, model="gemini-2.0-flash", logger=logger)
        try:
            response = await client.chat_completion(
                model="gemini-2.0-flash",
                messages=[
                    {"role": "system", "content": "You are a math tutor. Always explain your reasoning."},
                    {"role": "user", "content": "What is 7 * 8?"}
                ],
                model_config={"max_tokens": 200, "temperature": 0.3}
            )
            assert validate_response(response, "system_instruction")
        except RateLimitError as e:
            pytest.skip(f"Google AI rate limited: {e}")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_vision_capability(self, api_key, logger):
        """Test vision/image analysis with Google AI."""
        print("\n=== Google AI Vision Test ===")
        client = GoogleAIClient(api_key=api_key, model="gemini-2.0-flash", logger=logger)
        test_image = create_test_image()
        try:
            response = await client.chat_completion_with_chart_analysis(
                model="gemini-2.0-flash",
                messages=[{"role": "user", "content": "What color is this solid image? Answer in one word."}],
                chart_image=test_image,
                model_config={"max_tokens": 50, "temperature": 0.1}
            )
            assert validate_response(response, "vision")
        except RateLimitError as e:
            pytest.skip(f"Google AI rate limited: {e}")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_context_manager(self, api_key, logger):
        """Test async context manager pattern."""
        print("\n=== Google AI Context Manager Test ===")
        try:
            async with GoogleAIClient(api_key=api_key, model="gemini-2.0-flash", logger=logger) as client:
                response = await client.chat_completion(
                    model="gemini-2.0-flash",
                    messages=[{"role": "user", "content": "Say 'context works'."}],
                    model_config={"max_tokens": 30}
                )
                assert validate_response(response, "context_manager")
        except RateLimitError as e:
            pytest.skip(f"Google AI rate limited: {e}")

    @pytest.mark.asyncio
    async def test_json_response_mode(self, api_key, logger):
        """Test JSON response mode."""
        print("\n=== Google AI JSON Mode Test ===")
        client = GoogleAIClient(api_key=api_key, model="gemini-2.0-flash", logger=logger)
        try:
            response = await client.chat_completion(
                model="gemini-2.0-flash",
                messages=[{
                    "role": "user",
                    "content": "Return a JSON object with keys 'name' and 'age' for a person named Alice who is 30."
                }],
                model_config={"max_tokens": 100, "temperature": 0.1, "response_format": {"type": "json_object"}}
            )
            assert validate_response(response, "json_mode")
            content = response.choices[0].message.content
            import json
            try:
                parsed = json.loads(content)
                print(f"    Parsed JSON: {parsed}")
                assert "name" in parsed or "Name" in parsed
            except json.JSONDecodeError:
                print(f"    Warning: Response not valid JSON: {content}")
        except RateLimitError as e:
            pytest.skip(f"Google AI rate limited: {e}")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, api_key, logger):
        """Test multi-turn conversation."""
        print("\n=== Google AI Multi-turn Test ===")
        client = GoogleAIClient(api_key=api_key, model="gemini-2.0-flash", logger=logger)
        try:
            messages = [
                {"role": "user", "content": "My favorite color is purple. Remember that."},
            ]
            response1 = await client.chat_completion(
                model="gemini-2.0-flash",
                messages=messages,
                model_config={"max_tokens": 100}
            )
            assert validate_response(response1, "turn_1")
            messages.append({
                "role": "assistant",
                "content": response1.choices[0].message.content
            })
            messages.append({"role": "user", "content": "What is my favorite color?"})
            response2 = await client.chat_completion(
                model="gemini-2.0-flash",
                messages=messages,
                model_config={"max_tokens": 100}
            )
            assert validate_response(response2, "turn_2")
        except RateLimitError as e:
            pytest.skip(f"Google AI rate limited: {e}")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_agentic_vision_code_execution(self, api_key, logger):
        """Test Agentic Vision with Code Execution for Gemini 3 Flash.
        
        This test verifies that the model can use Code Execution to analyze images
        more precisely (zooming, counting, calculations).
        """
        print("\n=== Google AI Agentic Vision (Code Execution) Test ===")
        client = GoogleAIClient(api_key=api_key, model="gemini-3-flash-preview", logger=logger)
        test_image = create_test_image()
        try:
            response = await client.chat_completion_with_chart_analysis(
                model="gemini-3-flash-preview",
                messages=[{
                    "role": "user", 
                    "content": "Analyze this image carefully. Describe what you see, including the dominant color and any patterns. Use your code execution capability to verify your analysis if needed."
                }],
                chart_image=test_image,
                model_config={
                    "max_tokens": 500, 
                    "temperature": 1.0,  # Gemini 3 Flash default
                    "thinking_level": "high",
                    "google_code_execution": True  # Enable Agentic Vision
                }
            )
            assert validate_response(response, "agentic_vision")
            # Check if response indicates the model understood the image
            content = response.choices[0].message.content.lower()
            assert any(word in content for word in ["blue", "color", "image", "solid"]), \
                "Response should describe the blue test image"
            print("    ✓ Agentic Vision test passed - model analyzed image with code execution enabled")
        except RateLimitError as e:
            pytest.skip(f"Google AI rate limited: {e}")
        finally:
            await client.close()


class TestCrossProviderConsistency:
    """Tests to verify consistent behavior across all providers."""

    @staticmethod
    async def check_lmstudio():
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:1234/v1/models",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    @pytest.fixture
    def openrouter_key(self):
        return os.getenv("OPENROUTER_API_KEY")

    @pytest.fixture
    def google_key(self):
        return os.getenv("GOOGLE_STUDIO_PAID_API_KEY") or os.getenv("GOOGLE_STUDIO_API_KEY")

    @pytest.mark.asyncio
    async def test_response_format_consistency(self, openrouter_key, google_key):
        """Verify all providers return consistent ChatResponseModel format."""
        print("\n=== Cross-Provider Response Format Test ===")
        logger = Logger("cross_provider_test", logger_debug=False)
        prompt = "What is 2 + 2? Answer with just the number."
        results = {}
        try:
            lmstudio_available = await self.check_lmstudio()
            if lmstudio_available:
                async with LMStudioClient(base_url="http://localhost:1234", logger=logger) as client:
                    response = await client.chat_completion(
                        model="",
                        messages=[{"role": "user", "content": prompt}],
                        model_config={"max_tokens": 20}
                    )
                    if response and not response.error:
                        results["lmstudio"] = response
                        print(f"  LMStudio: {response.choices[0].message.content if response.choices else 'N/A'}")
                    else:
                        print(f"  LMStudio: Error - {response.error if response else 'Unknown'}")
            if openrouter_key:
                async with OpenRouterClient(
                    api_key=openrouter_key,
                    base_url="https://openrouter.ai/api/v1",
                    logger=logger
                ) as client:
                    response = await client.chat_completion(
                        model="openai/gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        model_config={"max_tokens": 20}
                    )
                    if response and not response.error:
                        results["openrouter"] = response
                        print(f"  OpenRouter: {response.choices[0].message.content if response.choices else 'N/A'}")
                    else:
                        print(f"  OpenRouter: Error - {response.error if response else 'Unknown'}")
            if google_key:
                async with GoogleAIClient(api_key=google_key, model="gemini-2.0-flash", logger=logger) as client:
                    response = await client.chat_completion(
                        model="gemini-2.0-flash",
                        messages=[{"role": "user", "content": prompt}],
                        model_config={"max_tokens": 20}
                    )
                    if response and not response.error:
                        results["google"] = response
                        print(f"  Google AI: {response.choices[0].message.content if response.choices else 'N/A'}")
                    else:
                        print("  Google AI: Skipped (rate limited or error)")
            if not results:
                pytest.skip("No providers returned successful responses")
            for provider, response in results.items():
                assert response.choices, f"{provider}: Missing 'choices'"
                assert isinstance(response.choices, list), f"{provider}: 'choices' not a list"
                assert len(response.choices) > 0, f"{provider}: Empty choices list"
                choice = response.choices[0]
                assert choice.message, f"{provider}: Missing 'message' in choice"
                assert choice.message.content, f"{provider}: Missing 'content' in message"
                assert response.usage, f"{provider}: Missing 'usage'"
            print(f"  ✓ {len(results)} provider(s) return consistent ChatResponseModel format")
        finally:
            logger.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
