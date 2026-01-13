"""Test script for OpenRouter SDK integration."""
import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from src.logger.logger import Logger
from src.platforms.ai_providers import OpenRouterClient

# Load environment variables from keys.env
load_dotenv("keys.env")


def format_cost(cost: float) -> str:
    """Format cost in human-readable format."""
    if cost == 0:
        return "Free"
    elif cost < 0.0001:
        return f"${cost:.8f} ({cost * 100:.6f}¢)"
    elif cost < 0.01:
        return f"${cost:.6f} ({cost * 100:.4f}¢)"
    elif cost < 1:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"


async def test_openrouter_sdk():
    """Test OpenRouter SDK chat completion."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in environment")
        return False
    logger = Logger("test_openrouter")
    client = OpenRouterClient(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        logger=logger
    )
    print("\n=== Testing OpenRouter SDK Chat Completion ===")
    try:
        response = await client.chat_completion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'Hello SDK!' in exactly 3 words."}],
            model_config={"max_tokens": 50, "temperature": 0.5}
        )
        if response:
            print("✓ Response received")
            if 'choices' in response:
                content = response['choices'][0].get('message', {}).get('content', '')
                print(f"  Content: {content}")
            if 'usage' in response:
                usage = response['usage']
                prompt = int(usage.get('prompt_tokens', 0))
                completion = int(usage.get('completion_tokens', 0))
                total = prompt + completion
                print(f"  Prompt token count: {prompt:,}")
                print(f"  Response token count: {completion:,}")
                print(f"  Total tokens used: {total:,}")
            if 'id' in response:
                cost_data = await client.get_generation_cost(response['id'])
                if cost_data:
                    cost_str = format_cost(cost_data.get('total_cost', 0))
                    print(f"  Request cost: {cost_str}")
            return True
        else:
            print("✗ No response received")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    finally:
        await client.close()


async def test_openrouter_with_image():
    """Test OpenRouter SDK with image input."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in environment")
        return False
    logger = Logger("test_openrouter_image")
    client = OpenRouterClient(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        logger=logger
    )
    print("\n=== Testing OpenRouter SDK with Image ===")
    # Create a simple test image (100x100 red pixel PNG)
    import io
    from PIL import Image
    img = Image.new('RGB', (100, 100), color='red')
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    try:
        response = await client.chat_completion_with_chart_analysis(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "What color is this image? Answer in one word."}],
            chart_image=img_buffer,
            model_config={"max_tokens": 50, "temperature": 0.5}
        )
        if response:
            print("✓ Response received")
            if 'choices' in response:
                content = response['choices'][0].get('message', {}).get('content', '')
                print(f"  Content: {content}")
            if 'usage' in response:
                usage = response['usage']
                prompt = int(usage.get('prompt_tokens', 0))
                completion = int(usage.get('completion_tokens', 0))
                total = prompt + completion
                print(f"  Prompt token count: {prompt:,}")
                print(f"  Response token count: {completion:,}")
                print(f"  Total tokens used: {total:,}")
            return True
        else:
            print("✗ No response received")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await client.close()


async def test_smart_retry():
    """Test smart retry logic with unsupported parameters."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in environment")
        return False
        
    logger = Logger("test_smart_retry")
    client = OpenRouterClient(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        logger=logger
    )
    
    print("\n=== Testing Smart Parameter Retry ===")
    print("Sending request with unsupported parameter 'top_k' (known to cause SDK TypeError)...")
    
    try:
        # 'top_k' is rejected by the Python SDK's send_async method signature
        # We expect the client to catch the TypeError, log a warning, remove 'top_k', and succeed.
        response = await client.chat_completion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'Retry works'"}],
            model_config={"max_tokens": 20, "top_k": 50} 
        )
        
        if response:
            content = response['choices'][0].get('message', {}).get('content', '')
            print(f"✓ Success! Response received: '{content}'")
            print("  This confirms the client automatically removed the bad parameter.")
            return True
        else:
            print("✗ Failed: No response received")
            return False
            
    except Exception as e:
        print(f"✗ Failed: Exception was raised instead of handled: {e}")
        return False
    finally:
        await client.close()


if __name__ == "__main__":
    t1 = asyncio.run(test_openrouter_sdk())
    t2 = asyncio.run(test_openrouter_with_image()) 
    t3 = asyncio.run(test_smart_retry())
    
    success = t1 and t2 and t3
    sys.exit(0 if success else 1)
