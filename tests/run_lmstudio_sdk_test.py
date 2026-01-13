"""Test script for LM Studio SDK integration."""
import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger.logger import Logger
from src.platforms.ai_providers import LMStudioClient


async def test_lmstudio_sdk():
    """Test LM Studio SDK chat completion."""
    # Default LM Studio server URL
    base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")
    logger = Logger("test_lmstudio")
    # LMStudioClient wrapper still accepts base_url and maps it to api_host internally
    client = LMStudioClient(base_url=base_url, logger=logger)
    print("\n=== Testing LM Studio SDK Chat Completion ===")
    print(f"Connecting to: {base_url}")
    try:
        response = await client.chat_completion(
            model="",  # LM Studio auto-selects loaded model when empty
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
                print(f"  Usage: {usage.get('prompt_tokens', 0)} in / {usage.get('completion_tokens', 0)} out")
            return True
        else:
            print("✗ No response received")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        print("  (Make sure LM Studio is running with a model loaded)")
        return False
    finally:
        await client.close()


async def test_lmstudio_streaming():
    """Test LM Studio SDK streaming."""
    base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")
    logger = Logger("test_lmstudio_stream")
    client = LMStudioClient(base_url=base_url, logger=logger)
    print("\n=== Testing LM Studio SDK Streaming ===")
    try:
        chunks_received = 0
        async def count_chunks(chunk):
            nonlocal chunks_received
            chunks_received += 1
            print(chunk, end='', flush=True)
        response = await client.stream_chat_completion(
            model="",
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            model_config={"max_tokens": 100, "temperature": 0.5},
            callback=count_chunks
        )
        print()  # newline after streaming
        if response and chunks_received > 0:
            print(f"✓ Streaming completed with {chunks_received} chunks")
            return True
        else:
            print("✗ Streaming failed or no chunks received")
            return False
    except Exception as e:
        print(f"✗ Streaming error: {e}")
        return False
    finally:
        await client.close()


async def test_lmstudio_vision():
    """Test LM Studio SDK vision capabilities."""
    base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")
    logger = Logger("test_lmstudio_vision")
    client = LMStudioClient(base_url=base_url, logger=logger)
    
    print("\n=== Testing LM Studio SDK Vision ===")
    
    # Create a simple red 100x100 dummy image for testing
    import io
    from PIL import Image
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    try:
        response = await client.chat_completion_with_chart_analysis(
            model="", # Auto-detect
            messages=[{"role": "user", "content": "What color is this image? Reply in 1 word."}],
            chart_image=img_byte_arr,
            model_config={"max_tokens": 50, "temperature": 0.1}
        )
        
        if response:
            content = response['choices'][0].get('message', {}).get('content', '')
            print(f"✓ Response received")
            print(f"  Content: {content}")
            return True
        else:
            print("✗ No response received")
            return False
    except Exception as e:
        print(f"✗ Vision Error: {e}")
        return False
    finally:
        await client.close()


if __name__ == "__main__":
    import time
    
    print("\n--- Starting Sequential Tests ---")
    t1 = asyncio.run(test_lmstudio_sdk())
    time.sleep(2) # Wait for server to recover/clean up
    
    t2 = asyncio.run(test_lmstudio_streaming())
    time.sleep(2) # Wait for server to recover/clean up
    
    t3 = asyncio.run(test_lmstudio_vision())
    
    success = t1 and t2 and t3
    print(f"\nOverall Test Result: {'SUCCESS' if success else 'FAILURE'}")
    sys.exit(0 if success else 1)
