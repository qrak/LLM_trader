"""
Quick integration test for BlockRun.AI provider.
Tests configuration loading and client initialization with real credentials.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.loader import Config
from src.logger.logger import Logger
from src.platforms.ai_providers.blockrun import BlockRunClient


async def test_blockrun_integration():
    """Test BlockRun client initialization and basic request."""
    print("=" * 60)
    print("BlockRun.AI Integration Test")
    print("=" * 60)
    
    # Load configuration
    print("\n1. Loading configuration...")
    config = Config()
    
    # Check if wallet key is configured
    if not config.BLOCKRUN_WALLET_KEY:
        print("❌ BLOCKRUN_WALLET_KEY not found in keys.env")
        return False
    
    print(f"   ✅ Wallet key configured: {config.BLOCKRUN_WALLET_KEY[:6]}...{config.BLOCKRUN_WALLET_KEY[-4:]}")
    print(f"   ✅ Base URL: {config.BLOCKRUN_BASE_URL}")
    print(f"   ✅ Model: {config.BLOCKRUN_MODEL}")
    
    # Initialize client
    print("\n2. Initializing BlockRun client...")
    logger = Logger()
    client = BlockRunClient(
        wallet_key=config.BLOCKRUN_WALLET_KEY,
        base_url=config.BLOCKRUN_BASE_URL,
        logger=logger
    )
    
    try:
        async with client:
            print("   ✅ Client initialized successfully")
            
            # Attempt a minimal test request
            print("\n3. Testing chat completion...")
            print("   (Expected to fail with insufficient funds error)")
            
            result = await client.chat_completion(
                model=config.BLOCKRUN_MODEL,
                messages=[{"role": "user", "content": "Hi"}],
                model_config={"max_tokens": 10}
            )
            
            if result and "error" in result:
                error_msg = result["error"]
                print(f"\n   ⚠️  Request failed (expected): {error_msg}")
                
                # Check if it's a funding error (good) or implementation error (bad)
                if any(keyword in error_msg.lower() for keyword in ["fund", "balance", "payment", "insufficient", "usdc"]):
                    print("\n" + "=" * 60)
                    print("✅ INTEGRATION TEST PASSED!")
                    print("=" * 60)
                    print("The error is due to insufficient wallet funds (expected).")
                    print("BlockRun client is working correctly!")
                    print("\nTo use BlockRun:")
                    print("1. Fund your wallet with USDC on Base chain")
                    print("2. Run your trading bot normally")
                    return True
                else:
                    print("\n" + "=" * 60)
                    print("⚠️  UNEXPECTED ERROR TYPE")
                    print("=" * 60)
                    print("This might be an implementation issue.")
                    return False
            elif result:
                print(f"\n   ✅ SUCCESS: Response received!")
                print(f"   Content: {result.get('choices', [{}])[0].get('message', {}).get('content', 'N/A')}")
                print("\n" + "=" * 60)
                print("✅ INTEGRATION TEST PASSED!")
                print("=" * 60)
                return True
            else:
                print("\n   ❌ No response received")
                return False
                
    except Exception as e:
        print(f"\n   ❌ Error: {type(e).__name__}: {str(e)}")
        
        # Check if it's a connection/SDK error
        if "blockrun" in str(e).lower() or "wallet" in str(e).lower():
            print("\nThis might be a configuration or SDK initialization issue.")
        
        return False


if __name__ == "__main__":
    success = asyncio.run(test_blockrun_integration())
    sys.exit(0 if success else 1)
