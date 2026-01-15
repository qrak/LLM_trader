
import asyncio
import sys
import os
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

class MockRequest:
    def __init__(self):
        self.app = MagicMock()
        self.app.state = MagicMock()
        self.app.state.config = MagicMock()
        self.app.state.config.DATA_DIR = "data"
        # Logger is no longer expected

async def test_performance_security():
    print("Testing performance.py security...")
    from src.dashboard.routers.performance import get_performance_history
    
    request = MockRequest()
    
    with patch("builtins.open", side_effect=Exception("Secret DB Info Leaked")):
        with patch("pathlib.Path.exists", return_value=True):
            response = await get_performance_history(request)
            
    print(f"Response: {response}")
    
    if "Secret DB Info Leaked" in str(response):
        print("FAIL: Exception details leaked in response")
        return False
    
    if response.get("error") != "Failed to load trade history":
        print(f"FAIL: Unexpected error message: {response.get('error')}")
        return False
        
    print("PASS: Performance history endpoint secure")
    return True

async def test_monitor_security():
    print("\nTesting monitor.py security...")
    from src.dashboard.routers.monitor import get_last_response
    
    request = MockRequest()
    request.app.state.analysis_engine = MagicMock()
    # Ensure memory check fails to hit disk path
    setattr(request.app.state.analysis_engine, "last_llm_response", None)
    
    with patch("builtins.open", side_effect=Exception("Internal Path Revealed")):
        with patch("pathlib.Path.exists", return_value=True):
            response = await get_last_response(request)

    print(f"Response: {response}")

    if "Internal Path Revealed" in str(response):
        print("FAIL: Exception details leaked in response")
        return False
        
    if response.get("response") != "Error reading previous response":
        print(f"FAIL: Unexpected error message: {response.get('response')}")
        return False

    print("PASS: Monitor last_response endpoint secure")
    return True

async def test_brain_security():
    print("\nTesting brain.py security...")
    from src.dashboard.routers.brain import get_vector_details
    
    request = MockRequest()
    
    # Use a simple class to raise the exception reliably on property access
    class BrokenVectorMemory:
        @property
        def trade_count(self):
            raise Exception("SQL Injection Info")
        
    request.app.state.vector_memory = BrokenVectorMemory()

    response = await get_vector_details(request)
    print(f"Response: {response}")

    if "SQL Injection Info" in str(response):
        print("FAIL: Exception details leaked in response")
        return False

    if response.get("error") != "Internal error retrieving vector details":
        print(f"FAIL: Unexpected error message: {response.get('error')}")
        return False

    print("PASS: Brain vector details endpoint secure")
    return True

async def main():
    success = True
    success &= await test_performance_security()
    success &= await test_monitor_security()
    success &= await test_brain_security()
    
    if success:
        print("\nALL SECURITY TESTS PASSED")
    else:
        print("\nSOME TESTS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
