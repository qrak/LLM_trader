import asyncio
import sys
from pathlib import Path
from dataclasses import dataclass, field

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import DashboardState
from src.dashboard.dashboard_state import DashboardState

async def test_dashboard_state_caching():
    print("Testing DashboardState caching logic...")
    
    state = DashboardState()
    
    # Test 1: Cache Miss
    val = state.get_cached("memory")
    assert val is None, "Should be None initially"
    print("  [Pass] Initial cache miss")
    
    # Test 2: Cache Set & Hit
    data = {"foo": "bar"}
    state.set_cached("memory", data)
    val = state.get_cached("memory")
    assert val == data, "Should return cached data"
    assert val["foo"] == "bar", "Data integrity check"
    print("  [Pass] Cache set and hit")
    
    # Test 3: TTL Expiration (Mocking time)
    # We can't easily wait 30s in a test, but we can manually verify the timestamp logic
    # or override the time module if we mocked it. 
    # Since we didn't inject a time provider, we'll trust the logic:
    # if time.time() - cached_time > ttl_seconds: return None
    
    # Let's set a very short TTL and wait
    import time
    state.set_cached("costs", {"cost": 100})
    val = state.get_cached("costs", ttl_seconds=0.1)
    assert val == {"cost": 100}, "Should be valid immediately"
    
    print("  Waiting 0.2s for expiration...")
    time.sleep(0.2)
    
    val = state.get_cached("costs", ttl_seconds=0.1)
    assert val is None, "Should be expired"
    print("  [Pass] Cache expiration works")

    print("\nALL DASHBOARD STATE TESTS PASSED!")

if __name__ == "__main__":
    asyncio.run(test_dashboard_state_caching())
