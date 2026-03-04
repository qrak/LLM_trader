import asyncio
import time
import json
import os
from pathlib import Path

# Mock dependencies
class MockConfig:
    DATA_DIR = "data_bench"

class MockLogger:
    def error(self, *args, **kwargs):
        pass

class MockDashboardState:
    def get_cached(self, *args, **kwargs):
        return None
    def set_cached(self, *args, **kwargs):
        pass

from src.dashboard.routers.monitor import MonitorRouter

async def measure_event_loop_block(router):
    # Create a large dummy JSON file
    os.makedirs("data_bench/news_cache", exist_ok=True)
    dummy_data = {"articles": [{"title": f"Title {i}", "content": "x" * 100} for i in range(500000)]} # lots of small items
    with open("data_bench/crypto_news.json", "w", encoding="utf-8") as f:
        json.dump(dummy_data, f)

    max_block_time = 0
    keep_measuring = True

    async def event_loop_monitor():
        nonlocal max_block_time
        while keep_measuring:
            start = time.perf_counter()
            await asyncio.sleep(0.01)
            elapsed = time.perf_counter() - start
            if elapsed - 0.01 > max_block_time:
                max_block_time = elapsed - 0.01

    monitor_task = asyncio.create_task(event_loop_monitor())

    # warmup loop monitor
    await asyncio.sleep(0.1)

    # Run the get_news function
    start_time = time.perf_counter()
    for _ in range(5):
        # We need to bypass the cache to force reading from disk
        await router.get_news()
        # Sleep to let monitor run
        await asyncio.sleep(0)
    end_time = time.perf_counter()

    keep_measuring = False
    await monitor_task

    print(f"Total time for 5 iterations: {end_time - start_time:.4f}s")
    print(f"Max event loop block time: {max_block_time:.4f}s")

    # Cleanup
    os.remove("data_bench/crypto_news.json")
    os.rmdir("data_bench/news_cache")
    os.rmdir("data_bench")

async def main():
    router = MonitorRouter(MockConfig(), MockLogger(), MockDashboardState(), None, None)
    await measure_event_loop_block(router)

if __name__ == "__main__":
    asyncio.run(main())
