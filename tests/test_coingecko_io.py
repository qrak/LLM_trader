import asyncio
import json
import os
import unittest
from unittest.mock import MagicMock
from datetime import datetime, timezone
import tempfile
import shutil

# Adjust path to import src
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.platforms.coingecko import CoinGeckoAPI

class TestCoinGeckoIO(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.logger = MagicMock()
        self.test_dir = tempfile.mkdtemp()
        self.api = CoinGeckoAPI(self.logger, cache_dir=self.test_dir)

    async def asyncTearDown(self):
        if self.api.session:
            await self.api.close()
        shutil.rmtree(self.test_dir)

    async def test_read_write_cache_file(self):
        test_data = {"test": "data", "number": 123}

        # Write
        await self.api._write_cache_file(test_data)

        # Verify file exists and content
        self.assertTrue(os.path.exists(self.api.coingecko_cache_file))
        with open(self.api.coingecko_cache_file, 'r', encoding='utf-8') as f:
            content = json.load(f)
            self.assertEqual(content, test_data)

        # Read back using async method
        read_data = await self.api._read_cache_file()
        self.assertEqual(read_data, test_data)

    async def test_read_nonexistent_file(self):
        # Remove file if exists
        if os.path.exists(self.api.coingecko_cache_file):
            os.remove(self.api.coingecko_cache_file)

        # Read should fail gracefully (log error and return None)
        result = await self.api._read_cache_file()
        self.assertIsNone(result)
        self.logger.error.assert_called()

    async def test_initialize_loads_cache(self):
        # Create a cache file manually
        cache_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"some": "data"}
        }
        with open(self.api.coingecko_cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f)

        # Mock _fetch_all_coins to avoid network call
        self.api._fetch_all_coins = MagicMock()
        async def mock_fetch(): return []
        self.api._fetch_all_coins.side_effect = mock_fetch

        await self.api.initialize()

        self.assertIsNotNone(self.api.last_update)
        # Verify timestamp matches (ignoring microsecond precision differences if any)
        self.assertEqual(self.api.last_update.timestamp(), datetime.fromisoformat(cache_data["timestamp"]).timestamp())

    async def test_concurrent_writes(self):
        # Test that concurrent writes are serialized properly (no crashes/corruption)
        write_tasks = []
        for i in range(10):
            data = {"iteration": i, "timestamp": datetime.now(timezone.utc).isoformat()}
            write_tasks.append(self.api._write_cache_file(data))

        await asyncio.gather(*write_tasks)

        # Verify file is valid JSON
        cached_data = await self.api._read_cache_file()
        self.assertIsNotNone(cached_data)
        self.assertIn("iteration", cached_data)

if __name__ == "__main__":
    unittest.main()
