import pytest
import asyncio
import json
import sys
from unittest.mock import MagicMock, AsyncMock, patch

# Mock src.config.loader before importing anything that uses it
sys.modules['src.config.loader'] = MagicMock()

from src.platforms.cryptocompare.categories_api import CryptoCompareCategoriesAPI  # noqa: E402

@pytest.mark.asyncio
class TestCryptoCompareCategoriesAPIAsync:

    @pytest.fixture
    def api(self, tmp_path):
        logger = MagicMock()
        config = MagicMock()
        data_processor = MagicMock()
        collision_resolver = MagicMock()

        # Configure config
        config.RAG_CATEGORIES_API_URL = "http://mock-api.com"
        config.CRYPTOCOMPARE_API_KEY = "dummy_key"
        config.RAG_CATEGORIES_UPDATE_INTERVAL_HOURS = 24

        # Use a temporary directory for data_dir
        api = CryptoCompareCategoriesAPI(
            logger=logger,
            config=config,
            data_processor=data_processor,
            collision_resolver=collision_resolver,
            data_dir=str(tmp_path),
            categories_update_interval_hours=24
        )
        return api

    async def test_load_cached_categories_reads_file(self, api, tmp_path):
        """Test that _load_cached_categories correctly reads a file"""
        # Create a dummy categories file
        categories_file = tmp_path / "categories.json"
        data = {"categories": [{"categoryName": "BTC", "wordsAssociatedWithCategory": ["Bitcoin"]}]}
        with open(categories_file, "w") as f:
            json.dump(data, f)

        # Call the async method
        await api._load_cached_categories()

        # Verify data loaded
        assert api.api_categories == data["categories"]
        api.data_processor.normalize_categories_data.assert_called()

    async def test_get_categories_writes_file_async(self, api, tmp_path):
        """Test that get_categories writes to file correctly"""
        # Mock API response data
        mock_data = [{"categoryName": "ETH", "wordsAssociatedWithCategory": ["Ethereum"]}]

        # Response object
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = mock_data

        # Response context manager (returned by session.get())
        mock_get_cm = AsyncMock()
        mock_get_cm.__aenter__.return_value = mock_resp
        mock_get_cm.__aexit__.return_value = None

        # Session object (yielded by ClientSession CM)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_get_cm

        # ClientSession context manager (returned by ClientSession())
        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session
        mock_session_cm.__aexit__.return_value = None

        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            # Call get_categories with force_refresh=True
            result = await api.get_categories(force_refresh=True)

            assert result == mock_data

            # Verify file was written
            categories_file = tmp_path / "categories.json"
            assert categories_file.exists()

            with open(categories_file, "r") as f:
                saved_data = json.load(f)
                assert saved_data["categories"] == mock_data

    async def test_load_cached_categories_non_blocking_check(self, api, tmp_path):
        """
        Verify that _load_cached_categories allows other tasks to run.
        """
        # Mock _read_cache_file_sync to take some time
        original_read = api._read_cache_file_sync

        def slow_read():
            import time
            time.sleep(0.2)
            return original_read()

        api._read_cache_file_sync = slow_read

        # Create a dummy file so it has something to read
        categories_file = tmp_path / "categories.json"
        with open(categories_file, "w") as f:
            json.dump({"categories": []}, f)

        # Background task that records time
        task_end_time = 0

        async def background_task():
            nonlocal task_end_time
            await asyncio.sleep(0.05)
            task_end_time = asyncio.get_running_loop().time()
            return "ran"

        bg_task = asyncio.create_task(background_task())
        load_task = asyncio.create_task(api._load_cached_categories())

        start_time = asyncio.get_running_loop().time()
        await asyncio.gather(bg_task, load_task)

        # Background task should have finished well before the slow read (0.2s) completed
        # If blocking, bg_task would finish after 0.2s
        assert task_end_time - start_time < 0.15
        assert bg_task.result() == "ran"
