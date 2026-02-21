import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional

import aiohttp

from src.logger.logger import Logger
from src.utils.decorators import retry_api_call


class AlternativeMeAPI:
    """
    API client for Alternative.me services.
    Primarily handles the Fear & Greed Index data.
    """
    # API endpoints
    FEAR_GREED_INDEX_URL = "https://api.alternative.me/fng/"
    FEAR_GREED_HISTORY_URL = "https://api.alternative.me/fng/?limit={limit}&format=json"

    def __init__(
        self,
        logger: Logger,
        data_dir: str = 'data/market_data',
        cache_update_hours: int = 12
    ) -> None:
        self.logger = logger
        self.data_dir = data_dir
        self.update_interval = timedelta(hours=cache_update_hours)
        self.fear_greed_cache_file = os.path.join(data_dir, "fear_greed_index.json")
        self.last_update: Optional[datetime] = None
        self.current_index: Optional[Dict[str, Any]] = None
        self.session: Optional[aiohttp.ClientSession] = None

        # Ensure cache directory exists
        os.makedirs(data_dir, exist_ok=True)

    async def initialize(self) -> None:
        """Initialize the API client and load cached data"""
        self.session = aiohttp.ClientSession()

        # Check if we have cached fear & greed data
        if await asyncio.to_thread(os.path.exists, self.fear_greed_cache_file):
            try:
                cached_data = await asyncio.to_thread(self._read_cache_file)
                if "timestamp" in cached_data and "data" in cached_data:
                    loaded_time = datetime.fromisoformat(cached_data["timestamp"])
                    # Ensure timezone-aware (old caches may be naive)
                    if loaded_time.tzinfo is None:
                        loaded_time = loaded_time.replace(tzinfo=timezone.utc)
                    self.last_update = loaded_time
                    self.current_index = cached_data["data"]
                    self.logger.debug(f"Loaded Fear & Greed cache from {self.last_update.isoformat()}")
            except Exception as e:
                self.logger.error(f"Error loading Fear & Greed cache: {e}")

    def _read_cache_file(self) -> Dict[str, Any]:
        """Read and parse the cache file. Executed in a thread."""
        with open(self.fear_greed_cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _write_cache_file(self, data: Dict[str, Any]) -> None:
        """Write data to cache file. Executed in a thread."""
        with open(self.fear_greed_cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @retry_api_call(max_retries=3)
    async def get_fear_greed_index(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get current Fear & Greed Index data

        Args:
            force_refresh: Force refresh from API instead of using cache

        Returns:
            Dictionary containing Fear & Greed Index data
        """
        current_time = datetime.now(timezone.utc)

        # Check if we should use cached data
        if not force_refresh and self.last_update and self.current_index and \
           current_time - self.last_update < self.update_interval:
            self.logger.debug(f"Using cached Fear & Greed data from {self.last_update.isoformat()}")
            return self.current_index

        # Fetch fresh data
        self.logger.debug("Fetching fresh Fear & Greed Index data")

        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

        try:
            async with self.session.get(self.FEAR_GREED_INDEX_URL, timeout=30) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data and "data" in data and len(data["data"]) > 0:
                        index_data = data["data"][0]

                        # Format the data consistently
                        result = {
                            "value": int(index_data.get("value", 0)),
                            "value_classification": index_data.get("value_classification", "Unknown"),
                            "timestamp": int(index_data.get("timestamp", 0)),
                            "time": datetime.fromtimestamp(int(index_data.get("timestamp", 0)), tz=timezone.utc).isoformat()
                        }

                        # Cache the result
                        cache_data = {
                            "timestamp": current_time.isoformat(),
                            "data": result
                        }

                        await asyncio.to_thread(self._write_cache_file, cache_data)

                        self.last_update = current_time
                        self.current_index = result
                        self.logger.debug(f"Updated Fear & Greed cache with value: {result['value']} - {result['value_classification']}")

                        return result
                    else:
                        self.logger.warning("Invalid Fear & Greed Index API response format")
                else:
                    self.logger.error(f"Fear & Greed API request failed with status {resp.status}")
        except Exception as e:
            self.logger.error(f"Error fetching Fear & Greed data: {e}")

        # If API call fails, try to use cached data
        if self.current_index:
            self.logger.warning("Using cached Fear & Greed data as fallback after API failure")
            return self.current_index

        # Return default data if cache is also unavailable
        return {
            "value": 0,
            "value_classification": "Unknown",
            "timestamp": 0,
            "time": current_time.isoformat()
        }

    @retry_api_call(max_retries=3)
    async def get_historical_fear_greed(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get historical Fear & Greed Index data

        Args:
            days: Number of days of historical data to retrieve

        Returns:
            List of Fear & Greed Index data points, sorted by date (newest first)
        """
        limit = min(max(days, 1), 365)  # Limit between 1 and 365
        url = self.FEAR_GREED_HISTORY_URL.format(limit=limit)

        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

        try:
            async with self.session.get(url, timeout=45) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data and "data" in data:
                        # Format the data consistently
                        history = []
                        for item in data["data"]:
                            history.append({
                                "value": int(item.get("value", 0)),
                                "value_classification": item.get("value_classification", "Unknown"),
                                "timestamp": int(item.get("timestamp", 0)),
                                "time": datetime.fromtimestamp(int(item.get("timestamp", 0)), tz=timezone.utc).isoformat()
                            })

                        # Sort by timestamp (newest first)
                        history.sort(key=lambda x: x["timestamp"], reverse=True)

                        return history
                    else:
                        self.logger.warning("Invalid Fear & Greed History API response format")
                else:
                    self.logger.error(f"Fear & Greed History API request failed with status {resp.status}")
        except Exception as e:
            self.logger.error(f"Error fetching Fear & Greed history: {e}")

        # Return empty list if API call fails
        return []

    async def close(self) -> None:
        """Close the API client session."""
        if self.session and not self.session.closed:
            await self.session.close()
