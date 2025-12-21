import asyncio
import json
import os
from datetime import datetime, timedelta
from os.path import exists, getsize
from typing import Dict, List, Any, Literal, Optional

from aiohttp_client_cache import CachedSession, SQLiteBackend

from src.logger.logger import Logger
from src.utils.decorators import retry_api_call


class CoinGeckoAPI:
    GLOBAL_API_URL = "https://api.coingecko.com/api/v3/global"
    COINS_LIST_URL = "https://api.coingecko.com/api/v3/coins/list"
    COIN_DATA_URL_TEMPLATE = "https://api.coingecko.com/api/v3/coins/{coin_id}"
    COINS_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
    GLOBAL_DEFI_URL = "https://api.coingecko.com/api/v3/global/decentralized_finance_defi"
    
    def __init__(
        self,
        logger: Logger,
        cache_name: str = 'cache/coingecko_cache.db',
        cache_dir: str = 'data/market_data',
        expire_after: int = -1
    ) -> None:
        self.cache_backend = SQLiteBackend(cache_name=cache_name, expire_after=expire_after)
        self.session: Optional[CachedSession] = None
        self.symbol_to_id_map: Dict[str, List[Dict[str, str]]] = {}
        self.logger = logger
        self.cache_dir = cache_dir
        self.coingecko_cache_file = os.path.join(self.cache_dir, "coingecko_global.json")
        self.update_interval = timedelta(hours=4)  # Default update interval
        self.last_update: Optional[datetime] = None

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    async def initialize(self) -> None:
        """Initialize the API client and load cached data"""
        try:
            if self.session:
                await self.session.close()

            self.session = CachedSession(cache=self.cache_backend)
            coins = await self._fetch_all_coins()
            if coins:
                self._update_symbol_map(coins)
                self.logger.info(f"Loaded {len(self.symbol_to_id_map)} unique symbols from coingecko.")
                self._log_cache_info()
        except Exception as e:
            self.logger.error(f"Error initializing coin mappings: {e}")
            self.symbol_to_id_map = {}
        
        # Check if we have cached global data
        if os.path.exists(self.coingecko_cache_file):
            try:
                with open(self.coingecko_cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    if "timestamp" in cached_data:
                        self.last_update = datetime.fromisoformat(cached_data["timestamp"])
                        self.logger.debug(f"Loaded CoinGecko cache from {self.last_update.isoformat()}")
            except Exception as e:
                self.logger.error(f"Error loading CoinGecko cache: {e}")

    async def get_coin_image(self,
                             base_symbol: str,
                             exchange_name: str,
                             size: Literal['thumb', 'small', 'large'] = 'small') -> str:
        try:
            base_symbol = base_symbol.upper()
            coin_data_list = self.symbol_to_id_map.get(base_symbol, [])
            if not coin_data_list:
                return ''

            if not self.session:
                self.session = CachedSession(cache=self.cache_backend)

            for coin_data in coin_data_list:
                coin_info = await self._fetch_coin_data(coin_data['id'])
                if self._coin_traded_on_exchange(coin_info, exchange_name):
                    image_url = coin_info.get('image', {}).get(size, '')
                    if image_url:
                        coin_data['image'] = image_url
                        return image_url

            if coin_data_list:
                first_coin_data = coin_data_list[0]
                if first_coin_data.get('image'):
                    return first_coin_data['image']

                coin_info = await self._fetch_coin_data(first_coin_data['id'])
                image_url = coin_info.get('image', {}).get(size, '')
                if image_url:
                    first_coin_data['image'] = image_url
                    return image_url

        except Exception as e:
            self.logger.error(f"Error fetching coin image for {base_symbol} on {exchange_name}: {e}")
        return ''

    def _get_dominance_coin_ids(self, dominance_data: Optional[Dict[str, float]] = None) -> List[str]:
        """
        Map dominance symbols to CoinGecko coin IDs dynamically.
        
        Args:
            dominance_data: Optional dominance dictionary from API. If not provided,
                          uses a default mapping for top coins.
        
        Returns:
            List of CoinGecko coin IDs
        """
        # Standard mapping for common symbols
        symbol_to_id = {
            "btc": "bitcoin",
            "eth": "ethereum",
            "usdt": "tether",
            "xrp": "ripple",
            "bnb": "binancecoin",
            "sol": "solana",
            "usdc": "usd-coin",
            "steth": "staked-ether",
            "doge": "dogecoin",
            "trx": "tron",
            "ada": "cardano",
            "avax": "avalanche-2",
            "shib": "shiba-inu",
            "link": "chainlink",
            "dot": "polkadot",
            "matic": "matic-network",
            "ltc": "litecoin",
            "dai": "dai",
            "uni": "uniswap"
        }
        
        # If dominance data provided, use those symbols
        if dominance_data:
            coin_ids = []
            for symbol in dominance_data.keys():
                symbol_lower = symbol.lower()
                if symbol_lower in symbol_to_id:
                    coin_ids.append(symbol_to_id[symbol_lower])
            return coin_ids if coin_ids else list(symbol_to_id.values())[:10]
        
        # Default: return top 10 most common coins
        return list(symbol_to_id.values())[:10]

    async def get_top_coins_by_dominance(self, dominance_coins: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch market data for top coins by dominance.
        
        Args:
            dominance_coins: List of coin IDs (e.g., ['bitcoin', 'ethereum', 'tether'])
        
        Returns:
            List of coin market data objects
        """
        if not dominance_coins:
            return []
        
        if not self.session:
            self.session = CachedSession(cache=self.cache_backend)
        
        ids_str = ",".join(dominance_coins)
        params = {
            "vs_currency": "usd",
            "ids": ids_str,
            "order": "market_cap_desc",
            "per_page": 100,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "1h,24h,7d",
            "precision": "full"  # Get full precision from API, we'll format later
        }
        
        try:
            async with self.session.get(
                self.COINS_MARKETS_URL,
                params=params
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.error(f"Failed to fetch coins/markets. Status: {response.status}")
                    return []
        except Exception as e:
            self.logger.error(f"Error fetching coins/markets: {e}")
            return []

    async def get_defi_market_data(self) -> Dict[str, Any]:
        """
        Fetch global DeFi market data.
        
        Returns:
            Dictionary containing DeFi metrics
        """
        if not self.session:
            self.session = CachedSession(cache=self.cache_backend)
        
        try:
            async with self.session.get(self.GLOBAL_DEFI_URL) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.error(f"Failed to fetch global/defi. Status: {response.status}")
                    return {}
        except Exception as e:
            self.logger.error(f"Error fetching DeFi data: {e}")
            return {}

    @retry_api_call(max_retries=3)
    async def get_global_market_data(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get global market data, top coins, and DeFi metrics from CoinGecko.
        Caches everything in coingecko_global.json every 4h.
        
        Args:
            force_refresh: If True, bypass cache and fetch fresh data
        
        Returns:
            Dictionary containing processed market data
        """
        current_time = datetime.now()
        
        # Check if we should use cached data
        if not force_refresh and self.last_update and \
           current_time - self.last_update < self.update_interval:
            try:
                with open(self.coingecko_cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    if "data" in cached_data:
                        self.logger.debug(f"Using cached CoinGecko data from {self.last_update.isoformat()}")
                        return cached_data["data"]
            except Exception as e:
                self.logger.warning(f"Failed to read cached data: {e}")
        
        # Fetch fresh data from all endpoints in parallel
        self.logger.debug("Fetching fresh CoinGecko global, top coins, and DeFi data")
        if not self.session:
            self.session = CachedSession(cache=self.cache_backend)
        
        try:
            # First fetch global data to get dominance info
            global_data = await self._fetch_global()
            
            if isinstance(global_data, Exception):
                self.logger.error(f"Error fetching global data: {global_data}")
                return await self._get_cached_global_data()
            
            # Process global data to extract dominance
            processed_global = self._process_global_data(global_data)
            dominance_data = processed_global.get("dominance", {})
            
            # Get coin IDs based on current dominance
            dominance_coin_ids = self._get_dominance_coin_ids(dominance_data)
            
            # Now fetch top coins and DeFi in parallel
            top_coins, defi_data = await asyncio.gather(
                self.get_top_coins_by_dominance(dominance_coin_ids),
                self.get_defi_market_data(),
                return_exceptions=True
            )
            
            # Start with processed global data
            processed_data = processed_global
            
            # Add top coins if successful
            if top_coins and not isinstance(top_coins, Exception):
                processed_data["top_coins"] = top_coins
            elif isinstance(top_coins, Exception):
                self.logger.warning(f"Error fetching top coins: {top_coins}")
            
            # Add DeFi data if successful (with precision cleanup)
            if defi_data and not isinstance(defi_data, Exception):
                defi_dict = defi_data.get("data", {})
                # Clean up precision on string numbers
                if defi_dict:
                    for key in ["defi_market_cap", "eth_market_cap", "defi_to_eth_ratio", 
                               "trading_volume_24h", "defi_dominance"]:
                        if key in defi_dict and isinstance(defi_dict[key], str):
                            try:
                                # Convert to float and round to reasonable precision
                                defi_dict[key] = round(float(defi_dict[key]), 2)
                            except (ValueError, TypeError):
                                pass
                processed_data["defi"] = defi_dict
            elif isinstance(defi_data, Exception):
                self.logger.warning(f"Error fetching DeFi data: {defi_data}")
            
            # Save to cache
            cache_data = {
                "timestamp": current_time.isoformat(),
                "data": processed_data
            }
            with open(self.coingecko_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            self.last_update = current_time
            self.logger.debug("Updated CoinGecko global data cache with top coins and DeFi metrics")
            return processed_data
        except Exception as e:
            self.logger.error(f"Error fetching global market data: {e}")
            return await self._get_cached_global_data()
    
    async def _fetch_global(self) -> Dict[str, Any]:
        """Fetch /global endpoint."""
        try:
            async with self.session.get(self.GLOBAL_API_URL) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.error(f"Failed to fetch /global. Status: {response.status}")
                    return {}
        except Exception as e:
            self.logger.error(f"Error fetching /global: {e}")
            return {}
    
    async def _get_cached_global_data(self) -> Dict[str, Any]:
        """Retrieve cached global data as fallback"""
        try:
            if os.path.exists(self.coingecko_cache_file):
                with open(self.coingecko_cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    if "data" in cached_data:
                        self.logger.warning("Using cached CoinGecko global data as fallback")
                        return cached_data["data"]
        except Exception as e:
            self.logger.error(f"Error reading cached global data: {e}")
        
        # Return empty dict if cache read fails
        return {}
    
    def _process_global_data(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw API data into a standardized format"""
        if not api_data or "data" not in api_data:
            return {}
            
        data = api_data["data"]
        
        return {
            "market_cap": {
                "total_usd": data.get("total_market_cap", {}).get("usd", 0),
                "change_24h": data.get("market_cap_change_percentage_24h_usd", 0)
            },
            "volume": {
                "total_usd": data.get("total_volume", {}).get("usd", 0)
            },
            "dominance": data.get("market_cap_percentage", {}),
            "stats": {
                "active_coins": data.get("active_cryptocurrencies", 0),
                "active_markets": data.get("markets", 0)
            }
        }
    
    @retry_api_call(max_retries=2)
    async def get_market_cap_data(self) -> Dict[str, Any]:
        """Get market cap specific data"""
        market_data = await self.get_global_market_data()
        if not market_data:
            return {}
            
        return {
            "total_market_cap": market_data.get("market_cap", {}).get("total_usd", 0),
            "market_cap_change_24h": market_data.get("market_cap", {}).get("change_24h", 0),
            "total_volume_24h": market_data.get("volume", {}).get("total_usd", 0)
        }
    
    @retry_api_call(max_retries=2)
    async def get_coin_dominance(self, limit: int = 5) -> Dict[str, float]:
        """
        Get coin dominance percentages
        
        Args:
            limit: Number of top coins to include
            
        Returns:
            Dictionary mapping coin symbols to dominance percentages
        """
        market_data = await self.get_global_market_data()
        if not market_data or "dominance" not in market_data:
            return {}
            
        dominance = market_data["dominance"]
        # Sort by dominance percentage and take top 'limit' coins
        sorted_coins = sorted(dominance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_coins[:limit])
        
    async def _fetch_all_coins(self) -> List[Dict[str, str]]:
        if not self.session:
            self.session = CachedSession(cache=self.cache_backend)
            
        async with self.session.get(self.COINS_LIST_URL) as response:
            if response.status == 200:
                return await response.json()
            else:
                self.logger.error(f"Failed to fetch coin list. Status: {response.status}")
                return []

    @retry_api_call(max_retries=2)
    async def _fetch_coin_data(self, coin_id: str) -> Dict[str, Any]:
        if not self.session:
            self.session = CachedSession(cache=self.cache_backend)
            
        url = self.COIN_DATA_URL_TEMPLATE.format(coin_id=coin_id)
        async with self.session.get(url) as response:
            if response.status == 200:
                return await response.json()
            return {}

    def _update_symbol_map(self, coins: List[Dict[str, str]]) -> None:
        for coin in coins:
            symbol = coin['symbol'].upper()
            if symbol not in self.symbol_to_id_map:
                self.symbol_to_id_map[symbol] = []
            self.symbol_to_id_map[symbol].append({
                'id': coin['id'],
                'name': coin['name'],
                'image': ''
            })

    @staticmethod
    def _coin_traded_on_exchange(coin_data: Dict[str, Any], exchange_name: str) -> bool:
        return any(
            ticker.get('market', {}).get('name') == exchange_name
            for ticker in coin_data.get('tickers', [])
        )

    def _log_cache_info(self) -> None:
        cache_file_path = self.cache_backend.name
        if exists(cache_file_path):
            cache_size = getsize(cache_file_path)
            cache_size_mb = cache_size / (1024 * 1024)
            self.logger.info(f"Cache file size: {cache_size_mb:.2f} MB")
        else:
            self.logger.info("Cache file does not exist yet.")

    async def close(self) -> None:
        if self.session:
            try:
                await asyncio.wait_for(self.session.close(), timeout=1.0)
            except asyncio.TimeoutError:
                self.logger.error("CoinGecko session close timed out")
            except Exception as e:
                self.logger.error(f"Error closing CoinGecko session: {e}")
            finally:
                self.session = None