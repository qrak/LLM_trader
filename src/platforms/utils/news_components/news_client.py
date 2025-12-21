"""
CryptoCompare News API Client
Handles direct API interactions with the CryptoCompare news service.
"""
import asyncio
from typing import Dict, List, Any, Optional

import aiohttp

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.contracts.config import ConfigProtocol
from src.logger.logger import Logger
from src.utils.decorators import retry_api_call


class CryptoCompareNewsClient:
    """Handles direct API communication with CryptoCompare news service."""
    
    def __init__(self, logger: Logger, config: "ConfigProtocol"):
        self.logger = logger
        self.config = config
    
    @retry_api_call(max_retries=3)
    async def fetch_news(
        self,
        session: Optional[aiohttp.ClientSession] = None,
        api_categories: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch crypto news from CryptoCompare API"""
        articles = []
        
        # Add optional query parameters based on categories
        categories_param = ""
        if api_categories:
            important_cats = [cat['categoryName'] for cat in api_categories 
                              if cat.get('categoryName', '') in self._get_important_categories()]
            if important_cats:
                categories_param = f"&categories={','.join(important_cats[:5])}"
                
        url = f"{self.config.RAG_NEWS_API_URL}{categories_param}"
        
        # Use provided session if available, otherwise create temporary one
        session_to_use = session or aiohttp.ClientSession()
        use_temp_session = session is None
        
        try:
            async with session_to_use.get(url, timeout=45) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data and "Data" in data:
                        articles = data["Data"]
                        self.logger.debug(f"Fetched {len(articles)} news articles from CryptoCompare")
                else:
                    self.logger.error(f"News API request failed with status {resp.status}")
        except asyncio.TimeoutError:
            self.logger.error("Timeout fetching news from CryptoCompare")
        except Exception as e:
            self.logger.error(f"Error fetching CryptoCompare news: {e}")
        finally:
            # Only close if we created a temporary session
            if use_temp_session:
                await session_to_use.close()
        
        return articles
    
    @staticmethod
    def _get_important_categories() -> List[str]:
        """Get list of important categories to prioritize in API requests"""
        return ["BTC", "ETH", "DeFi", "NFT", "Layer 2", "Stablecoin", "Altcoin"]
