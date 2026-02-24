"""
CryptoCompare News API Client
Handles direct API interactions with the CryptoCompare news service.
"""
import asyncio
from typing import Dict, List, Any, Optional, TYPE_CHECKING

import aiohttp

from src.logger.logger import Logger
from src.utils.decorators import retry_api_call

if TYPE_CHECKING:
    from src.config.protocol import ConfigProtocol

RELIABLE_NEWS_FEEDS = [
    'cointelegraph',
    'coindesk',
    'theblock',
    'decrypt',
    'bitcoinmagazine',
    'cryptoslate',
]


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
        url = self._build_news_url(api_categories)
        session_to_use = session or aiohttp.ClientSession()
        use_temp_session = session is None

        try:
            client_timeout = aiohttp.ClientTimeout(total=45)
            async with session_to_use.get(url, timeout=client_timeout) as resp:
                articles = await self._process_response(resp)
        except asyncio.TimeoutError:
            self.logger.error("Timeout fetching news from CryptoCompare")
            articles = []
        except Exception as e:
            self.logger.error("Error fetching CryptoCompare news: %s", e)
            articles = []
        finally:
            if use_temp_session:
                await session_to_use.close()

        return articles

    def _build_news_url(self, api_categories: Optional[List[Dict[str, Any]]] = None) -> str:
        """Build the news API URL with appropriate query parameters"""
        categories_param = self._build_categories_param(api_categories)
        feeds_param = self._build_feeds_param()
        url = f"{self.config.RAG_NEWS_API_URL}{categories_param}{feeds_param}"
        return self._append_api_key(url)

    def _build_categories_param(self, api_categories: Optional[List[Dict[str, Any]]]) -> str:
        """Build categories query parameter"""
        if not api_categories:
            return ""
        
        important_cats = [cat['categoryName'] for cat in api_categories
                          if cat.get('categoryName', '') in self._get_important_categories()]
        if important_cats:
            return f"&categories={','.join(important_cats[:5])}"
        return ""

    def _build_feeds_param(self) -> str:
        """Build feeds query parameter for source filtering"""
        if not getattr(self.config, 'RAG_NEWS_FILTER_SOURCES', True):
            return ""
        
        allowed_feeds = getattr(self.config, 'RAG_NEWS_ALLOWED_FEEDS', None)
        if allowed_feeds is None:
            allowed_feeds = RELIABLE_NEWS_FEEDS
        if allowed_feeds:
            return f"&feeds={','.join(allowed_feeds)}"
        return ""

    def _append_api_key(self, url: str) -> str:
        """Append API key to URL if available and not already present"""
        if self.config.CRYPTOCOMPARE_API_KEY and "api_key=" not in url:
            connector = "&" if "?" in url else "?"
            return f"{url}{connector}api_key={self.config.CRYPTOCOMPARE_API_KEY}"
        return url

    async def _process_response(self, resp) -> List[Dict[str, Any]]:
        """Process API response and extract articles"""
        if resp.status == 200:
            data = await resp.json()
            if data and "Data" in data:
                articles = data["Data"]
                self.logger.debug("Fetched %s news articles from CryptoCompare", len(articles))
                return articles
        else:
            self.logger.error("News API request failed with status %s", resp.status)
        return []

    @staticmethod
    def _get_important_categories() -> List[str]:
        """Get list of important categories to prioritize in API requests"""
        return ["BTC", "ETH", "DeFi", "NFT", "Layer 2", "Stablecoin", "Altcoin"]
