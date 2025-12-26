"""
Category fetching and API operations for cryptocurrency categories.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.logger.logger import Logger


class CategoryFetcher:
    """Handles fetching cryptocurrency categories from external APIs."""
    
    def __init__(self, logger: Logger, cryptocompare_api=None):
        self.logger = logger
        self.cryptocompare_api = cryptocompare_api
        self.categories_last_update: Optional[datetime] = None
    
    async def fetch_cryptocompare_categories(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Fetch cryptocurrency categories from CryptoCompare API."""
        if self.cryptocompare_api is None:
            self.logger.debug("CryptoCompare API client not initialized, using fallback categories")
            return []
            
        response = await self.cryptocompare_api.get_categories(force_refresh=force_refresh)
        
        # Debug logging to inspect the returned data
        self.logger.debug(f"Categories response type: {type(response)}")
        
        # Extract categories from the response structure
        categories = self._extract_categories_from_response(response)
        
        if categories:
            self.categories_last_update = datetime.now()
            
        return categories
    
    def _extract_categories_from_response(self, response) -> List[Dict[str, Any]]:
        """Extract categories from API response structure."""
        categories = []
        
        if not response:
            return categories
            
        # Handle direct list of categories
        if isinstance(response, list):
            categories = response
            self.logger.debug(f"Found {len(categories)} categories in list format")
        # Handle dictionary with nested structure
        elif isinstance(response, dict):
            categories = self._extract_from_dict_response(response)
        
        return categories
    
    def _extract_from_dict_response(self, response: dict) -> List[Dict[str, Any]]:
        """Extract categories from dictionary response structure."""
        if "Response" in response and "Data" in response and response["Response"] == "Success":
            categories = response["Data"]
            self.logger.debug(f"Found categories in Response/Data structure: {len(categories) if isinstance(categories, list) else 'dict'}")
        elif "Data" in response:
            categories = response["Data"]
            self.logger.debug(f"Found categories in Data key: {len(categories) if isinstance(categories, list) else 'dict'}")
        else:
            categories = response
            self.logger.debug("Using response directly as categories")
        
        return categories if isinstance(categories, list) else []
    
    async def ensure_categories_updated(self, force_refresh: bool = False) -> bool:
        """Ensure categories are loaded and up to date."""
        try:
            # Check if categories need updating
            if force_refresh or self.categories_last_update is None:
                categories = await self.fetch_cryptocompare_categories(force_refresh)
                return len(categories) > 0
            
            return True
            
        except Exception as e:
            self.logger.exception(f"Error ensuring categories updated: {e}")
            return False
