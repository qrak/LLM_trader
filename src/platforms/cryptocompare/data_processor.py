import json
from typing import Dict, List, Any, Optional

from src.logger.logger import Logger


class CryptoCompareDataProcessor:
    """
    Handles data processing and normalization utilities for CryptoCompare API responses
    """
    
    def __init__(self, logger: Logger) -> None:
        self.logger = logger
    
    def normalize_categories_data(self, api_categories: Any) -> Optional[List]:
        """Normalize various API category data formats to a consistent list structure"""
        # Handle string data - possible serialized JSON
        if isinstance(api_categories, str):
            try:
                api_categories = json.loads(api_categories)
                self.logger.debug("Converted string to JSON object")
            except json.JSONDecodeError:
                self.logger.warning("Received string data that is not valid JSON")
                return None
        
        # Handle dictionary data with nested structures
        if isinstance(api_categories, dict):
            return self._extract_data_from_dict(api_categories)
        
        # Handle list data directly
        if isinstance(api_categories, list):
            self.logger.debug(f"Processing list with {len(api_categories)} items")
            return api_categories
        
        self.logger.warning(f"Unexpected data type for api_categories: {type(api_categories)}")
        return None
    
    def _extract_data_from_dict(self, data_dict: Dict) -> Optional[List]:
        """Extract category data from dictionary structures"""
        self.logger.debug(f"Processing dictionary with keys: {list(data_dict.keys())}")
        
        # Check CoinDesk format with Response, Message, Type, Data structure
        if "Response" in data_dict and "Data" in data_dict:
            if data_dict["Response"] in ("Success", "success"):
                self.logger.debug(f"Using data from 'Data' key: {type(data_dict['Data'])}")
                return data_dict["Data"]
            else:
                self.logger.warning(f"API response not successful: {data_dict.get('Message', 'Unknown error')}")
                return None
        
        # Simple Data key structure
        elif "Data" in data_dict:
            self.logger.debug(f"Using data from 'Data' key: {type(data_dict['Data'])}")
            return data_dict["Data"]
        
        return None
    
    @staticmethod
    def get_important_categories() -> List[str]:
        """Get list of important categories to prioritize in API requests"""
        return ["BTC", "ETH", "DeFi", "NFT", "Layer 2", "Stablecoin", "Altcoin"]
