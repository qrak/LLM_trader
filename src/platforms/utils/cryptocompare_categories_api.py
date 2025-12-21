import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import aiohttp

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.contracts.config import ConfigProtocol
from src.logger.logger import Logger
from src.utils.decorators import retry_api_call
from src.utils.collision_resolver import CategoryCollisionResolver
from .cryptocompare_data_processor import CryptoCompareDataProcessor


class CryptoCompareCategoriesAPI:
    """
    Handles CryptoCompare categories API operations including fetching, caching, and processing categories
    """
    
    def __init__(
        self,
        logger: Logger,
        config: "ConfigProtocol",
        data_dir: str = 'data',
        categories_update_interval_hours: int = 24
    ) -> None:
        self.logger = logger
        self.config = config
        self.data_dir = data_dir
        self.categories_update_interval = timedelta(hours=categories_update_interval_hours)
        self.categories_last_update: Optional[datetime] = None
        self.api_categories: List[Dict[str, Any]] = []
        self.category_word_map: Dict[str, str] = {}
        self.categories_file = os.path.join(data_dir, "categories.json")
        
        # Initialize data processor and collision resolver
        self.data_processor = CryptoCompareDataProcessor(logger)
        self.collision_resolver = CategoryCollisionResolver()
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
    
    async def initialize(self) -> None:
        """Initialize the categories API and load cached data"""
        await self._load_cached_categories()
    
    async def _load_cached_categories(self) -> None:
        """Load cached categories data - supports both list and object formats"""
        if os.path.exists(self.categories_file):
            try:
                with open(self.categories_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    
                    # Handle both list and object formats
                    if isinstance(cached_data, list):
                        # Direct list format
                        self.api_categories = cached_data
                        self._process_api_categories(self.api_categories)
                        self.logger.debug(f"Loaded {len(self.api_categories)} categories from cache (list format)")
                    elif isinstance(cached_data, dict):
                        # Object format with timestamp and categories
                        if "timestamp" in cached_data:
                            self.categories_last_update = datetime.fromisoformat(cached_data["timestamp"])
                        
                        if "categories" in cached_data:
                            self.api_categories = cached_data["categories"]
                        else:
                            # Fallback: treat the dict as the categories data itself
                            self.api_categories = [cached_data]
                            
                        self._process_api_categories(self.api_categories)
                        self.logger.debug(f"Loaded {len(self.api_categories)} categories from cache (object format)")
                    else:
                        self.logger.warning(f"Unexpected cache format: {type(cached_data)}")
            except Exception as e:
                self.logger.error(f"Error loading categories cache: {e}")
    
    @retry_api_call(max_retries=3)
    async def get_categories(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get cryptocurrency categories data
        
        Args:
            force_refresh: Force refresh from API instead of using cache
            
        Returns:
            List of category objects
        """
        current_time = datetime.now()
        
        # Check if we need to refresh
        if not force_refresh and self.categories_last_update and \
           current_time - self.categories_last_update < self.categories_update_interval:
            return self.api_categories
            
        self.logger.debug(f"Fetching categories from CryptoCompare API: {self.config.RAG_CATEGORIES_API_URL}")
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.config.RAG_CATEGORIES_API_URL, timeout=30) as resp:
                    self.logger.debug(f"Categories API response status: {resp.status}")
                    if resp.status == 200:
                        data = await resp.json()
                        self.logger.debug(f"Categories API raw response: {str(data)[:500]}...")
                        if data:
                            # Save to cache with proper structure
                            cache_data = {
                                "timestamp": current_time.isoformat(),
                                "categories": data
                            }
                            
                            with open(self.categories_file, 'w', encoding='utf-8') as f:
                                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                                
                            # Update internal data
                            self.api_categories = data
                            self._process_api_categories(data)
                            self.categories_last_update = current_time
                            
                            return data
                    else:
                        self.logger.error(f"Categories API request failed with status {resp.status}")
                        self.logger.error(f"Response body: {await resp.text()}")
            except Exception as e:
                self.logger.error(f"Error fetching CryptoCompare categories: {e}")
        
        # Return cached data as fallback if API call fails
        return self.api_categories
    
    def _process_api_categories(self, api_categories: Any) -> None:
        """Process API categories and update internal data structures"""
        if not api_categories:
            return
            
        self.category_word_map = {}
        
        try:
            # Debug logging for the received categories data
            self.logger.debug(f"Processing categories data of type: {type(api_categories)}")
            
            # Normalize data format using the data processor
            normalized_data = self.data_processor.normalize_categories_data(api_categories)
            if normalized_data is None:
                return
                
            # Process the normalized data
            self._extract_category_mappings(normalized_data)
            
        except Exception as e:
            self.logger.error(f"Error processing API categories: {e}")
    
    def _extract_category_mappings(self, categories_list: List) -> None:
        """Extract word-to-category mappings from the categories list"""
        # Process items based on expected CryptoCompare category structure
        # Example: {"categoryName": "BTC", "wordsAssociatedWithCategory": ["BTC", "Bitcoin"], "includedPhrases": ["..."]}
        for cat in categories_list:
            if isinstance(cat, dict) and 'categoryName' in cat:
                self._process_category_dict(cat)
            elif isinstance(cat, str):
                # For simple string categories, just log them
                self.logger.debug(f"Adding string category: {cat}")
            else:
                self.logger.debug(f"Skipping category with unexpected structure: {type(cat)}")
        
        self.logger.debug(f"Processed {len(self.category_word_map)} category-word associations")
    
    def _process_category_dict(self, cat: Dict) -> None:
        """Process a single category dictionary to extract word mappings"""
        category_name = cat.get('categoryName', '')
        if not category_name:
            return
        
        # Process wordsAssociatedWithCategory
        words = cat.get('wordsAssociatedWithCategory', [])
        if isinstance(words, list):
            self._add_words_to_mapping(words, category_name)
        
        # Process includedPhrases
        phrases = cat.get('includedPhrases', [])
        if isinstance(phrases, list):
            self._add_words_to_mapping(phrases, category_name)
    
    def _add_words_to_mapping(self, words: List, category_name: str) -> None:
        """Add a list of words/phrases to the category mapping with priority-based collision resolution"""
        collision_count = 0
        excluded_count = 0
        
        for word in words:
            if not isinstance(word, str):
                continue
                
            word_stripped = word.strip()
            if len(word_stripped) < 2:
                # Skip single-character tokens
                excluded_count += 1
                continue
            elif len(word_stripped) == 2:
                # Allow 2-character tokens if they are uppercase (likely tickers) or contain digits
                if not (word_stripped.isupper() or any(c.isdigit() for c in word_stripped)):
                    excluded_count += 1
                    continue
            
            word_lower = word_stripped.lower()
            if word_lower in self.category_word_map:
                # Use shared collision resolver
                existing_category = self.category_word_map[word_lower]
                winner = self.collision_resolver.resolve_collision(existing_category, category_name, word_lower)
                
                if winner != existing_category:
                    # New category wins, update mapping
                    self.category_word_map[word_lower] = winner
                # collisions intentionally not logged (user requested removal)
                collision_count += 1
            else:
                self.category_word_map[word_lower] = category_name
        
        # Collision and exclusion counts are kept for internal use but not logged per user request
    
    def get_category_word_map(self) -> Dict[str, str]:
        """Get the current category word mapping"""
        return self.category_word_map.copy()
    
    def get_api_categories(self) -> List[Dict[str, Any]]:
        """Get the current API categories list"""
        return self.api_categories.copy()
