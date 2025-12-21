"""
Ticker management and validation operations.
"""
from typing import Set, List, Dict, Any, Optional
from src.logger.logger import Logger


class TickerManager:
    """Handles ticker tracking, validation, and management operations."""
    
    def __init__(self, logger: Logger, file_handler=None, exchange_manager=None):
        self.logger = logger
        self.file_handler = file_handler
        self.exchange_manager = exchange_manager
        self.known_tickers: Set[str] = set()
    
    async def load_known_tickers(self) -> None:
        """Load known cryptocurrency tickers from disk."""
        try:
            if self.file_handler:
                tickers_data = self.file_handler.load_known_tickers()
                if tickers_data and isinstance(tickers_data, list):
                    self.known_tickers = set(tickers_data)
                    self.logger.debug(f"Loaded {len(self.known_tickers)} known tickers")
        except Exception as e:
            self.logger.exception(f"Error loading known tickers: {e}")
            self.known_tickers = set()
    
    async def update_known_tickers(self, news_database: List[Dict[str, Any]]) -> None:
        """Update known tickers from news database and validation."""
        try:
            # Extract coins from news database
            detected_coins = self._extract_detected_coins(news_database)
            category_coins = self._extract_category_coins(news_database)
            
            # Combine all discovered coins
            all_discovered_coins = detected_coins.union(category_coins)
            
            # Filter and validate coins
            filtered_coins = {coin for coin in all_discovered_coins 
                            if self._is_potential_valid_coin(coin)}
            
            self.logger.debug(f"Found {len(filtered_coins)} potential new tickers")
            
            # Validate and add new coins
            await self._validate_and_add_coins(filtered_coins)
            
            # Save updated tickers
            await self.save_tickers()
            
        except Exception as e:
            self.logger.exception(f"Error updating known tickers: {e}")
    
    def _extract_detected_coins(self, news_database: List[Dict[str, Any]]) -> set:
        """Extract coins that were detected in news articles."""
        detected_coins = set()
        
        for article in news_database:
            # Handle both list and legacy string formats
            coins_mentioned = article.get('detected_coins', [])
            if isinstance(coins_mentioned, list):
                for coin in coins_mentioned:
                    if isinstance(coin, str) and len(coin) >= 2:
                        detected_coins.add(coin.upper())
            elif isinstance(coins_mentioned, str) and coins_mentioned:
                # Handle string format
                for coin in coins_mentioned.split('|'):
                    coin = coin.strip()
                    if coin and len(coin) >= 2:
                        detected_coins.add(coin.upper())
        
        return detected_coins
    
    def _extract_category_coins(self, news_database: List[Dict[str, Any]]) -> set:
        """Extract coins from category information in news articles."""
        category_coins = set()
        
        for article in news_database:
            categories = article.get('categories', '')
            if isinstance(categories, str):
                # Look for ticker-like categories
                category_parts = categories.split(',')
                for category in category_parts:
                    category = category.strip()
                    if self._is_valid_ticker_category(category, article):
                        # Extract potential ticker from category
                        ticker = self._extract_ticker_from_category(category)
                        if ticker:
                            category_coins.add(ticker)
        
        return category_coins
    
    def _is_valid_ticker_category(self, category: str, article: dict) -> bool:
        """Check if a category represents a valid ticker."""
        if not category or len(category) < 2:
            return False
        
        # Skip obviously non-ticker categories
        skip_categories = {
            'bitcoin', 'ethereum', 'blockchain', 'cryptocurrency', 'trading',
            'market', 'price', 'analysis', 'news', 'defi', 'nft'
        }
        
        category_lower = category.lower()
        return not any(skip in category_lower for skip in skip_categories)
    
    def _extract_ticker_from_category(self, category: str) -> Optional[str]:
        """Extract ticker symbol from category string."""
        # Look for patterns that might be tickers
        category_upper = category.upper()
        
        # Remove common prefixes/suffixes
        for remove in ['-USD', '-USDT', '-BTC', '-ETH']:
            if category_upper.endswith(remove):
                category_upper = category_upper[:-len(remove)]
        
        # Basic validation - should be 2-10 characters, mostly letters
        if 2 <= len(category_upper) <= 10 and category_upper.isalnum():
            return category_upper
        
        return None
    
    def _is_potential_valid_coin(self, coin: str) -> bool:
        """Basic validation for potential coin tickers."""
        if not coin or len(coin) < 2 or len(coin) > 10:
            return False
        
        # Skip obvious non-tickers
        skip_terms = {'USD', 'EUR', 'GBP', 'JPY', 'NEWS', 'MARKET', 'PRICE'}
        return coin.upper() not in skip_terms
    
    async def _validate_and_add_coins(self, filtered_coins: set) -> None:
        """Validate coins against exchange data and add valid ones."""
        if not filtered_coins:
            return
        
        # Get valid exchange symbols for validation
        valid_exchange_symbols = set()
        if self.exchange_manager:
            try:
                exchange_symbols = await self.exchange_manager.get_all_symbols()
                valid_exchange_symbols = set(exchange_symbols)
            except Exception as e:
                self.logger.warning(f"Could not get exchange symbols for validation: {e}")
        
        new_coins_added = 0
        for coin in filtered_coins:
            if self._should_add_coin(coin, valid_exchange_symbols):
                self.known_tickers.add(coin)
                new_coins_added += 1
        
        self.logger.debug(f"Added {new_coins_added} new tickers")
    
    def _should_add_coin(self, coin: str, valid_exchange_symbols: set) -> bool:
        """Determine if a coin should be added to known tickers."""
        # Don't add if already known
        if coin in self.known_tickers:
            return False
        
        # If we have exchange data, validate against it
        if valid_exchange_symbols:
            # Check if coin appears in any trading pair
            coin_in_pairs = any(coin in symbol for symbol in valid_exchange_symbols)
            return coin_in_pairs
        
        # If no exchange data, add by default (will be filtered later if invalid)
        return True
    
    async def save_tickers(self) -> None:
        """Save known tickers to disk."""
        try:
            if self.file_handler:
                tickers_list = sorted(list(self.known_tickers))
                self.file_handler.save_known_tickers(tickers_list)
                self.logger.debug(f"Saved {len(tickers_list)} known tickers")
        except Exception as e:
            self.logger.exception(f"Error saving tickers: {e}")
    
    def get_known_tickers(self) -> Set[str]:
        """Get the set of known cryptocurrency tickers."""
        return self.known_tickers.copy()
