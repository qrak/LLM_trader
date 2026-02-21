"""
Category processing and normalization operations.
"""
from typing import Dict, Any, List, Set, Tuple
from src.logger.logger import Logger
from src.rag.collision_resolver import CategoryCollisionResolver


class CategoryProcessor:
    """Handles processing and normalization of cryptocurrency categories."""

    def __init__(self, logger: Logger, collision_resolver: CategoryCollisionResolver, file_handler=None):
        self.logger = logger
        self.file_handler = file_handler

        # Category data storage
        self.category_word_map: Dict[str, str] = {}
        self.general_categories: Set[str] = set()
        self.ticker_categories: Set[str] = set()

        # Load RAG priorities config once
        rag_config = self._load_rag_config()

        # Load configurations from cached config
        self.important_categories: Set[str] = self._load_important_categories(rag_config)
        self.generic_priorities: Dict[str, int] = self._load_generic_priorities(rag_config)

        # Injected collision resolver
        self.collision_resolver = collision_resolver
        self._update_collision_resolver()

    def _load_rag_config(self) -> Dict:
        """Load RAG priorities config once for reuse."""
        if self.file_handler:
            try:
                return self.file_handler.load_rag_priorities() or {}
            except Exception as e:
                self.logger.error(f"Failed to load RAG config: {e}")
        return {}


    def _load_important_categories(self, config_data: Dict) -> Set[str]:
        """Load important categories from pre-loaded config."""
        if config_data and 'important_categories' in config_data:
            categories = set(config_data['important_categories'])
            if categories:
                return categories
            else:
                self.logger.warning("Config file has empty important_categories list")
        else:
            self.logger.warning("No important_categories in config data")
        return set()

    def _load_generic_priorities(self, config_data: Dict) -> Dict[str, int]:
        """Load generic category priorities from pre-loaded config."""
        if config_data and 'generic_priorities' in config_data:
            priorities = config_data['generic_priorities']
            if priorities:
                return priorities
            else:
                self.logger.warning("Config file has empty generic_priorities dict")
        else:
            self.logger.warning("No generic_priorities in config data")
        return {}


    def process_api_categories(self, api_categories: List[Dict[str, Any]]) -> None:
        """Process API categories and update internal indices."""
        if not api_categories:
            return

        # DO NOT clear existing mappings to preserve first-write-wins across calls
        # self.category_word_map.clear()

        # Categorize the data
        general_categories, ticker_categories = self._categorize_api_data(api_categories)

        # Process category words for mapping
        for category in api_categories:
            # Handle both camelCase and PascalCase field names
            category_name = category.get('categoryName') or category.get('CategoryName', '')
            category_name = category_name.lower()
            if category_name:
                self._process_category_words(category_name)

        # Update internal category sets and collision resolver
        self._update_category_sets(general_categories, ticker_categories)
        self._update_collision_resolver()


    def _categorize_api_data(self, api_categories: List[Dict[str, Any]]) -> Tuple[Set[str], Set[str]]:
        """Categorize API data into general and ticker categories."""
        general_categories = set()
        ticker_categories = set()

        for category in api_categories:
            category_name = (category.get('categoryName') or category.get('CategoryName', '')).lower()

            if category_name:
                if self._is_ticker_category(category_name):
                    ticker_categories.add(category_name)
                else:
                    general_categories.add(category_name)

        return general_categories, ticker_categories

    def _is_ticker_category(self, category_name: str) -> bool:
        """Determine if a category represents individual tickers."""
        ticker_indicators = ['symbol', 'coin', 'token', 'currency', '-usd', '-btc', '-eth']
        return any(indicator in category_name for indicator in ticker_indicators)

    def _process_category_words(self, category_name: str) -> None:
        """Process category words and create mappings with priority-based collision resolution."""
        # Extract words from category for search mapping
        words = category_name.replace('-', ' ').split()
        collision_count = 0
        excluded_count = 0

        for word in words:
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

                collision_count += 1
            else:
                self.category_word_map[word_lower] = category_name


    def _update_category_sets(self, general_categories: Set[str], ticker_categories: Set[str]) -> None:
        """Update internal category sets."""
        self.general_categories = general_categories
        self.ticker_categories = ticker_categories

    def _update_collision_resolver(self) -> None:
        """Update collision resolver with current category sets."""
        self.collision_resolver.update_category_sets(
            important_categories=self.important_categories,
            ticker_categories=self.ticker_categories,
            general_categories=self.general_categories,
            generic_priorities=self.generic_priorities
        )

    def get_api_categories(self, base_coin: str) -> set:
        """Get categories for a coin from the API category data."""
        matching_categories = set()

        # Check if the coin matches any ticker categories directly
        coin_lower = base_coin.lower()

        for category in self.ticker_categories:
            if coin_lower in category:
                matching_categories.add(category)

        # Check word-based mappings
        for word, category in self.category_word_map.items():
            if coin_lower == word or (len(coin_lower) > 3 and word in coin_lower):
                matching_categories.add(category)

        return matching_categories

    def extract_base_coin(self, symbol: str) -> str:
        """Extract base coin from trading pair symbol."""
        if not symbol:
            return ""

        # Handle different formats
        if '/' in symbol:
            return symbol.split('/')[0].upper()
        elif '-' in symbol:
            return symbol.split('-')[0].upper()
        else:
            # Try to extract base from common pairs
            # NOTE: Order matters! Longer quotes must come before shorter ones (e.g., BUSD before USD)
            common_quotes = ['USDT', 'USDC', 'BUSD', 'USD', 'BTC', 'ETH', 'BNB']
            symbol_upper = symbol.upper()

            for quote in common_quotes:
                if symbol_upper.endswith(quote):
                    base = symbol_upper[:-len(quote)]
                    # Special handling for BNB/BUSD ambiguity: BNBUSD -> BN (via BUSD) is wrong, should be BNB (via USD)
                    if quote == 'BUSD' and base == 'BN':
                        continue
                    return base

            return symbol_upper
