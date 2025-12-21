"""
Collision resolution utilities for category-word mappings.
Centralizes priority-based collision resolution logic to avoid code duplication.
"""
from typing import Set


class CategoryCollisionResolver:
    """Handles priority-based collision resolution for category-word mappings."""
    
    def __init__(self, important_categories: Set[str] = None, ticker_categories: Set[str] = None, general_categories: Set[str] = None):
        self.important_categories = important_categories or set()
        self.ticker_categories = ticker_categories or set()
        self.general_categories = general_categories or set()
    
    def resolve_collision(self, existing_category: str, new_category: str, word: str) -> str:
        """
        Resolve mapping collision using priority-based rules.
        
        Returns the category that should win based on priority hierarchy:
        1. Specific ticker categories (BTC, ETH, etc.) - Highest priority
        2. Important categories - High priority  
        3. Ticker categories - Medium-high priority
        4. Other specific categories - Medium priority
        5. General categories - Low priority
        6. Exchange/regulatory categories - Lowest priority
        """
        existing_priority = self._get_category_priority(existing_category)
        new_priority = self._get_category_priority(new_category)
        
        # Return the category with higher priority
        if new_priority > existing_priority:
            return new_category
        else:
            return existing_category
    
    def update_category_sets(self, important_categories: Set[str], ticker_categories: Set[str], general_categories: Set[str]) -> None:
        """Update category sets for priority calculation."""
        self.important_categories = important_categories
        self.ticker_categories = ticker_categories
        self.general_categories = general_categories
    
    def _get_category_priority(self, category: str) -> int:
        """Get priority score for a category (higher = more specific/important)."""
        category_upper = category.upper()
        category_lower = category.lower()
        
        # Specific ticker categories get highest priority
        # Short uppercase categories are likely specific tickers (BTC, ETH, AAVE, etc.)
        if len(category_upper) <= 10 and category_upper.isupper() and '-' not in category_upper:
            return 100
        
        # Important categories get high priority
        if category_lower in self.important_categories:
            return 80
        
        # Ticker categories get medium-high priority
        if category_lower in self.ticker_categories:
            return 70
        
        # Generic/broad categories get lower priority
        generic_categories = {
            'cryptocurrency': 10,
            'exchange': 15,
            'regulation': 20,
            'macroeconomics': 20,
            'token listing and delisting': 25,
            'token sale': 25,
            'digital asset treasury': 30
        }
        
        if category_lower in generic_categories:
            return generic_categories[category_lower]
        
        # General categories get low-medium priority
        if category_lower in self.general_categories:
            return 50
        
        # Default priority for unknown categories
        return 60