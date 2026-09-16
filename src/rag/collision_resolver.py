"""
Collision resolution utilities for category-word mappings.
Centralizes priority-based collision resolution logic to avoid code duplication.
"""


class CategoryCollisionResolver:
    """Handles priority-based collision resolution for category-word mappings."""

    def __init__(self, important_categories: set[str] | None = None, ticker_categories: set[str] | None = None,
                 general_categories: set[str] | None = None, generic_priorities: dict[str, int] | None = None):
        self.important_categories = important_categories or set()
        self.ticker_categories = ticker_categories or set()
        self.general_categories = general_categories or set()

        self.generic_priorities = generic_priorities or {
            "cryptocurrency": 10,
            "exchange": 15,
            "regulation": 20,
            "macroeconomics": 20,
            "token listing and delisting": 25,
            "token sale": 25,
            "digital asset treasury": 30
        }

    def resolve_collision(self, existing_category: str, new_category: str, _word: str) -> str:
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

        if new_priority > existing_priority:
            return new_category
        return existing_category

    def update_category_sets(self, important_categories: set[str], ticker_categories: set[str], general_categories: set[str], generic_priorities: dict[str, int] | None = None) -> None:
        """Update category sets for priority calculation."""
        self.important_categories = important_categories
        self.ticker_categories = ticker_categories
        self.general_categories = general_categories
        if generic_priorities:
            self.generic_priorities = generic_priorities

    def _get_category_priority(self, category: str) -> int:
        """Get priority score for a category (higher = more specific/important)."""
        category_upper = category.upper()
        category_lower = category.lower()

        if len(category_upper) <= 10 and category_upper.isupper() and "-" not in category_upper:
            return 100

        if category_lower in self.important_categories:
            return 80

        if category_lower in self.ticker_categories:
            return 70

        if category_lower in self.generic_priorities:
            return self.generic_priorities[category_lower]

        if category_lower in self.general_categories:
            return 50

        return 60
