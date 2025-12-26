"""
Timestamp utilities for news components.
Provides shared timestamp extraction and normalization functions.
"""
from datetime import datetime
from typing import Dict, Any


def get_article_timestamp(article: Dict[str, Any]) -> float:
    """Extract timestamp from article in a consistent format"""
    pub_time = 0.0
    published_on = article.get('published_on', 0)

    if isinstance(published_on, (int, float)):
        pub_time = float(published_on)
    elif isinstance(published_on, str):
        try:
            if published_on.isdigit():
                pub_time = float(published_on)
            else:
                pub_time = datetime.fromisoformat(published_on.replace('Z', '+00:00')).timestamp()
        except ValueError:
            pub_time = 0.0

    return pub_time
