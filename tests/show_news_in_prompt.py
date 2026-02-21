"""
Demonstration: How News Context Appears in AI Prompt
Shows the actual formatted output that gets sent to the AI model.
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from unittest.mock import MagicMock
from src.rag.context_builder import ContextBuilder
import json


def main():
    print("=" * 80)
    print("NEWS CONTEXT IN AI PROMPT - Real Example")
    print("=" * 80)
    print()

    # Setup context builder
    mock_logger = MagicMock()
    mock_token_counter = MagicMock()
    mock_token_counter.count_tokens.side_effect = lambda t: len(t.split()) // 3 * 4

    mock_article_processor = MagicMock()
    mock_article_processor.format_article_date.return_value = "2 hours ago"

    builder = ContextBuilder(
        logger=mock_logger,
        token_counter=mock_token_counter,
        article_processor=mock_article_processor,
    )

    # Load real news
    json_path = project_root / 'data' / 'news_cache' / 'recent_news.json'
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            news_data = json.load(f)
            articles = news_data if isinstance(news_data, list) else news_data.get('articles', [])
            print(f"✓ Loaded {len(articles)} articles from cache")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return

    # Take first 3 articles (typical for a prompt)
    sample_articles = articles[:5]

    print("\n" + "=" * 80)
    print("BUILDING NEWS CONTEXT (max_tokens=600, ~3 articles)")
    print("=" * 80)

    # Build context as it appears in the prompt
    news_context = builder.build_context(sample_articles, max_tokens=600)

    print("\n" + "─" * 80)
    print("THIS IS WHAT THE AI RECEIVES IN THE PROMPT:")
    print("─" * 80)
    print()
    print("## RELEVANT NEWS & MARKET CONTEXT")
    print()
    print(news_context)
    print()
    print("─" * 80)
    print(f"Total characters: {len(news_context)}")
    print(f"Approx tokens: ~{len(news_context) // 4}")
    print("=" * 80)


if __name__ == "__main__":
    main()
