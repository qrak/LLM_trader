"""
Verification script to demonstrate the improved context building logic.
Shows before/after comparison of sentence selection.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Force UTF-8 output
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from unittest.mock import MagicMock
from src.rag.context_builder import ContextBuilder
import json

def main():
    print("=" * 80)
    print("VERIFICATION: Testing Context Builder on REAL Cached Data")
    print("=" * 80)
    print()
    
    # Setup
    mock_logger = MagicMock()
    mock_token_counter = MagicMock()
    # Simple token counting approximation for mock
    mock_token_counter.count_tokens.side_effect = lambda t: len(t.split()) // 3 * 4 
    
    mock_article_processor = MagicMock()
    mock_article_processor.format_article_date.return_value = "Today"
    
    # New simple builder (no sentence splitter)
    builder = ContextBuilder(
        logger=mock_logger,
        token_counter=mock_token_counter,
        article_processor=mock_article_processor
    )

    # Load real data
    json_path = project_root / 'data' / 'news_cache' / 'recent_news.json'
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            news_data = json.load(f)
            articles = news_data
            if isinstance(news_data, dict) and 'articles' in news_data:
                articles = news_data['articles']
            
            print(f"📂 Loaded {len(articles)} articles from {json_path.name}")
    except Exception as e:
        print(f"❌ Failed to load cache: {e}")
        return
    print("-" * 80)
    
    processed_count = 0
    for article in articles:
        if processed_count >= 5: break
        
        # Test the new simple processing method
        # We give it a generous token limit to see the lead paragraph
        processed_text = builder._process_article_simple(article, max_tokens=100)
        
        if not processed_text: continue
        
        print(f"\n{processed_text}")
        print("-" * 40)
            
        processed_count += 1
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
