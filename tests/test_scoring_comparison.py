"""
Scoring Comparison Test for wtpsplit Sentence Selection

This script compares sentence selection BEFORE and AFTER the scoring penalty,
using real news from crypto_news.json.

Usage:
    python tests/test_scoring_comparison.py [--num-articles=5]
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
import random
from typing import List, Dict, Tuple
from src.logger.logger import Logger
from src.rag.text_splitting import SentenceSplitter
from src.utils.token_counter import TokenCounter
from src.rag.article_processor import ArticleProcessor
from src.parsing.unified_parser import UnifiedParser
from src.utils.format_utils import FormatUtils
from src.analyzer.data_processor import DataProcessor


def load_random_btc_articles(num_articles: int = 5) -> List[Dict]:
    """Load random BTC-related articles from crypto_news.json."""
    news_path = ROOT / "data" / "crypto_news.json"
    
    with open(news_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        all_articles = data.get('articles', [])
    
    # Filter for BTC-related articles
    btc_articles = [
        article for article in all_articles
        if 'BTC' in article.get('categories', '').upper() 
        or 'BITCOIN' in article.get('tags', '').upper()
        or 'bitcoin' in article.get('title', '').lower()
        or 'btc' in article.get('title', '').lower()
    ]
    
    if not btc_articles:
        print("⚠️  No BTC-related articles found, using all articles")
        btc_articles = all_articles
    
    # Sample random articles
    sample_size = min(num_articles, len(btc_articles))
    return random.sample(btc_articles, sample_size)


def select_sentences_without_penalty(sentences: List[str], keywords: set, max_count: int) -> Tuple[List[str], List[float]]:
    """Original scoring logic WITHOUT the keyword-only penalty."""
    import re
    
    if not sentences or not keywords:
        return sentences[:max_count], [0.0] * min(len(sentences), max_count)
    
    scored = []
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        kw_count = sum(1 for kw in keywords if kw in sent_lower)
        score = float(kw_count)
        
        # Boost for numbers
        number_groups = len(re.findall(r'\d+(?:\.\d+)?', sent))
        if number_groups > 0:
            score += (number_groups * 0.5)
        
        # Boost for percentage signs
        percent_count = sent.count('%')
        if percent_count > 0:
            score += (percent_count * 2.0)
        
        # Boost for currency symbols
        if '$' in sent:
            score += 2.0

        # Boost for "Key: Value" patterns
        if ':' in sent:
            score += 1.5

        # SYNERGY BOOST (original)
        if kw_count > 0 and ('$' in sent or percent_count > 0):
            score += 2.0
        
        # NO PENALTY HERE - this is the old version
            
        scored.append((i, score, sent))
    
    # Sort by score descending, take top N
    scored.sort(key=lambda x: x[1], reverse=True)
    top_sentences = scored[:max_count]
    
    # Re-sort by original index to preserve narrative order
    top_sentences.sort(key=lambda x: x[0])
    
    return [sent for _, _, sent in top_sentences], [score for _, score, _ in top_sentences]


def select_sentences_with_penalty(sentences: List[str], keywords: set, max_count: int) -> Tuple[List[str], List[float]]:
    """New scoring logic WITH the keyword-only penalty."""
    import re
    
    if not sentences or not keywords:
        return sentences[:max_count], [0.0] * min(len(sentences), max_count)
    
    scored = []
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        kw_count = sum(1 for kw in keywords if kw in sent_lower)
        score = float(kw_count)
        
        # Boost for numbers
        number_groups = len(re.findall(r'\d+(?:\.\d+)?', sent))
        if number_groups > 0:
            score += (number_groups * 0.5)
        
        # Boost for percentage signs
        percent_count = sent.count('%')
        if percent_count > 0:
            score += (percent_count * 2.0)
        
        # Boost for currency symbols
        if '$' in sent:
            score += 2.0

        # Boost for "Key: Value" patterns
        if ':' in sent:
            score += 1.5

        # SYNERGY BOOST
        if kw_count > 0 and ('$' in sent or percent_count > 0):
            score += 2.0
        
        # NEW PENALTY: Deprioritize keyword-only matches
        if kw_count > 0 and number_groups == 0 and percent_count == 0 and '$' not in sent:
            score -= 1.0
            
        scored.append((i, score, sent))
    
    # Sort by score descending, take top N
    scored.sort(key=lambda x: x[1], reverse=True)
    top_sentences = scored[:max_count]
    
    # Re-sort by original index to preserve narrative order
    top_sentences.sort(key=lambda x: x[0])
    
    return [sent for _, _, sent in top_sentences], [score for _, score, _ in top_sentences]


def analyze_article(article: Dict, logger: Logger) -> None:
    """Analyze a single article with both scoring methods."""
    title = article.get('title', 'No Title')
    body = article.get('body', '')
    categories = article.get('categories', '')
    
    # Check if BTC-related
    is_btc = ('BTC' in categories.upper() or 
              'BITCOIN' in title.upper() or 
              'btc' in title.lower())
    
    btc_marker = "✓ BTC" if is_btc else "✗ NOT BTC"
    
    print(f"\n{'='*80}")
    print(f"Article: {title[:70]}...")
    print(f"BTC-Related: {btc_marker}")
    print(f"Categories: {categories}")
    print(f"{'='*80}")
    
    # Split sentences
    splitter = SentenceSplitter(logger)
    sentences = splitter.split_text(body.replace('\n', ' ').strip())
    
    print(f"\nTotal sentences: {len(sentences)}")
    
    # Keywords relevant to BTC trading
    keywords = {'btc', 'bitcoin', 'price', 'long', 'short', 'ratio', 'trading', 'market'}
    
    # Compare both methods
    old_selected, old_scores = select_sentences_without_penalty(sentences, keywords, max_count=5)
    new_selected, new_scores = select_sentences_with_penalty(sentences, keywords, max_count=5)
    
    # Show comparison
    print(f"\n{'─'*80}")
    print("BEFORE (Old Scoring - No Penalty):")
    print(f"{'─'*80}")
    for i, (sent, score) in enumerate(zip(old_selected, old_scores), 1):
        has_data = any(c in sent for c in ['$', '%']) or any(c.isdigit() for c in sent)
        data_marker = "📊" if has_data else "📝"
        print(f"{i}. [{score:+.1f}] {data_marker} {sent[:120]}...")
    
    print(f"\n{'─'*80}")
    print("AFTER (New Scoring - With Penalty):")
    print(f"{'─'*80}")
    for i, (sent, score) in enumerate(zip(new_selected, new_scores), 1):
        has_data = any(c in sent for c in ['$', '%']) or any(c.isdigit() for c in sent)
        data_marker = "📊" if has_data else "📝"
        
        # Highlight if this sentence changed position
        changed = ""
        if sent not in old_selected:
            changed = " 🆕 NEW"
        elif old_selected.index(sent) != i - 1:
            changed = f" ⬆️ moved from #{old_selected.index(sent) + 1}"
        
        print(f"{i}. [{score:+.1f}] {data_marker} {sent[:120]}...{changed}")
    
    # Analysis
    print(f"\n{'─'*80}")
    print("ANALYSIS:")
    print(f"{'─'*80}")
    
    # Count data-rich sentences
    old_data_rich = sum(1 for s in old_selected if any(c in s for c in ['$', '%']) or any(c.isdigit() for c in s))
    new_data_rich = sum(1 for s in new_selected if any(c in s for c in ['$', '%']) or any(c.isdigit() for c in s))
    
    print(f"Data-rich sentences (with numbers/$/%):")
    print(f"  Old: {old_data_rich}/5 ({old_data_rich * 20}%)")
    print(f"  New: {new_data_rich}/5 ({new_data_rich * 20}%)")
    
    if new_data_rich > old_data_rich:
        print("  ✅ IMPROVED: More data-rich sentences selected")
    elif new_data_rich == old_data_rich:
        print("  ➡️  NO CHANGE: Same number of data-rich sentences")
    else:
        print("  ⚠️  REGRESSION: Fewer data-rich sentences selected")
    
    # Check for changes
    differences = set(new_selected) - set(old_selected)
    if differences:
        print(f"\n{len(differences)} sentences changed:")
        for sent in differences:
            print(f"  + {sent[:80]}...")
    else:
        print("\n✓ Same sentences selected (order may differ)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare sentence scoring before/after penalty')
    parser.add_argument('--num-articles', type=int, default=5, help='Number of articles to analyze')
    args = parser.parse_args()
    
    logger = Logger("scoring_comparison", logger_debug=False)
    
    print("="*80)
    print("WTPSPLIT SCORING COMPARISON TEST")
    print("="*80)
    print(f"\nLoading {args.num_articles} random BTC-related articles from crypto_news.json...")
    
    articles = load_random_btc_articles(args.num_articles)
    
    print(f"✓ Loaded {len(articles)} articles")
    
    for idx, article in enumerate(articles, 1):
        print(f"\n\n{'█'*80}")
        print(f"ARTICLE {idx}/{len(articles)}")
        print(f"{'█'*80}")
        analyze_article(article, logger)
    
    print(f"\n\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}")
    print("\nLegend:")
    print("  📊 = Data-rich sentence (has numbers, %, or $)")
    print("  📝 = Narrative sentence (keywords only, no data)")
    print("  🆕 = New sentence (not in old selection)")
    print("  ⬆️  = Sentence moved up in ranking")


if __name__ == '__main__':
    main()
