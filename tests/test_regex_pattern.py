"""
Quick test to verify the regex patterns match "previous weekend"
"""
import re

_TEMPORAL_RELATIVE_PATTERN = re.compile(
    r'\b(yesterday|today|tomorrow|now|currently|recently|previously|earlier|later|following|weekend|'
    r'Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|'
    r'this\s+(week|weekend|month|year|morning|afternoon|evening)|'
    r'last\s+(week|weekend|month|year|hour|day|night)|'
    r'next\s+(week|weekend|month|year|day)|'
    r'past\s+(week|weekend|month|year|day|hour)|'
    r'previous\s+(week|weekend|month|year|day)|'
    r'hours?\s+ago|days?\s+ago|minutes?\s+ago|seconds?\s+ago)\b',
    re.IGNORECASE
)

test_sentences = [
    "The previous weekend brought unexpected volatility to the cryptocurrency markets.",
    "BTC Taps $70K.",
    "The largest of the bunch dumped from $84,000 to under $76,000 on Saturday night and tried to recover to $79,000 on Sunday.",
    "The rather calm behavior during the weekend has worked in favor of bitcoin"
]

for sent in test_sentences:
    matches = _TEMPORAL_RELATIVE_PATTERN.findall(sent)
    print(f"Sentence: {sent[:60]}...")
    print(f"  Matches found: {matches}")
    print(f"  Match count: {len(matches)}")
    print()
