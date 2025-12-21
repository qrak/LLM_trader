"""
Unified parsing system that consolidates all parsing functionality.
Eliminates duplication and unnecessary delegation layers.
"""
import json
import re
from typing import Dict, Any, Set, Optional, Union, List

from src.logger.logger import Logger


class UnifiedParser:
    """
    Consolidated parser that handles all parsing needs across the application.
    Replaces multiple scattered parsing components with a single, comprehensive solution.
    """
    
    def __init__(self, logger: Logger, format_utils=None):
        self.logger = logger
        self.format_utils = format_utils
        
        # Numeric fields that should be converted from strings with their defaults
        self._numeric_fields = {
            'risk_ratio': 1.0,
            'trend_strength': 50,
            'confidence_score': 50,
            'bullish_scenario': 0.0,
            'bearish_scenario': 0.0
        }
    
    # ============================================================================
    # AI RESPONSE PARSING
    # ============================================================================
    
    def parse_ai_response(self, raw_text: str) -> Dict[str, Any]:
        """
        Parse AI model response from raw string to structured data.
        Supports all AI providers: OpenRouter, Google AI, LM Studio.
        """
        try:
            cleaned_text = self._clean_tool_response_tags(raw_text)
            
            if "```json" in cleaned_text:
                json_start = cleaned_text.find("```json") + 7
                json_end = cleaned_text.find("```", json_start)
                if json_end > json_start:
                    try:
                        result = json.loads(cleaned_text[json_start:json_end].strip())
                        return self._normalize_numeric_fields(result)
                    except json.JSONDecodeError:
                        pass
            
            first_brace = cleaned_text.find('{')
            if first_brace != -1:
                depth = 0
                for i in range(first_brace, len(cleaned_text)):
                    if cleaned_text[i] == '{':
                        depth += 1
                    elif cleaned_text[i] == '}':
                        depth -= 1
                        if depth == 0:
                            try:
                                result = json.loads(cleaned_text[first_brace:i+1])
                                return self._normalize_numeric_fields(result)
                            except json.JSONDecodeError:
                                pass
                            break
            
            heuristic = self._heuristic_extract_analysis_from_text(cleaned_text)
            if heuristic:
                self.logger.debug("Extracted analysis from markdown text")
                return self._normalize_numeric_fields(heuristic)
            
            self.logger.warning(f"Unable to parse AI response, using fallback. Preview: {cleaned_text[:200]}")
            return self._create_fallback_response(cleaned_text)
            
        except Exception as e:
            self.logger.error(f"Failed to parse AI response: {e}")
            return self._create_error_response(str(e), raw_text)
    
    def validate_ai_response(self, response: Dict[str, Any]) -> bool:
        """Validate that AI response has required structure."""
        return (isinstance(response, dict) and 
                "analysis" in response and 
                isinstance(response["analysis"], dict))
    
    # ============================================================================
    # TIMESTAMP PARSING (consolidates multiple timestamp parsers)
    # ============================================================================
    
    def parse_timestamp(self, timestamp_field: Union[int, float, str, None]) -> float:
        """
        Universal timestamp parser supporting all formats used across the application.
        Consolidates timestamp parsing from MarketDataProcessor and ArticleProcessor.
        """
        if timestamp_field is None:
            return 0.0

        if isinstance(timestamp_field, (int, float)):
            return float(timestamp_field)
        
        if isinstance(timestamp_field, str):
            return self._parse_timestamp_string(timestamp_field)
        
        return 0.0
    
    def _parse_timestamp_string(self, timestamp_str: str) -> float:
        """Parse timestamp string to float timestamp.
        
        Uses centralized FormatUtils for consistency.
        """
        return self.format_utils.timestamp_from_iso(timestamp_str)
    
    # ============================================================================
    # CATEGORY PARSING (consolidates category-related parsing)
    # ============================================================================
    
    def parse_article_categories(self, categories_string: str) -> Set[str]:
        """Parse categories from article category string."""
        if not categories_string:
            return set()
        
        categories = set()
        
        # Split by common separators
        for separator in [',', ';', '|']:
            if separator in categories_string:
                parts = categories_string.split(separator)
                for part in parts:
                    clean_category = part.strip().lower()
                    if clean_category and len(clean_category) > 2:
                        categories.add(clean_category)
                break
        else:
            # No separator found, use as single category
            clean_category = categories_string.strip().lower()
            if clean_category and len(clean_category) > 2:
                categories.add(clean_category)
        
        return categories
    
    # ============================================================================
    # SYMBOL AND COIN PARSING
    # ============================================================================
    
    def extract_base_coin(self, symbol: str) -> str:
        """Extract base coin from trading pair symbol."""
        if not symbol:
            return ""
        
        # Handle symbols with explicit separators first
        if '/' in symbol:
            return symbol.split('/')[0].upper()
        if '-' in symbol:
            return symbol.split('-')[0].upper()
        
        # Handle concatenated symbols by removing common quote currencies
        common_quotes = ['USDT', 'USD', 'BTC', 'ETH', 'BNB', 'BUSD']
        symbol_upper = symbol.upper()
        
        for quote in common_quotes:
            if symbol_upper.endswith(quote):
                return symbol_upper[:-len(quote)]
        
        return symbol_upper
    
    def detect_coins_in_text(self, text: str, known_tickers: Set[str]) -> Set[str]:
        """Detect cryptocurrency mentions in text content."""
        if not text:
            return set()
            
        coins_mentioned = set()
        text_upper = text.upper()
        
        # Find potential tickers using regex
        potential_tickers = set(re.findall(r'\b[A-Z]{2,6}\b', text_upper))
        
        # Validate against known tickers
        for ticker in potential_tickers:
            if ticker in known_tickers:
                coins_mentioned.add(ticker)
        
        # Special handling for major cryptocurrencies
        text_lower = text.lower()
        if 'bitcoin' in text_lower:
            coins_mentioned.add('BTC')
        if 'ethereum' in text_lower:
            coins_mentioned.add('ETH')
            
        return coins_mentioned
    
    # ============================================================================
    # PRIVATE HELPER METHODS
    # ============================================================================
    
    def _clean_tool_response_tags(self, text: str) -> str:
        """Remove tool_response tags from AI responses."""
        if "<tool_response>" in text:
            self.logger.warning("Found tool_response tags in response, cleaning up")
            cleaned = re.sub(r'<tool_response>[\s\n]*</tool_response>|<tool_response>|</tool_response>', '', text)
            return re.sub(r'\n\s*\n', '\n', cleaned).strip()
        return text

    def _heuristic_extract_analysis_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Attempt to extract a minimal `analysis` object from plain text.

        This provides graceful degradation when the model returns a natural
        language analysis instead of strict JSON. It only extracts a few key
        fields (current_price, RSI, ADX, summary) and leaves the rest for
        downstream fallback handling.
        """
        if not text or len(text) < 100:
            return None

        def _find_currency(label_patterns: List[str]) -> Optional[float]:
            for pat in label_patterns:
                m = re.search(pat + r"\s*\$?([0-9,]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
                if m:
                    try:
                        return float(m.group(1).replace(',', ''))
                    except Exception:
                        return None
            return None

        def _find_number(label_patterns: List[str]) -> Optional[float]:
            for pat in label_patterns:
                m = re.search(pat + r"\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
                if m:
                    try:
                        return float(m.group(1))
                    except Exception:
                        return None
            return None

        analysis: Dict[str, Any] = {}

        current_price = _find_currency([
            r"\*\*Current Price:\*\*",  # Markdown bold
            r"Current Price:",
            r"Current price",
            r"price is"
        ])
        if current_price is not None:
            analysis['current_price'] = current_price

        rsi = _find_number([
            r"\*\*Momentum \(RSI\):\*\*",
            r"RSI\(14\):",
            r"RSI:",
            r"rsi "
        ])
        if rsi is not None:
            analysis['rsi'] = rsi

        adx = _find_number([
            r"\*\*Trend Strength \(ADX\):\*\*",
            r"Trend Strength \(ADX\):",
            r"ADX:",
            r"trend strength \(adx\)"
        ])
        if adx is not None:
            analysis['trend_strength'] = adx

        summary = None
        m = re.search(r"\*\*What this means:(.*?)\n\n", text)
        if m:
            summary = m.group(1).strip()
        else:
            m2 = re.split(r"\n\n|## |### ", text)
            if m2 and len(m2[0]) > 20:
                summary = m2[0].strip()

        if summary:
            analysis['summary'] = summary

        if analysis:
            return {"analysis": analysis}
        return None
    
    def _normalize_numeric_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure numeric fields are properly typed at the data source."""
        if not isinstance(data, dict):
            return data
            
        # Check analysis section
        analysis = data.get('analysis', {})
        for field, default_value in self._numeric_fields.items():
            if field in analysis and isinstance(analysis[field], str):
                try:
                    analysis[field] = float(analysis[field])
                except ValueError:
                    # Use default value for invalid strings (fix at source)
                    analysis[field] = default_value
        
        # Normalize key_levels arrays (support/resistance)
        key_levels = analysis.get('key_levels', {})
        if isinstance(key_levels, dict):
            for level_type in ['support', 'resistance']:
                levels = key_levels.get(level_type, [])
                if isinstance(levels, list):
                    normalized_levels = []
                    for level in levels:
                        if isinstance(level, (int, float)):
                            normalized_levels.append(float(level))
                        elif isinstance(level, str):
                            try:
                                normalized_levels.append(float(level))
                            except ValueError:
                                # Skip invalid string values - don't add them to the list
                                continue
                    key_levels[level_type] = normalized_levels
        
        # Check root level
        for field, default_value in self._numeric_fields.items():
            if field in data and isinstance(data[field], str):
                try:
                    data[field] = float(data[field])
                except ValueError:
                    data[field] = default_value
        
        return data
    
    def _create_fallback_response(self, cleaned_text: str) -> Dict[str, Any]:
        """Create fallback response when parsing fails."""
        return {
            "analysis": {
                "summary": "Unable to parse the AI response. The analysis may have been in an invalid format.",
                "observed_trend": "NEUTRAL",
                "trend_strength": 50,
                "confidence_score": 0
            },
            "raw_response": cleaned_text,
            "parse_error": "Failed to parse response"
        }
    
    def _create_error_response(self, error_message: str, raw_text: str) -> Dict[str, Any]:
        """Create error response for parsing exceptions."""
        return {
            "error": error_message, 
            "raw_response": raw_text,
            "analysis": {
                "summary": "Error parsing response",
                "observed_trend": "NEUTRAL",
                "confidence_score": 0
            }
        }