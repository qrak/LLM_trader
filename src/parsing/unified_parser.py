"""
Unified parsing system that consolidates all parsing functionality.
Eliminates duplication and unnecessary delegation layers.
"""
import json
import re
from typing import Dict, Any, Set, Optional

from src.logger.logger import Logger


class UnifiedParser:
    """
    Consolidated parser that handles all parsing needs across the application.
    Replaces multiple scattered parsing components with a single, comprehensive solution.
    """

    _CATEGORY_SEPARATORS = (',', ';', '|')

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




    def parse_ai_response(self, raw_text: str) -> Dict[str, Any]:
        """
        Parse AI model response from raw string to structured data.
        Requires models to produce valid JSON (no heuristic parsing for low-quality models).
        """
        try:
            if "```json" in raw_text:
                json_start = raw_text.find("```json") + 7
                json_end = raw_text.find("```", json_start)
                if json_end > json_start:
                    try:
                        result = json.loads(raw_text[json_start:json_end].strip())
                        return self._normalize_numeric_fields(result)
                    except json.JSONDecodeError:
                        pass

            first_brace = raw_text.find('{')
            if first_brace != -1:
                try:
                    decoder = json.JSONDecoder()
                    result, _ = decoder.raw_decode(raw_text, first_brace)
                    return self._normalize_numeric_fields(result)
                except json.JSONDecodeError:
                    pass

            self.logger.warning("Unable to parse AI response, using fallback. Preview: %s", raw_text[:200])
            return self._create_fallback_response(raw_text)

        except Exception as e:
            self.logger.error("Failed to parse AI response: %s", e)
            return self._create_error_response(str(e), raw_text)

    def validate_ai_response(self, response: Dict[str, Any]) -> bool:
        """Validate that AI response has required structure."""
        return (isinstance(response, dict) and
                "analysis" in response and
                isinstance(response["analysis"], dict))

    def extract_json_block(self, text: str, unwrap_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Extract JSON block from markdown-formatted text.

        Reusable utility for extracting ```json ... ``` blocks from AI responses.
        Used by notifiers, position extractors, and other components.

        Args:
            text: Raw text containing JSON block
            unwrap_key: If specified, unwrap this key from the result (e.g., 'analysis')

        Returns:
            Parsed JSON dict or None if extraction fails
        """
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)

                if unwrap_key and unwrap_key in data and isinstance(data[unwrap_key], dict):
                    return data[unwrap_key]
                return data
        except (json.JSONDecodeError, Exception) as e:
            self.logger.debug("JSON block extraction failed: %s", e)
        return None

    def extract_text_before_json(self, text: str) -> str:
        """Extract text content before a JSON block.

        Useful for separating reasoning/explanation from structured JSON data.

        Args:
            text: Raw text potentially containing JSON block

        Returns:
            Text before the JSON block, or full text if no JSON found
        """
        json_match = re.search(r'```json', text, re.IGNORECASE)
        if json_match:
            reasoning = text[:json_match.start()].strip()
            return reasoning
        return text.strip()




    @staticmethod
    def format_error_response(error_message: str) -> str:
        """Create a standardized error response in JSON format.

        Args:
            error_message: Error message to include in the response

        Returns:
            Formatted error response with JSON structure and markdown
        """
        json_fallback = {
            "analysis": {
                "summary": f"Analysis unavailable: {error_message}. Please try again later.",
            }
        }
        return f"```json\n{json.dumps(json_fallback, indent=2)}\n```\n\nThe analysis failed due to a technical issue. Please try again later."

    @staticmethod
    def format_provider_error(provider: str, error_detail: str) -> Dict[str, Any]:
        """Create a standardized provider error dictionary.

        Args:
            provider: Provider name that failed
            error_detail: Details about the error

        Returns:
            Error dictionary for use in ChatResponseModel
        """
        return {"error": f"{provider} failed: {error_detail}"}

    @staticmethod
    def format_final_fallback_error(provider: str, error_detail: str) -> Dict[str, Any]:
        """Create error response for final fallback failure.

        Args:
            provider: Last provider attempted
            error_detail: Details about the error

        Returns:
            Error dictionary indicating all providers failed
        """
        return {"error": f"All models failed. Last attempt ({provider}): {error_detail}"}


    def parse_article_categories(self, categories_string: str) -> Set[str]:
        """Parse categories from article category string."""
        if not categories_string:
            return set()

        categories = set()

        # Split by common separators
        for separator in self._CATEGORY_SEPARATORS:
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
        # Check longer quotes first to avoid partial matches (e.g. matching USD in BUSD)
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






    def _normalize_numeric_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure numeric fields are properly typed at the data source."""
        if not isinstance(data, dict):
            return data

        # Check analysis section
        analysis = data.get('analysis', {})
        for field, default_value in self._numeric_fields.items():
            if field in analysis:
                val = analysis[field]
                if self.format_utils:
                    parsed = self.format_utils.parse_value(val, default=None)
                    if parsed is not None:
                        analysis[field] = parsed
                    else:
                        analysis[field] = default_value
                elif isinstance(val, str):
                    try:
                        analysis[field] = float(val)
                    except ValueError:
                        analysis[field] = default_value

        # Normalize confluence_factors (new Chain-of-Thought scoring)
        confluence_factors = analysis.get('confluence_factors', {})
        if isinstance(confluence_factors, dict):
            for factor_key in ['trend_alignment', 'momentum_strength', 'volume_support',
                              'pattern_quality', 'support_resistance_strength']:
                if factor_key in confluence_factors:
                    value = confluence_factors[factor_key]
                    if self.format_utils:
                        # Default to 50.0 (Neutral) if parsing fails
                        parsed = self.format_utils.parse_value(value, default=None)
                        confluence_factors[factor_key] = parsed if parsed is not None else 50.0
                    elif isinstance(value, str):
                        try:
                            confluence_factors[factor_key] = float(value)
                        except ValueError:
                            confluence_factors[factor_key] = 50.0
                    elif isinstance(value, (int, float)):
                        confluence_factors[factor_key] = float(value)
                    else:
                        confluence_factors[factor_key] = 50.0

        # Normalize key_levels arrays (support/resistance)
        key_levels = analysis.get('key_levels', {})
        if isinstance(key_levels, dict):
            for level_type in ['support', 'resistance']:
                levels = key_levels.get(level_type, [])
                if isinstance(levels, list):
                    normalized_levels = []
                    for level in levels:
                        if self.format_utils:
                            val = self.format_utils.parse_value(level, default=None)
                            if val is not None:
                                normalized_levels.append(val)
                        elif isinstance(level, (int, float)):
                            normalized_levels.append(float(level))
                        elif isinstance(level, str):
                            try:
                                normalized_levels.append(float(level))
                            except ValueError:
                                continue
                    key_levels[level_type] = normalized_levels

        # Check root level
        for field, default_value in self._numeric_fields.items():
            if field in data:
                if self.format_utils:
                    parsed = self.format_utils.parse_value(data[field], default=None)
                    if parsed is not None:
                        data[field] = parsed
                    else:
                        data[field] = default_value
                elif isinstance(data[field], str):
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
