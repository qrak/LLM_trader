
import json
import math
import re
from typing import Any, ClassVar

from pydantic import ValidationError

from src.logger.logger import Logger
from src.platforms.ai_providers.response_models import TradingAnalysisResponseModel


class UnifiedParser:
    """
    Consolidated parser that handles all parsing needs across the application.
    Replaces multiple scattered parsing components with a single, comprehensive solution.
    """

    def __init__(self, logger: Logger, format_utils=None):
        self.logger = logger
        self.format_utils = format_utils

        self._numeric_fields = {
            "risk_ratio": 1.0,
            "risk_reward_ratio": None,
            "trend_strength": 50,
            "confidence": 50,
            "confidence_score": 50,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "position_size": None,
            "bullish_scenario": 0.0,
            "bearish_scenario": 0.0
        }


    def parse_ai_response(self, raw_text: str) -> dict[str, Any]:

        try:
            if "```json" in raw_text:
                json_start = raw_text.find("```json") + 7
                json_end = raw_text.find("```", json_start)
                if json_end > json_start:
                    try:
                        result = json.loads(raw_text[json_start:json_end].strip())
                        result = self._normalize_numeric_fields(result)
                        return self._attach_response_validation(result)
                    except json.JSONDecodeError:
                        pass

            first_brace = raw_text.find("{")
            if first_brace != -1:
                try:
                    decoder = json.JSONDecoder()
                    result, _ = decoder.raw_decode(raw_text, first_brace)
                    result = self._normalize_numeric_fields(result)
                    return self._attach_response_validation(result)
                except json.JSONDecodeError:
                    pass

            self.logger.warning("Unable to parse AI response, using fallback. Preview: %s", raw_text[:200])
            return self._create_fallback_response(raw_text)

        except Exception as e:  # noqa: BLE001
            self.logger.error("Failed to parse AI response: %s", e)
            return self._create_error_response(str(e), raw_text)

    def validate_ai_response(self, response: dict[str, Any]) -> bool:

        return (isinstance(response, dict) and
                "analysis" in response and
                isinstance(response["analysis"], dict))

    def validate_trading_response(self, response: dict[str, Any]) -> dict[str, Any]:

        validation = {
            "schema": TradingAnalysisResponseModel.schema_version,
            "status": "skipped",
            "valid": None,
            "errors": [],
        }
        if not isinstance(response, dict):
            return {
                **validation,
                "status": "invalid",
                "valid": False,
                "errors": [{"field": "response", "message": "Response is not an object", "type": "type_error"}],
            }

        analysis = response.get("analysis")
        if not isinstance(analysis, dict) or "signal" not in analysis:
            return validation

        try:
            TradingAnalysisResponseModel.model_validate(response)
        except ValidationError as error:
            return {
                **validation,
                "status": "invalid",
                "valid": False,
                "errors": self._format_validation_errors(error),
            }
        return {**validation, "status": "valid", "valid": True}

    @staticmethod
    def _iter_json_blocks(text: str) -> list[str]:

        blocks: list[str] = []
        text_lower = text.lower()
        index = 0
        while True:
            start = text_lower.find("```json", index)
            if start == -1:
                return blocks
            body_start = start + 7
            end = text_lower.find("```", body_start)
            if end == -1:
                return blocks
            blocks.append(text[body_start:end].strip())
            index = end + 3

    _CONTROL_CHAR_ESCAPES: ClassVar[dict[str, str]] = {
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
        "\b": "\\b",
        "\f": "\\f",
    }

    @classmethod
    def repair_raw_control_characters(cls, text: str) -> str:

        if not text:
            return text
        result: list[str] = []
        in_string = False
        escaped = False
        for char in text:
            if not in_string:
                result.append(char)
                if char == '"':
                    in_string = True
                continue
            if escaped:
                result.append(char)
                escaped = False
                continue
            if char == "\\":
                result.append(char)
                escaped = True
                continue
            if char == '"':
                result.append(char)
                in_string = False
                continue
            replacement = cls._CONTROL_CHAR_ESCAPES.get(char)
            if replacement is not None:
                result.append(replacement)
            elif char < " ":
                result.append(f"\\u{ord(char):04x}")
            else:
                result.append(char)
        return "".join(result)

    @classmethod
    def _loads_lenient(cls, text: str) -> dict[str, Any] | None:

        if not text:
            return None
        for candidate in (text, cls.repair_raw_control_characters(text)):
            stripped = candidate.strip()
            if not stripped:
                continue
            try:
                decoded = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                decoded = None
            if isinstance(decoded, dict):
                return decoded
            for brace in range(len(stripped)):
                if stripped[brace] != "{":
                    continue
                try:
                    decoded, _end = json.JSONDecoder().raw_decode(stripped, brace)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(decoded, dict):
                    return decoded
        return None

    @classmethod
    def parse_json_object(cls, text: str, unwrap_key: str | None = None) -> dict[str, Any] | None:

        if not text or not text.strip():
            return None
        stripped = text.strip()
        candidates = list(reversed(cls._iter_json_blocks(text)))
        candidates.append(stripped)
        brace = stripped.find("{")
        if brace > 0:
            candidates.append(stripped[brace:])
        for candidate in candidates:
            data = cls._loads_lenient(candidate)
            if data is None:
                continue
            if unwrap_key and isinstance(data.get(unwrap_key), dict):
                return data[unwrap_key]
            return data
        return None

    @staticmethod
    def extract_json_block(text: str, unwrap_key: str | None = None) -> dict[str, Any] | None:

        if not text:
            return None
        for block in reversed(UnifiedParser._iter_json_blocks(text)):
            try:
                data = json.loads(block)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            if unwrap_key and isinstance(data.get(unwrap_key), dict):
                return data[unwrap_key]
            return data
        return None

    def extract_text_before_json(self, text: str) -> str:

        if not text:
            return ""
        start = text.lower().find("```json")
        if start != -1:
            return text[:start].strip()
        return text.strip()


    @staticmethod
    def format_error_response(error_message: str) -> str:

        json_fallback = {
            "analysis": {
                "summary": f"Analysis unavailable: {error_message}. Please try again later.",
            }
        }
        return f"```json\n{json.dumps(json_fallback, indent=2)}\n```\n\nThe analysis failed due to a technical issue. Please try again later."

    def extract_base_coin(self, symbol: str) -> str:

        if not symbol:
            return ""

        if "/" in symbol:
            return symbol.split("/")[0].upper()
        if "-" in symbol:
            return symbol.split("-")[0].upper()

        common_quotes = ["USDT", "USDC", "BUSD", "USD", "BTC", "ETH", "BNB"]
        symbol_upper = symbol.upper()

        for quote in common_quotes:
            if symbol_upper.endswith(quote):
                base = symbol_upper[:-len(quote)]
                if quote == "BUSD" and base == "BN":
                    continue
                return base

        return symbol_upper

    def detect_coins_in_text(self, text: str, known_tickers: set[str]) -> set[str]:

        if not text:
            return set()

        coins_mentioned = set()
        text_upper = text.upper()

        potential_tickers = set(re.findall(r"\b[A-Z]{2,6}\b", text_upper))

        for ticker in potential_tickers:
            if ticker in known_tickers:
                coins_mentioned.add(ticker)

        text_lower = text.lower()
        if "bitcoin" in text_lower:
            coins_mentioned.add("BTC")
        if "ethereum" in text_lower:
            coins_mentioned.add("ETH")

        return coins_mentioned


    def _normalize_numeric_fields(self, data: dict[str, Any]) -> dict[str, Any]:


        analysis = data.get("analysis", {})
        if not isinstance(analysis, dict):
            analysis = {}
            data["analysis"] = analysis
        for field, default_value in self._numeric_fields.items():
            if field in analysis:
                analysis[field] = self._parse_numeric_field(field, analysis[field], default_value)

        confluence_factors = analysis.get("confluence_factors", {})
        if isinstance(confluence_factors, dict):
            for factor_key in ["trend_alignment", "momentum_strength", "volume_support",
                              "pattern_quality", "support_resistance_strength"]:
                if factor_key in confluence_factors:
                    confluence_factors[factor_key] = self._parse_finite_number(
                        confluence_factors[factor_key], 50.0
                    )

        key_levels = analysis.get("key_levels", {})
        if isinstance(key_levels, dict):
            for level_type in ["support", "resistance"]:
                levels = key_levels.get(level_type, [])
                if isinstance(levels, list):
                    normalized_levels = []
                    for level in levels:
                        val = self._parse_finite_number(level, None)
                        if val is not None:
                            normalized_levels.append(val)
                    key_levels[level_type] = normalized_levels

        for field, default_value in self._numeric_fields.items():
            if field in data:
                data[field] = self._parse_numeric_field(field, data[field], default_value)

        return data

    def _parse_numeric_field(self, field: str, value: Any, default_value: Any) -> float | int | None:

        numeric_value = self._parse_finite_number(value, None)
        if numeric_value is None:
            return default_value
        if field == "position_size":
            if isinstance(value, str) and value.strip().endswith("%"):
                numeric_value = numeric_value / 100
            if not 0.0 <= numeric_value <= 1.0:
                self.logger.warning(
                    "position_size %s is outside the 0.0-1.0 contract; treating it as unset", value
                )
                return None
            return numeric_value
        if isinstance(default_value, int) and float(numeric_value).is_integer():
            return int(numeric_value)
        return numeric_value

    def _parse_finite_number(self, value: Any, default_value: Any) -> float | None:

        if isinstance(value, bool):
            return default_value
        if self.format_utils:
            parsed = self.format_utils.parse_value(value, default=None)
        elif isinstance(value, (int, float)):
            parsed = float(value)
        elif isinstance(value, str):
            try:
                parsed = float(value.strip().replace(",", ""))
            except ValueError:
                return default_value
        else:
            return default_value

        if parsed is None:
            return default_value
        try:
            numeric_value = float(parsed)
        except (TypeError, ValueError):
            return default_value
        return numeric_value if math.isfinite(numeric_value) else default_value

    def _attach_response_validation(self, data: dict[str, Any]) -> dict[str, Any]:

        data["response_validation"] = self.validate_trading_response(data)
        return data

    @staticmethod
    def _format_validation_errors(error: ValidationError) -> list[dict[str, str]]:

        formatted_errors = []
        for item in error.errors():
            field_path = ".".join(str(part) for part in item.get("loc", ()))
            formatted_errors.append({
                "field": field_path or "response",
                "message": str(item.get("msg", "Invalid value")),
                "type": str(item.get("type", "value_error")),
            })
        return formatted_errors

    def _create_validation_error_metadata(self, message: str, error_type: str) -> dict[str, Any]:

        return {
            "schema": TradingAnalysisResponseModel.schema_version,
            "status": "invalid",
            "valid": False,
            "errors": [{"field": "response", "message": message, "type": error_type}],
        }

    def _create_fallback_response(self, cleaned_text: str) -> dict[str, Any]:

        return {
            "analysis": {
                "summary": "Unable to parse the AI response. The analysis may have been in an invalid format.",
                "observed_trend": "NEUTRAL",
                "trend_strength": 50,
                "confidence_score": 0
            },
            "raw_response": cleaned_text,
            "parse_error": "Failed to parse response",
            "response_validation": self._create_validation_error_metadata(
                "Failed to parse response JSON",
                "json_parse_error"
            )
        }

    def _create_error_response(self, error_message: str, raw_text: str) -> dict[str, Any]:

        return {
            "error": error_message,
            "raw_response": raw_text,
            "analysis": {
                "summary": "Error parsing response",
                "observed_trend": "NEUTRAL",
                "confidence_score": 0
            },
            "response_validation": self._create_validation_error_metadata(error_message, "parser_error")
        }

