"""Extract trading signals from AI responses."""

import math
import re
from re import Pattern
from typing import TYPE_CHECKING, Any

from src.logger.logger import Logger

if TYPE_CHECKING:
    from src.parsing.unified_parser import UnifiedParser


class PositionExtractor:
    """Extracts trading signals, stop loss, take profit from AI responses."""

    def __init__(self, logger: Logger | None = None, unified_parser: "UnifiedParser" = None):  # type: ignore
        """Initialize the position extractor.

        Args:
            logger: Optional logger instance
            unified_parser: UnifiedParser for JSON extraction (DRY)
        """
        self.logger = logger  # type: ignore[reportOptionalMemberAccess]
        self.unified_parser = unified_parser

        # Regex patterns for extracting trading information
        self.signal_pattern: Pattern = re.compile(
            r'signal["\s:]*\[?(BUY|SELL|LONG|SHORT|HOLD|CLOSE|CLOSE_LONG|CLOSE_SHORT|UPDATE)\]?',
            re.IGNORECASE
        )
        self.confidence_pattern: Pattern = re.compile(
            r'confidence["\s:]*\[?(HIGH|MEDIUM|LOW)\]?',
            re.IGNORECASE
        )
        self.stop_loss_pattern: Pattern = re.compile(  # type: ignore[reportOptionalMemberAccess]
            r'stop[_\s]?loss["\s:]*\$?([0-9,]+(?:\.[0-9]+)?)',
            re.IGNORECASE
        )
        self.take_profit_pattern: Pattern = re.compile(  # type: ignore[reportOptionalMemberAccess]
            r'take[_\s]?profit["\s:]*\$?([0-9,]+(?:\.[0-9]+)?)',
            re.IGNORECASE
        )
        self.position_size_pattern: Pattern = re.compile(
            r'position[_\s]?size["\s:]*\[?([0-9]+(?:\.[0-9]+)?)(?:\s*(%))?\]?',
            re.IGNORECASE
        )
        self.reasoning_pattern: Pattern = re.compile(
            r'reasoning["\s:]*["\']?([^"\'}\]]+)["\']?',
            re.IGNORECASE
        )

    def extract_from_json(self, text: str) -> dict[str, Any] | None:
        """Try to extract trading decision from JSON in response.

        Uses UnifiedParser.extract_json_block() for JSON extraction (DRY).

        Args:
            text: AI response text

        Returns:
            Extracted JSON trading decision or None
        """
        if not self.unified_parser:
            if self.logger:  # type: ignore[reportOptionalMemberAccess]
                self.logger.warning("No UnifiedParser provided, cannot extract JSON")  # type: ignore[reportOptionalMemberAccess]
            return None

        # Bolt: extract JSON block once and unwrap key candidates in memory instead of parsing string 4 times
        data = self.unified_parser.extract_json_block(text)
        if not data or not isinstance(data, dict):
            return data

        for key in ("analysis", "trading_decision", "decision"):
            val = data.get(key)
            if isinstance(val, dict):
                return val

        return data

    def extract_trading_info(self, text: str) -> tuple[str, str, float | None, float | None, float | None, str]:
        """Extract trading information from AI response.

        Args:
            text: AI response text

        Returns: tuple of (signal, confidence, stop_loss, take_profit, position_size, reasoning)
        """
        # First try JSON extraction
        json_data = self.extract_from_json(text)
        if json_data:
            return self._extract_from_dict(json_data)

        # Fall back to regex extraction
        return self._extract_from_text(text)

    def _extract_from_dict(self, data: dict[str, Any]) -> tuple[str, str, float | None, float | None, float | None, str]:
        """Extract trading info from a dictionary.

        Args:
            data: Dictionary containing trading decision

        Returns: tuple of (signal, confidence, stop_loss, take_profit, position_size, reasoning)
        """
        signal = str(data.get("signal", data.get("action", "HOLD"))).upper()

        # Confidence can be numeric (0-100) or string (HIGH/MEDIUM/LOW)
        confidence_raw = data.get("confidence", "MEDIUM")
        if isinstance(confidence_raw, (int, float)):
            # Convert numeric confidence to string
            confidence = self._numeric_to_confidence_string(confidence_raw)
        else:
            confidence = str(confidence_raw).upper()

        stop_loss = data.get("stop_loss")
        if stop_loss is not None:
            stop_loss = self._parse_finite_float(stop_loss)

        take_profit = data.get("take_profit")
        if take_profit is not None:
            take_profit = self._parse_finite_float(take_profit)

        position_size = data.get("position_size")
        if position_size is not None:
            position_size = self._normalize_position_size(position_size)

        reasoning = str(data.get("reasoning", data.get("rationale", "")))

        return signal, confidence, stop_loss, take_profit, position_size, reasoning

    def _numeric_to_confidence_string(self, confidence: float) -> str:
        """Convert numeric confidence (0-100) to string (HIGH/MEDIUM/LOW).

        Args:
            confidence: Numeric confidence 0-100

        Returns:
            String confidence level
        """
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            return "MEDIUM"
        if not math.isfinite(confidence):
            return "MEDIUM"
        if confidence >= 70:
            return "HIGH"
        if confidence >= 50:
            return "MEDIUM"
        return "LOW"

    def _parse_finite_float(self, value: Any) -> float | None:
        """Parse a trade numeric field and reject NaN/Infinity payloads."""
        try:
            numeric_value = float(str(value).replace(",", "").replace("$", ""))
        except (TypeError, ValueError):
            if self.logger:  # type: ignore[reportOptionalMemberAccess]
                self.logger.warning("Invalid numeric value from AI response: %s", value)  # type: ignore[reportOptionalMemberAccess]
            return None
        if not math.isfinite(numeric_value):
            if self.logger:  # type: ignore[reportOptionalMemberAccess]
                self.logger.warning("Non-finite numeric value from AI response: %s", value)  # type: ignore[reportOptionalMemberAccess]
            return None
        return numeric_value

    def _normalize_position_size(self, value: Any, explicit_percent: bool = False) -> float | None:
        """Normalize position size values to decimal capital fractions."""
        value_text = str(value).strip()
        if not value_text:
            return None

        has_percent = explicit_percent or value_text.endswith("%")
        try:
            numeric_value = float(value_text.replace("%", "").replace(",", ""))
        except (TypeError, ValueError):
            if self.logger:  # type: ignore[reportOptionalMemberAccess]
                self.logger.warning("Invalid position_size value from AI response: %s", value)  # type: ignore[reportOptionalMemberAccess]
            return None
        if not math.isfinite(numeric_value) or numeric_value < 0:
            if self.logger:  # type: ignore[reportOptionalMemberAccess]
                self.logger.warning("Invalid position_size value from AI response: %s", value)  # type: ignore[reportOptionalMemberAccess]
            return None

        if has_percent:
            return numeric_value / 100
        # Without % suffix, numbers > 1.0 are ambiguous:
        # "50" could mean "50% of capital" or "50x leverage".
        # Reject — require explicit % suffix.
        if numeric_value > 1.0:
            if self.logger:  # type: ignore[reportOptionalMemberAccess]
                self.logger.warning(  # type: ignore[reportOptionalMemberAccess]
                    "Ambiguous position_size without %% suffix: %s — "
                    "treating as None (rejected). "
                    "LLM should use %% suffix for percentage values "
                    "(e.g., '5%%').",
                    value,
                )
            return None
        return numeric_value

    def _extract_from_text(self, text: str) -> tuple[str, str, float | None, float | None, float | None, str]:
        """Extract trading info using regex patterns.

        Args:
            text: Raw text response

        Returns: tuple of (signal, confidence, stop_loss, take_profit, position_size, reasoning)
        """
        signal_match = self.signal_pattern.search(text)
        signal = signal_match.group(1).upper() if signal_match else "HOLD"

        confidence_match = self.confidence_pattern.search(text)
        confidence = confidence_match.group(1).upper() if confidence_match else "MEDIUM"

        stop_loss_match = self.stop_loss_pattern.search(text)  # type: ignore[reportOptionalMemberAccess]
        stop_loss = float(stop_loss_match.group(1).replace(",", "")) if stop_loss_match else None

        take_profit_match = self.take_profit_pattern.search(text)  # type: ignore[reportOptionalMemberAccess]
        take_profit = float(take_profit_match.group(1).replace(",", "")) if take_profit_match else None

        position_size_match = self.position_size_pattern.search(text)
        position_size = None
        if position_size_match:
            position_size = self._normalize_position_size(
                position_size_match.group(1),
                explicit_percent=position_size_match.group(2) is not None,
            )

        reasoning_match = self.reasoning_pattern.search(text)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        return signal, confidence, stop_loss, take_profit, position_size, reasoning

    def validate_signal(self, signal: str) -> bool:
        """Validate if signal is a recognized trading action.

        Args:
            signal: Trading signal to validate

        Returns:
            True if valid signal
        """
        valid_signals = {"BUY", "SELL", "LONG", "SHORT", "HOLD", "CLOSE", "CLOSE_LONG", "CLOSE_SHORT", "UPDATE"}
        return signal.upper() in valid_signals
