"""Read trading fields from the parsed AI analysis payload."""

from typing import Any


class PositionExtractor:
    """Reads signal, stop loss, take profit and sizing from a parsed analysis payload.

    The payload is produced ONCE per cycle by ``UnifiedParser.parse_ai_response``
    (numeric fields normalized, NaN/Infinity rejected, JSON contract validated), so
    this class never re-parses the raw response: one parser, one normalization.
    """

    VALID_SIGNALS = frozenset(
        {"BUY", "SELL", "LONG", "SHORT", "HOLD", "CLOSE", "CLOSE_LONG", "CLOSE_SHORT", "UPDATE"}
    )

    def extract_trading_info(
        self, analysis: dict[str, Any]
    ) -> tuple[str, str, float | None, float | None, float | None, str]:
        """Extract trading information from a parsed analysis payload.

        Args:
            analysis: The ``analysis`` object of a UnifiedParser parse result

        Returns: tuple of (signal, confidence, stop_loss, take_profit, position_size, reasoning)
        """
        signal = str(analysis.get("signal", "HOLD")).upper()
        confidence = self._confidence_label(analysis.get("confidence", "MEDIUM"))
        reasoning = str(analysis.get("reasoning", "") or "")

        return (
            signal,
            confidence,
            analysis.get("stop_loss"),
            analysis.get("take_profit"),
            analysis.get("position_size"),
            reasoning,
        )

    def _confidence_label(self, value: Any) -> str:
        """Map a numeric confidence (0-100) or an already-labelled string to HIGH/MEDIUM/LOW."""
        if isinstance(value, bool):
            return "MEDIUM"
        if isinstance(value, (int, float)):
            return self._numeric_to_confidence_string(float(value))
        return str(value).upper()

    @staticmethod
    def _numeric_to_confidence_string(confidence: float) -> str:
        """Convert numeric confidence (0-100) to string (HIGH/MEDIUM/LOW)."""
        if confidence >= 70:
            return "HIGH"
        if confidence >= 50:
            return "MEDIUM"
        return "LOW"

    def validate_signal(self, signal: str) -> bool:
        """Validate if signal is a recognized trading action.

        Args:
            signal: Trading signal to validate

        Returns:
            True if valid signal
        """
        return signal.upper() in self.VALID_SIGNALS
