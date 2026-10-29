

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

        if isinstance(value, bool):
            return "MEDIUM"
        if isinstance(value, (int, float)):
            return self._numeric_to_confidence_string(float(value))
        return str(value).upper()

    @staticmethod
    def _numeric_to_confidence_string(confidence: float) -> str:

        if confidence >= 70:
            return "HIGH"
        if confidence >= 50:
            return "MEDIUM"
        return "LOW"

    def validate_signal(self, signal: str) -> bool:

        return signal.upper() in self.VALID_SIGNALS
