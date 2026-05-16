"""Protocol definition for RiskManager interface"""

from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from src.trading.data_models import RiskAssessment, MarketConditions

class RiskManagerProtocol(Protocol):
    """
    Protocol defining the interface for Risk Management.
    """

    def calculate_entry_parameters(
        self,
        signal: str,
        current_price: float,
        capital: float,
        confidence: str,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        position_size: float | None = None,
        market_conditions: "MarketConditions | None" = None
    ) -> "RiskAssessment":
        """
        Calculate all risk parameters for a new position entry.

        Returns:
            RiskAssessment dataclass with calculated SL, TP, size, quantity, fees, etc.
        """
        ...

    def validate_signal(self, signal: str) -> bool:
        """Validate if a signal is actionable."""
        ...

    def get_and_clear_frictions(self) -> list[dict[str, Any]]:
        """Return accumulated risk frictions and clear the buffer."""
        ...
