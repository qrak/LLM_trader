"""Guard that restricts orders to the configured trading pair."""

from __future__ import annotations

from . import GuardResult


class ConfiguredSymbolGuard:
    """Reject intents for symbols other than the configured trading pair."""

    name = "configured_symbol"

    def check(self, intent, /, *, capital: float, config) -> GuardResult:
        symbol = intent.symbol
        configured_symbol = config.CRYPTO_PAIR

        if symbol != configured_symbol:
            return GuardResult(
                guard_name=self.name,
                passed=False,
                reason=f"Symbol '{symbol}' does not match configured trading pair '{configured_symbol}'",
                metadata={"symbol": symbol, "configured_symbol": configured_symbol},
            )

        return GuardResult(
            guard_name=self.name,
            passed=True,
            reason=f"Symbol '{symbol}' matches configured trading pair",
            metadata={"symbol": symbol, "configured_symbol": configured_symbol},
        )
