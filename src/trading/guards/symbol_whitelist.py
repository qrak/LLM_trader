"""Symbol Whitelist Guard — ensures the trading symbol is in the allowlist."""

from __future__ import annotations

from . import GuardResult


class SymbolWhitelistGuard:
    """Reject intents for symbols not present in the configured whitelist.

    By default, the bot only trades its configured crypto_pair. Additional
    symbols can be added to the config as a comma-separated whitelist.
    """

    name = "symbol_whitelist"

    def check(self, intent, /, *, capital: float, config) -> GuardResult:
        symbol = intent.symbol

        # Primary: the configured trading pair
        allowed = {config.CRYPTO_PAIR}

        extra_symbols = config.SYMBOL_WHITELIST
        if extra_symbols:
            allowed.update(extra_symbols)

        if symbol not in allowed:
            return GuardResult(
                guard_name=self.name,
                passed=False,
                reason=f"Symbol '{symbol}' not in whitelist: {sorted(allowed)}",
                metadata={
                    "symbol": symbol,
                    "whitelist": sorted(allowed),
                },
            )

        return GuardResult(
            guard_name=self.name,
            passed=True,
            reason=f"Symbol '{symbol}' is in whitelist",
            metadata={"symbol": symbol, "whitelist": sorted(allowed)},
        )
