"""ExecutorHandler — builds and forwards trading decision payloads to
the llm_trader_executor service.

Single responsibility: translate a strategy decision into the executor's
wire format, persist it as a file fallback, and HTTP-forward it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import httpx

from src.logger.logger import Logger
from .data_models import TradeDecision

if TYPE_CHECKING:
    from src.managers.persistence_manager import PersistenceManager


ACTIONABLE_SIGNALS = frozenset({"BUY", "SELL", "CLOSE", "UPDATE"})
EXECUTOR_HTTP_TIMEOUT = 5.0


class ExecutorHandler:
    """Builds, persists, and forwards decision payloads to llm_trader_executor."""

    def __init__(
        self,
        persistence: PersistenceManager,
        config: Any,
        logger: Logger,
    ) -> None:
        self._persistence = persistence
        self._config = config
        self._logger = logger

    # ── public API ────────────────────────────────────────────────────────

    async def handle(
        self,
        analysis: dict[str, Any],
        strategy_decision: TradeDecision | None,
        symbol: str,
    ) -> None:
        """Full pipeline: build payload, persist to file, HTTP-forward.

        When ``strategy_decision.action == "HOLD"`` the payload is suppressed
        entirely — no file write, no HTTP call.
        """
        payload = self._build(analysis, strategy_decision, symbol)
        if payload is None:
            return
        self._persist(payload)
        await self._forward(payload)

    # ── internal ──────────────────────────────────────────────────────────

    def _build(
        self,
        analysis: dict[str, Any],
        strategy_decision: TradeDecision | None,
        symbol: str,
    ) -> dict[str, Any] | None:
        """Produce a CCXT-ready payload dict, or None if suppressed."""
        if not analysis or not symbol:
            return None
        signal = analysis.get("signal")
        if signal not in ACTIONABLE_SIGNALS:
            return None
        if strategy_decision is not None and strategy_decision.action == "HOLD":
            return None
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "signal": signal,
            "order_type": analysis.get("order_type", "limit"),
            "quantity": analysis.get("quantity", 0.0),
            "entry_price": analysis.get("entry_price"),
            "stop_loss": analysis.get("stop_loss"),
            "take_profit": analysis.get("take_profit"),
            "reduce_only": analysis.get("reduce_only", False),
            "leverage": analysis.get("leverage", 1),
            "confidence": analysis.get("confidence"),
            "reasoning": analysis.get("reasoning", ""),
        }

    def _persist(self, payload: dict[str, Any]) -> None:
        """Atomically write payload to latest_decision.json (file fallback)."""
        try:
            self._persistence.save_latest_decision(payload)
        except Exception:
            self._logger.error(
                "Failed to persist latest_decision.json for %s %s",
                payload.get("signal"),
                payload.get("symbol"),
                exc_info=True,
            )

    async def _forward(self, payload: dict[str, Any]) -> None:
        """POST the payload to the llm_trader_executor HTTP endpoint."""
        if not self._config.EXECUTOR_API_ENABLED:
            return
        url: str = self._config.EXECUTOR_API_URL
        if not url:
            self._logger.warning(
                "EXECUTOR_API_ENABLED but EXECUTOR_API_URL is empty"
            )
            return
        signal = payload.get("signal")
        symbol = payload.get("symbol")
        try:
            async with httpx.AsyncClient(timeout=EXECUTOR_HTTP_TIMEOUT) as client:
                resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                self._logger.info("Executor queued: %s %s", signal, symbol)
            else:
                self._logger.warning(
                    "Executor returned %s for %s %s: %s",
                    resp.status_code, signal, symbol, resp.text,
                )
        except Exception:
            self._logger.error(
                "Executor forward failed for %s %s (file fallback exists)",
                signal, symbol, exc_info=True,
            )
