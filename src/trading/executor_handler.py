"""ExecutorHandler — builds and forwards trading decision payloads to
the llm_trader_executor service.

Single responsibility: translate a strategy decision into the executor's
wire format, persist it as a file fallback, and HTTP-forward it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from src.logger.logger import Logger
from src.utils.decorators import retry_async
from .data_models import TradeDecision

if TYPE_CHECKING:
    from src.managers.persistence_manager import PersistenceManager


ACTIONABLE_SIGNALS = frozenset({"BUY", "SELL", "CLOSE", "UPDATE", "LONG", "SHORT"})
EXECUTOR_HTTP_TIMEOUT = 5.0
DEAD_LETTER_PATH = Path("data/trading/failed_forwards.jsonl")


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
        self.logger = logger  # named 'logger' for @retry_async convention

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
        try:
            await self._forward(payload)
        except Exception:
            # @retry_async exhausted all retries — executor is unreachable.
            # Write to dead-letter so we can replay later.
            signal = payload.get("signal")
            symbol_name = payload.get("symbol")
            self.logger.error(
                "Executor forward failed after all retries for %s %s — "
                "writing to dead-letter",
                signal, symbol_name,
            )
            self._write_dead_letter(payload)

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
        # NEVER forward when strategy_decision is None — the analysis was
        # never validated by TradingStrategy (processing error, guard failure, etc.).
        if strategy_decision is None:
            return None
        signal = analysis.get("signal")
        if signal not in ACTIONABLE_SIGNALS:
            return None
        if strategy_decision.action == "HOLD":
            return None
        return {
            "timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            ),
            "symbol": symbol,
            "signal": signal,
            "order_type": self._config.ENTRY_ORDER_TYPE,
            "quantity": (
                strategy_decision.quantity
                or analysis.get("quantity", 0.0)
            ),
            "entry_price": strategy_decision.price or analysis.get("entry_price"),
            "stop_loss": strategy_decision.stop_loss or analysis.get("stop_loss"),
            "take_profit": strategy_decision.take_profit or analysis.get("take_profit"),
            "reduce_only": analysis.get("reduce_only", False),
            "leverage": analysis.get("leverage", 1),
            "confidence": strategy_decision.confidence,
            "reasoning": strategy_decision.reasoning or "",
        }

    def _persist(self, payload: dict[str, Any]) -> None:
        """Atomically write payload to latest_decision.json (file fallback)."""
        try:
            self._persistence.save_latest_decision(payload)
        except Exception:
            self.logger.error(
                "Failed to persist latest_decision.json for %s %s",
                payload.get("signal"),
                payload.get("symbol"),
                exc_info=True,
            )

    @retry_async(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def _forward(self, payload: dict[str, Any]) -> None:
        """POST the payload to the llm_trader_executor HTTP endpoint.

        Connection failures, timeouts, and network blips are retried
        automatically by @retry_async.  Non-200 responses are NOT retried
        (re-sending a rejected payload won't help).
        """
        if not self._config.EXECUTOR_API_ENABLED:
            return
        url: str = self._config.EXECUTOR_API_URL
        if not url:
            self.logger.warning(
                "EXECUTOR_API_ENABLED but EXECUTOR_API_URL is empty"
            )
            return
        signal = payload.get("signal")
        symbol = payload.get("symbol")

        async with httpx.AsyncClient(timeout=EXECUTOR_HTTP_TIMEOUT) as client:
            resp = await client.post(url, json=payload)

        if resp.status_code == 200:
            self.logger.info("Executor queued: %s %s", signal, symbol)
            # Executor is reachable — replay any previously failed forwards
            await self._replay_dead_letters()
        else:
            self.logger.warning(
                "Executor returned %s for %s %s: %s",
                resp.status_code, signal, symbol, resp.text,
            )

    def _write_dead_letter(self, payload: dict[str, Any]) -> None:
        """Append a failed forward to the dead-letter journal."""
        try:
            DEAD_LETTER_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(DEAD_LETTER_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            self.logger.error("Failed to write dead-letter entry", exc_info=True)

    async def _replay_dead_letters(self) -> None:
        """Replay previously failed forwards now that executor is reachable."""
        if not DEAD_LETTER_PATH.exists():
            return
        url: str = self._config.EXECUTOR_API_URL
        if not url:
            return
        try:
            entries: list[dict[str, Any]] = []
            with open(DEAD_LETTER_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))

            if not entries:
                return

            self.logger.info("Replaying %d dead-letter entries...", len(entries))
            for entry in entries:
                sig = entry.get("signal")
                sym = entry.get("symbol")
                try:
                    async with httpx.AsyncClient(
                        timeout=EXECUTOR_HTTP_TIMEOUT
                    ) as client:
                        r = await client.post(url, json=entry)
                    if r.status_code == 200:
                        self.logger.info(
                            "Dead-letter replayed: %s %s", sig, sym,
                        )
                    else:
                        self.logger.warning(
                            "Dead-letter replay failed (%s): %s %s",
                            r.status_code, sig, sym,
                        )
                except Exception:
                    self.logger.warning(
                        "Dead-letter replay exception: %s %s", sig, sym,
                    )

            DEAD_LETTER_PATH.unlink(missing_ok=True)
        except Exception:
            self.logger.error("Dead-letter replay failed", exc_info=True)
