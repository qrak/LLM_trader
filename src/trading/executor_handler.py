"""ExecutorHandler — builds and forwards trading decision payloads to
the llm_trader_executor service.

Single responsibility: translate a strategy decision into the executor's
wire format, persist it as a file fallback, and HTTP-forward it.
"""

from __future__ import annotations

import asyncio
import json
import math
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
        self._http_client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Bolt: reuse persistent httpx.AsyncClient to keep TCP connection pools alive and avoid per-request setup overhead."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=EXECUTOR_HTTP_TIMEOUT)
        return self._http_client

    async def close(self) -> None:
        """Close persistent HTTP client resources."""
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    # ── public API ────────────────────────────────────────────────────────

    async def handle(
        self,
        analysis: dict[str, Any],
        strategy_decision: TradeDecision | None,
        symbol: str,
    ) -> bool:
        """Full pipeline: build payload, persist to file, HTTP-forward.

        When ``strategy_decision.action == "HOLD"`` the payload is suppressed
        entirely — no file write, no HTTP call.

        When the executor is disabled in config (``EXECUTOR_API_ENABLED=False``)
        nothing is produced at all — no HTTP forward AND no file fallback. A
        live executor process polls ``latest_decision.json``; writing the
        fallback while "disabled" would still deliver decisions to it.

        Returns:
            True when the payload was delivered via the HTTP fast path
            (executor reachable and queued). False when it was suppressed,
            written to the file fallback, or dead-lettered — callers must
            NOT treat a False as "order will never execute".
        """
        if not self._config.EXECUTOR_API_ENABLED:
            return False
        payload = self._build(analysis, strategy_decision, symbol)
        if payload is None:
            return False

        forward_success = False
        try:
            forward_success = await self._forward(payload)
        except Exception:  # noqa: BLE001
            # @retry_async exhausted all retries — executor is unreachable.
            # Write to dead-letter so we can replay later.
            signal = payload.get("signal")
            symbol_name = payload.get("symbol")
            self.logger.error(
                "Executor forward failed after all retries for %s %s — "
                "writing to dead-letter",
                signal, symbol_name,
            )
            await self._write_dead_letter(payload)
            forward_success = False

        # Bolt: skip disk file write when HTTP primary path succeeds (executor reachable)
        if not forward_success:
            self._persist(payload)
        else:
            # HTTP path delivered the decision — remove any stale fallback file
            # so the executor's file poller doesn't re-read an old decision on
            # every restart ("Already executed (duplicate content hash)").
            try:
                self._persistence.clear_latest_decision()
            except Exception:
                self.logger.warning(
                    "Failed to clear stale latest_decision.json after successful forward",
                    exc_info=True,
                )
        return forward_success

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
        raw_qty = strategy_decision.quantity if strategy_decision.quantity is not None else analysis.get("quantity", 0.0)
        raw_price = strategy_decision.price if strategy_decision.price is not None else analysis.get("entry_price")
        raw_sl = strategy_decision.stop_loss if strategy_decision.stop_loss is not None else analysis.get("stop_loss")
        raw_tp = strategy_decision.take_profit if strategy_decision.take_profit is not None else analysis.get("take_profit")

        def _to_float(val: Any, default: float | None = None) -> float | None:
            if val is None:
                return default
            try:
                f_val = float(val)
                return f_val if math.isfinite(f_val) else default
            except (ValueError, TypeError):
                return default

        def _to_int(val: Any, default: int = 1) -> int:
            """Parse an LLM-supplied integer safely; never raises.

            LLMs emit ``null``, ``"5x"``, ``NaN`` or floats for leverage —
            none of those may crash order forwarding with a TypeError.
            """
            if val is None:
                return default
            try:
                f_val = float(val)
                if not math.isfinite(f_val) or f_val < 1:
                    return default
                return int(f_val)
            except (ValueError, TypeError):
                return default

        def _to_bool(val: Any, default: bool = False) -> bool:
            """Parse an LLM-supplied boolean safely.

            LLMs frequently emit string booleans (``"false"``), which
            ``bool("false")`` would wrongly coerce to True and flip order
            semantics (e.g. a BUY forwarded as reduce-only). Only actual
            truthy tokens are accepted.
            """
            if isinstance(val, bool):
                return val
            if val is None:
                return default
            if isinstance(val, (int, float)):
                return val != 0
            return str(val).strip().lower() in ("true", "1", "yes", "on")

        return {
            "timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            ),
            "symbol": str(symbol),
            "signal": str(signal),
            "order_type": str(self._config.ENTRY_ORDER_TYPE),
            "quantity": _to_float(raw_qty, 0.0) or 0.0,
            "entry_price": _to_float(raw_price),
            "stop_loss": _to_float(raw_sl),
            "take_profit": _to_float(raw_tp),
            "reduce_only": _to_bool(analysis.get("reduce_only", False)),
            "leverage": _to_int(analysis.get("leverage", 1)),
            "confidence": str(strategy_decision.confidence) if strategy_decision.confidence else "MEDIUM",
            "reasoning": str(strategy_decision.reasoning or ""),
        }

    def _persist(self, payload: dict[str, Any]) -> None:
        """Atomically write payload to latest_decision.json (file fallback)."""
        try:
            self._persistence.save_latest_decision(payload)
        except Exception:
            self.logger.error(  # noqa: G201
                "Failed to persist latest_decision.json for %s %s",
                payload.get("signal"),
                payload.get("symbol"),
                exc_info=True,
            )

    @retry_async(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def _forward(self, payload: dict[str, Any]) -> bool:
        """POST the payload to the llm_trader_executor HTTP endpoint.

        Connection failures, timeouts, and network blips are retried
        automatically by @retry_async.  Non-200 responses are NOT retried
        (re-sending a rejected payload won't help).
        """
        if not self._config.EXECUTOR_API_ENABLED:
            return False
        url: str = self._config.EXECUTOR_API_URL
        if not url:
            self.logger.warning(
                "EXECUTOR_API_ENABLED but EXECUTOR_API_URL is empty"
            )
            return False
        signal = payload.get("signal")
        symbol = payload.get("symbol")

        client = self._get_client()
        resp = await client.post(url, json=payload)

        if resp.status_code == 200:
            self.logger.info("Executor queued: %s %s", signal, symbol)
            # Executor is reachable — replay any previously failed forwards
            await self._replay_dead_letters()
            return True

        self.logger.warning(
            "Executor returned %s for %s %s: %s",
            resp.status_code, signal, symbol, resp.text,
        )
        return False

    async def _write_dead_letter(self, payload: dict[str, Any]) -> None:
        """Append a failed forward to the dead-letter journal (off the event loop)."""
        try:
            await asyncio.to_thread(self._write_dead_letter_sync, payload)
        except Exception:
            self.logger.error("Failed to write dead-letter entry", exc_info=True)  # noqa: G201

    def _write_dead_letter_sync(self, payload: dict[str, Any]) -> None:
        """Synchronous dead-letter append (runs in a worker thread)."""
        DEAD_LETTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DEAD_LETTER_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    async def _replay_dead_letters(self) -> None:
        """Replay previously failed forwards now that executor is reachable.

        Entries that fail to replay are KEPT in the journal — the file is
        only deleted once every entry has been acknowledged by the executor.
        A transient replay failure must never silently drop a trade signal
        (a lost CLOSE would strand an unprotected live position).
        """
        url: str = self._config.EXECUTOR_API_URL
        if not url:
            return
        try:
            entries = await asyncio.to_thread(self._read_dead_letters)
            if not entries:
                return

            self.logger.info("Replaying %d dead-letter entries...", len(entries))
            client = self._get_client()
            failed: list[dict[str, Any]] = []
            for entry in entries:
                sig = entry.get("signal")
                sym = entry.get("symbol")
                try:
                    r = await client.post(url, json=entry)
                    if r.status_code == 200:
                        self.logger.info(
                            "Dead-letter replayed: %s %s", sig, sym,
                        )
                    else:
                        self.logger.error(
                            "Dead-letter replay failed (%s): %s %s — keeping entry",
                            r.status_code, sig, sym,
                        )
                        failed.append(entry)
                except Exception:
                    self.logger.exception(
                        "Dead-letter replay exception: %s %s — keeping entry",
                        sig, sym,
                    )
                    failed.append(entry)

            if failed:
                await asyncio.to_thread(self._write_dead_letter_batch, failed)
                self.logger.error(
                    "Dead-letter: %d/%d entries could not be replayed and were kept",
                    len(failed), len(entries),
                )
            else:
                await asyncio.to_thread(self._delete_dead_letters)
        except Exception:
            self.logger.error("Dead-letter replay failed", exc_info=True)  # noqa: G201

    def _read_dead_letters(self) -> list[dict[str, Any]]:
        """Read all journal entries (synchronous, runs in a worker thread)."""
        if not DEAD_LETTER_PATH.exists():
            return []
        entries: list[dict[str, Any]] = []
        with open(DEAD_LETTER_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def _write_dead_letter_batch(self, entries: list[dict[str, Any]]) -> None:
        """Rewrite the journal with only the entries that still need replaying."""
        DEAD_LETTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DEAD_LETTER_PATH, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(entry) + "\n" for entry in entries)

    def _delete_dead_letters(self) -> None:
        """Remove the journal once every entry has been replayed."""
        DEAD_LETTER_PATH.unlink(missing_ok=True)

