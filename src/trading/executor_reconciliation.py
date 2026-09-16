"""Executor reconciliation for the trading strategy.

Verifies that an entry the bot recorded locally actually reached the executor, correlates
per-order outcomes through the executor's verdict journal, and rolls back phantom positions
the executor silently blocked. Split out of trading_strategy.py; the methods use
``self`` state provided by TradingStrategy at MRO resolution time.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data_models import Position, TradeDecision

ENTRY_CONFIRM_ATTEMPTS = 10
ENTRY_CONFIRM_DELAY = 2.5
ENTRY_CONFIRM_MIN_FALSE_REPORTS = 6


class ExecutorReconciliationMixin:
    """Executor position verification, verdict correlation and ghost rollback."""

    logger: Any
    config: Any
    persistence: Any
    current_position: Position | None
    _http_client: Any
    _record_trade_decision: Any

    def _get_http_client(self):
        """Reuse one httpx client across executor position queries (keeps TCP pools alive)."""
        if self._http_client is None or self._http_client.is_closed:
            import httpx
            self._http_client = httpx.AsyncClient(timeout=3.0)
        return self._http_client

    async def close(self) -> None:
        """Close persistent HTTP client resources."""
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    async def _executor_has_position(self, symbol: str) -> bool | None:
        """Query executor API to confirm the position is actually open on exchange.

        Returns:
            True if position is open (or executor API disabled).
            False if executor explicitly returned open: False.
            None if query failed (network error, timeout, HTTP error).
        """
        if not self.config.EXECUTOR_API_ENABLED:
            return True
        url: str = self.config.EXECUTOR_API_URL
        if not url:
            return True
        try:
            base = url.rstrip("/").removesuffix("/decision")
            pos_url = base + "/position"
            client = self._get_http_client()
            resp = await client.get(pos_url, params={"symbol": symbol})
            if resp.status_code == 200:
                data = resp.json()
                return bool(data.get("open", False))
            self.logger.warning(
                "Executor returned HTTP %s for position query: %s",
                resp.status_code, resp.text,
            )
            return None
        except Exception:
            self.logger.error(# noqa: G201
                "CRITICAL: Failed to query executor position for %s — "
                "cannot verify position state.",
                symbol, exc_info=True,
            )
            return None

    async def confirm_entry_with_executor(self, symbol: str, order_id: str | None = None) -> bool:
        """Confirm a forwarded entry executed, using the verdict journal.

        The executor appends one verdict line per processed decision, keyed by
        the bot's ``order_id`` (written on /decision → queue → main loop →
        SafetyGuard / execution). Polling this journal answers "what happened
        to MY order" definitively — unlike polling /position, which only says
        whether ANY position exists and races the executor's 10s queue tick.

        Returns:
            True if the executor reports ``executed`` (or the journal is
            unreadable/absent — fail-open: never roll back a possibly-live
            order because a log file hiccuped).
            False only when the executor explicitly recorded ``blocked`` or
            ``error`` for THIS order_id.
        """
        if not order_id:
            false_reports = 0
            polls = 0
            for _ in range(ENTRY_CONFIRM_ATTEMPTS):
                polls += 1
                state = await self._executor_has_position(symbol)
                if state is True:
                    return True
                if state is False:
                    false_reports += 1
                    if false_reports >= ENTRY_CONFIRM_MIN_FALSE_REPORTS:
                        break
                await asyncio.sleep(ENTRY_CONFIRM_DELAY)
            if false_reports >= ENTRY_CONFIRM_MIN_FALSE_REPORTS:
                self.logger.warning(
                    "Executor reports no position for %s after %d polls — entry was likely blocked",
                    symbol, polls,
                )
                return False
            self.logger.warning(
                "Could not verify executor position for %s after %d polls — "
                "keeping local position (fail-open)",
                symbol, polls,
            )
            return True

        for _ in range(ENTRY_CONFIRM_ATTEMPTS):
            verdict = self._read_executor_verdict(order_id)
            if verdict == "executed":
                return True
            if verdict in ("blocked", "error"):
                self.logger.warning(
                    "Executor verdict for %s: %s — entry was %s",
                    order_id, verdict,
                    "blocked" if verdict == "blocked" else "rejected with error",
                )
                return False
            await asyncio.sleep(ENTRY_CONFIRM_DELAY)

        self.logger.warning(
            "No executor verdict for %s after %d polls — keeping local position (fail-open)",
            order_id, ENTRY_CONFIRM_ATTEMPTS,
        )
        return True

    def _read_executor_verdict(self, order_id: str) -> str | None:
        """Read the executor's verdict journal for one order_id.

        Returns ``"executed"`` / ``"blocked"`` / ``"error"``, or None when the
        journal has no entry for this order yet (or is unreadable — treated as
        "no verdict yet", the caller fails open).
        """
        path = self._executor_verdict_path()
        try:
            if not path.exists():
                return None
            for line in reversed(path.read_text(encoding="utf-8").splitlines()):
                entry = json.loads(line)
                if entry.get("order_id") == order_id:
                    return entry.get("verdict")
        except (OSError, json.JSONDecodeError):
            self.logger.warning(
                "Failed to read executor verdict journal at %s", path,
            )
        return None

    def _executor_verdict_path(self) -> Path:
        """Filesystem path of the executor's verdict journal."""
        configured = getattr(self.config, "EXECUTOR_VERDICT_PATH", "")
        if configured:
            return Path(configured)
        return Path("data/trading/executor_verdicts.jsonl")

    async def rollback_blocked_entry(self, symbol: str, forward_delivered: bool, order_id: str | None = None) -> None:
        """After forwarding an entry, roll back the local position if the
        executor rejected the order.

        The bot persists a Position (and records the BUY/SELL row) BEFORE the
        executor processes the order. If the executor then blocks it (silent
        ``Blocked`` on its console), the bot would manage a phantom position
        forever. This verification runs right after the forward:

        - ``forward_delivered=False`` → the order went to the file fallback and
          may still execute later; never roll back a possibly-live order.
        - executor confirms the position (via verdict journal or /position) →
          nothing to do.
        - executor explicitly reports the order blocked/error → roll back the
          phantom and record a compensating CLOSE so trade history stays
          paired/truthful.
        """
        if self.current_position is None:
            return
        if not forward_delivered:
            return
        if await self.confirm_entry_with_executor(symbol, order_id=order_id):
            return
        entry = self.current_position
        self.current_position = None
        await self.persistence.async_save_position(None)
        await self._record_blocked_entry_close(entry)
        self.logger.warning(
            "Executor blocked %s entry for %s (no position after forward) — "
            "rolled back local phantom position and recorded compensating CLOSE.",
            entry.direction, symbol,
        )

    async def _record_blocked_entry_close(self, entry: Position) -> None:
        """Record a compensating CLOSE row for an executor-blocked entry."""
        decision = TradeDecision(
            timestamp=datetime.now(timezone.utc),
            symbol=entry.symbol,
            action="CLOSE",
            confidence=entry.confidence,
            price=entry.entry_price,
            stop_loss=entry.stop_loss,
            take_profit=entry.take_profit,
            position_size=entry.size_pct,
            quote_amount=entry.quote_amount,
            quantity=entry.size,
            fee=0.0,
            reasoning=(
                f"Executor blocked the {entry.direction} entry (no position on exchange). "
                f"Local phantom rolled back; entry recorded {entry.entry_time.isoformat()}."
            ),
        )
        await self._record_trade_decision(decision)
