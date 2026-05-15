"""Vector memory service for trading experiences using ChromaDB."""

from datetime import datetime, timezone
from typing import Any

from src.logger.logger import Logger
from .vector_memory_analytics import VectorMemoryAnalyticsMixin
from .vector_memory_context import VectorMemoryContextMixin
from .vector_memory_rules import VectorMemoryRulesMixin


class VectorMemoryService(
    VectorMemoryContextMixin,
    VectorMemoryRulesMixin,
    VectorMemoryAnalyticsMixin,
):
    """Service for storing and retrieving trading experiences via vector similarity.

    Uses ChromaDB for local vector storage and sentence-transformers for embeddings.
    Provides semantic search to find past trades similar to current market conditions.
    """

    COLLECTION_NAME = "trading_experiences"
    SEMANTIC_RULES_COLLECTION = "semantic_rules"
    BLOCKED_TRADES_COLLECTION = "system_constraints_rejections"
    RR_THRESHOLDS = (1.3, 1.5, 1.8)
    DECAY_HALF_LIFE_DIVISOR = 17
    MAX_DECAY_HALF_LIFE_DAYS = 30
    MAX_AGE_MULTIPLIER = 4
    RETRIEVAL_OVERFETCH_MULTIPLIER = 5

    FACTOR_BUCKETS = ("LOW", "MEDIUM", "HIGH")
    FACTOR_NAMES = (
        "trend_alignment",
        "momentum_strength",
        "volume_support",
        "pattern_quality",
        "support_resistance",
    )

    @classmethod
    def _derive_decay_window(cls, timeframe_minutes: int) -> tuple[int, int]:
        """Derive default decay half-life and max relevance age from timeframe."""
        half_life_days = min(
            cls.MAX_DECAY_HALF_LIFE_DAYS,
            max(1, round(timeframe_minutes / cls.DECAY_HALF_LIFE_DIVISOR)),
        )
        return half_life_days, half_life_days * cls.MAX_AGE_MULTIPLIER

    def __init__(
        self,
        logger: Logger,
        chroma_client: Any,
        embedding_model: Any = None,
        timeframe_minutes: int = 240,
    ):
        """Initialize vector memory service.

        Args:
            logger: Logger instance
            chroma_client: Injected ChromaDB client instance
            embedding_model: SentenceTransformer instance (injected)
            timeframe_minutes: Active analysis timeframe in minutes.
        """
        self.logger = logger
        self._client = chroma_client
        self._collection: Any | None = None
        self._semantic_rules_collection: Any | None = None
        self._blocked_collection: Any | None = None
        self._embedding_model = embedding_model
        self._initialized = False
        self._decay_half_life_days, self._max_age_days = self._derive_decay_window(
            timeframe_minutes
        )

    def _ensure_initialized(self) -> bool:
        """Lazy setup of collections (client is already injected).

        Returns:
            True if initialization succeeded, False otherwise.
        """
        if self._initialized:
            return True

        try:
            self.logger.info("Setting up VectorMemoryService collections...")

            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            self._semantic_rules_collection = self._client.get_or_create_collection(
                name=self.SEMANTIC_RULES_COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )
            self._blocked_collection = self._client.get_or_create_collection(
                name=self.BLOCKED_TRADES_COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )

            self._initialized = True
            collection = self._collection
            if collection is None:
                self.logger.error("VectorMemoryService collection setup returned None")
                self._initialized = False
                return False
            self.logger.info("VectorMemoryService collections ready: %s experiences stored", collection.count())
            return True

        except ImportError as e:
            self.logger.warning("VectorMemoryService unavailable (missing dependency): %s", e)
            return False
        except Exception as e:
            self.logger.error("Failed to initialize VectorMemoryService: %s", e, exc_info=True)
            return False

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Remove None values from metadata dict.

        ChromaDB rejects NoneType values in metadata. This filters them out
        while preserving all valid primitive values (str, int, float, bool).
        """
        return {k: v for k, v in metadata.items() if v is not None}

    def store_experience(
        self,
        trade_id: str,
        market_context: str,
        outcome: str,
        pnl_pct: float,
        direction: str,
        confidence: str,
        reasoning: str,
        metadata: dict[str, Any] | None = None,
        symbol: str = "",
        close_reason: str = "",
    ) -> bool:
        """Store a completed trade experience.

        Args:
            trade_id: Unique identifier for the trade.
            market_context: Categorical market description (used as header).
            outcome: "WIN", "LOSS", or "UPDATE".
            pnl_pct: Profit/loss percentage.
            direction: "LONG" or "SHORT".
            confidence: "HIGH", "MEDIUM", or "LOW".
            reasoning: AI reasoning for the trade.
            metadata: Additional metadata stored alongside the embedding.
            symbol: Trading pair, e.g. "BTC/USDC".
            close_reason: How the position closed (e.g. "stop_loss").

        Returns:
            True if stored successfully, False otherwise.
        """
        if not self._ensure_initialized():
            self.logger.warning("VectorMemoryService not initialized, cannot store experience.")
            return False

        try:
            collection = self._collection
            if collection is None:
                self.logger.warning("VectorMemoryService collection missing after initialization.")
                return False

            meta = dict(metadata or {})
            document = self._build_experience_document(
                direction=direction,
                symbol=symbol,
                outcome=outcome,
                pnl_pct=pnl_pct,
                confidence=confidence,
                reasoning=reasoning,
                close_reason=close_reason,
                market_context=market_context,
                adx=meta.get("adx_at_entry"),
                rsi=meta.get("rsi_at_entry"),
                atr_pct=meta.get("atr_at_entry"),
                volatility=meta.get("volatility_level", ""),
                macd_signal=meta.get("macd_signal", ""),
                bb_position=meta.get("bb_pos", ""),
                rr_ratio=meta.get("rr_ratio"),
                sl_pct=meta.get("sl_distance_pct"),
                tp_pct=meta.get("tp_distance_pct"),
                market_sentiment=meta.get("market_sentiment", ""),
                order_book_bias=meta.get("order_book_bias", ""),
                max_profit_pct=meta.get("max_profit_pct"),
                max_drawdown_pct=meta.get("max_drawdown_pct"),
                factor_scores={k: v for k, v in meta.items() if k.endswith("_score")},
                exit_execution_context=meta,
            )

            embedding = self._embedding_model.encode(document).tolist()

            trade_metadata: dict[str, Any] = {
                "outcome": outcome,
                "pnl_pct": pnl_pct,
                "direction": direction,
                "confidence": confidence,
                "market_context": market_context,
                "reasoning": reasoning,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if meta:
                market_regime = meta.pop("market_regime", "NEUTRAL")
                trade_metadata["market_regime"] = market_regime
                trade_metadata.update(meta)
            if close_reason:
                trade_metadata["close_reason"] = close_reason
            if symbol:
                trade_metadata["symbol"] = symbol

            trade_metadata = self._sanitize_metadata(trade_metadata)

            collection.upsert(
                ids=[trade_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[trade_metadata]
            )

            self.logger.info("Stored experience: %s (%s, %s%%)", trade_id, outcome, f"{pnl_pct:+.2f}")
            return True

        except Exception as e:
            self.logger.error("Failed to store experience: %s", e)
            return False

    def store_blocked_trade(
        self,
        guard_type: str,
        direction: str,
        confidence: str,
        suggested_rr: float,
        required_rr: float,
        suggested_sl_pct: float,
        suggested_tp_pct: float,
        suggested_sl: float,
        suggested_tp: float,
        current_price: float,
        volatility_level: str,
        reasoning_snippet: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store a blocked/rejected trade event for closed-loop LLM feedback.

        Args:
            guard_type: Which guard blocked the trade (e.g. 'rr_minimum', 'sl_clamp', 'tp_clamp')
            direction: LONG, SHORT, or N/A
            confidence: HIGH, MEDIUM, LOW
            suggested_rr: R/R ratio the LLM suggested
            required_rr: Minimum R/R required by the guard
            suggested_sl_pct: SL distance as decimal of entry
            suggested_tp_pct: TP distance as decimal of entry
            suggested_sl: Absolute SL price
            suggested_tp: Absolute TP price
            current_price: Price at rejection time
            volatility_level: HIGH, MEDIUM, LOW
            reasoning_snippet: First 200 chars of AI reasoning
            metadata: Additional metadata

        Returns:
            True if stored successfully, False otherwise.
        """
        if not self._ensure_initialized():
            self.logger.warning("VectorMemoryService not initialized, cannot store blocked trade.")
            return False

        try:
            collection = self._blocked_collection
            if collection is None:
                self.logger.warning("Blocked trades collection missing after initialization.")
                return False

            import math
            rr_delta = suggested_rr - required_rr if (math.isfinite(suggested_rr) and math.isfinite(required_rr)) else 0.0

            # Build discriminative document for semantic retrieval
            document_parts = [
                f"BLOCKED {direction} trade by {guard_type} guard.",
                f"Volatility: {volatility_level}.",
                f"LLM suggested R/R: {suggested_rr:.2f}, System requires: {required_rr:.2f} (delta: {rr_delta:+.2f}).",
                f"SL: {suggested_sl_pct*100:.2f}% from entry, TP: {suggested_tp_pct*100:.2f}% from entry.",
            ]
            if reasoning_snippet:
                document_parts.append(f"AI reasoning: {reasoning_snippet}")
            document = " ".join(document_parts)

            embedding = self._embedding_model.encode(document).tolist()

            block_metadata: dict[str, Any] = {
                "guard_type": guard_type,
                "direction": direction,
                "confidence": confidence,
                "suggested_rr": suggested_rr,
                "required_rr": required_rr,
                "rr_delta": rr_delta,
                "suggested_sl_pct": suggested_sl_pct,
                "suggested_tp_pct": suggested_tp_pct,
                "suggested_sl": suggested_sl,
                "suggested_tp": suggested_tp,
                "current_price": current_price,
                "volatility_level": volatility_level,
                "reasoning_snippet": reasoning_snippet,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "system_rejection",
            }
            if metadata:
                block_metadata.update(metadata)

            block_metadata = self._sanitize_metadata(block_metadata)

            block_id = f"blocked_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
            collection.upsert(
                ids=[block_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[block_metadata],
            )

            self.logger.info(
                "Stored blocked trade: %s | guard=%s | LLM R/R=%.2f vs Required=%.2f",
                block_id, guard_type, suggested_rr, required_rr,
            )
            return True

        except Exception as e:
            self.logger.error("Failed to store blocked trade: %s", e)
            return False

    def get_recent_blocked_trades(
        self,
        n: int = 10,
        guard_type: str | None = None,
        max_age_hours: int = 168,
    ) -> list[dict[str, Any]]:
        """Retrieve recent blocked trade events for feedback injection.

        Args:
            n: Maximum number of blocked trades to return
            guard_type: Optional filter by guard type (e.g. 'rr_minimum')
            max_age_hours: Only return events within this many hours

        Returns:
            List of blocked trade dicts sorted by recency (newest first).
        """
        if not self._ensure_initialized():
            return []

        try:
            collection = self._blocked_collection
            if collection is None or collection.count() == 0:
                return []

            cutoff = datetime.now(timezone.utc).isoformat()
            all_blocks = collection.get(include=["metadatas", "documents"])

            if not all_blocks or not all_blocks.get("ids"):
                return []

            blocked: list[dict[str, Any]] = []
            cutoff_dt = datetime.now(timezone.utc)

            for i, block_id in enumerate(all_blocks["ids"]):
                meta = all_blocks["metadatas"][i] if all_blocks["metadatas"] else {}
                doc = all_blocks["documents"][i] if all_blocks["documents"] else ""

                # Filter by guard type
                if guard_type and meta.get("guard_type") != guard_type:
                    continue

                # Filter by age
                ts = meta.get("timestamp", "")
                try:
                    event_dt = datetime.fromisoformat(ts)
                    if event_dt.tzinfo is None:
                        event_dt = event_dt.replace(tzinfo=timezone.utc)
                    age_hours = (cutoff_dt - event_dt).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        continue
                except (TypeError, ValueError):
                    pass

                blocked.append({
                    "id": block_id,
                    "document": doc,
                    **meta,
                })

            blocked.sort(
                key=lambda b: b.get("timestamp", ""),
                reverse=True,
            )
            return blocked[:n]

        except Exception as e:
            self.logger.error("Failed to retrieve blocked trades: %s", e)
            return []

    def get_blocked_trade_count(self) -> int:
        """Get total number of blocked trade events."""
        if not self._ensure_initialized():
            return 0
        try:
            return self._blocked_collection.count()
        except Exception:
            return 0

    def get_blocked_trade_feedback(
        self,
        n: int = 5,
        max_age_hours: int = 168,
    ) -> str:
        """Build actionable feedback string from recent blocked trades for LLM prompt injection.

        Groups rejections by guard type and presents deltas between LLM suggestions
        and system requirements so the LLM can self-correct its parameters.

        Args:
            n: Maximum number of blocked trades to summarize
            max_age_hours: Only consider events within this many hours

        Returns:
            Formatted feedback string, or empty string if no recent blocks.
        """
        import math

        blocked = self.get_recent_blocked_trades(n=n, max_age_hours=max_age_hours)
        if not blocked:
            return ""

        lines = [
            "## CRITICAL FEEDBACK: System Rejections",
            "",
            "The following trade suggestions were BLOCKED by risk guards. "
            "ADJUST your parameters before proposing the next trade.",
            "",
        ]

        # Group by guard type for pattern detection
        by_guard: dict[str, list[dict[str, Any]]] = {}
        for b in blocked:
            gt = b.get("guard_type", "unknown")
            by_guard.setdefault(gt, []).append(b)

        for guard, events in sorted(by_guard.items()):
            guard_label = {
                "rr_minimum": "R/R Minimum Guard",
                "sl_clamp": "Stop-Loss Clamping",
                "tp_clamp": "Take-Profit Clamping",
                "sl_below_entry": "Invalid SL (wrong side of entry)",
                "tp_below_entry": "Invalid TP (wrong side of entry)",
                "sl_distance_max": "SL Too Far (max 10%)",
                "sl_distance_min": "SL Too Tight (min 1%)",
                "update_frequency": "Update Frequency Guard",
                "sl_tightening": "Premature SL Tightening Guard",
            }.get(guard, guard.replace("_", " ").title())

            lines.append(f"### {guard_label} ({len(events)} recent):")

            for i, ev in enumerate(events[:3]):
                direction = ev.get("direction", "N/A")
                confidence = ev.get("confidence", "?")
                suggested_rr = ev.get("suggested_rr", 0)
                required_rr = ev.get("required_rr", 0)
                rr_delta = ev.get("rr_delta", 0)
                sl_pct = ev.get("suggested_sl_pct", 0)
                tp_pct = ev.get("suggested_tp_pct", 0)
                vol = ev.get("volatility_level", "?")
                reasoning = ev.get("reasoning_snippet", "")

                lines.append(f"  {i+1}. {direction} ({confidence} confidence, {vol} volatility):")
                if math.isfinite(suggested_rr) and math.isfinite(required_rr):
                    lines.append(
                        f"     - Your R/R: {suggested_rr:.2f} | Required: {required_rr:.2f} "
                        f"(gap: {rr_delta:+.2f})"
                    )
                if sl_pct > 0:
                    lines.append(f"     - Your SL: {sl_pct*100:.2f}% from entry")
                if tp_pct > 0:
                    lines.append(f"     - Your TP: {tp_pct*100:.2f}% from entry")
                if reasoning:
                    lines.append(f'     - Your thesis: "{reasoning}"')

        lines.extend([
            "",
            "### PRE-FLIGHT CHECKLIST (MANDATORY):",
            "- Before outputting BUY/SELL, verify: R/R >= required minimum (see Response Format).",
            "- If volatility is HIGH, widen SL to >1x ATR to achieve required R/R.",
            "- If volatility is LOW, do not use 2x+ ATR SL — tighten to keep R/R viable.",
            "- Compare your proposed SL/TP against the last rejection patterns above.",
            "- If you cannot meet the R/R requirement with a reasonable SL/TP, output HOLD.",
            "",
        ])

        return "\n".join(lines)
