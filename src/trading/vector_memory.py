"""Vector memory service for trading experiences using ChromaDB."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

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
    DEFAULT_DECAY_HALF_LIFE_DAYS = 90
    RR_THRESHOLDS = (1.3, 1.5, 1.8)

    FACTOR_BUCKETS = ("LOW", "MEDIUM", "HIGH")
    FACTOR_NAMES = (
        "trend_alignment",
        "momentum_strength",
        "volume_support",
        "pattern_quality",
        "support_resistance",
    )

    def __init__(self, logger: Logger, chroma_client: Any, embedding_model: Any = None):
        """Initialize vector memory service.

        Args:
            logger: Logger instance
            chroma_client: Injected ChromaDB client instance
            embedding_model: SentenceTransformer instance (injected)
        """
        self.logger = logger
        self._client = chroma_client
        self._collection: Optional[Any] = None
        self._semantic_rules_collection: Optional[Any] = None
        self._embedding_model = embedding_model
        self._initialized = False

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

            self._initialized = True
            self.logger.info("VectorMemoryService collections ready: %s experiences stored", self._collection.count())
            return True

        except ImportError as e:
            self.logger.warning("VectorMemoryService unavailable (missing dependency): %s", e)
            return False
        except Exception as e:
            self.logger.error("Failed to initialize VectorMemoryService: %s", e, exc_info=True)
            return False

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
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
        metadata: Optional[Dict[str, Any]] = None,
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
            )

            embedding = self._embedding_model.encode(document).tolist()

            trade_metadata: Dict[str, Any] = {
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

            self._collection.upsert(
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
