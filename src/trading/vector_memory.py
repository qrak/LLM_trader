"""Vector memory service for trading experiences using ChromaDB.

Provides semantic search over historical trades to find relevant past experiences
for context-aware decision making. Implements Temporal Awareness and Decay Engine
for recency-weighted retrieval.
"""

import math
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.logger.logger import Logger


class VectorMemoryService:
    """Service for storing and retrieving trading experiences via vector similarity.
    
    Uses ChromaDB for local vector storage and sentence-transformers for CPU embeddings.
    Provides semantic search to find past trades similar to current market conditions.
    """
    
    COLLECTION_NAME = "trading_experiences"
    SEMANTIC_RULES_COLLECTION = "semantic_rules"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_DECAY_HALF_LIFE_DAYS = 90
    
    def __init__(self, logger: Logger, data_dir: str = "data/brain_vector_db"):
        """Initialize vector memory service.
        
        Args:
            logger: Logger instance
            data_dir: Directory for ChromaDB persistence
        """
        self.logger = logger
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._semantic_rules_collection: Optional[Any] = None
        self._embedding_model: Optional[Any] = None
        self._initialized = False
    
    def _ensure_initialized(self) -> bool:
        """Lazy initialization of ChromaDB and embedding model.
        
        Returns:
            True if initialization succeeded, False otherwise.
        """
        if self._initialized:
            return True
        
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            
            self.logger.info("Initializing VectorMemoryService...")
            
            self._client = chromadb.PersistentClient(path=str(self.data_dir))
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            self._semantic_rules_collection = self._client.get_or_create_collection(
                name=self.SEMANTIC_RULES_COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )
            
            self._embedding_model = SentenceTransformer(
                self.EMBEDDING_MODEL,
                device="cpu"
            )
            
            self._initialized = True
            self.logger.info(
                f"VectorMemoryService initialized: {self._collection.count()} experiences stored"
            )
            return True
            
        except ImportError as e:
            self.logger.warning(
                f"VectorMemoryService unavailable (missing dependency): {e}"
            )
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize VectorMemoryService: {e}")
            return False
    
    def store_experience(
        self,
        trade_id: str,
        market_context: str,
        outcome: str,
        pnl_pct: float,
        direction: str,
        confidence: str,
        reasoning: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store a completed trade experience.
        
        Args:
            trade_id: Unique identifier for the trade (e.g., "trade_2026-01-04T17:00:00")
            market_context: Description of market state (e.g., "High ADX + Uptrend + Low Vol")
            outcome: "WIN" or "LOSS"
            pnl_pct: Profit/loss percentage
            direction: "LONG" or "SHORT"
            confidence: "HIGH", "MEDIUM", or "LOW"
            reasoning: AI's reasoning for the trade
            metadata: Additional metadata to store
            
        Returns:
            True if stored successfully, False otherwise.
        """
        if not self._ensure_initialized():
            return False
        
        try:
            document = (
                f"{direction} trade. Market: {market_context}. "
                f"Result: {outcome} ({pnl_pct:+.2f}%). "
                f"Confidence: {confidence}. Reasoning: {reasoning}"
            )
            
            embedding = self._embedding_model.encode(document).tolist()
            
            trade_metadata = {
                "outcome": outcome,
                "pnl_pct": pnl_pct,
                "direction": direction,
                "confidence": confidence,
                "market_context": market_context,
                "reasoning": reasoning,
                "timestamp": datetime.utcnow().isoformat(),
            }
            if metadata:
                market_regime = metadata.pop("market_regime", "NEUTRAL")
                trade_metadata["market_regime"] = market_regime
                trade_metadata.update(metadata)
            
            self._collection.add(
                ids=[trade_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[trade_metadata]
            )
            
            self.logger.info(
                f"Stored experience: {trade_id} ({outcome}, {pnl_pct:+.2f}%)"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store experience: {e}")
            return False
    
    def _calculate_recency_score(
        self,
        trade_timestamp: str,
        half_life_days: int = DEFAULT_DECAY_HALF_LIFE_DAYS
    ) -> float:
        """Calculate recency weight using exponential decay.

        Args:
            trade_timestamp: ISO format timestamp of the trade.
            half_life_days: Days until weight decays to 0.5.

        Returns:
            Recency weight between 0 and 1.
        """
        try:
            trade_dt = datetime.fromisoformat(trade_timestamp)
            age_days = (datetime.utcnow() - trade_dt).days
            decay_rate = math.log(2) / half_life_days
            return math.exp(-decay_rate * age_days)
        except (ValueError, TypeError):
            return 0.5

    def retrieve_similar_experiences(
        self,
        current_context: str,
        k: int = 5,
        use_decay: bool = True,
        decay_half_life_days: int = DEFAULT_DECAY_HALF_LIFE_DAYS
    ) -> List[Dict[str, Any]]:
        """Retrieve past experiences similar to the current market context.

        Uses hybrid scoring: similarity * recency_weight for temporal awareness.

        Args:
            current_context: Description of current market conditions
            k: Number of similar experiences to retrieve
            use_decay: Whether to apply recency decay weighting
            decay_half_life_days: Half-life for recency decay

        Returns:
            List of dicts with keys: id, document, similarity, hybrid_score, metadata
        """
        if not self._ensure_initialized():
            return []
        
        try:
            if self._collection.count() == 0:
                return []
            
            query_embedding = self._embedding_model.encode(current_context).tolist()
            
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self._collection.count())
            )
            
            experiences = []
            if results and results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    similarity = 1 - results["distances"][0][i] if results["distances"] else 0
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}

                    if use_decay:
                        timestamp = meta.get("timestamp", "")
                        recency = self._calculate_recency_score(timestamp, decay_half_life_days)
                        hybrid_score = similarity * 0.7 + recency * 0.3
                    else:
                        recency = 1.0
                        hybrid_score = similarity

                    experiences.append({
                        "id": doc_id,
                        "document": results["documents"][0][i] if results["documents"] else "",
                        "similarity": round(similarity * 100, 1),
                        "recency": round(recency * 100, 1),
                        "hybrid_score": round(hybrid_score * 100, 1),
                        "metadata": meta
                    })

            if use_decay:
                experiences.sort(key=lambda x: x["hybrid_score"], reverse=True)
                experiences = experiences[:k]

            self.logger.debug(
                f"Retrieved {len(experiences)} similar experiences (decay={use_decay})"
            )
            return experiences
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve experiences: {e}")
            return []
    
    def get_context_for_prompt(
        self,
        current_context: str,
        k: int = 5
    ) -> str:
        """Get formatted context string for prompt injection.
        
        Args:
            current_context: Current market context description
            k: Number of experiences to include
            
        Returns:
            Formatted string ready for prompt injection
        """
        experiences = self.retrieve_similar_experiences(current_context, k)
        
        if not experiences:
            return ""
        
        lines = [
            f"RELEVANT PAST EXPERIENCES (Context: {current_context}):",
            ""
        ]
        
        for i, exp in enumerate(experiences, 1):
            meta = exp.get("metadata", {})
            outcome = meta.get("outcome", "UNKNOWN")
            pnl = meta.get("pnl_pct", 0)
            direction = meta.get("direction", "?")
            
            lines.append(
                f"{i}. [SIMILARITY {exp['similarity']:.0f}%] {direction} trade"
            )
            lines.append(f"   - Result: {outcome} ({pnl:+.2f}%)")
            lines.append(f"   - Context: {meta.get('market_context', 'N/A')}")
            if meta.get("reasoning"):
                lines.append(f"   - Key Insight: \"{meta.get('reasoning')}\"")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_stats_for_context(
        self,
        current_context: str,
        k: int = 20
    ) -> Dict[str, Any]:
        """Calculate statistics from similar past experiences.
        
        Args:
            current_context: Current market context description
            k: Number of experiences to analyze
            
        Returns:
            Dict with win_rate, avg_pnl, total_trades for similar contexts
        """
        experiences = self.retrieve_similar_experiences(current_context, k)
        
        if not experiences:
            return {"win_rate": 0, "avg_pnl": 0, "total_trades": 0}
        
        wins = sum(1 for e in experiences if e["metadata"].get("outcome") == "WIN")
        pnls = [e["metadata"].get("pnl_pct", 0) for e in experiences]
        
        return {
            "win_rate": (wins / len(experiences)) * 100 if experiences else 0,
            "avg_pnl": sum(pnls) / len(pnls) if pnls else 0,
            "total_trades": len(experiences)
        }
    
    @property
    def experience_count(self) -> int:
        """Get total number of stored experiences."""
        if not self._ensure_initialized():
            return 0
        return self._collection.count()

    def store_semantic_rule(
        self,
        rule_id: str,
        rule_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store a semantic trading rule learned from trade clusters.

        Args:
            rule_id: Unique identifier for the rule (e.g., "rule_2026-01-06")
            rule_text: Human readable rule text
            metadata: Optional metadata (source_trades, win_rate, etc.)

        Returns:
            True if stored successfully
        """
        if not self._ensure_initialized():
            return False

        try:
            embedding = self._embedding_model.encode(rule_text).tolist()
            rule_meta = {
                "timestamp": datetime.utcnow().isoformat(),
                "active": True,
            }
            if metadata:
                rule_meta.update(metadata)

            self._semantic_rules_collection.upsert(
                ids=[rule_id],
                embeddings=[embedding],
                documents=[rule_text],
                metadatas=[rule_meta]
            )

            self.logger.info(f"Stored semantic rule: {rule_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store semantic rule: {e}")
            return False

    def get_active_rules(self, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve active semantic rules for prompt injection.

        Args:
            n_results: Maximum number of rules to return

        Returns:
            List of dicts with rule_id, text, metadata
        """
        if not self._ensure_initialized():
            return []

        try:
            count = self._semantic_rules_collection.count()
            if count == 0:
                return []

            all_rules = self._semantic_rules_collection.get(
                where={"active": True},
                limit=n_results
            )

            rules = []
            if all_rules and all_rules["ids"]:
                for i, rule_id in enumerate(all_rules["ids"]):
                    rules.append({
                        "rule_id": rule_id,
                        "text": all_rules["documents"][i] if all_rules["documents"] else "",
                        "metadata": all_rules["metadatas"][i] if all_rules["metadatas"] else {}
                    })

            return rules

        except Exception as e:
            self.logger.error(f"Failed to get active rules: {e}")
            return []

    @property
    def semantic_rule_count(self) -> int:
        """Get total number of stored semantic rules."""
        if not self._ensure_initialized():
            return 0
        return self._semantic_rules_collection.count()

    def compute_confidence_stats(self) -> Dict[str, Dict[str, Any]]:
        """Compute confidence level statistics from all stored experiences.

        Returns:
            Dict with HIGH/MEDIUM/LOW keys containing win rates, trade counts, avg P&L.
        """
        if not self._ensure_initialized():
            return {}

        all_experiences = self._collection.get()
        if not all_experiences or not all_experiences["ids"]:
            return {}

        stats = {
            "HIGH": {"total_trades": 0, "winning_trades": 0, "pnl_sum": 0.0},
            "MEDIUM": {"total_trades": 0, "winning_trades": 0, "pnl_sum": 0.0},
            "LOW": {"total_trades": 0, "winning_trades": 0, "pnl_sum": 0.0},
        }

        for meta in all_experiences["metadatas"]:
            confidence = meta.get("confidence", "MEDIUM").upper()
            if confidence not in stats:
                confidence = "MEDIUM"

            pnl = meta.get("pnl_pct", 0)
            is_win = meta.get("outcome") == "WIN"

            stats[confidence]["total_trades"] += 1
            if is_win:
                stats[confidence]["winning_trades"] += 1
            stats[confidence]["pnl_sum"] += pnl

        result: Dict[str, Dict[str, Any]] = {}
        for level, data in stats.items():
            total = data["total_trades"]
            result[level] = {
                "total_trades": total,
                "winning_trades": data["winning_trades"],
                "win_rate": (data["winning_trades"] / total * 100) if total > 0 else 0.0,
                "avg_pnl_pct": (data["pnl_sum"] / total) if total > 0 else 0.0,
            }

        return result

    def compute_adx_performance(self) -> Dict[str, Dict[str, Any]]:
        """Compute ADX bucket performance from all stored experiences.

        Returns:
            Dict with LOW/MEDIUM/HIGH keys for ADX buckets.
        """
        if not self._ensure_initialized():
            return {}

        all_experiences = self._collection.get()
        if not all_experiences or not all_experiences["ids"]:
            return {}

        buckets = {
            "LOW": {"level": "ADX<20", "trades": []},
            "MEDIUM": {"level": "ADX20-25", "trades": []},
            "HIGH": {"level": "ADX>25", "trades": []},
        }

        for meta in all_experiences["metadatas"]:
            adx = meta.get("adx_at_entry", meta.get("adx", 0))
            pnl = meta.get("pnl_pct", 0)
            is_win = meta.get("outcome") == "WIN"

            if adx < 20:
                bucket = "LOW"
            elif adx < 25:
                bucket = "MEDIUM"
            else:
                bucket = "HIGH"

            buckets[bucket]["trades"].append({"pnl": pnl, "is_win": is_win})

        result: Dict[str, Dict[str, Any]] = {}
        for key, data in buckets.items():
            trades = data["trades"]
            total = len(trades)
            wins = sum(1 for t in trades if t["is_win"])
            pnl_sum = sum(t["pnl"] for t in trades)

            result[key] = {
                "level": data["level"],
                "total_trades": total,
                "winning_trades": wins,
                "win_rate": (wins / total * 100) if total > 0 else 0.0,
                "avg_pnl_pct": (pnl_sum / total) if total > 0 else 0.0,
            }

        return result

    def compute_factor_performance(self) -> Dict[str, Dict[str, Any]]:
        """Compute confluence factor performance from all stored experiences.

        Returns:
            Dict with factor_bucket keys (e.g., trend_alignment_HIGH).
        """
        if not self._ensure_initialized():
            return {}

        all_experiences = self._collection.get()
        if not all_experiences or not all_experiences["ids"]:
            return {}

        factor_names = [
            "trend_alignment",
            "momentum_strength",
            "volume_support",
            "pattern_quality",
            "support_resistance",
        ]

        factors: Dict[str, Dict[str, Any]] = {}
        for name in factor_names:
            for bucket in ["LOW", "MEDIUM", "HIGH"]:
                key = f"{name}_{bucket}"
                factors[key] = {
                    "factor_name": name,
                    "bucket": bucket,
                    "trades": [],
                    "scores": [],
                }

        for meta in all_experiences["metadatas"]:
            pnl = meta.get("pnl_pct", 0)
            is_win = meta.get("outcome") == "WIN"

            for name in factor_names:
                score_key = f"{name}_score"
                score = meta.get(score_key, 0)

                if score <= 0:
                    continue

                if score <= 30:
                    bucket = "LOW"
                elif score <= 69:
                    bucket = "MEDIUM"
                else:
                    bucket = "HIGH"

                key = f"{name}_{bucket}"
                factors[key]["trades"].append({"pnl": pnl, "is_win": is_win})
                factors[key]["scores"].append(score)

        result: Dict[str, Dict[str, Any]] = {}
        for key, data in factors.items():
            trades = data["trades"]
            scores = data["scores"]
            total = len(trades)

            if total == 0:
                continue

            wins = sum(1 for t in trades if t["is_win"])
            pnl_sum = sum(t["pnl"] for t in trades)

            result[key] = {
                "factor_name": data["factor_name"],
                "bucket": data["bucket"],
                "total_trades": total,
                "winning_trades": wins,
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
                "win_rate": (wins / total * 100) if total > 0 else 0.0,
                "avg_pnl_pct": (pnl_sum / total) if total > 0 else 0.0,
            }

        return result

    def compute_optimal_thresholds(self, min_sample_size: int = 5) -> Dict[str, Any]:
        """Compute optimal thresholds from vector store data.

        Args:
            min_sample_size: Minimum trades needed for reliable threshold.

        Returns:
            Dict with adx_strong_threshold, min_rr_recommended, avg_sl_pct, confidence_threshold.
        """
        if not self._ensure_initialized():
            return {}

        thresholds: Dict[str, Any] = {}

        adx_perf = self.compute_adx_performance()

        adx_high = adx_perf.get("HIGH", {})
        adx_med = adx_perf.get("MEDIUM", {})

        if adx_high.get("total_trades", 0) >= min_sample_size:
            if adx_med.get("total_trades", 0) >= min_sample_size:
                if adx_high.get("win_rate", 0) > adx_med.get("win_rate", 0) + 10:
                    thresholds["adx_strong_threshold"] = 25
                elif adx_med.get("win_rate", 0) > 55:
                    thresholds["adx_strong_threshold"] = 20

        all_experiences = self._collection.get()
        if all_experiences and all_experiences["metadatas"]:
            rr_wins = []
            rr_losses = []
            sl_distances = []

            for meta in all_experiences["metadatas"]:
                rr = meta.get("rr_ratio", 0)
                if rr > 0:
                    if meta.get("outcome") == "WIN":
                        rr_wins.append(rr)
                    else:
                        rr_losses.append(rr)

                if meta.get("outcome") == "WIN" and meta.get("sl_distance_pct", 0) > 0:
                    sl_distances.append(meta["sl_distance_pct"] * 100)

            if rr_wins:
                avg_winning_rr = sum(rr_wins) / len(rr_wins)
                thresholds["min_rr_recommended"] = round(avg_winning_rr * 0.8, 1)

            if sl_distances:
                thresholds["avg_sl_pct"] = round(sum(sl_distances) / len(sl_distances), 2)

        conf_stats = self.compute_confidence_stats()
        high_stats = conf_stats.get("HIGH", {})
        if high_stats.get("total_trades", 0) >= min_sample_size:
            if high_stats.get("win_rate", 0) < 55:
                thresholds["confidence_threshold"] = 75
            elif high_stats.get("win_rate", 0) > 70:
                thresholds["confidence_threshold"] = 65

        return thresholds

    def get_confidence_recommendation(self, min_sample_size: int = 5) -> Optional[str]:
        """Generate recommendation based on confidence calibration.

        Args:
            min_sample_size: Minimum trades for analysis.

        Returns:
            Insight string or None.
        """
        conf_stats = self.compute_confidence_stats()

        high_stats = conf_stats.get("HIGH", {})
        medium_stats = conf_stats.get("MEDIUM", {})

        if high_stats.get("total_trades", 0) >= min_sample_size:
            high_win_rate = high_stats.get("win_rate", 0)

            if high_win_rate < 60:
                return f"HIGH confidence win rate is only {high_win_rate:.0f}% - increase entry criteria"

            if medium_stats.get("total_trades", 0) >= min_sample_size:
                medium_win_rate = medium_stats.get("win_rate", 0)
                if medium_win_rate > high_win_rate:
                    return "MEDIUM confidence outperforming HIGH - current HIGH standards may be too loose"

        return None
