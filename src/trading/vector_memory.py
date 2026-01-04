"""Vector memory service for trading experiences using ChromaDB.

Provides semantic search over historical trades to find relevant past experiences
for context-aware decision making.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any

from src.logger.logger import Logger


class VectorMemoryService:
    """Service for storing and retrieving trading experiences via vector similarity.
    
    Uses ChromaDB for local vector storage and sentence-transformers for CPU embeddings.
    Provides semantic search to find past trades similar to current market conditions.
    """
    
    COLLECTION_NAME = "trading_experiences"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
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
            }
            if metadata:
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
    
    def retrieve_similar_experiences(
        self,
        current_context: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve past experiences similar to the current market context.
        
        Args:
            current_context: Description of current market conditions
            k: Number of similar experiences to retrieve
            
        Returns:
            List of dicts with keys: id, document, similarity, metadata
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
                    experiences.append({
                        "id": doc_id,
                        "document": results["documents"][0][i] if results["documents"] else "",
                        "similarity": round(similarity * 100, 1),
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
                    })
            
            self.logger.debug(
                f"Retrieved {len(experiences)} similar experiences for context"
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
