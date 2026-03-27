"""Semantic rule helpers for vector memory."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class VectorMemoryRulesMixin:
    """Semantic rule storage and retrieval behavior."""

    def store_semantic_rule(
        self,
        rule_id: str,
        rule_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store a semantic trading rule learned from trade clusters."""
        if not self._ensure_initialized():
            return False

        try:
            embedding = self._embedding_model.encode(rule_text).tolist()
            rule_meta = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "active": True,
            }
            if metadata:
                rule_meta.update(metadata)

            self._semantic_rules_collection.upsert(
                ids=[rule_id],
                embeddings=[embedding],
                documents=[rule_text],
                metadatas=[rule_meta],
            )

            self.logger.info("Stored semantic rule: %s", rule_id)
            return True

        except Exception as e:
            self.logger.error("Failed to store semantic rule: %s", e)
            return False

    def get_active_rules(self, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve active semantic rules for prompt injection."""
        if not self._ensure_initialized():
            return []

        try:
            count = self._semantic_rules_collection.count()
            if count == 0:
                return []

            all_rules = self._semantic_rules_collection.get(
                where={"active": True},
                limit=n_results,
            )

            rules: List[Dict[str, Any]] = []
            if all_rules and all_rules["ids"]:
                for i, rule_id in enumerate(all_rules["ids"]):
                    rules.append({
                        "rule_id": rule_id,
                        "text": all_rules["documents"][i] if all_rules["documents"] else "",
                        "metadata": all_rules["metadatas"][i] if all_rules["metadatas"] else {},
                    })

            return rules

        except Exception as e:
            self.logger.error("Failed to get active rules: %s", e)
            return []

    def get_relevant_rules(
        self,
        current_context: str,
        n_results: int = 3,
        min_similarity: float = 0.4,
    ) -> List[Dict[str, Any]]:
        """Retrieve semantic rules relevant to current market context."""
        if not self._ensure_initialized():
            return []

        try:
            count = self._semantic_rules_collection.count()
            if count == 0:
                return []

            query_embedding = self._embedding_model.encode(current_context).tolist()

            results = self._semantic_rules_collection.query(
                query_embeddings=[query_embedding],
                where={"active": True},
                n_results=min(n_results * 2, count),
            )

            rules: List[Dict[str, Any]] = []
            if results and results["ids"] and results["ids"][0]:
                for i, rule_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 1.0
                    similarity = 1 - distance

                    if similarity < min_similarity:
                        continue

                    rules.append({
                        "rule_id": rule_id,
                        "text": results["documents"][0][i] if results["documents"] else "",
                        "similarity": round(similarity * 100, 1),
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    })

            rules.sort(key=lambda rule: rule["similarity"], reverse=True)
            return rules[:n_results]

        except Exception as e:
            self.logger.error("Failed to get relevant rules: %s", e)
            return []

    @property
    def semantic_rule_count(self) -> int:
        """Get total number of stored semantic rules."""
        if not self._ensure_initialized():
            return 0
        return self._semantic_rules_collection.count()

    def get_anti_patterns_for_prompt(self, k: int = 3) -> str:
        """Get anti-pattern rules for prompt injection."""
        rules = self.get_active_rules(n_results=k * 2)
        anti_rules = [rule for rule in rules if rule.get("metadata", {}).get("rule_type") == "anti_pattern"]

        if not anti_rules:
            return ""

        lines = ["⚠️ AVOID PATTERNS (learned from losses):"]
        for rule in anti_rules[:k]:
            lines.append(f"  - {rule['text']}")

        return "\n".join(lines)