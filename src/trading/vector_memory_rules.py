"""Semantic rule helpers for vector memory."""

from datetime import datetime, timezone
from typing import Any


class VectorMemoryRulesMixin:
    """Semantic rule storage and retrieval behavior."""

    def store_semantic_rule(
        self,
        rule_id: str,
        rule_text: str,
        metadata: dict[str, Any] | None = None,
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

    def get_active_rules(self, n_results: int = 5) -> list[dict[str, Any]]:
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

            rules: list[dict[str, Any]] = []
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

    def deactivate_semantic_rules(self, rule_ids: list[str]) -> int:
        """Mark semantic rules inactive without deleting their history."""
        if not rule_ids or not self._ensure_initialized():
            return 0

        try:
            existing_rules = self._semantic_rules_collection.get(
                ids=rule_ids,
                include=["metadatas"],
            )
            existing_ids = existing_rules.get("ids", []) if existing_rules else []
            if not existing_ids:
                return 0

            existing_metadatas = existing_rules.get("metadatas") or []
            updated_metadatas: list[dict[str, Any]] = []
            deactivated_at = datetime.now(timezone.utc).isoformat()
            for index, rule_id in enumerate(existing_ids):
                source_metadata = existing_metadatas[index] if index < len(existing_metadatas) else {}
                metadata = dict(source_metadata or {})
                metadata["active"] = False
                metadata["deactivated_at"] = deactivated_at
                updated_metadatas.append(metadata)
                self.logger.info("Deactivated semantic rule: %s", rule_id)

            self._semantic_rules_collection.update(
                ids=existing_ids,
                metadatas=updated_metadatas,
            )
            return len(existing_ids)

        except Exception as e:
            self.logger.error("Failed to deactivate semantic rules: %s", e)
            return 0

    def get_relevant_rules(
        self,
        current_context: str,
        n_results: int = 3,
        min_similarity: float = 0.4,
    ) -> list[dict[str, Any]]:
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

            rules: list[dict[str, Any]] = []
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
        """Get loss, corrective, and AI-mistake rules for prompt injection."""
        rules = self.get_active_rules(n_results=k * 4)
        actionable = [
            rule for rule in rules
            if rule.get("metadata", {}).get("rule_type") in ("anti_pattern", "corrective", "ai_mistake")
        ]

        if not actionable:
            return ""

        lines = ["⚠️ AVOID / IMPROVE / AI MISTAKE PATTERNS:"]
        for rule in actionable[:k]:
            meta = rule.get("metadata", {})
            lines.append(f"  - {rule['text']}")
            failure = meta.get("failure_reason")
            if failure:
                lines.append(f"    → Why: {failure}")
            recommended = meta.get("recommended_adjustment")
            if recommended:
                lines.append(f"    → Fix: {recommended}")

        return "\n".join(lines)