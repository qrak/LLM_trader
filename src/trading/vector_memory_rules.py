"""Semantic rule helpers for vector memory."""

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any


class VectorMemoryRulesMixin:
    """Semantic rule storage and retrieval behavior."""

    if TYPE_CHECKING:
        logger: Any
        _semantic_rules_collection: Any
        _decay_half_life_days: int
        _max_age_days: int
        _timeframe_minutes: int

        def _ensure_initialized(self) -> bool: ...

        def _encode_embedding(self, text: str) -> list[float]: ...

        def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]: ...

    RULE_SIMILARITY_WEIGHT = 0.55
    RULE_EVIDENCE_WEIGHT = 0.25
    RULE_FRESHNESS_WEIGHT = 0.15
    RULE_CONTRADICTION_WEIGHT = 0.05
    RULE_EVIDENCE_FULL_SAMPLE_SIZE = 20

    @staticmethod
    def _clamp_score(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _timeframe_bucket(timeframe_minutes: int) -> str:
        if timeframe_minutes < 60:
            return "scalping"
        if timeframe_minutes < 240:
            return "intraday"
        if timeframe_minutes < 1440:
            return "swing"
        return "position"

    @staticmethod
    def _parse_rule_timestamp(timestamp: Any) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(str(timestamp))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
            return parsed if math.isfinite(parsed) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _rule_created_at(self, metadata: dict[str, Any]) -> datetime | None:
        return self._parse_rule_timestamp(
            metadata.get("last_validated_at")
            or metadata.get("created_at")
            or metadata.get("timestamp")
        )

    def _rule_age_days(self, metadata: dict[str, Any], now: datetime) -> float:
        created_at = self._rule_created_at(metadata)
        if created_at is None:
            return 0.0
        return max(0.0, (now - created_at).total_seconds() / 86400)

    def _rule_freshness_label(self, age_days: float) -> str:
        if age_days <= self._decay_half_life_days:
            return "fresh"
        if age_days <= self._decay_half_life_days * 2:
            return "maturing"
        if age_days <= self._max_age_days:
            return "stale"
        return "legacy"

    def _rule_freshness_score(self, age_days: float) -> float:
        decay_rate = math.log(2) / max(1, self._decay_half_life_days)
        return self._clamp_score(math.exp(-decay_rate * age_days))

    def _rule_support_count(self, metadata: dict[str, Any]) -> int:
        explicit = self._as_int(metadata.get("support_count"), -1)
        if explicit >= 0:
            return explicit
        source_trades = self._as_int(metadata.get("source_trades"), -1)
        if source_trades >= 0:
            return source_trades
        wins = self._as_int(metadata.get("wins"), 0)
        losses = self._as_int(metadata.get("losses"), 0)
        source_loss_count = self._as_int(metadata.get("source_loss_count"), 0)
        return max(wins + losses, source_loss_count)

    def _rule_evidence_score(self, metadata: dict[str, Any]) -> float:
        support_count = self._rule_support_count(metadata)
        sample_score = self._clamp_score(
            math.log1p(max(0, support_count)) / math.log1p(self.RULE_EVIDENCE_FULL_SAMPLE_SIZE)
        )

        rule_type = str(metadata.get("rule_type", "best_practice"))
        if rule_type in {"anti_pattern", "ai_mistake"}:
            quality = self._as_float(metadata.get("loss_rate"), 50.0) / 100.0
        else:
            quality = self._as_float(metadata.get("win_rate"), 50.0) / 100.0

        expectancy = self._as_float(metadata.get("expectancy_pct"), 0.0)
        expectancy_score = self._clamp_score((expectancy + 5.0) / 10.0)
        quality_score = self._clamp_score(quality * 0.8 + expectancy_score * 0.2)
        return self._clamp_score(quality_score * 0.7 + sample_score * 0.3)

    def _rule_contradiction_penalty(self, metadata: dict[str, Any]) -> float:
        contradictions = max(0, self._as_int(metadata.get("contradiction_count"), 0))
        validations = max(0, self._as_int(metadata.get("validation_hit_count"), 0))
        support = max(0, self._rule_support_count(metadata))
        denominator = max(1, support + validations + contradictions)
        return self._clamp_score(contradictions / denominator)

    def _score_rule_metadata(
        self,
        metadata: dict[str, Any],
        similarity: float,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Compute timeframe-aware semantic-rule influence scores."""
        current_time = now or datetime.now(timezone.utc)
        normalized_similarity = self._clamp_score(similarity)
        age_days = self._rule_age_days(metadata, current_time)
        freshness_score = self._rule_freshness_score(age_days)
        evidence_score = self._rule_evidence_score(metadata)
        contradiction_penalty = self._rule_contradiction_penalty(metadata)
        final_score = self._clamp_score(
            normalized_similarity * self.RULE_SIMILARITY_WEIGHT
            + evidence_score * self.RULE_EVIDENCE_WEIGHT
            + freshness_score * self.RULE_FRESHNESS_WEIGHT
            - contradiction_penalty * self.RULE_CONTRADICTION_WEIGHT
        )
        return {
            "age_days": round(age_days, 1),
            "freshness_score": round(freshness_score * 100, 1),
            "freshness_label": self._rule_freshness_label(age_days),
            "evidence_score": round(evidence_score * 100, 1),
            "contradiction_penalty": round(contradiction_penalty * 100, 1),
            "final_score": round(final_score * 100, 1),
            "support_count": self._rule_support_count(metadata),
        }

    def _metadata_with_rule_defaults(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Return metadata with lifecycle defaults for old and new rules."""
        enriched = dict(metadata or {})
        timestamp = enriched.get("timestamp") or datetime.now(timezone.utc).isoformat()
        enriched.setdefault("timestamp", timestamp)
        enriched.setdefault("created_at", timestamp)
        enriched.setdefault("support_count", self._rule_support_count(enriched))
        enriched.setdefault("validation_hit_count", 0)
        enriched.setdefault("contradiction_count", 0)
        enriched.setdefault("source_timeframe_minutes", getattr(self, "_timeframe_minutes", 240))
        enriched.setdefault(
            "source_timeframe_bucket",
            self._timeframe_bucket(self._as_int(enriched.get("source_timeframe_minutes"), 240)),
        )
        return enriched

    def _rule_result(
        self,
        rule_id: str,
        text: str,
        metadata: dict[str, Any],
        similarity: float,
        now: datetime,
    ) -> dict[str, Any]:
        enriched_metadata = self._metadata_with_rule_defaults(metadata)
        score_fields = self._score_rule_metadata(enriched_metadata, similarity, now)
        enriched_metadata.update(score_fields)
        return {
            "rule_id": rule_id,
            "text": text,
            "similarity": round(similarity * 100, 1),
            "final_score": score_fields["final_score"],
            "metadata": enriched_metadata,
        }

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
            now = datetime.now(timezone.utc)
            embedding = self._encode_embedding(rule_text)
            rule_meta = {
                "timestamp": now.isoformat(),
                "created_at": now.isoformat(),
                "active": True,
                "validation_hit_count": 0,
                "contradiction_count": 0,
                "source_timeframe_minutes": getattr(self, "_timeframe_minutes", 240),
                "source_timeframe_bucket": self._timeframe_bucket(
                    self._as_int(getattr(self, "_timeframe_minutes", 240), 240)
                ),
            }
            if metadata:
                rule_meta.update(metadata)
            rule_meta = self._metadata_with_rule_defaults(rule_meta)
            rule_meta = self._sanitize_metadata(rule_meta)

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

            fetch_limit = min(max(n_results * 4, n_results), count)
            all_rules = self._semantic_rules_collection.get(
                where={"active": True},
                limit=fetch_limit,
            )

            rules: list[dict[str, Any]] = []
            now = datetime.now(timezone.utc)
            if all_rules and all_rules["ids"]:
                for i, rule_id in enumerate(all_rules["ids"]):
                    rules.append(self._rule_result(
                        rule_id=rule_id,
                        text=all_rules["documents"][i] if all_rules["documents"] else "",
                        metadata=all_rules["metadatas"][i] if all_rules["metadatas"] else {},
                        similarity=1.0,
                        now=now,
                    ))

            rules.sort(key=lambda rule: rule["final_score"], reverse=True)
            return rules[:n_results]

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

            query_embedding = self._encode_embedding(current_context)

            results = self._semantic_rules_collection.query(
                query_embeddings=[query_embedding],
                where={"active": True},
                n_results=min(n_results * 2, count),
            )

            rules: list[dict[str, Any]] = []
            now = datetime.now(timezone.utc)
            if results and results["ids"] and results["ids"][0]:
                for i, rule_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 1.0
                    similarity = 1 - distance

                    if similarity < min_similarity:
                        continue

                    rules.append(self._rule_result(
                        rule_id=rule_id,
                        text=results["documents"][0][i] if results["documents"] else "",
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        similarity=similarity,
                        now=now,
                    ))

            rules.sort(key=lambda rule: rule["final_score"], reverse=True)
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

    def update_rule_validation_feedback(
        self,
        current_context: str,
        outcome: str,
        n_results: int = 5,
        min_similarity: float = 0.5,
    ) -> int:
        """Update matched semantic-rule validation counters after a closed trade."""
        if not self._ensure_initialized():
            return 0

        try:
            matched_rules = self.get_relevant_rules(
                current_context=current_context,
                n_results=n_results,
                min_similarity=min_similarity,
            )
            if not matched_rules:
                return 0

            now = datetime.now(timezone.utc).isoformat()
            updated_ids: list[str] = []
            updated_metadatas: list[dict[str, Any]] = []
            normalized_outcome = str(outcome).upper()
            for rule in matched_rules:
                metadata = self._metadata_with_rule_defaults(rule.get("metadata", {}))
                rule_type = str(metadata.get("rule_type", "best_practice"))
                supports_rule = normalized_outcome == "WIN"
                if rule_type in {"anti_pattern", "ai_mistake"}:
                    supports_rule = normalized_outcome == "LOSS"

                if supports_rule:
                    metadata["validation_hit_count"] = self._as_int(metadata.get("validation_hit_count"), 0) + 1
                    metadata["last_validated_at"] = now
                else:
                    metadata["contradiction_count"] = self._as_int(metadata.get("contradiction_count"), 0) + 1
                    metadata["last_contradicted_at"] = now

                updated_ids.append(rule["rule_id"])
                updated_metadatas.append(self._sanitize_metadata(metadata))

            self._semantic_rules_collection.update(
                ids=updated_ids,
                metadatas=updated_metadatas,
            )
            return len(updated_ids)
        except Exception as e:
            self.logger.error("Failed to update rule validation feedback: %s", e)
            return 0