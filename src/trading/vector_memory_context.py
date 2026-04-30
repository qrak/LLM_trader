"""Context and retrieval helpers for vector memory."""

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.utils.indicator_classifier import classify_rsi_label, format_exit_execution_context

from .data_models import VectorSearchResult


class VectorMemoryContextMixin:
    """Document building, retrieval, and prompt formatting behavior."""

    def _build_experience_document(
        self,
        direction: str,
        symbol: str,
        outcome: str,
        pnl_pct: float,
        confidence: str,
        reasoning: str,
        close_reason: str,
        market_context: str,
        adx: Optional[float],
        rsi: Optional[float],
        atr_pct: Optional[float],
        volatility: str,
        macd_signal: str,
        bb_position: str,
        rr_ratio: Optional[float],
        sl_pct: Optional[float],
        tp_pct: Optional[float],
        market_sentiment: str,
        order_book_bias: str,
        max_profit_pct: Optional[float],
        max_drawdown_pct: Optional[float],
        factor_scores: Dict[str, float],
        exit_execution_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the discriminative text document embedded for semantic search."""
        symbol_str = f" [{symbol}]" if symbol else ""
        header = f"{direction} trade{symbol_str}. {market_context}."

        indicator_parts: List[str] = []
        if adx is not None:
            indicator_parts.append(f"ADX={adx:.1f} ({self._adx_label(adx)})")
        if rsi is not None:
            indicator_parts.append(f"RSI={rsi:.1f} ({classify_rsi_label(rsi)})")
        if atr_pct is not None:
            indicator_parts.append(f"ATR=${atr_pct:.0f}")
        if volatility:
            indicator_parts.append(f"Vol={volatility}")
        if macd_signal:
            indicator_parts.append(f"MACD={macd_signal}")
        if bb_position:
            indicator_parts.append(f"BB={bb_position}")
        indicators_str = " | ".join(indicator_parts)

        structure_parts: List[str] = []
        if rr_ratio is not None:
            structure_parts.append(f"RR={rr_ratio:.1f}")
        if sl_pct is not None:
            structure_parts.append(f"SL={sl_pct * 100:.2f}%")
        if tp_pct is not None:
            structure_parts.append(f"TP={tp_pct * 100:.2f}%")
        if market_sentiment:
            structure_parts.append(f"Sentiment={market_sentiment}")
        if order_book_bias:
            structure_parts.append(f"OB={order_book_bias}")
        exit_execution_text = format_exit_execution_context(exit_execution_context)
        if exit_execution_text:
            structure_parts.append(exit_execution_text)
        structure_str = " | ".join(structure_parts)

        factor_parts: List[str] = []
        for key, score in sorted(factor_scores.items()):
            name = key.replace("_score", "")
            bucket = "HIGH" if score > 69 else "MED" if score > 30 else "LOW"
            factor_parts.append(f"{name}={score:.0f} ({bucket})")
        factors_str = ", ".join(factor_parts)

        result_str = f"{outcome} ({pnl_pct:+.2f}%)"
        if close_reason:
            result_str += f" via {close_reason}"

        post_parts: List[str] = []
        if max_profit_pct is not None and max_profit_pct > 0:
            post_parts.append(f"MFE=+{max_profit_pct:.1f}%")
        if max_drawdown_pct is not None and max_drawdown_pct > 0:
            post_parts.append(f"MAE=-{max_drawdown_pct:.1f}%")
        post_str = " | ".join(post_parts)

        lines = [header]
        if indicators_str:
            lines.append(f"Indicators: {indicators_str}")
        if structure_str:
            lines.append(f"Structure: {structure_str}")
        if factors_str:
            lines.append(f"Confluences: {factors_str}")
        if reasoning and reasoning not in ("N/A", ""):
            lines.append(f"Reasoning: {reasoning}")
        lines.append(f"Result: {result_str}. Confidence: {confidence}.")
        if post_str:
            lines.append(f"Post-trade: {post_str}")

        return " ".join(lines)

    @staticmethod
    def _adx_label(adx: float) -> str:
        """Map ADX value to descriptive label."""
        if adx >= 25:
            return "High ADX"
        if adx >= 20:
            return "Medium ADX"
        return "Low ADX"

    def _calculate_recency_score(
        self,
        trade_timestamp: str,
        half_life_days: int = 90,
    ) -> float:
        """Calculate recency weight using exponential decay."""
        try:
            trade_dt = datetime.fromisoformat(trade_timestamp)
            if trade_dt.tzinfo is None:
                trade_dt = trade_dt.replace(tzinfo=timezone.utc)

            age_days = (datetime.now(timezone.utc) - trade_dt).days
            decay_rate = math.log(2) / half_life_days
            return math.exp(-decay_rate * age_days)
        except (ValueError, TypeError):
            return 0.5

    def retrieve_similar_experiences(
        self,
        current_context: str,
        k: int = 5,
        use_decay: bool = True,
        decay_half_life_days: int = 90,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Retrieve past experiences similar to the current market context."""
        if not self._ensure_initialized():
            return []

        try:
            if self._collection.count() == 0:
                return []

            query_embedding = self._embedding_model.encode(current_context).tolist()

            query_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": min(k, self._collection.count()),
            }
            if where:
                query_kwargs["where"] = where
            results = self._collection.query(**query_kwargs)

            experiences: List[VectorSearchResult] = []
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

                    experiences.append(VectorSearchResult(
                        id=doc_id,
                        document=results["documents"][0][i] if results["documents"] else "",
                        similarity=round(similarity * 100, 1),
                        recency=round(recency * 100, 1),
                        hybrid_score=round(hybrid_score * 100, 1),
                        metadata=meta,
                    ))

            if use_decay:
                experiences.sort(key=lambda item: item.hybrid_score, reverse=True)
                experiences = experiences[:k]

            return experiences

        except Exception as e:
            self.logger.error("Failed to retrieve experiences: %s", e)
            return []

    def get_context_for_prompt(
        self,
        current_context: str,
        k: int = 5,
        display_context: str = "",
    ) -> str:
        """Get formatted context string for prompt injection."""
        display = display_context or current_context
        experiences = self.retrieve_similar_experiences(
            current_context, k, where={"outcome": {"$ne": "UPDATE"}}
        )

        if not experiences:
            return ""

        max_similarity = max(exp.similarity for exp in experiences)
        if len(experiences) <= 2 and max_similarity < 50:
            lines = [
                f"RELEVANT PAST EXPERIENCES (Context: {display}):",
                "",
                f"⚠️ LIMITED DATA: Only {len(experiences)} trade(s) with <50% similarity. Standard analysis recommended.",
                "",
            ]
        else:
            lines = [
                f"RELEVANT PAST EXPERIENCES (Context: {display}):",
                "",
            ]

        for i, exp in enumerate(experiences, 1):
            meta = exp.metadata
            outcome = meta.get("outcome", "UNKNOWN")
            pnl = meta.get("pnl_pct", 0)
            direction = meta.get("direction", "?")

            lines.append(f"{i}. [SIMILARITY {exp.similarity:.0f}%] {direction} trade")
            lines.append(f"   - Result: {outcome} ({pnl:+.2f}%)")
            lines.append(f"   - Context: {meta.get('market_context', 'N/A')}")

            match_factors = self._build_match_factors(meta, display)
            if match_factors:
                lines.append(f"   - Match Factors: {match_factors}")

            reasoning = meta.get("reasoning", "")
            if reasoning and reasoning != "N/A":
                lines.append(f'   - Key Insight: "{reasoning}"')
            else:
                lines.append(f'   - Key Insight: "{self._generate_synthetic_insight(meta)}"')
            lines.append("")

        anti_patterns = self.get_anti_patterns_for_prompt(k=2)
        if anti_patterns:
            lines.append("")
            lines.append(anti_patterns)

        return "\n".join(lines)

    def _generate_synthetic_insight(self, meta: Dict[str, Any]) -> str:
        """Generate synthetic insight from trade metadata when reasoning is unavailable."""
        parts: List[str] = []

        symbol = meta.get("symbol", "")
        if symbol:
            parts.append(f"Pair: {symbol}")

        context = meta.get("market_context", "")
        if context:
            parts.append(f"Entry: {context}")

        close_reason = meta.get("close_reason", "")
        if close_reason:
            parts.append(f"Exit: {close_reason}")

        sl_dist = meta.get("sl_distance_pct")
        tp_dist = meta.get("tp_distance_pct")
        if sl_dist is not None:
            parts.append(f"SL: {sl_dist * 100:.2f}%")
        if tp_dist is not None:
            parts.append(f"TP: {tp_dist * 100:.2f}%")

        rr = meta.get("rr_ratio")
        if rr is not None:
            parts.append(f"R/R: {rr:.1f}")

        exit_execution_text = format_exit_execution_context(meta)
        if exit_execution_text:
            parts.append(exit_execution_text)

        max_profit = meta.get("max_profit_pct")
        max_dd = meta.get("max_drawdown_pct")
        if max_profit is not None and max_profit > 0:
            parts.append(f"MaxProfit: +{max_profit:.1f}%")
        if max_dd is not None and max_dd > 0:
            parts.append(f"MaxDD: -{max_dd:.1f}%")

        return " | ".join(parts) if parts else "No additional data"

    def _build_match_factors(self, meta: Dict[str, Any], current_context: str) -> str:
        """Build a match factors line showing stored numeric features vs current context."""
        parts: List[str] = []
        ctx_upper = current_context.upper()

        adx = meta.get("adx_at_entry")
        if adx is not None:
            stored_label = self._adx_label(adx)
            stored_upper = stored_label.upper()
            current_adx_mismatch = (
                ("LOW ADX" in stored_upper) != ("LOW ADX" in ctx_upper)
                or ("HIGH ADX" in stored_upper) != ("HIGH ADX" in ctx_upper)
                or ("MEDIUM ADX" in stored_upper) != ("MEDIUM ADX" in ctx_upper)
            )
            flag = " ⚠️" if current_adx_mismatch else ""
            parts.append(f"ADX={adx:.0f} ({stored_label}){flag}")

        rsi = meta.get("rsi_at_entry")
        if rsi is not None:
            stored_rsi_label = classify_rsi_label(rsi).upper()
            rsi_mismatch = (
                (stored_rsi_label in ("OVERBOUGHT", "STRONG") and f"RSI {stored_rsi_label}" not in ctx_upper)
                or (stored_rsi_label in ("OVERSOLD", "WEAK") and f"RSI {stored_rsi_label}" not in ctx_upper)
            ) if stored_rsi_label != "NEUTRAL" else False
            flag = " ⚠️" if rsi_mismatch else ""
            parts.append(f"RSI={rsi:.0f} ({classify_rsi_label(rsi)}){flag}")

        vol = meta.get("volatility_level", "")
        if vol:
            vol_mismatch = vol.upper() not in ctx_upper
            flag = " ⚠️" if vol_mismatch else ""
            parts.append(f"Vol={vol}{flag}")

        macd = meta.get("macd_signal", "")
        if macd:
            macd_mismatch = (
                (macd == "BULLISH" and "MACD BULLISH" not in ctx_upper)
                or (macd == "BEARISH" and "MACD BEARISH" not in ctx_upper)
            ) if macd != "NEUTRAL" else False
            flag = " ⚠️" if macd_mismatch else ""
            parts.append(f"MACD={macd}{flag}")

        bb = meta.get("bb_pos", "")
        if bb and bb != "MIDDLE":
            bb_mismatch = f"BB {bb.upper()}" not in ctx_upper
            flag = " ⚠️" if bb_mismatch else ""
            parts.append(f"BB={bb}{flag}")

        ob = meta.get("order_book_bias", "")
        if ob and ob not in ("BALANCED", ""):
            ob_mismatch = f"ORDERBOOK {ob.upper()}" not in ctx_upper
            flag = " ⚠️" if ob_mismatch else ""
            parts.append(f"OB={ob}{flag}")

        fgi = meta.get("fear_greed_index")
        if fgi is not None:
            parts.append(f"F&G={fgi:.0f}")

        atr = meta.get("atr_at_entry")
        if atr is not None:
            parts.append(f"ATR=${atr:.0f}")

        if meta.get("is_weekend", False):
            parts.append("Weekend ⚠️")

        exit_execution_text = format_exit_execution_context(meta)
        if exit_execution_text:
            parts.append(exit_execution_text)

        return " | ".join(parts)

    def get_stats_for_context(
        self,
        current_context: str,
        k: int = 20,
    ) -> Dict[str, Any]:
        """Calculate statistics from similar past experiences."""
        experiences = self.retrieve_similar_experiences(
            current_context, k, where={"outcome": {"$ne": "UPDATE"}}
        )

        if not experiences:
            return {"win_rate": 0, "avg_pnl": 0, "total_trades": 0}

        wins = sum(1 for experience in experiences if experience.metadata.get("outcome") == "WIN")
        pnls = [experience.metadata.get("pnl_pct", 0) for experience in experiences]

        return {
            "win_rate": (wins / len(experiences)) * 100 if experiences else 0,
            "avg_pnl": sum(pnls) / len(pnls) if pnls else 0,
            "total_trades": len(experiences),
        }