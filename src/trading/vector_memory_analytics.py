"""Analytics helpers for vector memory."""

from typing import Any

from .data_models import VectorSearchResult


class VectorMemoryAnalyticsMixin:
    """Statistics, reporting, and threshold-learning behavior."""

    @property
    def experience_count(self) -> int:
        """Get total number of stored entries (includes UPDATE)."""
        if not self._ensure_initialized():
            return 0
        return self._collection.count()

    @property
    def trade_count(self) -> int:
        """Get count of actual trades (excludes UPDATE entries)."""
        if not self._ensure_initialized():
            return 0
        try:
            results = self._collection.get(where={"outcome": {"$ne": "UPDATE"}})
            return len(results["ids"]) if results and results["ids"] else 0
        except Exception:
            return self._collection.count()

    def get_direction_bias(self) -> dict[str, Any] | None:
        """Get count of LONG vs SHORT trades for bias detection."""
        metas = self._get_trade_metadatas(exclude_updates=True)
        if not metas:
            return None

        long_count = sum(1 for meta in metas if meta.get("direction") == "LONG")
        short_count = sum(1 for meta in metas if meta.get("direction") == "SHORT")
        total = long_count + short_count

        if total == 0:
            return None

        return {
            "long_count": long_count,
            "short_count": short_count,
            "long_pct": round(long_count / total * 100, 1),
            "short_pct": round(short_count / total * 100, 1),
        }

    def get_all_experiences(
        self,
        limit: int = 100,
        where: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Retrieve all experiences without vector similarity search."""
        if not self._ensure_initialized():
            return []

        try:
            query_where = where if where else {"outcome": {"$ne": "UPDATE"}}
            results = self._collection.get(
                where=query_where,
                limit=limit,
                include=["metadatas", "documents"],
            )

            experiences: list[VectorSearchResult] = []
            if results and results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    meta = results["metadatas"][i] if results["metadatas"] else {}
                    doc = results["documents"][i] if results["documents"] else ""
                    experiences.append(VectorSearchResult(
                        id=doc_id,
                        document=doc,
                        similarity=0,
                        recency=0,
                        hybrid_score=0,
                        metadata=meta,
                    ))

            return experiences

        except Exception as e:
            self.logger.error("Failed to retrieve all experiences: %s", e)
            return []

    def _get_trade_metadatas(self, exclude_updates: bool = True) -> list[dict[str, Any]]:
        """Retrieve metadatas for all stored trades, handling filtering."""
        if not self._ensure_initialized():
            return []

        all_experiences = self._collection.get(include=["metadatas"])
        if not all_experiences or not all_experiences["ids"] or not all_experiences["metadatas"]:
            return []

        metas = all_experiences["metadatas"]
        if exclude_updates:
            return [meta for meta in metas if meta.get("outcome") != "UPDATE"]
        return metas

    @staticmethod
    def _build_trade_snapshot(pnl: float, is_win: bool) -> dict[str, Any]:
        """Create a normalized trade snapshot for aggregation helpers."""
        return {"pnl": pnl, "is_win": is_win}

    @staticmethod
    def _summarize_trade_group(trades: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute common trade performance metrics for an aggregated group."""
        total = len(trades)
        wins = sum(1 for trade in trades if trade["is_win"])
        pnl_sum = sum(trade["pnl"] for trade in trades)

        return {
            "total_trades": total,
            "winning_trades": wins,
            "win_rate": (wins / total * 100) if total > 0 else 0.0,
            "avg_pnl_pct": (pnl_sum / total) if total > 0 else 0.0,
        }

    @staticmethod
    def _factor_bucket_for_score(score: float) -> str | None:
        """Map a factor score to the configured factor bucket."""
        if score <= 0:
            return None
        if score <= 30:
            return "LOW"
        if score <= 69:
            return "MEDIUM"
        return "HIGH"

    @staticmethod
    def _normalize_categorical_value(category_name: str, value: Any) -> str | None:
        """Normalize categorical factor values into stable reporting buckets."""
        if not value:
            return None

        normalized = str(value).upper()
        if "GREED" in normalized:
            normalized = "GREED"
        elif "FEAR" in normalized:
            normalized = "FEAR"

        if category_name.upper() == "VOLATILITY" and "VOLATILITY" not in normalized:
            normalized = f"{normalized} VOLATILITY"

        return normalized

    @staticmethod
    def _append_trade_to_group(
        groups: dict[str, list[dict[str, Any]]],
        key: str,
        pnl: float,
        is_win: bool,
    ) -> None:
        """Append a normalized trade snapshot to a keyed aggregation bucket."""
        if key not in groups:
            groups[key] = []
        groups[key].append(VectorMemoryAnalyticsMixin._build_trade_snapshot(pnl, is_win))

    def _build_factor_result(
        self,
        factor_name: str,
        bucket: str,
        trades: list[dict[str, Any]],
        avg_score: float = 0.0,
    ) -> dict[str, Any]:
        """Build a stable factor performance payload from aggregated trades."""
        summary = self._summarize_trade_group(trades)
        return {
            "factor_name": factor_name,
            "bucket": bucket,
            "avg_score": avg_score,
            **summary,
        }

    def compute_confidence_stats(self) -> dict[str, dict[str, Any]]:
        """Compute confidence level statistics from all stored experiences."""
        metas = self._get_trade_metadatas()
        if not metas:
            return {}

        stats = {
            "HIGH": {"total_trades": 0, "winning_trades": 0, "pnl_sum": 0.0},
            "MEDIUM": {"total_trades": 0, "winning_trades": 0, "pnl_sum": 0.0},
            "LOW": {"total_trades": 0, "winning_trades": 0, "pnl_sum": 0.0},
        }

        for meta in metas:
            confidence = meta.get("confidence", "MEDIUM").upper()
            if confidence not in stats:
                confidence = "MEDIUM"

            pnl = meta.get("pnl_pct", 0)
            is_win = meta.get("outcome") == "WIN"

            stats[confidence]["total_trades"] += 1
            if is_win:
                stats[confidence]["winning_trades"] += 1
            stats[confidence]["pnl_sum"] += pnl

        result: dict[str, dict[str, Any]] = {}
        for level, data in stats.items():
            total = data["total_trades"]
            result[level] = {
                "total_trades": total,
                "winning_trades": data["winning_trades"],
                "win_rate": (data["winning_trades"] / total * 100) if total > 0 else 0.0,
                "avg_pnl_pct": (data["pnl_sum"] / total) if total > 0 else 0.0,
            }

        return result

    def compute_adx_performance(self) -> dict[str, dict[str, Any]]:
        """Compute ADX bucket performance from all stored experiences."""
        metas = self._get_trade_metadatas()
        if not metas:
            return {}

        buckets = {
            "LOW": {"level": "ADX<20", "trades": []},
            "MEDIUM": {"level": "ADX20-25", "trades": []},
            "HIGH": {"level": "ADX>25", "trades": []},
        }

        for meta in metas:
            adx = meta.get("adx_at_entry", meta.get("adx", 0))
            pnl = meta.get("pnl_pct", 0)
            is_win = meta.get("outcome") == "WIN"

            if adx < 20:
                bucket = "LOW"
            elif adx < 25:
                bucket = "MEDIUM"
            else:
                bucket = "HIGH"

            buckets[bucket]["trades"].append(self._build_trade_snapshot(pnl, is_win))

        result: dict[str, dict[str, Any]] = {}
        for key, data in buckets.items():
            result[key] = {
                "level": data["level"],
                **self._summarize_trade_group(data["trades"]),
            }

        return result

    def compute_factor_performance(self) -> dict[str, dict[str, Any]]:
        """Compute confluence factor performance from all stored experiences."""
        metas = self._get_trade_metadatas()
        if not metas:
            return {}

        factors: dict[str, dict[str, Any]] = {}
        for name in self.FACTOR_NAMES:
            for bucket in self.FACTOR_BUCKETS:
                key = f"{name}_{bucket}"
                factors[key] = {
                    "factor_name": name,
                    "bucket": bucket,
                    "trades": [],
                    "scores": [],
                }

        for meta in metas:
            pnl = meta.get("pnl_pct", 0)
            is_win = meta.get("outcome") == "WIN"

            for name in self.FACTOR_NAMES:
                score = meta.get(f"{name}_score", 0)
                bucket = self._factor_bucket_for_score(score)
                if bucket is None:
                    continue

                key = f"{name}_{bucket}"
                factors[key]["trades"].append(self._build_trade_snapshot(pnl, is_win))
                factors[key]["scores"].append(score)

        result: dict[str, dict[str, Any]] = {}
        for key, data in factors.items():
            trades = data["trades"]
            scores = data["scores"]
            if not trades:
                continue

            result[key] = self._build_factor_result(
                factor_name=data["factor_name"],
                bucket=data["bucket"],
                trades=trades,
                avg_score=sum(scores) / len(scores) if scores else 0.0,
            )

        categorical_buckets: dict[str, list[dict[str, Any]]] = {}
        for meta in metas:
            pnl = meta.get("pnl_pct", 0)
            is_win = meta.get("outcome") == "WIN"
            categorical_inputs = (
                ("Sentiment", meta.get("market_sentiment_at_entry", meta.get("market_sentiment"))),
                ("Volatility", meta.get("volatility_level", meta.get("volatility"))),
                ("Trend", meta.get("trend_direction_at_entry", meta.get("trend_direction"))),
            )
            for category_name, raw_value in categorical_inputs:
                normalized = self._normalize_categorical_value(category_name, raw_value)
                if normalized is None:
                    continue
                bucket_key = f"{category_name}: {normalized}"
                self._append_trade_to_group(categorical_buckets, bucket_key, pnl, is_win)

        for category_name, trades in categorical_buckets.items():
            if not trades:
                continue
            result[f"cat_{category_name}"] = self._build_factor_result(
                factor_name=category_name,
                bucket=category_name.split(": ", 1)[1] if ": " in category_name else category_name,
                trades=trades,
            )

        return result

    def compute_optimal_thresholds(self, min_sample_size: int = 5) -> dict[str, Any]:
        """Compute optimal thresholds from vector store data."""
        if not self._ensure_initialized():
            return {}

        thresholds: dict[str, Any] = {}

        adx_perf = self.compute_adx_performance()
        adx_high = adx_perf.get("HIGH", {})
        adx_med = adx_perf.get("MEDIUM", {})
        adx_low = adx_perf.get("LOW", {})

        if adx_high.get("total_trades", 0) >= min_sample_size:
            if adx_med.get("total_trades", 0) >= min_sample_size:
                if adx_high.get("win_rate", 0) > adx_med.get("win_rate", 0) + 10:
                    thresholds["adx_strong_threshold"] = 25
                elif adx_med.get("win_rate", 0) > 55:
                    thresholds["adx_strong_threshold"] = 20

        if adx_low.get("total_trades", 0) >= min_sample_size:
            low_win_rate = adx_low.get("win_rate", 50)
            if low_win_rate < 40:
                thresholds["adx_weak_threshold"] = 22
            elif low_win_rate > 55:
                thresholds["adx_weak_threshold"] = 18

        all_experiences_raw = self._collection.get()
        all_experiences = {"ids": [], "metadatas": []}
        if all_experiences_raw and all_experiences_raw.get("metadatas"):
            raw_ids = all_experiences_raw.get("ids") or []
            for idx, meta in enumerate(all_experiences_raw["metadatas"]):
                outcome = meta.get("outcome")
                if outcome not in ("WIN", "LOSS"):
                    continue
                all_experiences["metadatas"].append(meta)
                if idx < len(raw_ids):
                    all_experiences["ids"].append(raw_ids[idx])

        if all_experiences["metadatas"]:
            rr_wins: list[float] = []
            rr_losses: list[float] = []
            sl_distances: list[float] = []

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
                raw_min_rr = round(avg_winning_rr * 0.8, 1)
                borderline = thresholds.get("rr_borderline_min")
                if borderline is not None and raw_min_rr >= borderline:
                    raw_min_rr = round(borderline - 0.3, 1)
                thresholds["min_rr_recommended"] = max(1.0, raw_min_rr)

                sorted_rr = sorted(rr_wins)
                p75_idx = int(len(sorted_rr) * 0.75)
                if p75_idx < len(sorted_rr):
                    thresholds["rr_strong_setup"] = round(sorted_rr[p75_idx], 1)

            if rr_wins and rr_losses:
                for test_rr in self.RR_THRESHOLDS:
                    wins = sum(1 for rr in rr_wins if rr < test_rr)
                    losses = sum(1 for rr in rr_losses if rr < test_rr)
                    total = wins + losses
                    if total >= 3:
                        below_win_rate = wins / total
                        if below_win_rate < 0.40:
                            thresholds["rr_borderline_min"] = test_rr
                            break

            if sl_distances:
                thresholds["avg_sl_pct"] = round(sum(sl_distances) / len(sl_distances), 2)

        conf_stats = self.compute_confidence_stats()
        high_stats = conf_stats.get("HIGH", {})
        if high_stats.get("total_trades", 0) >= min_sample_size:
            if high_stats.get("win_rate", 0) < 55:
                thresholds["confidence_threshold"] = 75
            elif high_stats.get("win_rate", 0) > 70:
                thresholds["confidence_threshold"] = 65

        self._learn_position_size_threshold(all_experiences, min_sample_size, thresholds)
        self._learn_confluence_thresholds(all_experiences, min_sample_size, thresholds)
        self._learn_alignment_thresholds(all_experiences, min_sample_size, thresholds)
        self._learn_sl_tightening_threshold(all_experiences_raw, min_sample_size, thresholds)

        return thresholds

    def _learn_position_size_threshold(
        self,
        all_experiences: dict[str, Any],
        min_sample_size: int,
        thresholds: dict[str, Any],
    ) -> None:
        """Learn min_position_size from small position performance."""
        if not all_experiences or not all_experiences.get("metadatas"):
            return
        small_positions: list[bool] = []
        for meta in all_experiences["metadatas"]:
            size_pct = meta.get("position_size_pct")
            if size_pct is not None and size_pct < 0.15:
                small_positions.append(meta.get("outcome") == "WIN")
        if len(small_positions) >= min_sample_size:
            small_win_rate = sum(small_positions) / len(small_positions)
            if small_win_rate >= 0.55:
                thresholds["min_position_size"] = 0.08
            elif small_win_rate < 0.40:
                thresholds["min_position_size"] = 0.15

    def _learn_confluence_thresholds(
        self,
        all_experiences: dict[str, Any],
        min_sample_size: int,
        thresholds: dict[str, Any],
    ) -> None:
        """Learn minimum confluence thresholds from historical performance."""
        if not all_experiences or not all_experiences.get("metadatas"):
            return
        confluence_buckets: dict[tuple, list[bool]] = {}
        for meta in all_experiences["metadatas"]:
            count = meta.get("confluence_count")
            adx = meta.get("adx_at_entry", 25)
            if count is not None:
                key = (count, adx < 20)
                if key not in confluence_buckets:
                    confluence_buckets[key] = []
                confluence_buckets[key].append(meta.get("outcome") == "WIN")
        for count in range(5, 1, -1):
            key = (count, True)
            if key in confluence_buckets and len(confluence_buckets[key]) >= min_sample_size:
                win_rate = sum(confluence_buckets[key]) / len(confluence_buckets[key])
                if win_rate >= 0.55:
                    thresholds["min_confluences_weak"] = count
                    break
        for count in range(4, 1, -1):
            key = (count, False)
            if key in confluence_buckets and len(confluence_buckets[key]) >= min_sample_size:
                win_rate = sum(confluence_buckets[key]) / len(confluence_buckets[key])
                if win_rate >= 0.55:
                    thresholds["min_confluences_standard"] = count
                    break

    def _learn_alignment_thresholds(
        self,
        all_experiences: dict[str, Any],
        min_sample_size: int,
        thresholds: dict[str, Any],
    ) -> None:
        """Learn position reduction thresholds from timeframe alignment performance."""
        if not all_experiences or not all_experiences.get("metadatas"):
            return
        alignment_pnl: dict[str, list[float]] = {"ALIGNED": [], "MIXED": [], "DIVERGENT": []}
        for meta in all_experiences["metadatas"]:
            alignment = meta.get("timeframe_alignment")
            pnl = meta.get("pnl_pct", 0)
            if alignment in alignment_pnl:
                alignment_pnl[alignment].append(pnl)
        aligned_avg = (
            sum(alignment_pnl["ALIGNED"]) / len(alignment_pnl["ALIGNED"])
            if alignment_pnl["ALIGNED"] else 0
        )
        if len(alignment_pnl["MIXED"]) >= min_sample_size and aligned_avg > 0:
            mixed_avg = sum(alignment_pnl["MIXED"]) / len(alignment_pnl["MIXED"])
            if mixed_avg < aligned_avg:
                reduction = min(0.40, max(0.10, 1 - (mixed_avg / aligned_avg)))
                thresholds["position_reduce_mixed"] = round(reduction, 2)
        if len(alignment_pnl["DIVERGENT"]) >= min_sample_size and aligned_avg > 0:
            divergent_avg = sum(alignment_pnl["DIVERGENT"]) / len(alignment_pnl["DIVERGENT"])
            if divergent_avg < aligned_avg:
                reduction = min(0.50, max(0.20, 1 - (divergent_avg / aligned_avg)))
                thresholds["position_reduce_divergent"] = round(reduction, 2)

    def _learn_sl_tightening_threshold(
        self,
        raw_snapshot: dict[str, Any] | None,
        min_sample_size: int,
        thresholds: dict[str, Any],
    ) -> None:
        """Learn optimal SL tightening progress threshold from paired update/close outcomes.

        Pairs each accepted SL-tightening UPDATE record with the eventual WIN/LOSS close
        for that position, then scans candidate thresholds to find the lowest value with
        positive expectancy.
        """
        if not raw_snapshot:
            return
        raw_ids: list[str] = raw_snapshot.get("ids") or []
        raw_metas: list[dict] = raw_snapshot.get("metadatas") or []
        if not raw_metas:
            return

        close_lookup: dict[str, dict] = {}
        update_records: list[dict] = []
        for idx, meta in enumerate(raw_metas):
            outcome = meta.get("outcome")
            if outcome in ("WIN", "LOSS"):
                trade_id = raw_ids[idx] if idx < len(raw_ids) else ""
                pid = meta.get("position_id", "")
                pet_id = meta.get("position_entry_trade_id", "")
                if trade_id:
                    close_lookup[trade_id] = meta
                if pid:
                    close_lookup.setdefault(pid, meta)
                if pet_id:
                    close_lookup.setdefault(pet_id, meta)
            elif outcome == "UPDATE":
                action = meta.get("action_type", "")
                if action in ("SL_TRAIL", "BOTH") and meta.get("is_tightening"):
                    pp = meta.get("price_progress")
                    if pp is not None and isinstance(pp, (int, float)) and pp == pp:
                        update_records.append(meta)

        if not update_records or not close_lookup:
            return

        best_per_position: dict[str, dict] = {}
        for meta in update_records:
            pos_id = meta.get("position_id", "")
            if not pos_id:
                continue
            existing = best_per_position.get(pos_id)
            if existing is None or meta.get("price_progress", 1.0) < existing.get("price_progress", 1.0):
                best_per_position[pos_id] = meta

        pairs: list[tuple[float, float, bool]] = []
        for pos_id, update_meta in best_per_position.items():
            close_meta = (
                close_lookup.get(pos_id)
                or close_lookup.get(update_meta.get("position_entry_trade_id", ""))
            )
            if close_meta is None:
                continue
            close_pnl = close_meta.get("pnl_pct", 0.0)
            is_win = close_meta.get("outcome") == "WIN"
            pp = update_meta.get("price_progress", 0.0)
            pairs.append((float(pp), float(close_pnl), is_win))

        if not pairs:
            return

        candidates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        learned_threshold: float | None = None
        best_stats: dict[str, Any] = {}
        for candidate in candidates:
            eligible = [(pp, pnl, w) for pp, pnl, w in pairs if pp >= candidate]
            if len(eligible) < min_sample_size:
                continue
            wins = [pnl for _, pnl, w in eligible if w]
            losses = [pnl for _, pnl, w in eligible if not w]
            win_rate = len(wins) / len(eligible)
            avg_win = sum(wins) / len(wins) if wins else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
            if expectancy > 0:
                learned_threshold = candidate
                best_stats = {
                    "learned_threshold": candidate,
                    "sample_count": len(pairs),
                    "paired_sample_count": len(eligible),
                    "win_rate": round(win_rate, 4),
                    "avg_win_pct": round(avg_win, 4),
                    "avg_loss_pct": round(avg_loss, 4),
                    "expectancy_pct": round(expectancy, 4),
                    "source": "brain",
                    "basis": "paired_update_outcomes",
                }
                break

        if learned_threshold is not None:
            thresholds["sl_tightening"] = best_stats

    def get_confidence_recommendation(self, min_sample_size: int = 5) -> str | None:
        """Generate recommendation based on confidence calibration."""
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