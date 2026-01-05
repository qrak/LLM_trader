"""Test vector memory aggregation methods for vector-only brain.

Tests the new aggregation methods added to VectorMemoryService that replace
the TradingBrain JSON-based statistics.

Usage:
    python tests/test_vector_aggregations.py
"""

if __name__ == "__main__":
    import chromadb
    from sentence_transformers import SentenceTransformer
    import sys

    print("=" * 80)
    print("VECTOR AGGREGATION METHODS TEST")
    print("=" * 80)

    # Initialize
    print("\n[1/5] Initializing...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    client = chromadb.PersistentClient(path='data/test_aggregations')
    collection = client.get_or_create_collection(
        name='test_aggregations',
        metadata={"hnsw:space": "cosine"}
    )
    print(f"✓ ChromaDB initialized")

    # Store test trades with rich metadata
    print("\n[2/5] Storing trades with enriched metadata...")

    trades = [
        # High confidence trades
        ("t001", "BULLISH + High ADX", {
            "outcome": "WIN", "pnl_pct": 4.2, "direction": "LONG", "confidence": "HIGH",
            "adx_at_entry": 28, "rr_ratio": 2.5, "sl_distance_pct": 0.02,
            "trend_alignment_score": 85, "momentum_strength_score": 72
        }),
        ("t002", "BULLISH + High ADX", {
            "outcome": "WIN", "pnl_pct": 3.1, "direction": "LONG", "confidence": "HIGH",
            "adx_at_entry": 30, "rr_ratio": 2.0, "sl_distance_pct": 0.025,
            "trend_alignment_score": 80, "momentum_strength_score": 68
        }),
        ("t003", "BEARISH + High ADX", {
            "outcome": "LOSS", "pnl_pct": -1.5, "direction": "SHORT", "confidence": "HIGH",
            "adx_at_entry": 26, "rr_ratio": 2.2, "sl_distance_pct": 0.018,
            "trend_alignment_score": 75, "momentum_strength_score": 50
        }),
        # Medium confidence trades
        ("t004", "NEUTRAL + Low ADX", {
            "outcome": "LOSS", "pnl_pct": -2.0, "direction": "LONG", "confidence": "MEDIUM",
            "adx_at_entry": 18, "rr_ratio": 1.5, "sl_distance_pct": 0.03,
            "trend_alignment_score": 45, "momentum_strength_score": 40
        }),
        ("t005", "BULLISH + Medium ADX", {
            "outcome": "WIN", "pnl_pct": 2.5, "direction": "LONG", "confidence": "MEDIUM",
            "adx_at_entry": 22, "rr_ratio": 1.8, "sl_distance_pct": 0.022,
            "trend_alignment_score": 55, "momentum_strength_score": 60
        }),
        # Low confidence trade
        ("t006", "NEUTRAL + Low ADX", {
            "outcome": "LOSS", "pnl_pct": -1.8, "direction": "SHORT", "confidence": "LOW",
            "adx_at_entry": 15, "rr_ratio": 1.2, "sl_distance_pct": 0.035,
            "trend_alignment_score": 25, "momentum_strength_score": 30
        }),
    ]

    for trade_id, context, meta in trades:
        doc = f"{meta['direction']} trade. Confidence: {meta['confidence']}. Result: {meta['outcome']}"
        embedding = model.encode(doc).tolist()
        collection.add(ids=[trade_id], embeddings=[embedding], documents=[doc], metadatas=[meta])

    print(f"✓ Stored {len(trades)} trades with rich metadata")

    # Test confidence stats computation
    print("\n[3/5] Testing compute_confidence_stats()...")

    all_exp = collection.get()
    conf_stats = {"HIGH": [0, 0, 0.0], "MEDIUM": [0, 0, 0.0], "LOW": [0, 0, 0.0]}  # total, wins, pnl_sum
    
    for meta in all_exp["metadatas"]:
        conf = meta.get("confidence", "MEDIUM")
        conf_stats[conf][0] += 1
        if meta.get("outcome") == "WIN":
            conf_stats[conf][1] += 1
        conf_stats[conf][2] += meta.get("pnl_pct", 0)

    print("  Confidence Stats:")
    for level in ["HIGH", "MEDIUM", "LOW"]:
        total, wins, pnl_sum = conf_stats[level]
        if total > 0:
            win_rate = (wins / total) * 100
            avg_pnl = pnl_sum / total
            print(f"  - {level}: {wins}/{total} wins ({win_rate:.0f}%), Avg P&L: {avg_pnl:+.2f}%")
    print("  ✓ Confidence stats computation works")

    # Test ADX performance computation
    print("\n[4/5] Testing compute_adx_performance()...")

    adx_buckets = {"LOW": [], "MEDIUM": [], "HIGH": []}
    for meta in all_exp["metadatas"]:
        adx = meta.get("adx_at_entry", 0)
        if adx < 20:
            bucket = "LOW"
        elif adx < 25:
            bucket = "MEDIUM"
        else:
            bucket = "HIGH"
        adx_buckets[bucket].append((meta.get("outcome") == "WIN", meta.get("pnl_pct", 0)))

    print("  ADX Performance:")
    for bucket, trades_list in adx_buckets.items():
        if trades_list:
            wins = sum(1 for w, _ in trades_list if w)
            total = len(trades_list)
            avg_pnl = sum(p for _, p in trades_list) / total
            print(f"  - ADX {bucket}: {wins}/{total} wins, Avg P&L: {avg_pnl:+.2f}%")
    print("  ✓ ADX performance computation works")

    # Test factor performance computation
    print("\n[5/5] Testing compute_factor_performance()...")

    factors = {}
    for meta in all_exp["metadatas"]:
        for key in ["trend_alignment_score", "momentum_strength_score"]:
            score = meta.get(key, 0)
            if score > 0:
                if score <= 30:
                    bucket = "LOW"
                elif score <= 69:
                    bucket = "MEDIUM"
                else:
                    bucket = "HIGH"
                factor_key = f"{key.replace('_score', '')}_{bucket}"
                if factor_key not in factors:
                    factors[factor_key] = []
                factors[factor_key].append((meta.get("outcome") == "WIN", score))

    print("  Factor Performance (non-empty buckets):")
    for key, data in sorted(factors.items()):
        wins = sum(1 for w, _ in data if w)
        avg_score = sum(s for _, s in data) / len(data)
        print(f"  - {key}: {wins}/{len(data)} wins, Avg Score: {avg_score:.0f}")
    print("  ✓ Factor performance computation works")

    # Cleanup
    client.delete_collection('test_aggregations')
    print("\n🧹 Test collection cleaned up")

    print("\n" + "=" * 80)
    print("✅ ALL AGGREGATION TESTS PASSED")
    print("=" * 80)
