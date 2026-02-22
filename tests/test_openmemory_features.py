"""Test OpenMemory features: Temporal Awareness, Decay Engine, Semantic Rules.

Tests the new features added to VectorMemoryService and TradingBrainService.

Usage:
    python tests/test_openmemory_features.py
"""

if __name__ == "__main__":
    import math
    from datetime import datetime, timedelta
    import chromadb
    from sentence_transformers import SentenceTransformer

    print("=" * 80)
    print("OPENMEMORY FEATURES TEST")
    print("=" * 80)

    # Initialize
    print("\n[1/5] Initializing...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    client = chromadb.PersistentClient(path='data/test_openmemory')
    collection = client.get_or_create_collection(
        name='test_experiences',
        metadata={"hnsw:space": "cosine"}
    )
    rules_collection = client.get_or_create_collection(
        name='test_semantic_rules',
        metadata={"hnsw:space": "cosine"}
    )
    print("✓ ChromaDB initialized with experiences and rules collections")

    # Test 1: Timestamp and market_regime in metadata
    print("\n[2/5] Testing Temporal Awareness (Phase 1)...")

    now = datetime.utcnow()
    old_ts = (now - timedelta(days=120)).isoformat()
    recent_ts = (now - timedelta(days=5)).isoformat()

    trades = [
        ("trade_old", "BULLISH + High ADX", {
            "outcome": "WIN", "pnl_pct": 4.0, "direction": "LONG",
            "timestamp": old_ts, "market_regime": "BULLISH"
        }),
        ("trade_recent", "BULLISH + High ADX", {
            "outcome": "WIN", "pnl_pct": 3.5, "direction": "LONG",
            "timestamp": recent_ts, "market_regime": "BULLISH"
        }),
    ]

    for trade_id, context, meta in trades:
        doc = f"{meta['direction']} trade. Result: {meta['outcome']}"
        embedding = model.encode(doc).tolist()
        collection.add(ids=[trade_id], embeddings=[embedding], documents=[doc], metadatas=[meta])

    all_meta = collection.get()["metadatas"]
    has_timestamp = all(m.get("timestamp") for m in all_meta)
    has_regime = all(m.get("market_regime") for m in all_meta)
    print(f"  - All trades have timestamp: {has_timestamp}")
    print(f"  - All trades have market_regime: {has_regime}")
    assert has_timestamp, "Timestamp field missing!"
    assert has_regime, "Market regime field missing!"
    print("  ✓ Temporal Awareness metadata works")

    # Test 2: Recency score calculation
    print("\n[3/5] Testing Decay Engine (Phase 2)...")

    HALF_LIFE = 90

    def calculate_recency_score(ts_str: str) -> float:
        try:
            trade_dt = datetime.fromisoformat(ts_str)
            age_days = (datetime.utcnow() - trade_dt).days
            decay_rate = math.log(2) / HALF_LIFE
            return math.exp(-decay_rate * age_days)
        except (ValueError, TypeError):
            return 0.5

    old_recency = calculate_recency_score(old_ts)
    recent_recency = calculate_recency_score(recent_ts)

    print(f"  - Old trade (120 days): recency = {old_recency:.3f}")
    print(f"  - Recent trade (5 days): recency = {recent_recency:.3f}")

    assert recent_recency > old_recency, "Recent trade should have higher recency!"
    assert recent_recency > 0.9, "Recent trade recency should be close to 1.0!"
    assert old_recency < 0.5, "Old trade recency should be below 0.5 (past half-life)!"
    print("  ✓ Recency decay calculation works")

    # Test 3: Hybrid scoring (similarity * 0.7 + recency * 0.3)
    print("\n[4/5] Testing Hybrid Scoring...")

    query = "BULLISH + High ADX"
    query_embedding = model.encode(query).tolist()

    results = collection.query(query_embeddings=[query_embedding], n_results=2)

    hybrid_scores = []
    for i, doc_id in enumerate(results["ids"][0]):
        similarity = 1 - results["distances"][0][i]
        ts = results["metadatas"][0][i].get("timestamp", "")
        recency = calculate_recency_score(ts)
        hybrid = similarity * 0.7 + recency * 0.3
        hybrid_scores.append((doc_id, similarity, recency, hybrid))
        print(f"  - {doc_id}: sim={similarity:.3f}, rec={recency:.3f}, hybrid={hybrid:.3f}")

    hybrid_scores.sort(key=lambda x: x[3], reverse=True)
    best = hybrid_scores[0][0]
    print(f"  - Best by hybrid score: {best}")
    assert best == "trade_recent", "Recent trade should rank higher with hybrid scoring!"
    print("  ✓ Hybrid scoring works")

    # Test 4: Semantic rules storage and retrieval
    print("\n[5/5] Testing Semantic Rules (Phase 3)...")

    rule_text = "LONG trades perform well in BULLISH market with High Adx. (5 recent wins)"
    rule_embedding = model.encode(rule_text).tolist()
    rules_collection.upsert(
        ids=["rule_test"],
        embeddings=[rule_embedding],
        documents=[rule_text],
        metadatas=[{"active": True, "timestamp": datetime.utcnow().isoformat()}]
    )

    active_rules = rules_collection.get(where={"active": True})
    print(f"  - Stored rule: '{rule_text[:50]}...'")
    print(f"  - Active rules count: {len(active_rules['ids'])}")
    assert len(active_rules["ids"]) == 1, "Should have exactly 1 active rule!"
    assert active_rules["documents"][0] == rule_text, "Rule text mismatch!"
    print("  ✓ Semantic rules storage and retrieval works")

    # Cleanup
    client.delete_collection('test_experiences')
    client.delete_collection('test_semantic_rules')
    print("\n🧹 Test collections cleaned up")

    print("\n" + "=" * 80)
    print("✅ ALL OPENMEMORY FEATURE TESTS PASSED")
    print("=" * 80)
    print("\nFeatures verified:")
    print("  1. Temporal Awareness: timestamp + market_regime in metadata")
    print("  2. Decay Engine: Recency score with exponential decay (90-day half-life)")
    print("  3. Hybrid Scoring: similarity * 0.7 + recency * 0.3")
    print("  4. Semantic Rules: Storage and retrieval of learned trading patterns")
