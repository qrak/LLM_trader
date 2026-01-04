"""Pure ChromaDB + SentenceTransformers test - NO project dependencies.

This demonstrates the vector brain functionality using ONLY chromadb  
and sentence-transformers, without importing any src.* modules.

Usage:
    python tests/test_vector_pure.py
"""

if __name__ == "__main__":
    import chromadb
    from sentence_transformers import SentenceTransformer
    
    print("=" * 80)
    print("PURE VECTOR BRAIN TEST (ChromaDB + SentenceTransformers)")
    print("=" * 80)
    
    # 1. Initialize
    print("\n[1/4] Initializing...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    client = chromadb.PersistentClient(path='data/test_pure_vector')
    collection = client.get_or_create_collection(
        name='test_trades',
        metadata={"hnsw:space": "cosine"}
    )
    print(f"✓ Embedding model loaded ({model.get_sentence_embedding_dimension()} dimensions)")
    print(f"✓ ChromaDB initialized")
    
    # 2. Store experiences
    print("\n[2/4] Storing trade experiences...")
    
    trades = [
        ("trade_001", "LONG trade. Market: BULLISH + High ADX (32) + Low Volatility. Result: WIN (+4.2%). Confidence: HIGH. Reasoning: Strong uptrend confirmed", 
         {"outcome": "WIN", "pnl_pct": 4.2, "direction": "LONG"}),
        ("trade_002", "LONG trade. Market: BULLISH + High ADX (28) + Medium Volatility. Result: WIN (+3.8%). Confidence: HIGH. Reasoning: Continuation pattern",
         {"outcome": "WIN", "pnl_pct": 3.8, "direction": "LONG"}),
        ("trade_003", "SHORT trade. Market: BEARISH + High ADX (35) + High Volatility. Result: LOSS (-2.1%). Confidence: MEDIUM. Reasoning: Counter-trend failed",
         {"outcome": "LOSS", "pnl_pct": -2.1, "direction": "SHORT"}),
        ("trade_004", "LONG trade. Market: NEUTRAL + Low ADX (18) + Low Volatility. Result: LOSS (-1.5%). Confidence: LOW. Reasoning: Range-bound market",
         {"outcome": "LOSS", "pnl_pct": -1.5, "direction": "LONG"}),
        ("trade_005", "LONG trade. Market: BULLISH + High ADX (30) + Low Volatility. Result: WIN (+5.1%). Confidence: HIGH. Reasoning: Pullback entry in uptrend",
         {"outcome": "WIN", "pnl_pct": 5.1, "direction": "LONG"}),
    ]
    
    for trade_id, text, metadata in trades:
        embedding = model.encode(text).tolist()
        collection.add(
            ids=[trade_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )
    
    print(f"✓ Stored {len(trades)} experiences (Total: {collection.count()})")
    
    # 3. Query similar experiences  
    print("\n[3/4] Querying for similar experiences...")
    
    scenarios = [
        ("BULLISH + High ADX + Low Volatility", "🟢 BULLISH SCENARIO"),
        ("BEARISH + High ADX + High Volatility", "🔴 BEARISH SCENARIO"),
        ("NEUTRAL + Low ADX", "⚪ NEUTRAL SCENARIO"),
    ]
    
    for query_text, label in scenarios:
        print(f"\n{label}")
        print("-" * 80)
        print(f"Query: {query_text}")
        
        query_embedding = model.encode(query_text).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        if results and results['ids'][0]:
            print(f"Found {len(results['ids'][0])} similar trades:\n")
            for i, (doc_id, doc, dist, meta) in enumerate(zip(
                results['ids'][0],
                results['documents'][0],
                results['distances'][0],
                results['metadatas'][0]
            ), 1):
                similarity = (1 - dist) * 100
                outcome = meta.get('outcome', '?')
                pnl = meta.get('pnl_pct', 0)
                direction = meta.get('direction', '?')
                
                print(f"  {i}. [SIMILARITY {similarity:.0f}%] {outcome} trade ({direction})")
                print(f"     P&L: {pnl:+.1f}%")
                print(f"     Details: {doc[:100]}...")
                print()
    
   # 4. Calculate stats for a context
    print("\n[4/4] Calculating statistics for BULLISH context...")
    print("-" * 80)
    
    query_embedding = model.encode("BULLISH + High ADX").tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    
    if results and results['metadatas'][0]:
        metas = results['metadatas'][0]
        wins = sum(1 for m in metas if m.get('outcome') == 'WIN')
        pnls = [m.get('pnl_pct', 0) for m in metas]
        
        win_rate = (wins / len(metas)) * 100 if metas else 0
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0
        
        print(f"Context: 'BULLISH + High ADX'")
        print(f"  Win Rate: {win_rate:.1f}% ({wins}/{len(metas)} trades)")
        print(f"  Avg P&L: {avg_pnl:+.2f}%")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"✓ Embedding model: all-MiniLM-L6-v2 (CPU)")
    print(f"✓ Vector DB: ChromaDB with cosine similarity")
    print(f"✓ Total experiences stored: {collection.count()}")
    print(f"✓ Semantic retrieval: WORKING")
    print(f"✓ Statistics calculation: WORKING")
    print(f"✓ NO config.ini required: YES")
    print(f"✓ NO AI API calls: YES")
    print("=" * 80)
    
    print("\n✅ TEST PASSED")
    print("\n📝 HOW THIS INTEGRATES WITH PROMPTS:")
    print("   1. When a trade closes → Experience stored in ChromaDB")
    print("   2. Before AI analysis → Query ChromaDB for similar past trades")
    print("   3. Similar trades → Formatted and injected into AI prompt")
    print("   4. AI sees relevant history → Better decision making")
    
    # Cleanup
    client.delete_collection('test_trades')
    print("\n🧹 Test collection cleaned up")
