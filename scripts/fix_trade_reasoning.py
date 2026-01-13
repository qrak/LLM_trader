"""Fix trade_2026-01-08 with correct reasoning from trade history."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger.logger import Logger
from src.trading.vector_memory import VectorMemoryService

logger = Logger(logger_name="Fix", logger_debug=False)
vm = VectorMemoryService(logger, data_dir="data/trading/brain_BTC_USDC_4h")
vm._ensure_initialized()

trade_id = "trade_2026-01-08T17:02:00"
correct_reasoning = "Bitcoin confirmed the 200-SMA support at $89.2k with a Stochastic bull cross and a TD9 buy signal, suggesting a high-probability mean-reversion move toward $94.8k."

# Get existing data
results = vm._collection.get(ids=[trade_id], include=["documents", "metadatas"])
if results and results["ids"]:
    meta = results["metadatas"][0]
    meta["reasoning"] = correct_reasoning
    
    # Rebuild document
    pnl = meta.get("pnl_pct", 0)
    doc = f"{meta['direction']} trade. Market: {meta['market_context']}. Result: {meta['outcome']} ({pnl:+.2f}%). Confidence: {meta['confidence']}. Reasoning: {correct_reasoning}"
    
    # Re-encode and update
    embedding = vm._embedding_model.encode(doc).tolist()
    vm._collection.upsert(ids=[trade_id], embeddings=[embedding], documents=[doc], metadatas=[meta])
    print("SUCCESS: Updated trade_2026-01-08T17:02:00 with correct reasoning")
    print(f"New reasoning: {correct_reasoning}")
else:
    print("Trade not found")
