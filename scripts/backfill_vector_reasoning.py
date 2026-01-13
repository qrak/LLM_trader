"""Backfill script to update vector database entries with N/A reasoning.

This script retrieves the correct reasoning from trade_history.json for trades
that were stored in the vector database with "N/A" as reasoning.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from datetime import datetime
from src.logger.logger import Logger
from src.trading.vector_memory import VectorMemoryService


def extract_trade_id_timestamp(trade_id: str) -> datetime:
    """Extract timestamp from trade ID format: trade_2026-01-12T17:00:42.993078"""
    try:
        # Remove 'trade_' prefix
        timestamp_str = trade_id.replace("trade_", "")
        return datetime.fromisoformat(timestamp_str)
    except Exception as e:
        raise ValueError(f"Could not parse trade ID: {trade_id}") from e


def load_trade_history(data_dir: Path) -> list:
    """Load trade history from JSON file."""
    history_file = data_dir / "trading" / "trade_history.json"
    if not history_file.exists():
        raise FileNotFoundError(f"Trade history not found at {history_file}")
    
    with open(history_file, 'r') as f:
        return json.load(f)


def find_entry_reasoning(trade_timestamp: datetime, trade_history: list) -> str:
    """Find the entry reasoning for a trade by matching timestamp."""
    entry_actions = {"BUY", "SELL"}
    
    for decision in trade_history:
        action = decision.get("action", "")
        timestamp_str = decision.get("timestamp", "")
        
        if action in entry_actions and timestamp_str:
            decision_time = datetime.fromisoformat(timestamp_str)
            
            # Match by timestamp (allowing 1 second tolerance)
            time_diff = abs((decision_time - trade_timestamp).total_seconds())
            if time_diff < 1.0:
                reasoning = decision.get("reasoning", "")
                return reasoning if reasoning else "No reasoning available"
    
    return None


def backfill_vector_reasoning():
    """Main backfill function."""
    logger = Logger(logger_name="Backfill", logger_debug=True)
    logger.info("Starting vector reasoning backfill...")
    
    # Initialize vector memory (use the same path as brain_service in start.py)
    data_dir = Path("data")
    brain_path = data_dir / "trading" / "brain_BTC_USDC_4h"
    logger.info(f"Using brain path: {brain_path}")
    
    vector_memory = VectorMemoryService(
        logger=logger,
        data_dir=str(brain_path)
    )
    
    if not vector_memory._ensure_initialized():
        logger.error("Failed to initialize vector memory service")
        return
    
    # Load trade history
    try:
        trade_history = load_trade_history(data_dir)
        logger.info(f"Loaded {len(trade_history)} entries from trade history")
    except Exception as e:
        logger.error(f"Failed to load trade history: {e}")
        return
    
    # Get all experiences from vector database
    try:
        all_experiences = vector_memory._collection.get(include=["documents", "metadatas"])
        if not all_experiences or not all_experiences["ids"]:
            logger.warning("No experiences found in vector database")
            return
        
        logger.info(f"Found {len(all_experiences['ids'])} experiences in vector database")
    except Exception as e:
        logger.error(f"Failed to retrieve experiences: {e}")
        return
    
    # Find and update N/A entries
    updated_count = 0
    not_found_count = 0
    
    for i, trade_id in enumerate(all_experiences["ids"]):
        document = all_experiences["documents"][i]
        metadata = all_experiences["metadatas"][i]
        
        # Check if reasoning is N/A
        reasoning = metadata.get("reasoning", "")
        if reasoning == "N/A":
            try:
                # Extract timestamp from trade ID
                trade_timestamp = extract_trade_id_timestamp(trade_id)
                
                # Find matching entry in trade history
                correct_reasoning = find_entry_reasoning(trade_timestamp, trade_history)
                
                if correct_reasoning:
                    # Update metadata
                    metadata["reasoning"] = correct_reasoning
                    
                    # Rebuild document with correct reasoning
                    direction = metadata.get("direction", "UNKNOWN")
                    market_context = metadata.get("market_context", "")
                    outcome = metadata.get("outcome", "UNKNOWN")
                    pnl_pct = metadata.get("pnl_pct", 0)
                    confidence = metadata.get("confidence", "MEDIUM")
                    
                    new_document = (
                        f"{direction} trade. Market: {market_context}. "
                        f"Result: {outcome} ({pnl_pct:+.2f}%). "
                        f"Confidence: {confidence}. Reasoning: {correct_reasoning}"
                    )
                    
                    # Re-encode document
                    embedding = vector_memory._embedding_model.encode(new_document).tolist()
                    
                    # Update in vector database
                    vector_memory._collection.upsert(
                        ids=[trade_id],
                        embeddings=[embedding],
                        documents=[new_document],
                        metadatas=[metadata]
                    )
                    
                    updated_count += 1
                    logger.info(f"✓ Updated {trade_id} with reasoning: {correct_reasoning[:80]}...")
                else:
                    not_found_count += 1
                    logger.warning(f"✗ Could not find reasoning for {trade_id} ({trade_timestamp})")
                    
            except Exception as e:
                logger.error(f"Error processing {trade_id}: {e}")
                continue
    
    logger.info("=" * 80)
    logger.info(f"Backfill complete!")
    logger.info(f"  - Total experiences: {len(all_experiences['ids'])}")
    logger.info(f"  - Updated: {updated_count}")
    logger.info(f"  - Not found in history: {not_found_count}")
    logger.info(f"  - Already had reasoning: {len(all_experiences['ids']) - updated_count - not_found_count}")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        backfill_vector_reasoning()
    except KeyboardInterrupt:
        print("\nBackfill interrupted by user")
    except Exception as e:
        print(f"Backfill failed: {e}")
        import traceback
        traceback.print_exc()
