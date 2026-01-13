"""Quick script to check current vector memory insights."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger.logger import Logger
from src.trading.vector_memory import VectorMemoryService

logger = Logger(logger_name="Check", logger_debug=False)
vm = VectorMemoryService(logger, data_dir="data/trading/brain_BTC_USDC_4h")
vm._ensure_initialized()

# Get all experiences
all_exp = vm._collection.get(include=["documents", "metadatas"])

output_lines = []
output_lines.append("=" * 80)
output_lines.append(f"TOTAL EXPERIENCES: {len(all_exp['ids'])}")
output_lines.append("=" * 80)

for i, trade_id in enumerate(all_exp["ids"]):
    meta = all_exp["metadatas"][i]
    reasoning = meta.get("reasoning", "N/A")
    outcome = meta.get("outcome", "?")
    pnl = meta.get("pnl_pct", 0)
    direction = meta.get("direction", "?")
    context = meta.get("market_context", "N/A")
    
    output_lines.append(f"\n{i+1}. {trade_id}")
    output_lines.append(f"   Outcome: {outcome} | P&L: {pnl:+.2f}% | Direction: {direction}")
    output_lines.append(f"   Context: {context}")
    output_lines.append(f"   Reasoning: {reasoning}")
    output_lines.append("-" * 80)

# Check semantic rules
output_lines.append(f"\nSEMANTIC RULES COUNT: {vm.semantic_rule_count}")
if vm.semantic_rule_count > 0:
    rules = vm.get_active_rules(n_results=5)
    for r in rules:
        output_lines.append(f"  - {r['text']}")

# Write to file
with open("scripts/insights_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print("Report written to scripts/insights_report.txt")
