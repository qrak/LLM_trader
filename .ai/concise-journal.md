# Concise Journal ✂️ — LLM_trader Code Line Reduction

## 2026-07-28 — Dimension Mismatch Error Predicate & Clean Collection Recovery

- **Target Files:** `src/trading/vector_memory.py`, `src/rag/code_vector_index.py`
- **Abstraction Pattern Used:** Concise tuple-membership error predicate (`any(kw in err_msg for kw in (...))`) + safe collection recreation helper (`_get_or_create_clean_collection`).
- **LOC Impact:** Centralized dimension mismatch recovery in 15 lines per service while removing duplicate try-except cascades.
- **Verification:** `pytest` passed 100% clean (48/48 vector memory, 134/134 dashboard & brain tests), `ruff` clean.
